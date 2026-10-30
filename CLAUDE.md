# CLAUDE.md: SCOUT CV model

Instructions for Claude working in this repo. The goal is the **most accurate cat detector with the fewest false positives** that still runs on a Raspberry Pi 5. Every experiment should be reproducible, comparable and documented.

## Project in one paragraph

SCOUT is an edge-AI add-on for humane TNR live traps (CS 4605, Georgia Tech, Fall 2026, Group 29).
- A Pi camera feeds YOLO26n (NCNN) on a Raspberry Pi 5.
- `edge/vision.py` turns detections into a `VisionReport` using per-class cutoffs, one label per animal, trap/doorway zones and a K-of-N vote.
- The hardware team's `edge/gate.py` closes the gate only when there is a cat in the trap **and** the pressure plate is pressed.
- No live animals are trapped this semester. Trap trials are staged with props.

Model team: Sudiksha Das, Henry Poulin. Hardware team: Keshav Sheth, Sowmiyan Nattamai Vijayakumar. Mentor: Christine Taylor.

The source of truth for the plan is the Google doc [SCOUT - Research/Notes](https://docs.google.com/document/d/1BFhBik9DCZgdWnCy8yDcnWfe-6-JuTAZf1hQFuI_Fq8/edit). Its **CV Model Architecture n Plan** tab is the plan; its **CV Team Updates** tab is the log and runbook.

## Targets

| Metric | Target | Measured on |
| --- | --- | --- |
| Cat recall | ≥ 75% | real-animal public pictures from **unseen camera locations** |
| Non-cat false-positive rate | ≤ 25% (aim far lower) | same |
| Decision time | < 5 s (aim < 2 s) | the Pi 5, camera to report |
| Staged trials | correct gate action in ≥ 90% of 50+ trials; never "cat in trap" while in doorway | props in the real trap |

Optimize recall first until 75% is met, while keeping FPR as low as possible. A false positive traps a raccoon or opossum, so do not spend the whole 25% budget just because it is allowed. After 75% is met, push FPR down at fixed recall.

## Current state (Oct 30)

- **Best model: E1b**, YOLO26n fine-tuned on Caltech Camera Traps + ENA24 for 80 epochs. Weights are at `results/E1b/weights/best.pt` on PACE only.
- **Scores (500 val images, 208 cats):**
  - cutoff 0.05: 73.1% recall, 9.9% FPR
  - cutoff 0.25: 69.2% recall, 4.8% FPR
- **E1b miss analysis** (`results/E1b_analysis/summary.json`, cutoff 0.25):
  - 64 cats missed. 47 were found but **given the wrong label**, mostly `other_animal`, `raccoon` and `opossum`. 14 were not found, 3 had low confidence.
  - The model finds *some* animal in 91.8% of cat images. Class confusion is the biggest lever.
  - Recall by size: tiny 50%, small 69%, medium 89%.
  - Daytime cats: 6 of 22 (27%). The sample is tiny, but it is a red flag.
- **Other runs:**
  - E1c (cat images ×2) did not beat E1b.
  - E2 (MobileNetV3 classifier) reached 27.9% recall. Dropped.
- **Trap settings** (`edge/vision_config.json`): cat cutoff 0.15, 3-of-5 vote. These were tuned for E1 and still need re-tuning for E1b with `vote_sim.py`.
- **Known gaps:**
  - The seen-location split has 0 cats (`results/E1_full_seen`), so the background-reliance gap is unmeasured.
  - The 500-image val sample covers only 3 cat camera locations.
  - Trap/doorway zones are placeholders.
  - No prop footage or Pi benchmark yet.

## Hard rules

1. **Never change `data/splits.json`** or the validation sample (`--split val --n 500 --seed 4605`). Comparability depends on them.
   - New eval sets are *added* alongside the old one, never swapped in.
2. **Never tune on the test split.** Choose models, cutoffs and votes on `val` (and held-out prop sessions). Run `--split test` once, at the model freeze (by **Nov 16**), and record it.
3. **One change per experiment.** Each run answers one question, so differences can be attributed to it.
4. **Never commit images, videos, weights or exported models.** `.gitignore` blocks them; don't work around it. Big files live on PACE `~/scratch/scout_data`.
5. **Prop footage never reports accuracy.**
   - Props check the camera angle, zones, vote and gate.
   - Accuracy is always reported on real-animal pictures.
   - Any model trained on prop footage (E5+) must be re-scored on the public val set too, so it can't trade real accuracy for prop accuracy.
6. **Respect ownership.** The model team owns `edge/vision.py`, `edge/vision_config.json`, `data/` and `training/`. The hardware team owns `edge/gate.py` and the hook functions in `edge/run_trap.py`. Don't change hardware-owned behavior; propose it instead.
7. **`VisionReport` is the contract between the teams.** Adding a field is fine. Renaming or removing one needs both teams to agree.
8. **Don't add ear-tip detection or mention it** anywhere in code or docs. It's out of scope.

## How to run an experiment

### Naming

- `E<n>` for a new idea; `E<n><letter>` for a variant of it (E1b, E1c).
- Outputs go in `results/<name>/`, `results/<name>_eval/`, `results/<name>_analysis/` and, where relevant, `results/<name>_vote/`.

### Before training, write down

- the question it answers
- the single change from its parent run
- the parent weights
- the expected effect

These go into the log entry (below).

### Train (on PACE)

Run jobs through `training/slurm/train_pace.sbatch`, never on the login node.

```bash
sbatch -J scout-E7 training/slurm/train_pace.sbatch E7 --weights results/E1b/weights/best.pt --epochs 40 [one change]
```

Useful `train_yolo.py` knobs: `--imgsz`, `--epochs`, `--patience`, `--batch`, `--cls`, `--mosaic`, `--close-mosaic`, `--data-yaml`, `--weights`, `--export-imgsz`.

### Evaluate

Every run gets the same three steps. `eval_pace.sbatch` does all of them:

```bash
sbatch training/slurm/eval_pace.sbatch E7
# eval.py (val, n=500, seed 4605, imgsz 640) -> analyze_errors.py -> vote_sim.py sweep
```

### Always report

| Field | Source |
| --- | --- |
| Cat recall and FPR at cutoff **0.05** and **0.25** | `threshold_sweep.csv` |
| Per-species FPR | `metrics.json` `fpr_by_species` |
| Miss reasons, plus recall by size and by day/night | `_analysis/summary.json` |
| Estimated trap-level recall/FPR at the chosen cutoff + K-of-N | `vote_sim.py` |
| Training time and epochs reached | `results/<name>/summary.json` |

Report both cutoffs every time; past logs mixed them, which made runs hard to compare.

### Decide

Promote a run to "best" only if it beats the current best on **val recall at a matched FPR** (or on FPR at matched recall). Also check that no single species' FPR jumps.

Note the gate-relevant species: raccoon, opossum, squirrel, dog.

## Optimization backlog (ranked by expected payoff)

Work top-down. Each item is one experiment unless noted.

1. **Re-tune cutoff + vote for E1b** (no training).
   - Run `python training/vote_sim.py sweep results/E1b_eval/threshold_sweep.csv`.
   - Update `edge/vision_config.json`, then run `python -m edge.stub --trials 50` and the tests.
   - Also revisit the non-cat cutoffs (0.4–0.5). They decide when a raccoon *holds the gate open*, which is the safety side.
2. **Fix cat ↔ other-animal confusion** (the 47-of-64 problem):
   - (a) Raise the classification loss weight (`--cls 1.0`) from E1b.
   - (b) Do hard-negative/hard-positive mining: oversample cat images E1b called `other_animal`/`raccoon`/`opossum`, and the non-cats it called cat. Build the list with `make_train_list.py`.
   - (c) Audit labels: check whether `common.py` maps any cat-like species (e.g. bobcat) to `other_animal` in a way that teaches the model "cat-shaped = other". Look at the contact sheets from `analyze_errors.py`.
   - (d) Backup: add a small cat/not-cat crop classifier on detected boxes. Only if (a)–(c) and E5 don't fix it, and only if the Pi latency budget allows.
3. **Small and tiny cats:** train/eval at `--imgsz 960`.
   - Trail-camera cats are small, but the trap camera is inches away, so only chase this if it also helps medium/large cats.
   - Always check the Pi speed cost (E6) before adopting a bigger input size.
4. **A bigger model (YOLO26s)** if latency allows.
   - Measure on the Pi with `edge/bench.py` first.
   - The budget is 3-of-5 frames well under 5 s.
5. **More real cat data:** add cat-rich LILA datasets alongside Caltech/ENA24, especially daytime and close-range cats.
   - Keep the location-based split rules.
   - New datasets without location info go to train only, like ENA24.
6. **E5: close-range prop footage from the real trap**, fine-tuned from the best model.
   - Requires the camera to be mounted.
   - Keep 1–2 whole sessions for test.
   - Balance cat props, non-cat props and empty trap in each session.
7. **E4: grayscale/IR robustness.** The NoIR camera at night is effectively black and white. Try grayscale augmentation, or a grayscale-only variant, once night footage exists.
8. **E6: Pi deployment check.** Export to NCNN at `imgsz` 416/320 and confirm accuracy holds after export. Re-run `eval.py` on the exported model.

Things already shown not to help: cat oversampling ×2 (E1c), and the MobileNetV3 classifier as the main model (E2).

## Evaluation rigor to-dos

- **Rebuild the seen-location split** so it contains cats: hold back some bursts from training cameras, then retrain. Report the seen-vs-new gap as the background-reliance check.
- **Score the best model on the full val split** (8,211 images, not just 500) to reduce noise. The 500-image sample has only 22 daytime cats and 3 cat locations. Keep reporting the 500-sample numbers for continuity.
- **Report day vs night separately.** Daytime recall looks weak.
- **Empty-image false positives** must stay ~0. Check `fpr_by_species.background`.
- **Test on real animal video** (Christine's videos, once sent). Split them into frame folders and run `vote_sim.py clips` to test the vote and gate on real motion. A frame-splitting script is still to be written.

## Testing

- Run `python -m pytest tests` before every commit. All tests must pass. The CI-equivalent dependency set is `requirements.txt`; the tests need `tqdm`.
- New logic in `data/`, `training/` or `edge/` gets a test in `tests/`.
- Any change to `edge/vision_config.json` or `edge/vision.py` also needs:
  - `python -m edge.stub --script c,c,c,c,c` (closes on frame K)
  - `python -m edge.stub --script c,c,r,c,c` (stays open)
  - `python -m edge.stub --trials 50`

## Documentation

- **Logging a run:** add an entry to the **CV Team Updates** tab (newest first) with:
  - date
  - question
  - change
  - results table (recall/FPR at 0.05 and 0.25)
  - what the miss analysis says
  - decision and next step
- **Update when a result changes them:** the README's *CV results* table and *Status*, and section 9 (Experiments) of the CV Model tab.
- **File headers:** every code file starts with a short header covering what it does, its status, and what still needs to change. Keep it current when editing.
- **Style:** plain language, short sentences, tables over paragraphs, concrete numbers. No filler, and no ear-tip mentions.
- **Commit small files only:** `metrics.json`, `threshold_sweep.csv`, `summary.json`, `vote_sweep.csv`.

## Git workflow

- Branch from `main` as `cv/<topic>` (e.g. `cv/e7-cls-loss`) and open a PR into `main`. Never push to `main` directly.
- Run `git pull` before starting. PACE has no GitHub login, so copy result folders to a laptop and commit from there.
- One PR per experiment or feature, with the results table in the PR description.
- Slurm job scripts live in `training/slurm/`, not the repo root.

## PACE ICE notes

- Use account `ece` and partition `coe-gpu` (see `training/slurm/e1_full.sbatch`).
- Data lives in `~/scratch/scout_data`. Jobs read `data/scout_pace.yaml`, which exists on PACE only.
- On out-of-memory errors, lower `--batch`. On time-limit errors, raise `--time` or lower `--epochs`.

## Key dates

| Date | Milestone |
| --- | --- |
| Sun Nov 1 | Halftime materials due |
| Nov 2–3 | Halftime presentation |
| Fri Nov 13 | **CP2:** camera → model → servo end to end on the bench |
| **Mon Nov 16** | **Model freeze:** final cutoff/vote chosen, test split run once. Sudiksha is away Nov 17–27 |
| Tue Nov 24 | **CP3:** 50+ staged prop trials logged |
| Dec 2–3, 7–8 | Demo Day |
| Sun Dec 13 | Final report |

With the freeze on Nov 16, prefer cheap, high-payoff experiments (backlog items 1–3) before slow ones.
