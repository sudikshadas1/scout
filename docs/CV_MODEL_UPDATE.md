# SCOUT CV model update

Experiment results, what limits the model, the plan to improve it, and how runs are evaluated. Updated Oct 30. The [README](../README.md) has the code map and setup. No model has improved yet: E7–E10 are written, not run.

## Results so far

Validation: 500 pictures (208 cats, 292 non-cats) from camera locations never used in training, scored at 640 pixels. The trap runs at 416 pixels, which hasn't been scored yet. Each cell is cats caught / false positives, where this table's false positives include the 67 empty pictures among the 292 non-cats, so runs compare with the old logs. **The goal counts animals only** (see below).

| Run | Change | Cutoff 0.05 | Cutoff 0.15 (trap now) | Cutoff 0.25 |
| --- | --- | --- | --- | --- |
| E0 | Stock YOLO26n | 5.3% / 5.1% | 4.8% / 2.7% | 4.3% / 2.1% |
| E1 | Fine-tuned on Caltech + ENA24 (stopped early at 24 of 30 epochs) | 63.5% / 4.5% | 59.6% / 2.4% | 54.8% / 2.1% |
| **E1b** | **Longer training (80 epochs), more patience, changed brightness and colour augmentation** | **73.1% / 9.9%** | **71.2% / 6.5%** | **69.2% / 4.8%** |
| E1c | E1b + cat images repeated 2× | 70.7% / 6.8% | 69.7% / 5.8% | 68.8% / 5.1% |
| E2 | MobileNetV3 classifier | 27.9% / 6.2% | 22.6% / 3.8% | 18.8% / 3.1% |

Planned: **E3** bigger model (YOLO26s), **E4** black-and-white training, **E5** prop footage from our trap, **E6** speed test on the Pi 3, **E7** Wellington data, **E8** crop check, **E9** higher classification loss, **E10** training at 416 pixels.

**What E1b's misses show (cutoff 0.25):**
- 47 of 64 missed cats were found but given the wrong label: other animal 18, raccoon 11, opossum 10, dog 5, squirrel 3.
- Small and tiny cats are missed most. **The set has no large cats**, and those are what our close-range trap camera should mostly see, so close-range recall is unmeasured.
- Only 6 of 22 colour (daytime) pictures of cats were caught, against 74% of black-and-white ones. The sample is small, but matching that would add about 10 cats, or most of the 12 needed for 75%. Training isn't short of daytime cats by the clock (925 of 2,071 Caltech cats), so the cause is still open.
- Recall ranges from 56% to 79% across the 3 cat locations, which is more than most experiments moved it.
- Most false positives come from `other_animal`, which is called a cat 12.9% of the time (9 of 70). Raccoon is 4.4%, opossum 3.0%, and dog, squirrel and empty images are 0%. Even the stock model E0 calls 8.6% of `other_animal` cats, so some of these may be true look-alikes.
- Repeating cats (E1c) only moved errors around: false positives on opossums went from 2 to 8, and missed cats called `other_animal` from 18 to 26.

**Settings.**
- The trap acts on a 3-of-5 vote at a 0.15 cutoff, not on single frames. Re-measure this on real and staged clips with `vote_sim.py clips`.
- If one box gets two labels, the weaker one is dropped only when it trails by at least 0.1 confidence. Close calls keep both, so the gate stays open.

## Reaching the goals

| | Goal | E1b today (cutoff 0.35) | Gap on the 500-picture set |
| --- | --- | --- | --- |
| Cats caught | at least 75% (aiming for 80%) | 67.8% (141 of 208) | 15 more cats for 75%, 26 more for 80% |
| False positives on animals | under 5%, as low as possible | 4.9% (11 of 225) | no room: a 12th false positive is 5.3% |

**The goal is about animals, so empty pictures don't count.** They made up 67 of the 292 non-cats and flattered the old number:

| E1b cutoff | Cats caught | False positives, with empties | On animals only |
| --- | --- | --- | --- |
| 0.15 (trap now) | 71.2% | 6.5% | about 8.4% |
| 0.25 | 69.2% | 4.8% | 6.2% (14 of 225) |
| 0.30 | 68.3% | 4.1% | 5.3% |
| **0.35** | **67.8%** | **3.8%** | **4.9% (11 of 225)** |
| 0.40 | 64.9% | 3.4% | 4.4% |

The 0.15 figure assumes no false positives on empty pictures there, which is confirmed only at 0.25. `eval.py` now reports `animal_fpr`.

**Changing the cutoff can't close this gap.** Under 5% on animals, the most E1b reaches is 67.8%. The model itself has to tell cats from other animals better, so the 47 mislabeled cats are the main target.

**Decision rule:** keep the model that reaches at least 75% of cats with under 5% of other animals called cats at 416 pixels and fits the time budget on the Pi 3. That model goes into E5 and is frozen by Nov 16.

## What limits the model

Counted from the public metadata; rebuilding the manifest matches PACE's totals, see `results/data_report.json`.

| | Training | Validation | Test |
| --- | --- | --- | --- |
| Pictures | 21,475 | 8,211 | 8,683 |
| Cat pictures | 2,553 | **208 (every val cat)** | 694 |
| Cameras with cats (Caltech) | **8; one has 71% of those cats** | 3 | 6 |
| Large cats | **7** | 0 | 15 |

- Scoring all 8,211 val pictures adds no cats. It only sharpens the false-positive number.
- `eval.py` looks at cat boxes only. The trap also drops weaker labels and holds the gate for a non-cat, so `score_pipeline.py` scores that too.
- Wellington Camera Traps (LILA) has 28,599 cat pictures with a usable MegaDetector box, from 187 cameras, 22,000 of them medium or large.

## Plan to improve the model

E0–E6 are from the CV Team Updates tab. E7–E10 are new: add them to the doc. The order below is the order to run them.

| Run | What changes | Why | Cost on the Pi 3 |
| --- | --- | --- | --- |
| E7 | Add 3,000 Wellington cats, plus non-cats and empties from the same cameras | Our cats come from too few cameras and are almost never large | None |
| E8 | Cat / not-cat check on each detected animal's crop | Most misses are found but mislabeled | One small classifier per detected animal; time it |
| E9 | Classification loss weight 1.0 | Most misses are wrong labels | None |
| E10 | Train at 416 pixels | Only if step 1 shows a drop at 416 | None |
| E3 | YOLO26s | Only if the speed test shows it fits | Slower, maybe too slow |
| E4 | Black-and-white training | NoIR camera, weak colour-picture cats | None |
| E5 | Prop footage, from the best model | Our camera angle and lighting | None |

1. **Get the Wellington data, then score E1b at the trap's size.** This also gives the trap-accurate score, the training curve and the new eval set.
   ```bash
   sbatch training/slurm/e7_data.sbatch
   sbatch training/slurm/eval_pace.sbatch E1b 500 416     # run after the data job finishes
   ```
2. **Re-tune the cutoff and vote** for under 5% false positives on animals (`animal_fpr`), from `results/E1b_pipeline416/pipeline_sweep.csv`, then update `edge/vision_config.json`.
3. **Look at the mistakes** in the contact sheets in `results/E1b_analysis416/` (on PACE). Are the `other_animal` false positives bobcats? Are the colour-picture misses from one camera?
4. **E6: Pi 3 speed.** The number decides the picture size and whether E3 fits.
5. **Run E7, E8 and E9.** They can run side by side. E7 is estimated at about 9.5 hours from E1's speed, so ask for more time, or use `--epochs 60` if PACE refuses.
   ```bash
   sbatch --time=12:00:00 -J scout-E7 training/slurm/train_pace.sbatch E7 --epochs 80 --patience 25 \
          --data-yaml ~/scratch/scout_data/scout_e7.yaml
   sbatch training/slurm/e8_crop.sbatch
   sbatch -J scout-E9 training/slurm/train_pace.sbatch E9 --cls 1.0 --epochs 80 --patience 25
   sbatch training/slurm/eval_pace.sbatch E7 500 416     # and E9, when each finishes
   ```
6. **E3, E4 and E10, as the results call for them.**
   ```bash
   sbatch -J scout-E3 training/slurm/train_pace.sbatch E3 --weights yolo26s.pt --epochs 80 --patience 25
   sbatch training/slurm/run.sbatch python data/make_grayscale.py --data ~/scratch/scout_data
   sbatch -J scout-E4 training/slurm/train_pace.sbatch E4 --epochs 80 --patience 25 --data-yaml ~/scratch/scout_data/scout_gray.yaml
   sbatch -J scout-E10 training/slurm/train_pace.sbatch E10 --imgsz 416 --epochs 80 --patience 25
   ```
   Score E4 on the original pictures with `eval_pace.sbatch E4 500 416`, and on the grayscale copy with `sbatch training/slurm/run.sbatch python training/eval.py --model results/E4/weights/best.pt --data ~/scratch/scout_data/gray --split val --n 500 --imgsz 416 --name E4_gray416`. If E4 wins, the Pi code must convert frames to grayscale first.
7. **Backyard videos: test the vote and gate on real motion** (see "Real-animal videos" below). Don't train on them as they are.
8. **Once the camera is on the trap:** footage pilot, prop labeling, then E5 from the best model.

## Risks to the plan

- **E7 could raise false positives.** Wellington has no raccoons or opossums, so 3,000 more cats take cats from 1.1× raccoons to 2.5×. When E1c raised the cats' share, opossum false positives went from 2 to 8. If E7 does well on cats and safely on animals, try more with `--cats`.
- **The trap misses large cats with the placeholder zones.** A cat's whole bottom edge must be inside the trap zone, so a close-up cat isn't "in trap" and also sets off the doorway check (`tests/test_vision.py` records this). Test with a close-up prop as soon as the camera is mounted, and change the rule if real zones don't fix it. `--full-frame` tests skip zones, so they won't catch it.
- **The Pi 3 is unmeasured.** The Pi code loads through Ultralytics and PyTorch on 1 GB of RAM. If that doesn't fit, the detector code has to change, and that comes before any experiment.
- **208 validation cats is thin.** Picking the best of many runs on them can flatter the winner, so use the Wellington set too and run the test split only once.

## How a run wins

A run wins if it beats E1b on cats caught with under 5% of other animals called cats at 416 pixels on the 500 pictures, also improves on the Wellington eval set, doesn't raise false positives on any one animal, and fits the time budget on the Pi 3. Differences of a few points on 208 cats can be noise. The test split runs once, at the freeze.

## Hand-off to the Pi 3

Export the model to NCNN on a laptop or PACE, not on the Pi 3, and give the hardware team the model folder and `edge/vision_config.json`.

```bash
python -m edge.bench --model best.pt --export ncnn --imgsz 416                  # laptop or PACE
python -m edge.bench --model best_ncnn_model --imgsz 416 --runs 200             # on the Pi 3 (E6)
```

## How to run the experiments

Everything trains on **PACE ICE**, where the data and GPUs already are. Run jobs with `sbatch`, not on the login node. Copy small result files to a laptop and commit from there, because PACE has no GitHub login.

| Where | What runs there |
| --- | --- |
| PACE | Training, scoring, Wellington download, backyard video test |
| Pi 3 | E6 speed test only |
| Laptop | Tests, demo, exporting a model to NCNN |

Chain the first jobs so nothing starts before its data is ready:

```bash
DATA=$(sbatch --parsable training/slurm/e7_data.sbatch)
SCORE=$(sbatch --parsable --dependency=afterok:$DATA training/slurm/eval_pace.sbatch E1b 500 416)
sbatch --dependency=afterok:$DATA --time=12:00:00 -J scout-E7 training/slurm/train_pace.sbatch E7 \
       --epochs 80 --patience 25 --data-yaml ~/scratch/scout_data/scout_e7.yaml
sbatch --dependency=afterok:$SCORE training/slurm/e8_crop.sbatch
sbatch -J scout-E9 training/slurm/train_pace.sbatch E9 --cls 1.0 --epochs 80 --patience 25
```

E7 is estimated at about 9.5 hours from E1's speed, and jobs default to 8. If PACE refuses `--time=12:00:00`, use `--epochs 60`: training can't resume after a time-out yet.

## Real-animal videos

`backyard_camera/` holds 46 videos from one Wyze camera (Oct 22–29), about 16 minutes in total. They aren't in git. `data/backyard_labels.csv` lists the animal in each and how to rotate the 8 sideways ones. The labels come from looking at sample frames, so check them before reporting numbers, especially the rows marked "check" and the one `unknown`.

| | Videos | When |
| --- | --- | --- |
| Cat | 25 | 23 in daylight or dusk, 2 in the dark |
| Raccoon | 9 | dusk to dawn |
| Opossum | 11 | dusk to dawn |
| Unknown | 1 | dark |

- **What they add:** daytime cats (E1b caught only 6 of 22) and cats that fill the frame (the 500-picture set has none). They also have real motion and a low camera angle, so they test the vote and gate on real animals.
- **Limits:** it's one yard with only a few different cats, so scores from it don't replace the public pictures. Cats are mostly in daylight and raccoons and opossums almost all in the dark, and there are no daytime non-cats. **If trained on as they are, the model could learn "daylight = cat".** Night footage is in colour, unlike our NoIR camera under infrared. Every frame has a Wyze logo and timestamp, so crop or mask them if training on these frames.

```bash
# from your laptop: scp -r backyard_camera <gt-username>@login-ice.pace.gatech.edu:~/scratch/scout_data/
V=~/scratch/scout_data/backyard_camera; C=~/scratch/scout_data/real/clips
JOB=$(sbatch --parsable training/slurm/run.sbatch python data/video_to_frames.py --videos $V --labels data/backyard_labels.csv --clips-out $C --fps 5)
sbatch --dependency=afterok:$JOB training/slurm/run.sbatch python training/vote_sim.py clips --model results/E1b/weights/best.pt --clips $C --full-frame --imgsz 416 --name E1b_backyard
# after E6: repeat with the Pi 3's measured frame rate (--fps on both commands)
# labeling pool: one frame per second, pre-labeled by E1b, for Roboflow
sbatch training/slurm/run.sbatch python data/video_to_frames.py --videos $V --labels data/backyard_labels.csv \
       --pool-out ~/scratch/scout_data/real/pool --pool-every 1 --prelabel results/E1b/weights/best.pt
```

Never put frames of one video in both training and test.

## Staged footage and E5

```bash
python -m edge.demo --model best.pt --source 0 --save-dir pilot/clips/cat_01      # r to record
python training/vote_sim.py clips --model best.pt --clips pilot/clips --full-frame
python data/add_trap_footage.py --data ~/scratch/scout_data --export session1_yolo --session s1_day --split train
python data/make_train_list.py --data ~/scratch/scout_data --cat-repeat 2 \
       --extra-dir ~/scratch/scout_data/trap/yolo/images/train --extra-repeat 3 --tag e5
sbatch -J scout-E5 training/slurm/train_pace.sbatch E5 --weights results/E1b/weights/best.pt --epochs 40 \
       --data-yaml ~/scratch/scout_data/scout_e5.yaml
python -m edge.draw_zones --image pilot/clips/empty_01/<frame>.jpg               # once camera is mounted
```

## Learning cats, not backgrounds

- Test images always come from unseen camera locations.
- About 10% of training images are empty, and near-duplicate burst photos are limited.
- Every staged session includes cat props, non-cat props and an empty trap.
- E5 is always re-scored on real-animal test pictures.
