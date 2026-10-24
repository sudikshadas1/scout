# SCOUT: Smart Cat Observation & Unmanned Trap

An edge-AI add-on for humane live traps used in Trap-Neuter-Return (TNR). A camera and a Raspberry Pi 5 recognize what walks into the trap, and the gate closes only for a cat that is fully inside. Everything runs on the Pi, with no Wi-Fi needed.

CS 4605 Mobile & Ubiquitous Computing, Fall 2026, Group 29. Mentor: Christine Taylor.
Team: Sudiksha Das and Henry Poulin (model), Keshav Sheth and Sowmiyan Nattamai Vijayakumar (hardware).

Design docs: [SCOUT - Research/Notes](https://docs.google.com/document/d/1BFhBik9DCZgdWnCy8yDcnWfe-6-JuTAZf1hQFuI_Fq8/edit). The **CV Model** tab is the plan, the **Code Base** tab maps this code to it, the **CV Team - Next Steps** tab is the step-by-step runbook, and the **High Level Plan** tab has the dated status.

**How we evaluate** (plan section 12). No live animals are used this semester; every trial in our trap is staged with props. So the evaluation has two parts:

1. **Model accuracy, on real animals:** at least 75% of cats caught and no more than 25% of other animals called cats, on held-out public camera-trap pictures from camera locations the model never trained on.
2. **Staged trap trials, with props:** stuffed or printed animals and tablet photos in our trap. The gate closes for cat props and stays open for non-cat props, decides in under 5 seconds, and never reports "cat in trap" while the cat is in the doorway.

Props are not real animals, so prop footage is used to check the camera angle, lighting, zones, vote and gate, not to report accuracy.

## Status (Oct 24)

- **Model:** E1 (YOLO26n fine-tuned on Caltech Camera Traps + ENA24) catches 63.5% of cats on new camera locations, with about 2% false positives. The recall runs E1b and E1c, the seen-location score and the E2 comparison are ready to run on PACE.
- **Hardware:** the stub, gate logic and Pi benchmark are written. Waiting on the trap, the camera mount position, the Pi with a camera, and an infrared light.

## How it works

```mermaid
flowchart LR
    PIR[PIR motion sensor] -->|wake| CAM[Pi Camera 3 NoIR]
    CAM -->|frame| DET[Detector<br/>YOLO26n, NCNN]
    DET -->|boxes + classes| AGG[Vision aggregator<br/>K-of-N voting, zones]
    AGG -->|VisionReport| GATE[Gate controller]
    PLATE[Pressure plate] --> GATE
    GATE -->|CLOSE / KEEP_OPEN| SERVO[Servo on gate]
    GATE --> LED[Status LEDs / alerts]

    subgraph Model team
      DET
      AGG
    end
    subgraph Hardware team
      PIR
      CAM
      PLATE
      GATE
      SERVO
      LED
    end
```

1. The PIR sensor wakes the system when something moves.
2. The camera captures frames at about 5 fps.
3. The **detector** finds animals in each frame: cat, raccoon, opossum, squirrel, dog, other animal, person.
4. The **vision aggregator** reports what it sees. A cat counts as "in the trap" only when it is inside the trap zone in 3 of the last 5 frames. It also reports anything in the doorway and any non-cat. If one animal gets two different labels on the same box, only the clearly stronger label counts.
5. The **gate controller** decides whether to close, using the vision report, the pressure plate and its safety rules (non-cat cooldown, doorway check, fault handling).

The vote, cutoffs and zones live in `edge/vision_config.json`.

### Who decides what

| Question | Decided by | Where |
| --- | --- | --- |
| What animal is this, and where is its box? | Model team | `edge/vision.py` `Detector` |
| Is a cat really in the trap zone (not a one-frame glitch)? | Model team | `edge/vision.py` `VisionAggregator` |
| Is anything in the doorway? Is a non-cat present? | Model team (reports it) | `VisionReport` |
| Where the trap and doorway zones are in the image | Both, once the camera is mounted | `edge/vision_config.json`, drawn with `edge/draw_zones.py` |
| Should the gate close now? | Hardware team | `edge/gate.py` `GateController` |
| Pressure plate, cooldown after a non-cat, stay closed until reset, faults | Hardware team | `edge/gate.py` |
| Servo, LEDs, PIR wiring, power, alerts | Hardware team | hardware hooks in `edge/run_trap.py` |

The model team **reports observations**; the hardware team **decides actions**. `gate.py` is a reference version so both teams can test now; the hardware team owns it and can change it.

### The interface

```python
from edge.vision import Detector, VisionAggregator
from edge.gate import GateController, Command

detector = Detector("best_ncnn_model", imgsz=416)
vision = VisionAggregator.from_config()       # edge/vision_config.json
gate = GateController(require_plate=False)
gate.arm()

report = vision.update(detector.detect(frame))   # model team
status = gate.tick(report, plate_pressed=False)  # hardware team
if status.command == Command.CLOSE:
    close_gate()
```

`VisionReport` fields: `cat_in_trap`, `cat_confidence`, `cat_hits`, `animal_in_doorway`, `non_cat`, `non_cat_confidence`, `cat_box`, `n_detections`, `timestamp`.

### Test the hardware without a model

`edge/stub.py` feeds fake detections through the real vision and gate code. One token per frame: `c` cat in trap, `d` cat in doorway, `r` raccoon, `o` opossum, `n` nothing, `f` fault, `p` cat + plate pressed.

```bash
python -m edge.stub --script c,c,c,c,c            # closes on frame 3 (3-of-5 vote)
python -m edge.stub --script c,c,r,c,c            # raccoon: cooldown, stays open
python -m edge.stub --script c,c,c,c,p --plate    # closes only when the plate is pressed
python -m edge.stub --interactive
python -m edge.stub --trials 50                   # random staged trials, scored
```

## Code map

[![Architecture diagram of sudikshadas1/scout](https://gitdiagram.com/sudikshadas1/scout/diagram.png)](https://gitdiagram.com/sudikshadas1/scout?utm_source=readme&utm_medium=picture)

Generated automatically by [gitdiagram](https://gitdiagram.com/sudikshadas1/scout) from the `main` branch, so it can lag behind new files. The flowchart above is the exact runtime flow; the list below is the complete file list.

```
data/                    dataset tools (model team, run on PACE ICE)
  common.py                class list, species -> class mapping
  fetch_metadata.py        download LILA metadata JSONs only (~50 MB)
  stats.py                 images per species and per class, per dataset
  splits.py                location-based train/val/test -> data/splits.json (committed, never change)
  build_manifest.py        choose balanced images + boxes -> manifest.jsonl
  download_images.py       download images, write YOLO labels
  validate_dataset.py      check every YOLO label file
  build_seen_split.py      "seen locations" test split (new bursts from training cameras)
  make_train_list.py       training list with cat images repeated (E1c, E5) + matching yaml
  add_trap_footage.py      import one labeled session of staged trap footage (props, YOLO export)
  scout.yaml               Ultralytics dataset config
training/                training and evaluation (model team, PACE ICE)
  train_yolo.py            train one detector experiment (E1b, E1c, E5 ...)
  eval.py                  cat recall, per-species false positives, threshold sweep
  analyze_errors.py        why cats are missed, by size, day/night, dataset and location
  vote_sim.py              choose the cat cutoff + K-of-N vote at trap level; replays the real gate on clips
  train_classifier.py      E2: MobileNetV3 cat / not-cat comparison
  slurm/                   PACE jobs (see below)
edge/                    code that runs on the Pi
  vision.py                Detector + VisionAggregator + VisionReport (model team)
  vision_config.json       N, K, cutoffs, trap/doorway zones
  gate.py                  GateController (hardware team)
  run_trap.py              main loop + hardware hooks (PIR, plate, servo)
  stub.py                  fake detections for hardware testing
  bench.py                 per-stage latency on the Pi
  demo.py                  laptop prototype: live boxes, zones, report, simulated gate; records frames
  draw_zones.py            click the trap / doorway zones on a frame -> vision_config.json
tests/                   python -m pytest tests
results/                 one folder per experiment (metrics kept, weights git-ignored)
```

PACE jobs in `training/slurm/`:

| Job | What it runs |
| --- | --- |
| `setup_env.sh`, `smoke_test.sbatch` | one-time conda setup, GPU check |
| `data.sbatch` | metadata, manifest and image download |
| `e0.sbatch`, `e1_*.sbatch` | the E0 and E1 runs as they were done (yolo CLI) |
| `train_pace.sbatch <name> [args]` | any new detector run through `train_yolo.py` |
| `eval_pace.sbatch <name>` | eval + miss analysis + cutoff/vote estimate for one run |
| `seen.sbatch <name>` | seen-location score |
| `e2.sbatch` | E2 classifier |

The E1 jobs and `train_pace.sbatch` read `data/scout_pace.yaml`, a copy of `data/scout.yaml` with the absolute scratch path. It is only on PACE (`~/scout/data/`), not in git.

## Getting started

### Model team, on PACE ICE

First-time setup (already done once; repeat only on a new account):

```bash
git clone https://github.com/sudikshadas1/scout.git && cd scout
bash training/slurm/setup_env.sh                 # conda env "scout"
sbatch training/slurm/smoke_test.sbatch          # proves the GPU works
export DATA=~/scratch/scout_data
python data/fetch_metadata.py --out $DATA
python data/stats.py --data $DATA
python data/splits.py --data $DATA               # data/splits.json is already committed: don't rerun
sbatch training/slurm/data.sbatch                # manifest + image download
python data/validate_dataset.py
```

### Hardware team, on the Pi 5

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate   # keeps picamera2
pip install -r requirements-pi.txt
python -m edge.stub --interactive
python -m edge.bench --model yolo26n.pt --export ncnn --imgsz 416 --runs 200
python -m edge.run_trap --model yolo26n_ncnn_model
```

## CV results

Validation set: 500 pictures (208 cats) from camera locations the model never trained on.

| Run | What changed | Cat recall @0.05 | Cat recall @0.25 | Non-cat FPR @0.25 |
| --- | --- | --- | --- | --- |
| E0 | stock YOLO26n, no training | 4.3% | | 2.1% |
| E1 | fine-tuned on Caltech + ENA24, 24 epochs | 63.5% | 54.8% | 2.1% |
| E1b | E1, trained longer (80 epochs) | pending | | |
| E1c | E1b + cat pictures shown twice | pending | | |
| E2 | MobileNetV3 classifier (comparison only) | pending | | |

The plan's E3 ("add ENA24") is already part of E1. Only 2% of the 25% false-positive budget is used, so the next runs trade some of it for recall.

### Decisions on Oct 24

**Cutoff and vote.** The trap acts on a vote across frames, not on one picture. With the old settings (cat cutoff 0.5, 4-of-5 vote), `training/vote_sim.py` estimates E1 would trigger on only about 9% of cats. The config is now cutoff 0.15 with a 3-of-5 vote: an estimated 68% at the trap level and about 0.01% false positives. The estimate assumes frames are independent, so re-measure it on real clips with `vote_sim.py clips`, which also replays the real gate.

**One animal, one label.** If the model labeled the same cat "cat" and also "opossum" in one frame, the gate entered its 10-second non-cat cooldown and never closed on that cat. Now the weaker label on the same box is dropped, but only when the stronger one is at least 0.1 more confident. Close calls keep both, so a possible non-cat still holds the gate open (`same_animal_iou`, `same_animal_margin` in `vision_config.json`).

## Next CV runs

Step-by-step with explanations: the **CV Team - Next Steps** tab.

```bash
# 0. get the latest code on PACE
cd ~/scout && git fetch && git checkout cv/recall-analysis-demo && git reset --hard origin/cv/recall-analysis-demo
#    (after PR #1 is merged: git checkout main && git pull)

# 1. E1: why cats are missed + cutoff/vote estimate
sbatch training/slurm/eval_pace.sbatch E1_full
# 2. recall experiments, in parallel
sbatch -J scout-E1b training/slurm/train_pace.sbatch E1b --epochs 80 --patience 25
python data/make_train_list.py --data ~/scratch/scout_data --cat-repeat 2
sbatch -J scout-E1c training/slurm/train_pace.sbatch E1c --epochs 80 --patience 25 \
       --data-yaml ~/scratch/scout_data/scout_cat2x.yaml
# 3. seen-location score and the E2 comparison
sbatch training/slurm/seen.sbatch E1_full
sbatch training/slurm/e2.sbatch
# 4. when E1b / E1c finish
sbatch training/slurm/eval_pace.sbatch E1b && sbatch training/slurm/eval_pace.sbatch E1c
```

What to read: `results/<run>_analysis/summary.json`, where `recall_by_size.large` is the number closest to our close-range camera; `results/E1_full_seen/metrics.json` against `results/E1_full_eval/metrics.json`, where a big gap means the model learned backgrounds; `results/E2/metrics.json`.

### Staged trap footage (props, no live animals)

```bash
# record one staged trial per folder, named by what is in it (cat_, raccoon_, opossum_, empty_ ...)
python -m edge.demo --model yolo26n.pt --source 0 --save-dir pilot/clips/cat_01      # press r to record
python -m edge.demo --model yolo26n.pt --picam --headless --record --save-dir pilot/clips/empty_01  # on the Pi
# trap-level recall and gate behaviour on those clips, for every cutoff x vote setting
python training/vote_sim.py clips --model best.pt --clips pilot/clips --full-frame
# after labeling a session (YOLO export from Roboflow / CVAT / Label Studio); keep 1-2 whole sessions as test
python data/add_trap_footage.py --data ~/scratch/scout_data --export session1_yolo --session s1_day --split train
python data/add_trap_footage.py --data ~/scratch/scout_data --export session4_yolo --session s4_dark --split test
# E5: public data + our prop footage (repeated 3x), then the staged test on held-out trap sessions
python data/make_train_list.py --data ~/scratch/scout_data --cat-repeat 2 \
       --extra-dir ~/scratch/scout_data/trap/yolo/images/train --extra-repeat 3 --tag e5
sbatch -J scout-E5 training/slurm/train_pace.sbatch E5 --weights results/E1c/weights/best.pt --epochs 40 \
       --data-yaml ~/scratch/scout_data/scout_e5.yaml
python training/eval.py --model results/E5/weights/best.pt --data ~/scratch/scout_data/trap --split test --name E5_trap
# once the camera is mounted: draw the trap and doorway zones on a real frame, then commit vision_config.json
python -m edge.draw_zones --image pilot/clips/empty_01/<frame>.jpg
```

### Laptop demo

```bash
python -m edge.demo --model results/E1_full/weights/best.pt --source 0     # webcam
python -m edge.demo --model best.pt --source demo_pics/ --out demo_out/    # picture folder, no window
```

Keys: `q` quit, `space` pause, `s` save a frame, `r` start/stop recording, `a` re-arm the gate, `z` toggle zones.

## Project phases

Each phase ends at a class checkpoint. Dates updated Oct 24: recording round 1 moved to Oct 31, and the model freeze moved to Nov 16 because Sudiksha is away Nov 17 to 27.

| Phase | Dates | Finish line | Status |
| --- | --- | --- | --- |
| 1. Get set up | to Oct 23 (CP1) | data split; E0; footage pilot; Pi speed number; servo closes from the stub | data, E0 and laptop demo done; pilot, Pi number and servo still open |
| 2a. Public data + recording round 1 | to Oct 31 | about 750 labeled trap pictures; E1, E1b, E1c, E2 | E1 done; the rest ready to run |
| 2b. Fine-tune and fix what fails | Nov 2 to Nov 13 (CP2) | E5 on our footage; camera to servo end to end on the bench | |
| 3. Test and freeze | freeze by Nov 16; CP3 Nov 24 | final numbers; 50+ staged trials | |
| 4. Demo | Dec 2 to 8 | live labeling of new pictures; gate acts correctly | |

### Open items

- [ ] Add all team members as GitHub collaborators
- [ ] Footage pilot: webcam or camera in the real trap, stuffed cat, tablet photos, empty trap (`edge/demo.py`)
- [ ] Pi 5 + camera speed test: `python -m edge.bench`
- [ ] Hardware team closes the servo from `python -m edge.stub`
- [ ] Decide the camera position, buy an infrared light for night, decide whether the pressure plate is in
- [ ] Pick a labeling tool and record round 1

### Making sure the model learns cats, not backgrounds

Test pictures always come from camera locations the model never trained on (`splits.py`), about 10% of training pictures are empty, and near-identical burst photos are limited (`build_manifest.py`). The seen-location score (`build_seen_split.py`) measures how much the model still relies on backgrounds. In our staged trap footage, every session includes cat props, non-cat props and the empty trap, so "inside the trap" never means "cat". Because props can look different from real animals, E5 (trained with prop footage) is always also re-scored on the real-animal public test pictures, so it can't trade real accuracy for prop accuracy. Details: CV Model tab, section 7.

### Code status

- **Run on real data:** data pipeline, E0 and E1 training and evaluation.
- **Written and tested, not yet run on PACE:** miss analysis, cutoff/vote chooser, seen-location split, E2, cat oversampling, trap-footage import.
- **Written and tested on a laptop:** vision/gate interface, hardware stub, laptop demo and recorder, zone drawing, Pi benchmark and main loop (not yet on the Pi).
- **Still to write:** black-and-white training data (E4), second-camera support.
- **Placeholders:** the trap and doorway zones in `edge/vision_config.json`, until the camera is mounted.

Every file starts with a short header: what it does, its purpose, its status, and what still needs to change. Read it before editing a file.

## Later: ear-tip detection

Recognizing already-neutered (ear-tipped) cats is a later goal, after the cat detector works end to end.
