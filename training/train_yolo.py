"""Trains the YOLO26n detector for one experiment and exports it to NCNN for the Pi.

Purpose: One script for every training experiment in the plan (E1, E3, E5 ...), so runs are comparable.
Status:  E1 ran through the yolo CLI (training/slurm/e1_full.sbatch); E1b/E1c use this script (training/slurm/train_pace.sbatch).
Later stages (still to add or change):
  - Phase 2a: E1 (baseline fine-tune), E3 (augmentation choices) run as-is.
  - Phase 2b: E5 fine-tunes on trap footage and needs data/scout_trap.yaml, which does not exist yet.
  - If E2 (MobileNetV3 comparison) is run, it needs its own small script; this one is YOLO-only.

    python training/train_yolo.py --name E1 --data-yaml data/scout.yaml
    python training/train_yolo.py --name E1b --epochs 80 --patience 25                      # longer training
    python training/train_yolo.py --name E1c --epochs 80 --patience 25 \
        --data-yaml ~/scratch/scout_data/scout_cat2x.yaml                                    # + cats 2x (make_train_list.py)
    python training/train_yolo.py --name E5 --weights results/E3/weights/best.pt --data-yaml data/scout_trap.yaml --epochs 50

Outputs land in results/<name>/ (weights are git-ignored; metrics and plots are kept).
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="experiment id from the plan, e.g. E1")
    ap.add_argument("--weights", default="yolo26n.pt", help="starting weights (COCO-pretrained by default)")
    ap.add_argument("--data-yaml", default="data/scout.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--cls", type=float, default=0.5, help="classification loss gain (Ultralytics default 0.5)")
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--close-mosaic", type=int, default=10, help="turn mosaic off for the last N epochs")
    ap.add_argument("--export-imgsz", type=int, default=416, help="also export NCNN at this size (0 = skip)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=args.data_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        # augmentations matched to an in-trap camera (plan, "Training and experiments")
        hsv_v=0.5,          # strong brightness changes: IR LEDs, dusk
        hsv_s=0.5,
        degrees=0.0,
        flipud=0.0,         # animals are never upside down
        fliplr=0.5,
        scale=0.5,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        cls=args.cls,
        plots=True,
        project=str(Path("results").resolve()),
        name=args.name,
        exist_ok=True,
    )
    metrics = model.val(data=args.data_yaml, split="val", imgsz=args.imgsz)
    summary = {"name": args.name, "weights_start": args.weights, "map50": metrics.box.map50,
               "map50_95": metrics.box.map, "args": vars(args)}
    Path("results", args.name).mkdir(parents=True, exist_ok=True)
    Path("results", args.name, "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if args.export_imgsz:
        try:  # the weights are already saved; export on the Pi with edge/bench.py if this fails here
            model.export(format="ncnn", imgsz=args.export_imgsz)
        except Exception as e:  # noqa: BLE001
            print(f"NCNN export skipped ({e}); export on the Pi instead: python -m edge.bench --export ncnn")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
