"""Measures cat recall and per-species false-positive rate (including empty backgrounds), plus a threshold sweep.

Purpose: These are the numbers the plan and checkpoints report; they also pick the per-class thresholds in vision_config.json.
Status:  Written. Not yet run on real data (E0 is the first run).
Later stages (still to add or change):
  - Phase 2a: add a seen-location vs new-location breakdown once splits.py has a seen-location split.
  - Phase 2b: evaluate on our own trap footage and on the empty-trap check (no animal should be reported).
  - Phase 3: run once on the frozen test set and record the final numbers.

Works for any detector whose class names include "cat": the stock COCO model (experiment E0) or
our fine-tuned 7-class model (E1+). Ground truth comes from the YOLO label files.

    # E0: stock YOLO26n, no training, 500 val images
    python training/eval.py --model yolo26n.pt --data ~/scratch/scout_data --split val --n 500 --name E0
    # a trained run
    python training/eval.py --model results/E1/weights/best.pt --data ~/scratch/scout_data --split val --name E1

Writes results/<name>/metrics.json and results/<name>/threshold_sweep.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common import CLASSES  # noqa: E402

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def true_label(label_file: Path) -> str:
    """Primary class of an image: cat if any cat box, else the most common box class, else background."""
    if not label_file.exists() or not label_file.read_text().strip():
        return "background"
    ids = [int(line.split()[0]) for line in label_file.read_text().splitlines() if line.strip()]
    if 0 in ids:
        return "cat"
    return CLASSES[Counter(ids).most_common(1)[0][0]]


def sample(images: list[Path], labels_dir: Path, n: int, seed: int) -> list[tuple[Path, str]]:
    items = [(p, true_label(labels_dir / f"{p.stem}.txt")) for p in images]
    if not n or n >= len(items):
        return items
    rng = random.Random(seed)
    cats = [x for x in items if x[1] == "cat"]
    rest = [x for x in items if x[1] != "cat"]
    rng.shuffle(cats)
    rng.shuffle(rest)
    k = min(len(cats), n // 2)
    return cats[:k] + rest[: n - k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="data folder containing yolo/")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=0, help="sample N images (half cats); 0 = all")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--thr", type=float, default=0.25, help="threshold reported in metrics.json")
    ap.add_argument("--name", required=True, help="experiment id, e.g. E0")
    ap.add_argument("--seed", type=int, default=4605)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path(args.data).expanduser() / "yolo"
    img_dir, lbl_dir = root / "images" / args.split, root / "labels" / args.split
    items = sample(sorted(img_dir.glob("*.jpg")), lbl_dir, args.n, args.seed)
    if not items:
        sys.exit(f"no images in {img_dir}")
    model = YOLO(args.model)
    cat_ids = [i for i, n in model.names.items() if n == "cat"]
    if not cat_ids:
        sys.exit("model has no 'cat' class")

    # Highest cat confidence per image (0 if none).
    best = []
    for i in range(0, len(items), args.batch):
        chunk = items[i : i + args.batch]
        results = model([str(p) for p, _ in chunk], imgsz=args.imgsz, conf=min(THRESHOLDS),
                        classes=cat_ids, verbose=False)
        for (p, truth), r in zip(chunk, results):
            conf = max(r.boxes.conf.tolist(), default=0.0)
            best.append((p.name, truth, conf))

    def score(thr: float) -> dict:
        tp = sum(1 for _, t, c in best if t == "cat" and c >= thr)
        n_cat = sum(1 for _, t, _ in best if t == "cat")
        per = defaultdict(lambda: [0, 0])  # species -> [called cat, total]
        for _, t, c in best:
            if t != "cat":
                per[t][1] += 1
                per[t][0] += c >= thr
        fp = sum(v[0] for v in per.values())
        n_non = sum(v[1] for v in per.values())
        return {
            "threshold": thr,
            "cat_recall": tp / n_cat if n_cat else None,
            "noncat_fpr": fp / n_non if n_non else None,
            "fpr_by_species": {k: v[0] / v[1] for k, v in sorted(per.items()) if v[1]},
            "n_cat": n_cat,
            "n_noncat": n_non,
        }

    out = Path("results") / args.name
    out.mkdir(parents=True, exist_ok=True)
    main_score = score(args.thr)
    main_score.update(model=args.model, split=args.split, imgsz=args.imgsz, n_images=len(best))
    (out / "metrics.json").write_text(json.dumps(main_score, indent=1))
    with open(out / "threshold_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "cat_recall", "noncat_fpr"])
        for thr in THRESHOLDS:
            s = score(thr)
            w.writerow([thr, s["cat_recall"], s["noncat_fpr"]])
    with open(out / "per_image.csv", "w", newline="") as f:
        csv.writer(f).writerows([("image", "truth", "best_cat_conf"), *best])

    r, fpr = main_score["cat_recall"], main_score["noncat_fpr"]
    print(f"{args.name}: {len(best)} images ({main_score['n_cat']} cat / {main_score['n_noncat']} non-cat) @ thr {args.thr}")
    print(f"  cat recall   {r:.1%}   (target >= 75%)" if r is not None else "  no cat images")
    print(f"  non-cat FPR  {fpr:.1%}   (target <= 25%)" if fpr is not None else "  no non-cat images")
    for sp, v in main_score["fpr_by_species"].items():
        print(f"    {sp:13s} called cat {v:.1%}")
    print(f"  wrote {out}/metrics.json, threshold_sweep.csv, per_image.csv")


if __name__ == "__main__":
    main()
