"""Scores a model the way the trap decides, not just on cat boxes: one label per animal, then a non-cat holds the gate.

Purpose: eval.py only looks at cat boxes, so a raccoon that also gets a weak cat box counts as a false positive even
         though the trap would drop that label or hold the gate open. This shows the real cat / false-positive
         trade-off for one frame, with no training. It also saves every detection for training/train_crop_classifier.py.
Status:  Rules tested (tests/test_improvement.py). Not yet run on PACE. Runs inside eval_pace.sbatch.
Later stages (still to add or change):
  - The 10-second non-cat cooldown spans frames, so it is only measured by vote_sim.py clips.

Uses the same image sample as eval.py (same --n and --seed), and the rules in edge/vision.py.

    python training/score_pipeline.py --model results/E1b/weights/best.pt --data ~/scratch/scout_data \
        --split val --n 500 --imgsz 416 --name E1b_pipeline416

Writes results/<name>/pipeline_sweep.csv and detections.jsonl. Rules in the sweep:
    cat_box   any cat box over the cutoff (the same rule as eval.py)
    one_label also drop the weaker label when two boxes cover one animal (edge/vision.py)
    trap      also no confident non-cat in the frame, since that holds the gate open
animal_fpr is the goal (empty pictures left out). Cutoffs under 0.1 are only for comparison: the Pi's Detector drops
anything below min_conf=0.1.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from edge.vision import CONFIG, Detection, one_label_per_animal  # noqa: E402

CAT_CUTOFFS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
NONCAT_CUTOFFS = [0.3, 0.4, 0.5, 0.6, 0.7]
RULES = ("cat_box", "one_label", "trap")


def calls_cat(dets: list[Detection], rule: str, cat_cut: float, noncat_cut: float,
              same_iou: float, margin: float) -> bool:
    """One frame: would this rule report a cat? Mirrors VisionAggregator.update() without zones or the vote."""
    live = [d for d in dets if d.conf >= (cat_cut if d.cls == "cat" else noncat_cut)]
    if rule == "cat_box":
        return any(d.cls == "cat" for d in live)
    live = one_label_per_animal(live, same_iou, margin)
    has_cat = any(d.cls == "cat" for d in live)
    if rule == "one_label":
        return has_cat
    return has_cat and not any(d.cls != "cat" for d in live)


def sweep(records: list[dict], same_iou: float, margin: float) -> list[dict]:
    """records: [{"truth": str, "dets": [Detection]}] -> one row per rule x cutoffs."""
    n_cat = sum(r["truth"] == "cat" for r in records)
    n_non = len(records) - n_cat
    n_bg = sum(r["truth"] == "background" for r in records)
    rows = []
    for rule in RULES:
        for cat_cut in CAT_CUTOFFS:
            # the non-cat cutoff changes nothing for cat_box
            for noncat_cut in NONCAT_CUTOFFS if rule != "cat_box" else [NONCAT_CUTOFFS[0]]:
                called = [calls_cat(r["dets"], rule, cat_cut, noncat_cut, same_iou, margin) for r in records]
                tp = sum(c for c, r in zip(called, records) if r["truth"] == "cat")
                fp = sum(c for c, r in zip(called, records) if r["truth"] != "cat")
                fp_bg = sum(c for c, r in zip(called, records) if r["truth"] == "background")
                rows.append({"rule": rule, "cat_cutoff": cat_cut, "noncat_cutoff": noncat_cut,
                             "cat_recall": tp / n_cat if n_cat else None,
                             "animal_fpr": (fp - fp_bg) / (n_non - n_bg) if n_non > n_bg else None,
                             "noncat_fpr": fp / n_non if n_non else None, "cats_caught": tp, "false_positives": fp})
    return rows


def best_under(rows: list[dict], rule: str, max_fpr: float = 0.05) -> dict | None:
    ok = [r for r in rows if r["rule"] == rule and r["animal_fpr"] is not None and r["animal_fpr"] < max_fpr]
    return max(ok, key=lambda r: (r["cat_recall"], -r["animal_fpr"]), default=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="data folder containing yolo/")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "seen"])
    ap.add_argument("--n", type=int, default=0, help="same sampling as eval.py; 0 = all images")
    ap.add_argument("--seed", type=int, default=4605)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    from eval import sample
    from ultralytics import YOLO

    root = Path(args.data).expanduser() / "yolo"
    img_dir, lbl_dir = root / "images" / args.split, root / "labels" / args.split
    items = sample(sorted(img_dir.glob("*.jpg")), lbl_dir, args.n, args.seed)
    if not items:
        sys.exit(f"no images in {img_dir}")
    cfg = json.loads(CONFIG.read_text())
    model = YOLO(args.model)
    out = Path("results") / args.name
    out.mkdir(parents=True, exist_ok=True)

    records = []
    with open(out / "detections.jsonl", "w") as f:
        for i in range(0, len(items), args.batch):
            chunk = items[i : i + args.batch]
            results = model([str(p) for p, _ in chunk], imgsz=args.imgsz, conf=min(CAT_CUTOFFS), verbose=False)
            for (p, truth), r in zip(chunk, results):
                dets = [Detection(model.names[int(c)], float(s), tuple(b)) for b, s, c in
                        zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
                records.append({"truth": truth, "dets": dets})
                f.write(json.dumps({"image": p.name, "truth": truth,
                                    "dets": [[d.cls, round(d.conf, 4), [round(v, 5) for v in d.box]] for d in dets]}) + "\n")

    rows = sweep(records, cfg.get("same_animal_iou", 0.6), cfg.get("same_animal_margin", 0.1))
    with open(out / "pipeline_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{args.name}: {len(records)} images at imgsz {args.imgsz}. Best single-frame setting under 5% false positives:")
    for rule in RULES:
        b = best_under(rows, rule)
        print(f"  {rule:10s} " + (f"cat cutoff {b['cat_cutoff']:.2f}, non-cat cutoff {b['noncat_cutoff']:.1f}: "
                                  f"{b['cat_recall']:.1%} of cats, {b['animal_fpr']:.1%} false positives on animals" if b else "none"))
    print(f"wrote {out}/pipeline_sweep.csv, detections.jsonl")


if __name__ == "__main__":
    main()
