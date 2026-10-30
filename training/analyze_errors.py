"""Explains WHY cats are missed: what each missed cat was called instead, how big it was, day or night, where.

Purpose: E1b catches 144 of 208 val cats at threshold 0.25 (69.2%, goal at least 75%). Before changing training we need
         to know which kind of miss dominates, because each one has a different fix:
           wrong_class   the box is found but labelled raccoon/opossum/...  -> class balance, cls loss, more cats
           low_conf_cat  a cat box exists but under the threshold         -> threshold choice, longer training
           not_found     nothing at all on the cat                        -> tiny/far/dark cats; see size bins
         It also reports recall by box size. Our trap camera is inches away, so cats there fill the frame:
         recall on "large" boxes is the number closest to the real trap; tiny trail-cam cats matter much less.
Status:  Written; helper functions tested (tests/test_analysis.py). Run it on PACE after each eval.
Replaces: the old one-off inspect_missed.py contact sheet (removed; see git history) (this draws boxes and groups by failure type).
Later stages (still to add or change):
  - Phase 2b: run on our own trap footage (--split test on a trap-only yolo folder).
  - Phase 3: run once on the frozen test split for the final report.

Uses exactly the same image sample as training/eval.py (same --n and --seed), so numbers line up.

    python training/analyze_errors.py --model results/E1_full/weights/best.pt \
        --data ~/scratch/scout_data --split val --n 500 --name E1_full_analysis

Writes results/<name>/:
    summary.json        recall overall / by size / day-night / dataset / location, miss reasons, confusion
    cat_images.csv      one row per cat image: confidences, what it was called, size, night, location
    confusion.csv       image-level truth class vs predicted class (at --thr)
    sheets/<reason>.jpg contact sheets, ground truth green, predictions red (jpgs are git-ignored)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "data"))
sys.path.insert(0, str(HERE.parent))
from common import CLASSES  # noqa: E402

SIZE_BINS = [("tiny", 0.01), ("small", 0.05), ("medium", 0.20), ("large", 1.01)]  # fraction of frame area
MATCH_IOU = 0.3


# --------------------------------------------------------------------------- pure helpers (tested)
def read_labels(path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    """YOLO label file -> [(class name, (x1, y1, x2, y2) normalized)]."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
        out.append((CLASSES[c], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def size_bin(frac: float) -> str:
    for name, upper in SIZE_BINS:
        if frac < upper:
            return name
    return SIZE_BINS[-1][0]


def is_grayscale(rgb_pixels, tol: float = 4.0) -> bool:
    """Night/IR camera-trap frames are (near) grayscale. rgb_pixels: iterable of (r, g, b)."""
    n, diff = 0, 0.0
    for r, g, b in rgb_pixels:
        diff += abs(r - g) + abs(g - b)
        n += 1
    return n > 0 and diff / n < tol


def classify_cat_image(gt_cats, preds, thr: float, low: float):
    """Decide what happened to the cat(s) in one image.

    gt_cats: [(x1, y1, x2, y2)]; preds: [(cls, conf, box)].
    Returns (reason, called_as, called_conf, best_cat_conf, any_animal_found)
      reason: 'hit' | 'wrong_class' | 'low_conf_cat' | 'not_found'
    """
    best_cat = max((c for k, c, _ in preds if k == "cat"), default=0.0)
    if best_cat >= thr:
        return "hit", "cat", best_cat, best_cat, True
    overlapping = [(k, c) for k, c, b in preds if any(iou(b, g) >= MATCH_IOU for g in gt_cats)]
    animal_found = any(c >= thr for _, c in overlapping)
    wrong = [(k, c) for k, c in overlapping if k != "cat" and c >= thr]
    if wrong:
        k, c = max(wrong, key=lambda x: x[1])
        return "wrong_class", k, c, best_cat, animal_found
    if best_cat >= low:
        return "low_conf_cat", "cat", best_cat, best_cat, animal_found
    return "not_found", None, 0.0, best_cat, animal_found


def primary_pred(preds, thr: float) -> str:
    """Image-level prediction: cat if any confident cat, else the most confident class, else background."""
    live = [(k, c) for k, c, _ in preds if c >= thr]
    if not live:
        return "background"
    if any(k == "cat" for k, _ in live):
        return "cat"
    return max(live, key=lambda x: x[1])[0]


def recall_table(rows: list[dict], key: str) -> dict:
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        agg[r[key]][1] += 1
        agg[r[key]][0] += r["reason"] == "hit"
    return {str(k): {"recall": v[0] / v[1], "hit": v[0], "n": v[1]} for k, v in sorted(agg.items(), key=lambda x: str(x[0]))}


def location_lookup(meta: Path) -> dict[str, str]:
    """stem (as written by build_manifest.make_row) -> 'dataset:location'. Empty dict if metadata missing."""
    files = {"cct": meta / "cct_images.json", "ena24": meta / "ena24.json"}
    out = {}
    for ds, f in files.items():
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for img in d["images"]:
            stem = f"{ds}__" + img["file_name"].replace("/", "__").rsplit(".", 1)[0]
            out[stem] = f"{ds}:{img.get('location', 'unknown')}"
    return out


# --------------------------------------------------------------------------- contact sheets
def draw_sheet(items, img_dir: Path, out: Path, cols: int = 4, max_n: int = 24) -> None:
    from PIL import Image, ImageDraw
    items = items[:max_n]
    if not items:
        return
    w, h = 320, 250
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    d = ImageDraw.Draw(sheet)
    for i, it in enumerate(items):
        x0, y0 = (i % cols) * w, (i // cols) * h
        try:
            with Image.open(img_dir / it["image"]) as im:
                im = im.convert("RGB")
                im.thumbnail((w - 10, h - 30))
                tw, th = im.size
                sheet.paste(im, (x0 + 5, y0 + 25))
                for _, b in it["gt"]:
                    d.rectangle([x0 + 5 + b[0] * tw, y0 + 25 + b[1] * th, x0 + 5 + b[2] * tw, y0 + 25 + b[3] * th],
                                outline=(0, 200, 0), width=2)
                for k, c, b in it["preds"]:
                    if c < 0.1:
                        continue
                    bx = [x0 + 5 + b[0] * tw, y0 + 25 + b[1] * th, x0 + 5 + b[2] * tw, y0 + 25 + b[3] * th]
                    d.rectangle(bx, outline=(220, 0, 0), width=1)
                    d.text((bx[0] + 2, bx[1] + 2), f"{k} {c:.2f}", fill=(220, 0, 0))
        except OSError as e:
            d.text((x0 + 5, y0 + 30), f"error: {e}", fill="red")
        label = f"{it['reason']} {it['size']} {'night' if it['night'] else 'day'} cat={it['best_cat_conf']:.2f}"
        d.text((x0 + 5, y0 + 6), label, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="data folder containing yolo/ (and meta/ for locations)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "seen"])
    ap.add_argument("--n", type=int, default=0, help="same sampling as eval.py; 0 = all images")
    ap.add_argument("--seed", type=int, default=4605)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--thr", type=float, default=0.25, help="decision threshold for hit / wrong_class")
    ap.add_argument("--low", type=float, default=0.05, help="below this a cat box counts as not found")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO
    from eval import sample

    data = Path(args.data).expanduser()
    root = data / "yolo"
    img_dir, lbl_dir = root / "images" / args.split, root / "labels" / args.split
    items = sample(sorted(img_dir.glob("*.jpg")), lbl_dir, args.n, args.seed)
    if not items:
        sys.exit(f"no images in {img_dir}")
    locs = location_lookup(data / "meta")
    model = YOLO(args.model)
    names = model.names
    if "cat" not in names.values():
        sys.exit("model has no 'cat' class")

    cat_rows, confusion, sheet_items = [], Counter(), defaultdict(list)
    for i in range(0, len(items), args.batch):
        chunk = items[i : i + args.batch]
        results = model([str(p) for p, _ in chunk], imgsz=args.imgsz, conf=args.low, verbose=False)
        for (p, truth), r in zip(chunk, results):
            preds = [(names[int(c)], float(s), tuple(b)) for b, s, c in
                     zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            pred = primary_pred(preds, args.thr)
            confusion[(truth, pred if pred in CLASSES or pred == "background" else "other_coco")] += 1
            if truth != "cat":
                continue
            gt = read_labels(lbl_dir / f"{p.stem}.txt")
            gt_cats = [b for k, b in gt if k == "cat"]
            reason, called, called_conf, best_cat, found = classify_cat_image(gt_cats, preds, args.thr, args.low)
            with Image.open(p) as im:
                px = im.convert("RGB").resize((32, 32)).tobytes()
                night = is_grayscale(zip(px[0::3], px[1::3], px[2::3]))
            frac = max((area(b) for b in gt_cats), default=0.0)
            row = {
                "image": p.name, "reason": reason, "called_as": called or "", "called_conf": round(called_conf, 3),
                "best_cat_conf": round(best_cat, 3), "animal_found": found, "cat_area_frac": round(frac, 4),
                "size": size_bin(frac), "night": night, "n_cats": len(gt_cats),
                "others_in_frame": ",".join(sorted({k for k, _ in gt if k != "cat"})),
                "dataset": p.name.split("__", 1)[0], "location": locs.get(p.stem, "unknown"),
            }
            cat_rows.append(row)
            sheet_items[reason].append({**row, "gt": gt, "preds": preds})

    if not cat_rows:
        sys.exit("no cat images in the sample")
    out = Path("results") / args.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cat_images.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_rows[0]))
        w.writeheader()
        w.writerows(cat_rows)

    labels = ["cat", *[c for c in CLASSES if c != "cat"], "background", "other_coco"]
    with open(out / "confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["truth \\ predicted", *labels])
        for t in labels:
            if any(confusion[(t, p)] for p in labels):
                w.writerow([t, *[confusion[(t, p)] for p in labels]])

    n = len(cat_rows)
    reasons = Counter(r["reason"] for r in cat_rows)
    missed = [r for r in cat_rows if r["reason"] != "hit"]
    summary = {
        "model": args.model, "split": args.split, "threshold": args.thr, "n_cat_images": n,
        "cat_recall": reasons["hit"] / n,
        "animal_recall_any_class": sum(r["animal_found"] for r in cat_rows) / n,
        "miss_reasons": {k: v for k, v in reasons.items() if k != "hit"},
        "missed_cats_called": dict(Counter(r["called_as"] for r in missed if r["called_as"] and r["called_as"] != "cat")),
        "recall_by_size": recall_table(cat_rows, "size"),
        "recall_by_night": recall_table(cat_rows, "night"),
        "recall_by_dataset": recall_table(cat_rows, "dataset"),
        "recall_by_location": recall_table(cat_rows, "location"),
        "miss_reason_by_size": {s: dict(Counter(r["reason"] for r in cat_rows if r["size"] == s))
                                for s, _ in SIZE_BINS},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    for reason, its in sheet_items.items():
        its.sort(key=lambda x: x["best_cat_conf"])
        draw_sheet(its, img_dir, out / "sheets" / f"{reason}.jpg")

    print(f"{args.name}: {n} cat images @ thr {args.thr}")
    print(f"  cat recall              {summary['cat_recall']:.1%}")
    print(f"  found as any class      {summary['animal_recall_any_class']:.1%}   (gap to recall = cats boxed but mislabelled)")
    print(f"  misses                  {summary['miss_reasons']}")
    print(f"  missed cats called      {summary['missed_cats_called']}")
    for k in ("recall_by_size", "recall_by_night", "recall_by_dataset"):
        print(f"  {k:23s} " + "  ".join(f"{a}={v['recall']:.0%} ({v['n']})" for a, v in summary[k].items()))
    print(f"  wrote {out}/summary.json, cat_images.csv, confusion.csv, sheets/")


if __name__ == "__main__":
    main()
