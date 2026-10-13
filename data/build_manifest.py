"""Choose which images go into the YOLO dataset and with which boxes, without downloading anything.

Writes <data>/manifest.jsonl, one line per image:
    {"dataset", "file_name", "url", "split", "stem", "width", "height", "boxes": [[cls, x, y, w, h], ...]}
Boxes are absolute pixels, top-left x/y + width/height (COCO style). download_images.py turns them
into YOLO label files once the image is on disk.

Rules (see the plan, "Data plan"):
  * splits come from data/splits.json (location-based, frozen)
  * at most --per-seq frames per camera burst (sequence)
  * train only: each non-cat class capped at --ratio x the cat count; empties kept as
    ~--bg-frac background images
  * val/test: capped at --eval-cap images per class to keep evaluation fast
  * optional MegaDetector pseudo-boxes (--md) for single-species CCT images without human boxes

    python data/build_manifest.py --data ~/scratch/scout_data
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import CLASS_ID, CocoCT, is_empty, species_to_class

URL = {
    "cct": "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/{}",
    "ena24": "https://storage.googleapis.com/public-datasets-lila/ena24/images/{}",
}


def split_of(dataset: str, img: dict, splits: dict) -> str | None:
    key = str(img.get("location", "seq:" + str(img.get("seq_id", img["id"]))))
    for name, groups in splits[dataset].items():
        if key in groups:
            return name
    return None


def boxes_from_human(ds: CocoCT, img_id) -> list | None:
    """Human boxes for one image. Returns [] for an empty image, None if unusable."""
    anns = ds.anns_by_image.get(img_id, [])
    if not anns:
        return None
    out = []
    for a in anns:
        name = ds.cat_name[a["category_id"]]
        if is_empty(name):
            continue
        cls = species_to_class(name)
        if cls is None:
            return None  # e.g. a car in frame: skip the image
        if "bbox" not in a:
            return None  # species label without a box: can't use as a detection target
        out.append([CLASS_ID[cls], *a["bbox"]])
    return out


def load_md(path: Path, conf: float) -> dict[str, list]:
    """MegaDetector batch output -> {file: [[x, y, w, h] normalized, ...]} (animal boxes only)."""
    d = json.loads(path.read_text())
    out = {}
    for im in d.get("images", []):
        dets = [det["bbox"] for det in im.get("detections") or []
                if str(det.get("category")) == "1" and det.get("conf", 0) >= conf]
        out[im["file"]] = dets
    return out


def collect(dataset: str, ds: CocoCT, splits: dict, per_seq: int, rng: random.Random,
            image_level: CocoCT | None = None, md: dict | None = None) -> list[dict]:
    rows, seq_count = [], Counter()
    ids = list(ds.images)
    rng.shuffle(ids)
    for img_id in ids:
        img = ds.images[img_id]
        split = split_of(dataset, img, splits)
        if split is None:
            continue
        seq = img.get("seq_id")
        if seq is not None and seq_count[seq] >= per_seq:
            continue
        boxes = boxes_from_human(ds, img_id)
        if boxes is None:
            continue
        rows.append(make_row(dataset, img, split, boxes))
        if seq is not None:
            seq_count[seq] += 1

    # MegaDetector pseudo-labels for CCT images that have only an image-level species label.
    if image_level is not None and md is not None:
        have = {r["file_name"] for r in rows}
        for img_id, img in image_level.images.items():
            fn = img["file_name"]
            if fn in have or fn not in md:
                continue
            names = {n for n in image_level.species_of(img_id) if not is_empty(n)}
            classes = {species_to_class(n) for n in names}
            if len(classes) != 1 or None in classes or not md[fn]:
                continue
            split = split_of(dataset, img, splits)
            seq = img.get("seq_id")
            if split is None or (seq is not None and seq_count[seq] >= per_seq):
                continue
            W, H = img.get("width"), img.get("height")
            if not W or not H:
                continue
            cls = CLASS_ID[classes.pop()]
            boxes = [[cls, x * W, y * H, w * W, h * H] for x, y, w, h in md[fn]]
            row = make_row(dataset, img, split, boxes)
            row["pseudo"] = True
            rows.append(row)
            if seq is not None:
                seq_count[seq] += 1
    return rows


def collect_backgrounds(dataset: str, ds: CocoCT, splits: dict, have: set, rng: random.Random,
                        per_seq: int = 1) -> list[dict]:
    """Empty frames (no animal) to use as background images: one per burst."""
    rows, seen_seq = [], Counter()
    ids = list(ds.images)
    rng.shuffle(ids)
    for img_id in ids:
        img = ds.images[img_id]
        names = ds.species_of(img_id)
        if not names or not all(is_empty(n) for n in names) or img["file_name"] in have:
            continue
        seq = img.get("seq_id")
        if seq is not None and seen_seq[seq] >= per_seq:
            continue
        split = split_of(dataset, img, splits)
        if split is None:
            continue
        rows.append(make_row(dataset, img, split, []))
        if seq is not None:
            seen_seq[seq] += 1
    return rows


def make_row(dataset: str, img: dict, split: str, boxes: list) -> dict:
    fn = img["file_name"]
    return {
        "dataset": dataset,
        "file_name": fn,
        "url": URL[dataset].format(fn),
        "split": split,
        "stem": f"{dataset}__" + fn.replace("/", "__").rsplit(".", 1)[0],
        "width": img.get("width"),
        "height": img.get("height"),
        "boxes": boxes,
    }


def primary_class(row: dict) -> str:
    if not row["boxes"]:
        return "background"
    ids = [b[0] for b in row["boxes"]]
    return "cat" if CLASS_ID["cat"] in ids else str(Counter(ids).most_common(1)[0][0])


def balance(rows: list[dict], ratio: float, bg_frac: float, eval_cap: int, rng: random.Random) -> list[dict]:
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["split"]][primary_class(r)].append(r)
    out = []
    for split, groups in by.items():
        if split == "train":
            n_cat = len(groups.get("cat", []))
            cap = max(int(ratio * n_cat), 1)
            kept = []
            for cls, rs in groups.items():
                if cls in ("cat", "background"):
                    continue
                kept += rs[:cap]
            kept += groups.get("cat", [])
            n_bg = int(bg_frac * len(kept) / max(1 - bg_frac, 1e-6))
            kept += groups.get("background", [])[:n_bg]
        else:
            kept = [r for rs in groups.values() for r in rs[:eval_cap]]
        rng.shuffle(kept)
        out += kept
    return out


def report(rows: list[dict]) -> None:
    c = Counter((r["split"], primary_class(r)) for r in rows)
    names = {str(i): n for n, i in CLASS_ID.items()} | {"cat": "cat", "background": "background"}
    for split in ("train", "val", "test"):
        parts = {names.get(k, k): v for (s, k), v in c.items() if s == split}
        print(f"{split:5s} {sum(parts.values()):7,d}  {dict(sorted(parts.items()))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--per-seq", type=int, default=2)
    ap.add_argument("--ratio", type=float, default=2.0, help="max non-cat images per class / cat images (train)")
    ap.add_argument("--bg-frac", type=float, default=0.10)
    ap.add_argument("--eval-cap", type=int, default=2000)
    ap.add_argument("--md", help="MegaDetector results json for CCT (optional pseudo-labels)")
    ap.add_argument("--md-conf", type=float, default=0.5)
    ap.add_argument("--no-ena24", action="store_true")
    ap.add_argument("--seed", type=int, default=4605)
    args = ap.parse_args()

    data = Path(args.data).expanduser()
    meta = data / "meta"
    splits = json.loads(Path(args.splits).read_text())
    rng = random.Random(args.seed)

    cct_boxes = CocoCT(meta / "cct_bboxes.json")
    image_level = CocoCT(meta / "cct_images.json")
    md = load_md(Path(args.md).expanduser(), args.md_conf) if args.md else None
    rows = collect("cct", cct_boxes, splits, args.per_seq, rng, image_level, md)
    rows += collect_backgrounds("cct", image_level, splits, {r["file_name"] for r in rows}, rng)
    if not args.no_ena24:
        rows += collect("ena24", CocoCT(meta / "ena24.json"), splits, args.per_seq, rng)

    rows = balance(rows, args.ratio, args.bg_frac, args.eval_cap, rng)
    out = data / "manifest.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report(rows)
    print(f"wrote {out} ({len(rows):,} images)")


if __name__ == "__main__":
    main()
