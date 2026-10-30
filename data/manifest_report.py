"""Counts what the model actually sees: cats per split, per camera, by time of day and by size in the frame.

Purpose: Explains the model's limits from the metadata alone, with no images needed. Rebuilding the manifest from the
         public labels matches PACE's totals (21,475 / 8,211 / 8,683 pictures, 208 val cats), so these should be the real counts.
Status:  Tested (tests/test_improvement.py). Results are in results/data_report.json.

    python data/fetch_metadata.py --out ~/scratch/scout_data
    python data/build_manifest.py --data ~/scratch/scout_data
    python data/manifest_report.py --data ~/scratch/scout_data --out results/data_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import CLASS_ID

CAT = CLASS_ID["cat"]


def period(stamp) -> str:
    """Clock time to a period. Handles '2013-10-04 13:31:53' and '7/29/2016 11:40'."""
    try:
        hour = int(str(stamp).split(" ")[1].split(":")[0])
    except (IndexError, ValueError):
        return "unknown"
    if 7 <= hour < 18:
        return "day"
    return "dusk_dawn" if hour in (5, 6, 18, 19) else "night"


def size_name(frac: float) -> str:
    return "tiny" if frac < 0.01 else "small" if frac < 0.05 else "medium" if frac < 0.20 else "large"


def report(rows: list[dict], images: dict) -> dict:
    """rows: manifest lines. images: {(dataset, file_name): image dict from the dataset's metadata}."""
    out: dict = {}
    for split in sorted({r["split"] for r in rows}):
        part = [r for r in rows if r["split"] == split]
        cats = [r for r in part if any(b[0] == CAT for b in r["boxes"])]
        cams, when, size = Counter(), Counter(), Counter()
        for r in cats:
            img = images.get((r["dataset"], r["file_name"]), {})
            cams[f"{r['dataset']}:{img.get('location', 'none')}"] += 1
            when[period(img.get("date_captured") or img.get("datetime"))] += 1
            if r.get("width") and r.get("height"):
                area = max(b[3] * b[4] for b in r["boxes"] if b[0] == CAT) / (r["width"] * r["height"])
                size[size_name(area)] += 1
        top = cams.most_common(1)[0] if cams else ("", 0)
        out[split] = {"pictures": len(part), "cats": len(cats), "cat_cameras": len(cams),
                      "top_camera": top[0], "top_camera_cats": top[1], "cats_by_camera": dict(cams.most_common()),
                      "cats_by_time": dict(when), "cats_by_size": dict(size)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/data_report.json")
    args = ap.parse_args()
    data = Path(args.data).expanduser()
    rows = [json.loads(line) for line in open(data / "manifest.jsonl")]
    images: dict = defaultdict(dict)
    for ds, name in (("cct", "cct_images.json"), ("ena24", "ena24.json")):
        for img in json.loads((data / "meta" / name).read_text())["images"]:
            images[(ds, img["file_name"])] = img
    rep = report(rows, images)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    for split, r in rep.items():
        print(f"{split}: {r['cats']:,} cats, {r['cat_cameras']} cameras (top {r['top_camera']}: {r['top_camera_cats']:,}), "
              f"{r['cats_by_time']}, {r['cats_by_size']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
