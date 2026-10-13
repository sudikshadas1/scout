"""Download the images listed in manifest.jsonl and write matching YOLO label files.

Output (what data/scout.yaml points at):
    <data>/yolo/images/{train,val,test}/<stem>.jpg
    <data>/yolo/labels/{train,val,test}/<stem>.txt   one "cls cx cy w h" line per box, normalized

Safe to re-run: existing images are skipped. Run it on PACE (a login node is fine for a first small
batch; use a Slurm job for the full set).

    python data/download_images.py --data ~/scratch/scout_data --workers 16
    python data/download_images.py --data ~/scratch/scout_data --limit 500   # quick sample
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


def yolo_lines(boxes: list, W: float, H: float) -> list[str]:
    lines = []
    for cls, x, y, w, h in boxes:
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = w / W, h / H
        # clamp boxes that run slightly past the frame edge
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
        if nw <= 0 or nh <= 0:
            continue
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def handle(row: dict, root: Path, session: requests.Session) -> str:
    img_path = root / "images" / row["split"] / f"{row['stem']}.jpg"
    lbl_path = root / "labels" / row["split"] / f"{row['stem']}.txt"
    if not img_path.exists():
        r = session.get(row["url"], timeout=60)
        if r.status_code != 200:
            return f"fail {r.status_code} {row['url']}"
        img_path.write_bytes(r.content)
    W, H = row.get("width"), row.get("height")
    if not W or not H:
        W, H = image_size(img_path)
    lbl_path.write_text("\n".join(yolo_lines(row["boxes"], W, H)))
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (for a quick test)")
    args = ap.parse_args()

    data = Path(args.data).expanduser()
    root = data / "yolo"
    rows = [json.loads(line) for line in open(data / "manifest.jsonl")]
    if args.limit:
        rows = rows[: args.limit]
    for split in {r["split"] for r in rows}:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    fails = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(handle, r, root, session) for r in rows]
        for f in tqdm(as_completed(futs), total=len(futs)):
            res = f.result()
            if res != "ok":
                fails.append(res)
    print(f"done: {len(rows) - len(fails):,} ok, {len(fails):,} failed")
    if fails:
        (data / "download_failures.txt").write_text("\n".join(fails))
        print(f"see {data / 'download_failures.txt'}")


if __name__ == "__main__":
    main()
