"""Builds the "seen locations" test split: new pictures from camera locations the model DID train on.

Purpose: Plan section 7. Comparing this score with the normal val/test score (new locations) shows how much
         the model relies on backgrounds: a big gap means it learned scenery, not cats. No retraining needed.
Status:  Written; selection logic tested (tests/test_new_tools.py). Run on E1: cats exist only in bursts already used for training, so seen vs new could not be compared (reported as a limitation).
How it avoids leaks: it only takes pictures from camera bursts (sequences) that have NO picture in the
         training manifest, so a near-identical frame of the same animal was never trained on.

    python data/build_seen_split.py --data ~/scratch/scout_data            # picks + downloads (~1,000 images)
    python training/eval.py --model results/E1_full/weights/best.pt --data ~/scratch/scout_data \
        --split seen --n 500 --name E1_full_seen
    # compare cat recall with results/E1_full_eval (new locations)

Writes <data>/manifest_seen.jsonl and <data>/yolo/{images,labels}/seen/.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_manifest import boxes_from_human, make_row, primary_class
from common import CocoCT


def train_locations(splits: dict) -> set[str]:
    return {str(x) for x in splits["cct"]["train"]}


def pick_seen(ds: CocoCT, splits: dict, used_files: set[str], used_seqs: set, per_class: int,
              per_seq: int, rng: random.Random) -> list[dict]:
    locs = train_locations(splits)
    by_cls, seq_count = defaultdict(list), Counter()
    ids = list(ds.images)
    rng.shuffle(ids)
    for i in ids:
        img = ds.images[i]
        if str(img.get("location")) not in locs or img["file_name"] in used_files:
            continue
        seq = img.get("seq_id")
        if seq in used_seqs or (seq is not None and seq_count[seq] >= per_seq):
            continue
        boxes = boxes_from_human(ds, i)
        if not boxes:  # skip empties and unusable images; this split is for animals
            continue
        row = make_row("cct", img, "seen", boxes)
        cls = primary_class(row)
        if len(by_cls[cls]) >= per_class:
            continue
        by_cls[cls].append(row)
        if seq is not None:
            seq_count[seq] += 1
    return [r for rs in by_cls.values() for r in rs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--per-class", type=int, default=250, help="max images per class (cat gets the same cap)")
    ap.add_argument("--per-seq", type=int, default=1)
    ap.add_argument("--seed", type=int, default=4605)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    data = Path(args.data).expanduser()
    splits = json.loads(Path(args.splits).read_text())
    ds = CocoCT(data / "meta" / "cct_bboxes.json")
    manifest = [json.loads(line) for line in open(data / "manifest.jsonl")]
    used_files = {r["file_name"] for r in manifest if r["dataset"] == "cct"}
    by_file = {img["file_name"]: img for img in ds.images.values()}
    used_seqs = {by_file[f].get("seq_id") for f in used_files if f in by_file} - {None}

    rows = pick_seen(ds, splits, used_files, used_seqs, args.per_class, args.per_seq, random.Random(args.seed))
    out = data / "manifest_seen.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"seen split: {len(rows)} images {dict(Counter(primary_class(r) for r in rows))}")
    print("  (class ids: 1 raccoon, 2 opossum, 3 squirrel, 4 dog, 5 other_animal, 6 person)")
    print(f"wrote {out}")
    if args.no_download:
        return

    from concurrent.futures import ThreadPoolExecutor
    import requests
    from download_images import handle
    root = data / "yolo"
    (root / "images" / "seen").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "seen").mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    with ThreadPoolExecutor(args.workers) as pool:
        res = list(pool.map(lambda r: handle(r, root, session), rows))
    print(f"downloaded: {res.count('ok')} ok, {len(res) - res.count('ok')} failed")


if __name__ == "__main__":
    main()
