"""Prints images per species and per class (and how many have boxes) for each dataset.

Purpose: Gives the real counts the plan needs to choose class balance and confirm we have enough cats.
Status:  Written, tested on fake data and run on the real metadata.
Later stages (still to add or change):
  - Phase 2a: also count our own trap footage once it is labeled.

    python data/stats.py --data ~/scratch/scout_data
"""
import argparse
from collections import Counter
from pathlib import Path

from common import CocoCT, species_to_class


def summarize(path: Path, label: str) -> None:
    ds = CocoCT(path)
    species, classes, boxed = Counter(), Counter(), Counter()
    for img_id in ds.images:
        names = ds.species_of(img_id)
        for n in names:
            species[n] += 1
            cls = species_to_class(n)
            classes[cls or "empty/dropped"] += 1
            if any("bbox" in a for a in ds.anns_by_image[img_id]):
                boxed[cls or "empty/dropped"] += 1
    locations = {ds.location_of(i) for i in ds.images}
    print(f"\n=== {label}: {len(ds.images):,} images, {len(locations)} locations ===")
    print("Our classes (images / with boxes):")
    for cls, n in classes.most_common():
        print(f"  {cls:15s} {n:8,d}  {boxed[cls]:8,d}")
    print("Top raw species:")
    for name, n in species.most_common(25):
        print(f"  {name:25s} {n:8,d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    meta = Path(args.data).expanduser() / "meta"
    summarize(meta / "cct_images.json", "CCT image-level labels")
    summarize(meta / "cct_bboxes.json", "CCT bounding boxes")
    summarize(meta / "ena24.json", "ENA24")


if __name__ == "__main__":
    main()
