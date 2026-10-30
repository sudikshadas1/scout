"""Writes grayscale copies of the training images, plus a dataset yaml that uses them (experiment E4).

Purpose: The NoIR camera washes out colour, so E4 tests whether a model trained without colour holds up better.
         Val and test are untouched, so scores stay comparable with E1b.
Status:  Written; conversion and yaml tested (tests/test_make_grayscale.py). Not yet run on PACE.
Later stages (still to add or change):
  - Once night footage exists, also score E4 on it: public pictures have little true NoIR colour shift.

Images keep 3 channels so the same model and NCNN export work. Copies go next to the originals, in the data
folder, not the repo. Images already converted are skipped, so a stopped run can be restarted.

    python data/make_grayscale.py --data ~/scratch/scout_data
    # -> ~/scratch/scout_data/yolo/images/train_gray, labels/train_gray and ~/scratch/scout_data/scout_gray.yaml
    sbatch -J scout-E4 training/slurm/train_pace.sbatch E4 --epochs 80 --patience 25 \
        --data-yaml ~/scratch/scout_data/scout_gray.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from make_train_list import yaml_text


def to_gray(src: Path, dst: Path) -> None:
    import cv2
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not read {src}")
    cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def convert(root: Path, split: str = "train") -> int:
    """Copies images/<split> to images/<split>_gray, with labels alongside. Returns the number of images."""
    img_out, lbl_out = root / "images" / f"{split}_gray", root / "labels" / f"{split}_gray"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    images = sorted((root / "images" / split).glob("*.jpg"))
    for p in images:
        dst = img_out / p.name
        if not dst.exists():
            to_gray(p, dst)
        label = root / "labels" / split / f"{p.stem}.txt"
        if label.exists():   # empty images may have no label file
            shutil.copy(label, lbl_out / label.name)
    return len(images)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    data = Path(args.data).expanduser().resolve()
    root = data / "yolo"
    n = convert(root)
    if not n:
        raise SystemExit(f"no training images in {root / 'images' / 'train'}")
    yml = data / "scout_gray.yaml"
    yml.write_text(yaml_text(root, "images/train_gray"))
    print(f"{n:,} grayscale training images\nwrote {yml}")


if __name__ == "__main__":
    main()
