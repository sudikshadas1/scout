"""Writes grayscale copies of the training images, plus a dataset yaml that uses them (experiment E4).

Purpose: The NoIR camera washes out colour, so E4 tests whether a model trained without colour holds up better.
         The original val pictures are untouched, so scores stay comparable with E1b. A grayscale copy of val is
         also written, because a model trained on grayscale should be judged on grayscale too.
Status:  Written; conversion and yaml tested (tests/test_make_grayscale.py). Not yet run on PACE.
Later stages (still to add or change):
  - Once night footage exists, also score E4 on it: public pictures have little true NoIR colour shift.
  - If E4 wins, edge/vision.py must convert camera frames to grayscale before the detector. It doesn't yet.

Images keep 3 channels so the same model and NCNN export work. Copies go next to the originals, in the data
folder, not the repo. Images already converted are skipped, so a stopped run can be restarted.

    python data/make_grayscale.py --data ~/scratch/scout_data
    # -> ~/scratch/scout_data/yolo/images/train_gray, labels/train_gray, ~/scratch/scout_data/scout_gray.yaml,
    #    and ~/scratch/scout_data/gray/yolo/{images,labels}/val
    sbatch -J scout-E4 training/slurm/train_pace.sbatch E4 --epochs 80 --patience 25 \
        --data-yaml ~/scratch/scout_data/scout_gray.yaml
    # score on both: eval_pace.sbatch E4 500 416, then on the grayscale val copy, with the same pictures
    python training/eval.py --model results/E4/weights/best.pt --data ~/scratch/scout_data/gray --split val --n 500 \
        --imgsz 416 --name E4_gray416
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


def convert(root: Path, split: str = "train", out_root: Path | None = None) -> int:
    """Copies images/<split> to images/<split>_gray, or to the same split under out_root, with labels alongside.

    Returns the number of images.
    """
    img_out = out_root / "images" / split if out_root else root / "images" / f"{split}_gray"
    lbl_out = out_root / "labels" / split if out_root else root / "labels" / f"{split}_gray"
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
    n_val = convert(root, "val", data / "gray" / "yolo")
    print(f"{n:,} grayscale training images, {n_val:,} val images\nwrote {yml}")


if __name__ == "__main__":
    main()
