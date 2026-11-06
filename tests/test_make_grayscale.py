"""Tests for the E4 grayscale training copies.

Purpose: E4 is only comparable with E1b if the labels stay matched and val keeps its original pictures.
Status:  Passing where opencv is installed; skipped otherwise.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from make_grayscale import convert  # noqa: E402
from make_train_list import yaml_text  # noqa: E402

cv2 = pytest.importorskip("cv2")


def make_dataset(root: Path):
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    img = np.zeros((32, 32, 3), np.uint8)
    img[:, :, 2] = 200   # pure red
    cv2.imwrite(str(root / "images" / "train" / "a.jpg"), img)
    cv2.imwrite(str(root / "images" / "train" / "empty.jpg"), img)
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def test_convert_makes_gray_copies_with_labels(tmp_path):
    make_dataset(tmp_path)
    assert convert(tmp_path) == 2
    out = cv2.imread(str(tmp_path / "images" / "train_gray" / "a.jpg"))
    assert out.shape == (32, 32, 3)
    assert out[:, :, 0].mean() == pytest.approx(out[:, :, 2].mean(), abs=2)   # no colour left
    assert (tmp_path / "labels" / "train_gray" / "a.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert not (tmp_path / "labels" / "train_gray" / "empty.txt").exists()
    # originals untouched
    assert cv2.imread(str(tmp_path / "images" / "train" / "a.jpg"))[:, :, 2].mean() > 150


def test_convert_can_write_a_separate_eval_copy(tmp_path):
    make_dataset(tmp_path)
    out = tmp_path / "gray" / "yolo"
    assert convert(tmp_path, "train", out) == 2
    assert (out / "images" / "train" / "a.jpg").exists() and (out / "labels" / "train" / "a.txt").exists()


def test_yaml_trains_on_gray_and_keeps_val(tmp_path):
    text = yaml_text(tmp_path, "images/train_gray")
    assert "train: images/train_gray" in text and "val: images/val" in text
