"""Tests for the miss-analysis helpers (training/analyze_errors.py) and the cat-oversampling list (data/make_train_list.py).

Purpose: The failure buckets drive what we change next in training, so they must be computed correctly.
Status:  Passing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "training"))

from analyze_errors import (area, classify_cat_image, iou, is_grayscale, primary_pred,  # noqa: E402
                            read_labels, recall_table, size_bin)
from make_train_list import build_list, has_cat, yaml_text  # noqa: E402

CAT_BOX = (0.4, 0.4, 0.6, 0.6)


def test_iou_and_area():
    assert iou(CAT_BOX, CAT_BOX) == 1.0
    assert iou(CAT_BOX, (0.0, 0.0, 0.1, 0.1)) == 0.0
    assert abs(area(CAT_BOX) - 0.04) < 1e-9


def test_size_bins():
    assert size_bin(0.005) == "tiny" and size_bin(0.03) == "small"
    assert size_bin(0.1) == "medium" and size_bin(0.5) == "large" and size_bin(1.0) == "large"


def test_classify_hit():
    assert classify_cat_image([CAT_BOX], [("cat", 0.8, CAT_BOX)], 0.25, 0.05)[0] == "hit"


def test_classify_wrong_class():
    reason, called, conf, best_cat, found = classify_cat_image(
        [CAT_BOX], [("raccoon", 0.7, CAT_BOX), ("cat", 0.1, CAT_BOX)], 0.25, 0.05)
    assert (reason, called, found) == ("wrong_class", "raccoon", True) and best_cat == 0.1


def test_classify_low_conf_and_not_found():
    assert classify_cat_image([CAT_BOX], [("cat", 0.1, CAT_BOX)], 0.25, 0.05)[0] == "low_conf_cat"
    assert classify_cat_image([CAT_BOX], [], 0.25, 0.05)[0] == "not_found"
    # a confident raccoon somewhere else in the frame is not "the cat was called a raccoon"
    far = (0.0, 0.0, 0.1, 0.1)
    assert classify_cat_image([CAT_BOX], [("raccoon", 0.9, far)], 0.25, 0.05)[0] == "not_found"


def test_primary_pred():
    assert primary_pred([], 0.25) == "background"
    assert primary_pred([("opossum", 0.9, CAT_BOX), ("cat", 0.3, CAT_BOX)], 0.25) == "cat"
    assert primary_pred([("opossum", 0.9, CAT_BOX), ("cat", 0.1, CAT_BOX)], 0.25) == "opossum"


def test_grayscale():
    assert is_grayscale([(100, 100, 100), (30, 31, 30)])
    assert not is_grayscale([(200, 50, 40), (10, 120, 30)])


def test_recall_table():
    rows = [{"size": "large", "reason": "hit"}, {"size": "large", "reason": "not_found"},
            {"size": "tiny", "reason": "not_found"}]
    t = recall_table(rows, "size")
    assert t["large"] == {"recall": 0.5, "hit": 1, "n": 2} and t["tiny"]["recall"] == 0.0


def test_read_labels(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n")
    lab = read_labels(f)
    assert lab[0][0] == "cat" and lab[1][0] == "raccoon"
    assert all(abs(a - b) < 1e-9 for a, b in zip(lab[0][1], CAT_BOX))
    assert read_labels(tmp_path / "missing.txt") == []


def test_oversample_list(tmp_path):
    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    (lbls / "c.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (lbls / "r.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    (lbls / "e.txt").write_text("")
    paths = [imgs / f"{s}.jpg" for s in ("c", "r", "e")]
    assert has_cat(lbls / "c.txt") and not has_cat(lbls / "r.txt") and not has_cat(lbls / "e.txt")
    out = build_list(paths, lbls, cat_repeat=3)
    assert out.count(str(paths[0])) == 3 and out.count(str(paths[1])) == 1 and len(out) == 5
    y = yaml_text(tmp_path, "train_cat3x.txt")
    assert "train: train_cat3x.txt" in y and "0: cat" in y and "6: person" in y
