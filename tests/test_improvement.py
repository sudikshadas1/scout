"""Tests for the model improvement tools: trap-accurate scoring, Wellington selection, crop check, curve and data report.

Purpose: These decide which model we keep, so the rules have to match the trap and the new data has to stay clean.
Status:  Passing.
"""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "training"))

from add_wellington import (build_rows, capped, clearest_per_sequence, select, split_locations,  # noqa: E402
                            usable_box, wellington_class, yolo_line)
from curve import summarize  # noqa: E402
from manifest_report import period, report, size_name  # noqa: E402
from score_pipeline import best_under, calls_cat, sweep  # noqa: E402
from train_crop_classifier import crop_pixels, image_scores, pad_box  # noqa: E402

from edge.vision import Detection  # noqa: E402

BOX = (0.2, 0.2, 0.6, 0.6)
FAR = (0.7, 0.7, 0.9, 0.9)


# ---- score_pipeline
def call(dets, rule, cat_cut=0.25, noncat_cut=0.4):
    return calls_cat(dets, rule, cat_cut, noncat_cut, 0.6, 0.1)


def test_weak_cat_on_a_raccoon_is_only_a_false_positive_for_cat_box():
    dets = [Detection("raccoon", 0.9, BOX), Detection("cat", 0.3, BOX)]
    assert call(dets, "cat_box")
    assert not call(dets, "one_label")
    assert not call(dets, "trap")


def test_trap_rule_holds_for_a_non_cat_elsewhere_in_the_frame():
    dets = [Detection("cat", 0.8, BOX), Detection("raccoon", 0.5, FAR)]
    assert call(dets, "one_label")
    assert not call(dets, "trap")


def test_clear_cat_beats_a_weak_label_on_the_same_animal():
    assert call([Detection("cat", 0.9, BOX), Detection("opossum", 0.5, BOX)], "trap")


def test_sweep_counts_and_best_under():
    records = [{"truth": "cat", "dets": [Detection("cat", 0.3, BOX)]},
               {"truth": "cat", "dets": []},
               {"truth": "raccoon", "dets": [Detection("raccoon", 0.9, BOX), Detection("cat", 0.3, BOX)]}]
    rows = sweep(records, 0.6, 0.1)
    plain = next(r for r in rows if r["rule"] == "cat_box" and r["cat_cutoff"] == 0.25)
    trap = next(r for r in rows if r["rule"] == "trap" and r["cat_cutoff"] == 0.25 and r["noncat_cutoff"] == 0.4)
    assert (plain["cats_caught"], plain["false_positives"]) == (1, 1)
    assert (trap["cats_caught"], trap["false_positives"]) == (1, 0)
    assert best_under(rows, "cat_box") is not None and best_under(rows, "cat_box")["cat_recall"] == 0
    assert best_under(rows, "trap")["cat_recall"] == 0.5


def test_empty_pictures_do_not_count_toward_the_animal_goal():
    records = [{"truth": "cat", "dets": [Detection("cat", 0.9, BOX)]},
               {"truth": "raccoon", "dets": [Detection("cat", 0.9, BOX)]},
               {"truth": "raccoon", "dets": []},
               {"truth": "background", "dets": []},
               {"truth": "background", "dets": []}]
    r = next(x for x in sweep(records, 0.6, 0.1) if x["rule"] == "cat_box" and x["cat_cutoff"] == 0.25)
    assert r["animal_fpr"] == 0.5 and r["noncat_fpr"] == 0.25


# ---- add_wellington
def test_nz_possum_is_not_our_opossum():
    assert wellington_class("possum") == "other_animal"
    assert wellington_class("cat") == "cat"
    assert wellington_class("hedgehog") == "other_animal"
    assert wellington_class("unclassifiable") is None and wellington_class("empty") is None


def test_usable_box_rejects_footer_slivers_and_weak_boxes():
    good = {"category": "1", "conf": 0.9, "bbox": [0.2, 0.2, 0.3, 0.3]}
    assert usable_box(good, 0.7)
    assert not usable_box({**good, "conf": 0.5}, 0.7)
    assert not usable_box({**good, "category": "2"}, 0.7)
    assert not usable_box({**good, "bbox": [0.5, 0.93, 0.2, 0.05]}, 0.7)   # on the footer strip
    assert not usable_box({**good, "bbox": [0.5, 0.5, 0.01, 0.3]}, 0.7)    # sliver
    assert usable_box({**good, "bbox": [0.5, 0.5, 0.3, 0.5]}, 0.7)         # big animal reaching the bottom


def test_split_locations_keeps_whole_cameras_and_is_repeatable():
    locs = [f"c{i}" for i in range(10)] * 3
    a = split_locations(locs, 0.2, 1)
    assert a == split_locations(locs, 0.2, 1)
    assert sum(v == "val" for v in a.values()) == 2 and len(a) == 10


def test_clearest_per_sequence_and_cap():
    rows = [{"seq": "s", "conf": 0.7, "location": "a"}, {"seq": "s", "conf": 0.9, "location": "a"},
            {"seq": "t", "conf": 0.8, "location": "a"}]
    best = clearest_per_sequence(rows)
    assert sorted(r["conf"] for r in best) == [0.8, 0.9]
    assert len(capped(rows, 10, 2, random.Random(0))) == 2
    assert len(capped(rows, 1, 2, random.Random(0))) == 1


def fake_wct():
    meta = {"categories": [{"id": 0, "name": "cat"}, {"id": 1, "name": "bird"}, {"id": 2, "name": "empty"}],
            "images": [], "annotations": []}
    md = {"images": []}
    def add(i, loc, seq, cat, dets):
        fn = f"{i}.JPG"
        meta["images"].append({"id": str(i), "file_name": fn, "location": loc, "seq_id": seq})
        meta["annotations"].append({"image_id": str(i), "category_id": cat})
        md["images"].append({"file": fn, "detections": dets})
    det = {"category": "1", "conf": 0.9, "bbox": [0.1, 0.1, 0.4, 0.4]}
    for loc in ("a", "b", "c", "d", "e"):
        add(f"{loc}1", loc, f"{loc}s1", 0, [det])
        add(f"{loc}2", loc, f"{loc}s1", 0, [{**det, "conf": 0.8}])     # same sequence, less clear
        add(f"{loc}3", loc, f"{loc}s2", 1, [det])
        add(f"{loc}4", loc, f"{loc}s3", 2, [])                          # really empty
        add(f"{loc}5", loc, f"{loc}s4", 2, [det])                       # "empty" with an animal: dropped
        add(f"{loc}6", loc, f"{loc}s5", 0, [])                          # cat label, nothing in this frame: dropped
    return meta, md


def test_build_rows_and_select():
    meta, md = fake_wct()
    rows = build_rows(meta, md, 0.7, 0.2, 4605)
    assert len(rows) == 15                                              # per camera: 1 cat, 1 bird, 1 empty
    assert {r["file_name"] for r in rows if r["cls"] == "cat"} == {f"{c}1.JPG" for c in "abcde"}
    assert not {f"{c}5.JPG" for c in "abcde"} & {r["file_name"] for r in rows}
    train = select(rows, "train", 100, 100, 0.1, random.Random(0))
    assert {r["split"] for r in train} == {"train"}
    cams_val = {r["location"] for r in rows if r["split"] == "val"}
    assert cams_val and not cams_val & {r["location"] for r in train}
    assert sum(r["cls"] == "cat" for r in train) == 4 and sum(r["cls"] == "other_animal" for r in train) == 4


def test_yolo_line():
    assert yolo_line({"cls": "cat", "bbox": [0.1, 0.2, 0.4, 0.2]}) == "0 0.300000 0.300000 0.400000 0.200000"
    assert yolo_line({"cls": None, "bbox": None}) == ""


# ---- train_crop_classifier
def test_crop_box_stays_in_the_picture():
    assert pad_box((0.0, 0.0, 1.0, 1.0)) == (0.0, 0.0, 1.0, 1.0)
    left, top, right, bottom = crop_pixels((0.5, 0.5, 0.6, 0.6), 1000, 1000)
    assert (left, top, right, bottom) == (490, 490, 610, 610)
    l2, t2, r2, b2 = crop_pixels((0.5, 0.5, 0.5, 0.5), 100, 100)
    assert r2 > l2 and b2 > t2


def test_image_scores_uses_the_detector_cutoff():
    records = [{"image": "a", "truth": "cat", "dets": [["cat", 0.5, BOX], ["raccoon", 0.05, FAR]]},
               {"image": "b", "truth": "raccoon", "dets": []}]
    probs = {("a", 0): 0.7, ("a", 1): 0.99}
    assert image_scores(records, probs, 0.1) == [("a", "cat", 0.7), ("b", "raccoon", 0.0)]


# ---- curve
def test_curve_summary():
    rows = [{" epoch": str(i), "time": str(i * 60), "metrics/mAP50-95(B)": str(0.1 * i)} for i in range(1, 6)]
    s = summarize(rows, window=2)
    assert s["epochs_run"] == 5 and s["best_epoch"] == 5 and s["still_improving"] and s["train_minutes"] == 5.0
    rows[0]["metrics/mAP50-95(B)"] = "0.9"
    s = summarize(rows, window=2)
    assert s["best_epoch"] == 1 and not s["still_improving"]


# ---- manifest_report
def test_period_and_size():
    assert period("2013-10-04 13:31:53") == "day" and period("7/29/2016 02:40") == "night"
    assert period("7/29/2016 18:10") == "dusk_dawn" and period(None) == "unknown"
    assert size_name(0.005) == "tiny" and size_name(0.5) == "large"


def test_report_counts_cats_by_camera():
    rows = [{"dataset": "cct", "file_name": "a", "split": "train", "width": 100, "height": 100, "boxes": [[0, 0, 0, 50, 50]]},
            {"dataset": "cct", "file_name": "b", "split": "train", "width": 100, "height": 100, "boxes": [[1, 0, 0, 50, 50]]},
            {"dataset": "ena24", "file_name": "c", "split": "train", "width": 100, "height": 100, "boxes": [[0, 0, 0, 1, 1]]}]
    images = {("cct", "a"): {"location": 38, "date_captured": "2013-10-04 13:31:53"}}
    r = report(rows, images)["train"]
    assert r["cats"] == 2 and r["pictures"] == 3 and r["cat_cameras"] == 2
    assert r["cats_by_camera"]["cct:38"] == 1 and r["cats_by_time"]["day"] == 1
    assert r["cats_by_size"] == {"large": 1, "tiny": 1}


if __name__ == "__main__":
    pytest.main([__file__])
