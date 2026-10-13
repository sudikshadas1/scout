"""Unit tests for the zone geometry and K-of-N voting in edge/vision.py.

Purpose: Makes sure a cat is only reported in the trap when it really is, and doorway/non-cat cases are flagged.
Status:  Passing.
Later stages (still to add or change):
  - Add cases when zones change shape or a second camera is added.
"""
import json

import pytest

from edge.vision import (Detection, VisionAggregator, box_grounded_in, box_touches,
                         point_in_poly)

TRAP = [(0.35, 0.0), (1.0, 0.0), (1.0, 1.0), (0.35, 1.0)]
DOOR = [(0.0, 0.0), (0.2, 0.0), (0.2, 1.0), (0.0, 1.0)]
CAT_IN = Detection("cat", 0.9, (0.55, 0.3, 0.85, 0.8))
CAT_DOOR = Detection("cat", 0.9, (0.02, 0.3, 0.25, 0.8))
RACCOON = Detection("raccoon", 0.85, (0.5, 0.3, 0.85, 0.8))


def make():
    return VisionAggregator(N=5, K=4, trap_zone=TRAP, doorway_zone=DOOR)


def test_geometry():
    assert point_in_poly((0.5, 0.5), TRAP) and not point_in_poly((0.1, 0.5), TRAP)
    assert box_grounded_in(CAT_IN.box, TRAP)
    assert not box_grounded_in((0.3, 0.3, 0.6, 0.8), TRAP)  # straddles the trap edge
    assert box_touches(CAT_DOOR.box, DOOR) and not box_touches(CAT_IN.box, DOOR)


def test_cat_in_trap_after_k_frames():
    v = make()
    reports = [v.update([CAT_IN]) for _ in range(4)]
    assert [r.cat_in_trap for r in reports] == [False, False, False, True]
    assert reports[-1].cat_hits == 4 and reports[-1].cat_confidence == pytest.approx(0.9)


def test_k_of_n_tolerates_one_missed_frame():
    v = make()
    for dets in ([CAT_IN], [CAT_IN], [], [CAT_IN]):
        r = v.update(dets)
    assert not r.cat_in_trap
    assert v.update([CAT_IN]).cat_in_trap


def test_cat_in_doorway_never_counts():
    v = make()
    for _ in range(10):
        r = v.update([CAT_DOOR])
    assert not r.cat_in_trap and r.animal_in_doorway


def test_non_cat_is_reported_not_decided():
    v = make()
    r = v.update([CAT_IN, RACCOON])
    assert r.non_cat == "raccoon" and r.non_cat_confidence == pytest.approx(0.85)


def test_thresholds_and_unknown_classes():
    v = make()
    r = v.update([Detection("raccoon", 0.1, RACCOON.box), Detection("chair", 0.99, (0.5, 0.5, 0.6, 0.6))])
    assert r.non_cat is None and not r.anything_in_view


def test_bad_config():
    with pytest.raises(ValueError):
        VisionAggregator(N=3, K=4)


def test_config_file_loads():
    v = VisionAggregator.from_config()
    assert (v.N, v.K) == (5, 4) and v.thr["raccoon"] == 0.4
    assert json.load(open("edge/vision_config.json"))["trap_zone"]
