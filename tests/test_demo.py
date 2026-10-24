"""Tests for the laptop demo overlay (edge/demo.py) without loading a model.

Purpose: The demo screen must show the same report the gate sees.
Status:  Passing.
"""
import pytest

from edge.demo import draw, report_lines
from edge.gate import GateState
from edge.vision import Detection, VisionAggregator, VisionReport


def test_report_lines():
    r = VisionReport(cat_in_trap=True, cat_confidence=0.81, cat_hits=4, non_cat=None)
    lines = report_lines(r, GateState.CLOSED, "cat conf 0.81", 5)
    assert lines[0].startswith("GATE: CLOSED") and "YES" in lines[1] and "4/5" in lines[1]


def test_draw_shapes():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.zeros((240, 320, 3), np.uint8)
    v = VisionAggregator()
    dets = [Detection("cat", 0.9, (0.5, 0.3, 0.8, 0.9))]
    rep = v.update(dets)
    out = draw(frame, dets, rep, GateState.ARMED, "", v, fps=5.0, rec=True)
    assert out.shape == (240, 320 + 360, 3)
    assert frame.sum() == 0  # input frame is not modified
