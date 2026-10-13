"""Unit tests for the gate controller's rules (close, cooldown, latch, fault, pressure plate).

Purpose: Protects the safety rules from being broken by later changes.
Status:  Passing.
Later stages (still to add or change):
  - Hardware team: add a test for each new rule added to gate.py.
"""
from edge.gate import Command, GateController, GateState
from edge.vision import VisionReport


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


CAT = VisionReport(cat_in_trap=True, cat_confidence=0.9, cat_hits=4, n_detections=1)
CAT_DOOR = VisionReport(cat_in_trap=True, cat_hits=4, animal_in_doorway=True, n_detections=2)
RACCOON = VisionReport(non_cat="raccoon", non_cat_confidence=0.8, n_detections=1)
NOTHING = VisionReport()


def armed(**kw):
    clock = Clock()
    g = GateController(clock=clock, **kw)
    g.arm()
    return g, clock


def test_disarmed_never_closes():
    g = GateController()
    assert g.tick(CAT).command == Command.KEEP_OPEN


def test_closes_on_cat_and_latches():
    g, _ = armed()
    assert g.tick(CAT).command == Command.CLOSE
    assert g.tick(NOTHING).command == Command.CLOSE  # stays closed until a person resets
    g.reset()
    assert g.state == GateState.ARMED


def test_doorway_blocks_close():
    g, _ = armed()
    assert g.tick(CAT_DOOR).command == Command.KEEP_OPEN


def test_non_cat_starts_cooldown():
    g, clock = armed()
    assert g.tick(RACCOON).state == GateState.COOLDOWN
    clock.t = 5
    assert g.tick(CAT).command == Command.KEEP_OPEN
    clock.t = 11
    assert g.tick(CAT).command == Command.CLOSE


def test_plate_required():
    g, _ = armed(require_plate=True)
    assert g.tick(CAT, plate_pressed=False).command == Command.KEEP_OPEN
    assert g.tick(CAT, plate_pressed=True).command == Command.CLOSE


def test_vision_timeout_faults_open():
    g, clock = armed()
    g.tick(NOTHING)
    clock.t = 3
    s = g.tick(None)
    assert s.state == GateState.FAULT and s.command == Command.KEEP_OPEN
    assert g.tick(CAT).command == Command.KEEP_OPEN  # stays faulted until re-armed
    g.arm()
    assert g.tick(CAT).command == Command.CLOSE
