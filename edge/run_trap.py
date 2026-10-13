"""Main loop on the Pi: camera -> detector -> vision report -> gate controller -> hardware.

Purpose: The integration point where model and hardware code meet.
Status:  Loop runs on a laptop with --source. Hardware hooks at the bottom are placeholders.
Later stages (still to add or change):
  - Hardware team: fill in pir_motion(), plate_pressed() and apply(status) (servo, LEDs, alerts).
  - Phase 2b: add logging of frames/reports to disk for staged trials and debugging.
  - Phase 2b: idle/sleep mode woken by the PIR to save power.

This is the integration point. The model team owns the vision steps; the hardware team fills in
the three hardware hooks at the bottom (PIR, pressure plate, gate actuator / status).

    python -m edge.run_trap --model best_ncnn_model --imgsz 416
    python -m edge.run_trap --model yolo26n.pt --source test_cat.jpg   # laptop dry run
"""
from __future__ import annotations

import argparse
import logging
import time

from .bench import Camera
from .gate import Command, GateController, GateState, GateStatus
from .vision import Detector, VisionAggregator

log = logging.getLogger("scout")


def run(detector: Detector, cam: Camera, vision: VisionAggregator, gate: GateController,
        fps: float = 5.0, max_frames: int | None = None) -> None:
    gate.arm()
    frames, period = 0, 1.0 / fps
    while max_frames is None or frames < max_frames:
        t0 = time.monotonic()
        if gate.state == GateState.CLOSED:
            time.sleep(0.5)            # waiting for a person to reset the trap
            continue
        if not pir_motion():
            vision.reset()             # nothing moving: skip inference, save power
            time.sleep(0.2)
            continue
        try:
            report = vision.update(detector.detect(cam.read()))
        except Exception as e:         # any camera/model error -> fault, gate held open
            status = gate.report_fault(f"vision error: {e}")
        else:
            status = gate.tick(report, plate_pressed=plate_pressed())
            log.info("cat_in_trap=%s hits=%d non_cat=%s door=%s -> %s %s", report.cat_in_trap,
                     report.cat_hits, report.non_cat, report.animal_in_doorway,
                     status.state.value, status.reason)
        apply(status)
        frames += 1
        time.sleep(max(0.0, period - (time.monotonic() - t0)))


# ---- hardware hooks: hardware team replaces these ---------------------------------
def pir_motion() -> bool:
    """HC-SR501 output. Placeholder: always on."""
    return True


def plate_pressed() -> bool:
    """Pressure plate in front of the bait. Placeholder: not fitted."""
    return False


def apply(status: GateStatus) -> None:
    """Drive the DS3218 servo and status LEDs from the gate command."""
    if status.command == Command.CLOSE:
        log.warning("CLOSE GATE (%s)", status.reason)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--source", help="image file instead of the Pi camera")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--plate", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(Detector(args.model, args.imgsz), Camera(args.source), VisionAggregator.from_config(),
        GateController(require_plate=args.plate), args.fps, args.frames)


if __name__ == "__main__":
    main()
