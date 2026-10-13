"""Gate controller (owned by the HARDWARE team). Reference version written with the model team.

The gate controller decides WHEN to close. It combines:
  * the VisionReport from edge/vision.py (what the camera sees)
  * the pressure plate (is an animal at the bait?), if fitted
  * safety and timing rules: cooldown after a non-cat, fault on stale/missing vision,
    never close on an animal in the doorway, stay closed once closed (until a person resets it)

and outputs one command per tick: KEEP_OPEN or CLOSE. Driving the servo, status LEDs and remote
alerts happen in the hardware team's code around this class.

Hardware team: change any rule below; the only thing the model team relies on is that you call
`tick(report, ...)` with each VisionReport.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from .vision import VisionReport


class GateState(str, Enum):
    DISARMED = "DISARMED"    # trap not set; ignore everything
    ARMED = "ARMED"          # open, waiting for a cat
    COOLDOWN = "COOLDOWN"    # open, a non-cat was seen recently
    CLOSED = "CLOSED"        # gate closed; latched until reset() by a person
    FAULT = "FAULT"          # vision missing or broken; held open


class Command(str, Enum):
    KEEP_OPEN = "KEEP_OPEN"
    CLOSE = "CLOSE"


@dataclass
class GateStatus:
    state: GateState
    command: Command
    reason: str = ""


class GateController:
    def __init__(self, require_plate: bool = False, cooldown_s: float = 10.0,
                 vision_timeout_s: float = 2.0, clock=time.monotonic):
        self.require_plate = require_plate
        self.cooldown_s = cooldown_s
        self.vision_timeout_s = vision_timeout_s
        self.clock = clock
        self.state = GateState.DISARMED
        self.cooldown_until = 0.0
        self.last_report_t: float | None = None
        self.fault_msg: str | None = None

    # --- operator actions ------------------------------------------------
    def arm(self) -> None:
        self.state, self.fault_msg, self.last_report_t = GateState.ARMED, None, self.clock()

    def disarm(self) -> None:
        self.state = GateState.DISARMED

    def reset(self) -> None:
        """Person has handled the trap: reopen and re-arm."""
        self.arm()

    def report_fault(self, msg: str) -> GateStatus:
        if self.state != GateState.CLOSED:
            self.state, self.fault_msg = GateState.FAULT, msg
        return self.status(msg)

    # --- main loop -------------------------------------------------------
    def tick(self, report: VisionReport | None, plate_pressed: bool = False) -> GateStatus:
        """Call once per camera frame (report) or on a timer with report=None."""
        now = self.clock()
        if self.state in (GateState.DISARMED, GateState.CLOSED):
            return self.status()
        if report is not None:
            self.last_report_t = now
        elif self.last_report_t is not None and now - self.last_report_t > self.vision_timeout_s:
            return self.report_fault(f"no vision report for {now - self.last_report_t:.1f}s")
        if self.state == GateState.FAULT:
            return self.status(self.fault_msg or "fault")
        if report is None:
            return self.status()

        if report.non_cat:
            self.state, self.cooldown_until = GateState.COOLDOWN, now + self.cooldown_s
            return self.status(f"{report.non_cat} {report.non_cat_confidence:.2f}")
        if self.state == GateState.COOLDOWN:
            if now < self.cooldown_until:
                return self.status(f"cooldown {self.cooldown_until - now:.1f}s")
            self.state = GateState.ARMED

        if report.animal_in_doorway:
            return self.status("animal in doorway")
        if not report.cat_in_trap:
            return self.status(f"cat {report.cat_hits} frames" if report.cat_hits else "")
        if self.require_plate and not plate_pressed:
            return self.status("cat seen, waiting for plate")

        self.state = GateState.CLOSED
        return GateStatus(GateState.CLOSED, Command.CLOSE,
                          f"cat conf {report.cat_confidence:.2f}" + (" + plate" if self.require_plate else ""))

    def status(self, reason: str = "") -> GateStatus:
        cmd = Command.CLOSE if self.state == GateState.CLOSED else Command.KEEP_OPEN
        return GateStatus(self.state, cmd, reason)
