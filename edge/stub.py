"""Feeds scripted fake detections through the real VisionAggregator and GateController.

Purpose: Lets the hardware team test the servo and gate logic now, before any model is trained (CP1).
Status:  Done and tested.
Later stages (still to add or change):
  - Hardware team: pass your own on_status to run() so the stub drives the real servo/LEDs.

Fake detections go through the REAL VisionAggregator and GateController, so the hardware sees
exactly the behaviour it will see once the trained model is plugged in. One token per frame:

    c  cat inside the trap zone        d  cat in the doorway
    r  raccoon                         o  opossum
    n  nothing in view                 f  camera/model fault
    p  cat inside the trap zone AND pressure plate pressed

    python -m edge.stub --script c,c,c,c,c            # closes on frame 3 (K=3 of N=5, set in edge/vision_config.json)
    python -m edge.stub --script c,r,c,c,c,c          # raccoon -> cooldown, stays open
    python -m edge.stub --script c,c,c,c,p --plate    # plate required: closes on 'p'
    python -m edge.stub --interactive
    python -m edge.stub --trials 50 --seed 1          # random staged trials, scored (CP3 style)

Hardware: call run(tokens, on_status) with your own on_status that drives the servo / LEDs.
"""
from __future__ import annotations

import argparse
import random
import time
from typing import Callable, Iterable

from .gate import Command, GateController, GateStatus
from .vision import Detection, VisionAggregator

CAT_IN = Detection("cat", 0.9, (0.55, 0.30, 0.85, 0.80))
FAKE = {
    "c": [CAT_IN],
    "p": [CAT_IN],
    "d": [Detection("cat", 0.9, (0.02, 0.30, 0.25, 0.80))],
    "r": [Detection("raccoon", 0.85, (0.50, 0.30, 0.85, 0.80))],
    "o": [Detection("opossum", 0.85, (0.50, 0.30, 0.85, 0.80))],
    "n": [],
}


def run(tokens: Iterable[str], on_status: Callable[[GateStatus], None], fps: float = 5.0,
        vision: VisionAggregator | None = None, gate: GateController | None = None,
        require_plate: bool = False) -> GateStatus | None:
    vision = vision or VisionAggregator.from_config()
    if gate is None:
        gate = GateController(require_plate=require_plate)
        gate.arm()
    status = None
    for tok in (t.strip().lower() for t in tokens):
        if not tok:
            continue
        if tok == "f":
            status = gate.report_fault("stub fault")
        elif tok in FAKE:
            status = gate.tick(vision.update(FAKE[tok]), plate_pressed=(tok == "p"))
        else:
            print(f"unknown token {tok!r}; use c p d r o n f")
            continue
        on_status(status)
        if fps:
            time.sleep(1 / fps)
    return status


def print_status(s: GateStatus) -> None:
    gate = "CLOSE" if s.command == Command.CLOSE else "open "
    print(f"[{gate}] {s.state.value:9s} {s.reason}")


def random_trial(rng: random.Random) -> tuple[str, list[str]]:
    if rng.random() < 0.5:
        return "cat", ["n", "d", "d"] + ["c"] * 5
    return "non-cat", ["n"] + [rng.choice("ro")] * 7


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--script")
    g.add_argument("--interactive", action="store_true")
    g.add_argument("--trials", type=int)
    ap.add_argument("--plate", action="store_true", help="require the pressure plate to close")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.script:
        run(args.script.split(","), print_status, args.fps, require_plate=args.plate)
    elif args.interactive:
        vision, gate = VisionAggregator.from_config(), GateController(require_plate=args.plate)
        gate.arm()
        print("tokens: c p d r o n f   reset = reopen   q = quit")
        while (line := input("> ").strip()) != "q":
            if line == "reset":
                gate.reset(); vision.reset(); print("re-armed"); continue
            run(line.replace(" ", ",").split(","), print_status, 0, vision, gate)
    else:
        rng, correct = random.Random(args.seed), 0
        for i in range(args.trials):
            kind, toks = random_trial(rng)
            closed = False

            def on(s: GateStatus) -> None:
                nonlocal closed
                closed = closed or s.command == Command.CLOSE

            run(toks, on, fps=0)
            ok = closed == (kind == "cat")
            correct += ok
            print(f"trial {i + 1:3d} {kind:8s} closed={closed!s:5s} {'PASS' if ok else 'FAIL'}")
        print(f"{correct}/{args.trials} correct ({100 * correct / args.trials:.0f}%)")


if __name__ == "__main__":
    main()
