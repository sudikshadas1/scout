"""Runs the detector on each frame and turns detections into a VisionReport (cat in trap? animal in doorway? non-cat?).

Purpose: The model team's whole on-device output. It reports what it sees; it never decides to close the gate.
Status:  Done and tested (tests/test_vision.py). Detector runs with stock or trained weights.
Later stages (still to add or change):
  - Phase 2a: swap in our trained NCNN model; set thresholds from eval.py's threshold sweep.
  - Phase 2b: real trap and doorway zones go in vision_config.json once the camera is mounted.
  - If we add a second camera, the aggregator needs to merge two detection lists per tick.

The vision system answers one question per frame: "what is in front of the camera, and where?"
It does NOT decide whether to close the gate. That is the gate controller's job (edge/gate.py,
owned by the hardware team), which combines this report with the pressure plate, cooldowns,
timeouts and safety rules.

Pipeline per frame:
    frame --Detector.detect()--> [Detection, ...] --VisionAggregator.update()--> VisionReport

VisionAggregator smooths noisy single-frame detections with K-of-N voting, so a cat is only
reported as "in the trap" when it has been seen inside the trap zone in K of the last N frames.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

CLASSES = ("cat", "raccoon", "opossum", "squirrel", "dog", "other_animal", "person")
CONFIG = Path(__file__).with_name("vision_config.json")


# --------------------------------------------------------------------------- data types
@dataclass
class Detection:
    """One box from the detector in one frame."""
    cls: str                                   # one of CLASSES
    conf: float                                # 0..1
    box: tuple[float, float, float, float]     # x1, y1, x2, y2, normalized 0..1


@dataclass
class VisionReport:
    """What the vision system tells the gate controller, once per frame.

    These are observations, not commands. The gate controller decides what to do with them.
    """
    cat_in_trap: bool = False          # cat inside the trap zone in >= K of the last N frames
    cat_confidence: float = 0.0        # mean confidence of those cat detections
    cat_hits: int = 0                  # how many of the last N frames had a cat in the trap zone
    animal_in_doorway: bool = False    # any confident detection overlapping the doorway zone (this frame)
    non_cat: str | None = None         # most confident non-cat class this frame, if any
    non_cat_confidence: float = 0.0
    cat_box: tuple | None = None       # latest in-trap cat box (normalized), for logs / demo overlay
    n_detections: int = 0              # confident detections this frame (0 = nothing in view)
    timestamp: float = field(default_factory=time.time)

    @property
    def anything_in_view(self) -> bool:
        return self.n_detections > 0


# --------------------------------------------------------------------------- geometry
Point = tuple[float, float]


def point_in_poly(p: Point, poly: list[Point]) -> bool:
    """Ray casting; points on an edge count as inside."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if min(y1, y2) <= y <= max(y1, y2) and min(x1, x2) <= x <= max(x1, x2):
            if (x2 - x1) * (y - y1) == (y2 - y1) * (x - x1):
                return True
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def box_grounded_in(box, poly) -> bool:
    """Bottom edge of the box (both corners + middle) inside the polygon: the animal stands in the zone."""
    x1, _, x2, y2 = box
    return all(point_in_poly(p, poly) for p in ((x1, y2), ((x1 + x2) / 2, y2), (x2, y2)))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def one_label_per_animal(dets: list[Detection], same_iou: float, margin: float) -> list[Detection]:
    """Two boxes of different classes on the same animal (IoU >= same_iou): drop the weaker label, but only if
    the stronger one is at least `margin` more confident.

    Without this, one frame where a cat is ALSO labelled opossum puts the gate into its non-cat cooldown and
    the trap never closes on that cat. When the two labels are close (cat 0.90 vs raccoon 0.85) both are kept,
    so the non-cat still holds the gate open: when unsure, don't trap.
    """
    keep = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(d.cls == k.cls or iou(d.box, k.box) < same_iou or k.conf - d.conf < margin for k in keep):
            keep.append(d)
    return keep


def box_touches(box, poly) -> bool:
    """Approximate box/polygon overlap (fine for the convex zones we draw)."""
    x1, y1, x2, y2 = box
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), ((x1 + x2) / 2, (y1 + y2) / 2)]
    return any(point_in_poly(c, poly) for c in pts) or any(
        x1 <= px <= x2 and y1 <= py <= y2 for px, py in poly)


# --------------------------------------------------------------------------- aggregation
class VisionAggregator:
    """Turns per-frame detections into a smoothed VisionReport (K-of-N voting + zone checks)."""

    def __init__(self, N=5, K=4, thresholds=None, trap_zone=None, doorway_zone=None, same_animal_iou=0.6,
                 same_animal_margin=0.1):
        if not 1 <= K <= N:
            raise ValueError("need 1 <= K <= N")
        self.N, self.K = N, K
        # Only these classes count; anything else (e.g. a stock-COCO "chair") is ignored.
        self.thr = {**{c: 0.5 for c in CLASSES}, **(thresholds or {})}
        self.trap = [tuple(p) for p in (trap_zone or [(0, 0), (1, 0), (1, 1), (0, 1)])]
        self.door = [tuple(p) for p in (doorway_zone or [])]
        self.same_iou, self.same_margin = same_animal_iou, same_animal_margin
        self.window: deque[Detection | None] = deque(maxlen=N)

    @classmethod
    def from_config(cls, path: str | Path = CONFIG) -> "VisionAggregator":
        cfg = json.loads(Path(path).read_text())
        keys = ("N", "K", "thresholds", "trap_zone", "doorway_zone", "same_animal_iou", "same_animal_margin")
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    def reset(self) -> None:
        self.window.clear()

    def update(self, dets: list[Detection]) -> VisionReport:
        live = [d for d in dets if d.cls in self.thr and d.conf >= self.thr[d.cls]]
        live = one_label_per_animal(live, self.same_iou, self.same_margin)

        others = [d for d in live if d.cls != "cat"]
        worst = max(others, key=lambda d: d.conf) if others else None
        in_door = bool(self.door) and any(box_touches(d.box, self.door) for d in live)

        cats = [d for d in live if d.cls == "cat" and box_grounded_in(d.box, self.trap)]
        self.window.append(max(cats, key=lambda d: d.conf) if cats else None)
        hits = [d for d in self.window if d is not None]

        return VisionReport(
            cat_in_trap=len(hits) >= self.K,
            cat_confidence=mean(d.conf for d in hits) if hits else 0.0,
            cat_hits=len(hits),
            animal_in_doorway=in_door,
            non_cat=worst.cls if worst else None,
            non_cat_confidence=worst.conf if worst else 0.0,
            cat_box=hits[-1].box if hits else None,
            n_detections=len(live),
        )


# --------------------------------------------------------------------------- detector
class Detector:
    """Wraps an Ultralytics model (.pt, or an NCNN export folder on the Pi)."""

    def __init__(self, model_path: str = "yolo26n.pt", imgsz: int = 416, min_conf: float = 0.1):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="detect")
        self.imgsz, self.min_conf = imgsz, min_conf
        self.last_speed: dict = {}

    def detect(self, frame) -> list[Detection]:
        """frame: HxWx3 BGR numpy array (picamera2 'RGB888' and cv2 both give this)."""
        r = self.model(frame, imgsz=self.imgsz, conf=self.min_conf, verbose=False)[0]
        self.last_speed = r.speed
        return detections_from_ultralytics(r)


def detections_from_ultralytics(result) -> list[Detection]:
    names, boxes = result.names, result.boxes
    if boxes is None:
        return []
    return [Detection(names[int(c)], float(p), tuple(b))
            for b, p, c in zip(boxes.xyxyn.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]
