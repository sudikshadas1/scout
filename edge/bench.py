"""Times every stage (capture, detector, aggregation, gate) on the Pi or a laptop and writes a CSV.

Purpose: E6, the speed test on the Pi 3 (plan sections 9 and 11). Measures what the Raspberry Pi 3 can run. The vote needs 3 frames inside the 5 s decision limit, so one
         frame, camera included, must take under about 1.6 s (0.67 s for the 2 s aim). This sets the model and imgsz.
Status:  Written; tested on a laptop with --source. Not yet run on the Pi 3.
Later stages (still to add or change):
  - First run on the Pi 3: E1b (NCNN) at 416, then 320 if too slow. This decides the picture size and whether E3 fits.
  - Rerun with our trained model at the chosen size.
  - Phase 2a: a recording script for trap footage can reuse the Camera class here (not written yet).

Stages logged per frame: capture, detector (preprocess / inference / postprocess), vision
aggregation, gate controller, total. Writes a CSV and prints median / p95 in milliseconds.

Convert on a laptop or PACE, not on the Pi 3 (it has only 1 GB of memory), then copy the folder over:
    python -m edge.bench --model best.pt --export ncnn --imgsz 416
On the Pi 3 (add --source some_cat.jpg to test before the camera is ready):
    python -m edge.bench --model best_ncnn_model --imgsz 416 --runs 200
Anywhere, with a test image:
    python -m edge.bench --model yolo26n.pt --source some_cat.jpg --runs 50
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import time
from pathlib import Path

from .gate import GateController
from .vision import Detector, VisionAggregator


class Camera:
    def __init__(self, source: str | None, size=(1280, 720), index: int = 0):
        self.static = None
        if source:
            import cv2
            img = cv2.imread(source)
            if img is None:
                raise FileNotFoundError(source)
            self.static = img  # BGR, which is what ultralytics expects for numpy input
            return
        from picamera2 import Picamera2  # preinstalled on Raspberry Pi OS
        self.cam = Picamera2(index)
        # picamera2 "RGB888" is stored B,G,R in memory, which matches ultralytics' BGR numpy input
        cfg = self.cam.create_video_configuration(main={"size": size, "format": "RGB888"})
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.0)  # let auto-exposure settle

    def read(self):
        return self.static.copy() if self.static is not None else self.cam.capture_array()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--export", choices=["ncnn", "none"], default="none",
                    help="export .pt to NCNN first (on a laptop or PACE, then copy the folder to the Pi 3)")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--source", help="image file instead of the Pi camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="results/bench.csv")
    args = ap.parse_args()

    model_path = args.model
    if args.export == "ncnn" and model_path.endswith(".pt"):
        from ultralytics import YOLO
        model_path = YOLO(model_path).export(format="ncnn", imgsz=args.imgsz)
        if not args.source and importlib.util.find_spec("picamera2") is None:
            print(f"exported {model_path}; copy it to the Pi 3 and time it there")
            return
    detector = Detector(model_path, imgsz=args.imgsz)
    cam = Camera(args.source, index=args.camera)
    vision = VisionAggregator.from_config()
    gate = GateController()
    gate.arm()

    rows = []
    for i in range(args.warmup + args.runs):
        t0 = time.perf_counter()
        frame = cam.read()
        t1 = time.perf_counter()
        dets = detector.detect(frame)
        t2 = time.perf_counter()
        report = vision.update(dets)
        t3 = time.perf_counter()
        status = gate.tick(report)
        t4 = time.perf_counter()
        if status.state.value == "CLOSED":  # keep benchmarking: reopen
            gate.reset()
        if i < args.warmup:
            continue
        sp = detector.last_speed
        rows.append({
            "capture_ms": (t1 - t0) * 1e3,
            "preprocess_ms": sp["preprocess"],
            "inference_ms": sp["inference"],
            "postprocess_ms": sp["postprocess"],
            "detect_ms": (t2 - t1) * 1e3,
            "vision_ms": (t3 - t2) * 1e3,
            "gate_ms": (t4 - t3) * 1e3,
            "total_ms": (t4 - t0) * 1e3,
            "cat_in_trap": report.cat_in_trap,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{model_path} @ {args.imgsz}px, {len(rows)} runs -> {out}")
    for k in rows[0]:
        if k.endswith("_ms"):
            xs = [r[k] for r in rows]
            print(f"  {k:15s} median {statistics.median(xs):7.1f}   p95 {pct(xs, 95):7.1f}")
    fps = 1000 / statistics.median(r["total_ms"] for r in rows)
    print(f"  ~{fps:.1f} frames/s; {vision.K}-of-{vision.N} vote ≈ {vision.K / fps:.1f} s (limit 5 s, aim 2 s)")


if __name__ == "__main__":
    main()
