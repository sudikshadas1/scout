"""Per-stage latency benchmark on the Pi (or a laptop, with --source).

Stages logged per frame: capture, detector (preprocess / inference / postprocess), vision
aggregation, gate controller, total. Writes a CSV and prints median / p95 in milliseconds.

On the Pi 5 with the camera:
    python -m edge.bench --model yolo26n.pt --export ncnn --imgsz 416 --runs 200
Anywhere, with a test image:
    python -m edge.bench --model yolo26n.pt --source some_cat.jpg --runs 50
"""
from __future__ import annotations

import argparse
import csv
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
                    help="export .pt to NCNN first (do this on the Pi or copy the folder over)")
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
    print(f"  ~{fps:.1f} decisions/s; 5-frame vote ≈ {5000 / fps:.0f} ms")


if __name__ == "__main__":
    main()
