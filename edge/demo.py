"""Laptop prototype: camera or pictures -> detector -> boxes, zones, VisionReport and a simulated gate on screen.

Purpose: Step 1 of the plan ("prototype on a laptop"), the live view for the footage pilot, and the base of the
         Demo Day screen. Also saves raw frames, so it doubles as the recorder for our own trap footage.
Status:  Written; drawing/report code tested without a model (tests/test_demo.py).
Later stages (still to add or change):
  - Phase 2b: on the Pi, use --picam (picamera2) once a display or VNC is attached.
  - Phase 4: a bigger "demo mode" layout for the poster session.

Runs the REAL edge/vision.py aggregator and the reference edge/gate.py, so what you see is what the trap does.

    python -m edge.demo --model yolo26n.pt --source 0                       # webcam
    python -m edge.demo --model best.pt --source clip.mp4                   # a video
    python -m edge.demo --model best.pt --source demo_pics/ --out demo_out/ # folder, no window (headless)
    python -m edge.demo --model best.pt --source 0 --save-dir pilot/session1 # record raw frames for labelling

Keys (window mode): q quit, space pause, s save this raw frame, r start/stop recording (2 fps),
                    a re-arm the gate after it closes, z toggle zones.
Picture folders: each picture is shown for --hold frames with a fresh vote, i.e. one staged trial per picture.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .gate import GateController, GateState
from .vision import Detection, Detector, VisionAggregator, VisionReport

# BGR colours
GREEN, ORANGE, RED, WHITE, GREY, CYAN = (60, 200, 60), (0, 150, 255), (40, 40, 230), (255, 255, 255), (90, 90, 90), (220, 200, 0)
STATE_COLOUR = {GateState.ARMED: GREEN, GateState.COOLDOWN: ORANGE, GateState.CLOSED: RED,
                GateState.FAULT: RED, GateState.DISARMED: GREY}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def report_lines(report: VisionReport, state: GateState, reason: str, n: int) -> list[str]:
    """Text for the side panel (kept separate from drawing so it can be tested)."""
    return [
        f"GATE: {state.value}" + (f"  ({reason})" if reason else ""),
        f"cat in trap: {'YES' if report.cat_in_trap else 'no'}   votes {report.cat_hits}/{n}",
        f"cat conf (mean): {report.cat_confidence:.2f}",
        f"doorway: {'ANIMAL' if report.animal_in_doorway else 'clear'}",
        f"non-cat: {report.non_cat or '-'}" + (f" {report.non_cat_confidence:.2f}" if report.non_cat else ""),
        f"detections: {report.n_detections}",
    ]


def draw(frame, dets: list[Detection], report: VisionReport, state: GateState, reason: str,
         vision: VisionAggregator, show_zones: bool = True, fps: float | None = None, rec: bool = False):
    import cv2
    import numpy as np
    out = frame.copy()
    h, w = out.shape[:2]

    def px(poly):
        return np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32)

    if show_zones:
        cv2.polylines(out, [px(vision.trap)], True, CYAN, 2)
        cv2.putText(out, "trap zone", (int(vision.trap[0][0] * w) + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 2)
        if vision.door:
            cv2.polylines(out, [px(vision.door)], True, ORANGE, 2)
            cv2.putText(out, "doorway", (int(vision.door[0][0] * w) + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE, 2)

    for d in dets:
        live = d.cls in vision.thr and d.conf >= vision.thr[d.cls]
        col = (GREEN if d.cls == "cat" else ORANGE) if live else GREY
        x1, y1, x2, y2 = int(d.box[0] * w), int(d.box[1] * h), int(d.box[2] * w), int(d.box[3] * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 if live else 1)
        cv2.putText(out, f"{d.cls} {d.conf:.2f}", (x1 + 3, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    panel_w = 360
    panel = np.zeros((h, panel_w, 3), np.uint8)
    panel[:] = (30, 30, 30)
    cv2.rectangle(panel, (0, 0), (panel_w, 46), STATE_COLOUR.get(state, GREY), -1)
    for i, line in enumerate(report_lines(report, state, reason, vision.N)):
        y = 31 if i == 0 else 80 + 30 * (i - 1)
        cv2.putText(panel, line[:42], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 if i == 0 else 0.52, WHITE, 2 if i == 0 else 1)
    foot = []
    if fps:
        foot.append(f"{fps:.1f} fps")
    if rec:
        foot.append("REC")
    if foot:
        cv2.putText(panel, "  ".join(foot), (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if rec else WHITE, 2)
    return np.hstack([out, panel])


def iter_source(source: str, picam: bool):
    """Yields (name, BGR frame, is_still)."""
    import cv2
    if picam:
        from .bench import Camera
        cam = Camera(None)
        while True:
            yield "picam", cam.read(), False
    p = Path(source)
    if p.is_dir():
        for f in sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXT):
            img = cv2.imread(str(f))
            if img is not None:
                yield f.name, img, True
        return
    if p.is_file() and p.suffix.lower() in IMG_EXT:
        yield p.name, cv2.imread(str(p)), True
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        i += 1
        yield f"frame{i:06d}", frame, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--source", default="0", help="webcam index, video file, image or image folder")
    ap.add_argument("--picam", action="store_true", help="read the Raspberry Pi camera instead of --source")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--config", default=None, help="vision_config.json (default: edge/vision_config.json)")
    ap.add_argument("--hold", type=int, default=5, help="frames per still picture (one vote window)")
    ap.add_argument("--out", help="write annotated frames here instead of opening a window")
    ap.add_argument("--save-dir", help="where raw frames go when you press s / r")
    ap.add_argument("--plate", action="store_true", help="gate requires the pressure plate (always pressed here)")
    args = ap.parse_args()

    import cv2
    detector = Detector(args.model, imgsz=args.imgsz, min_conf=0.1)
    vision = VisionAggregator.from_config(args.config) if args.config else VisionAggregator.from_config()
    gate = GateController(require_plate=args.plate)
    gate.arm()
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir) if args.save_dir else None
    window = out_dir is None
    show_zones, paused, recording, last_rec, t_prev, fps = True, False, False, 0.0, time.time(), None
    trials = []

    for name, frame, still in iter_source(args.source, args.picam):
        reps = args.hold if still else 1
        if still:
            vision.reset()
            gate.reset()
        for k in range(reps):
            dets = detector.detect(frame)
            report = vision.update(dets)
            status = gate.tick(report, plate_pressed=True)
            now = time.time()
            fps = 0.8 * fps + 0.2 / max(now - t_prev, 1e-6) if fps else 1 / max(now - t_prev, 1e-6)
            t_prev = now
            canvas = draw(frame, dets, report, status.state, status.reason, vision, show_zones,
                          None if still else fps, recording)
            if recording and save_dir and now - last_rec >= 0.5:
                save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{int(now * 1000) % 1000:03d}.jpg"), frame)
                last_rec = now
            if out_dir and (not still or k == reps - 1):
                cv2.imwrite(str(out_dir / f"{Path(name).stem}.jpg"), canvas)
            if window:
                cv2.imshow("SCOUT demo", canvas)
                key = cv2.waitKey(0 if paused else (300 if still else 1)) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return _summary(trials)
                if key == ord(" "):
                    paused = not paused
                if key == ord("z"):
                    show_zones = not show_zones
                if key == ord("a"):
                    gate.reset()
                    vision.reset()
                if key in (ord("s"), ord("r")) and not save_dir:
                    print("pass --save-dir to save frames")
                elif key == ord("s"):
                    save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.jpg"), frame)
                elif key == ord("r"):
                    recording = not recording
                    print("recording" if recording else "stopped recording")
        if still:
            trials.append((name, status.state.value, report.non_cat or "", f"{report.cat_confidence:.2f}"))
            print(f"{name:40s} gate={status.state.value:8s} cat_votes={report.cat_hits}/{vision.N} "
                  f"non_cat={report.non_cat or '-'}")
    if window:
        cv2.destroyAllWindows()
    _summary(trials)


def _summary(trials) -> None:
    if not trials:
        return
    closed = sum(1 for t in trials if t[1] == "CLOSED")
    print(f"{len(trials)} pictures: gate closed on {closed}, stayed open on {len(trials) - closed}")


if __name__ == "__main__":
    main()
