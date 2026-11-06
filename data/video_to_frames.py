"""Turns real-animal videos (Christine's trail-camera videos) into frames the rest of the pipeline can use.

Purpose: Plan sections 9 and 12, and step 6 of the Oct 30 model plan. Real videos of cats, raccoons and opossums are the only real-animal footage
         we have with frames in order, so they are the honest test of the 3-of-5 vote and the gate. A sparse
         sample of their frames can also be labeled and added to training (as a held-out-by-video set first).
Status:  Written; frame sampling, naming, labels and rotation tested (tests/test_video_frames.py).
         data/backyard_labels.csv labels the 46 videos in backyard_camera/ (labels checked by eye only).

Two outputs, from the same videos:

1. Clip folders for the vote/gate test (training/vote_sim.py clips). One folder per video, frames at the trap
   camera's rate (--fps: use the rate edge/bench.py measures on the Pi 3), named <animal>_<video>. vote_sim reads the animal from the part before the
   first "_", so every video needs an animal: from --labels (a CSV of video,animal[,rotate]; rotate is cw or ccw
   for a sideways video), or from the file name
   if it starts with one of our classes (cat_01.mp4), or "unknown" (watch it, then rename the folder).

       python data/video_to_frames.py --videos ~/christine_videos --labels videos.csv --clips-out real/clips
       python training/vote_sim.py clips --model results/E1b/weights/best.pt --clips real/clips --full-frame

2. A labeling pool (--pool-out): one frame every --pool-every seconds, so near-identical frames don't swamp the
   labeler. With --prelabel <model>, E1b's boxes are written next to each frame as YOLO .txt plus classes.txt,
   so labeling becomes correcting (Roboflow, CVAT and Label Studio import this). After labeling, import whole
   videos with data/add_trap_footage.py --session <video> and keep some videos for --split test only.

       python data/video_to_frames.py --videos ~/christine_videos --labels videos.csv \
           --pool-out real/pool --pool-every 1 --prelabel results/E1b/weights/best.pt

Never put frames of one video in both training and test: frames of the same video are near-identical.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}
ANIMALS = ("cat", "raccoon", "opossum", "squirrel", "dog", "other_animal", "person", "empty")
CLASSES = ["cat", "raccoon", "opossum", "squirrel", "dog", "other_animal", "person"]   # same order as data/common.py


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "video"


def animal_for(video: Path, labels: dict[str, str]) -> str:
    """The animal for a video: the labels CSV first, then the file name prefix, else 'unknown'."""
    for key in (video.name, video.stem):
        if key in labels:
            return labels[key].strip().lower().replace(" ", "_")
    stem = video.stem.lower()
    for a in sorted(ANIMALS, key=len, reverse=True):
        if stem.startswith(a):
            return a
    return "unknown"


def read_labels(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, newline="") as f:
        return {row[0].strip(): row[1].strip() for row in csv.reader(f) if len(row) >= 2 and row[0].strip()
                and row[0].strip().lower() != "video"}


def read_rotations(path: str | None) -> dict[str, str]:
    """Third CSV column: cw or ccw for videos filmed sideways. The model never trains on sideways animals."""
    if not path:
        return {}
    with open(path, newline="") as f:
        return {row[0].strip(): row[2].strip().lower() for row in csv.reader(f)
                if len(row) >= 3 and row[2].strip().lower() in ("cw", "ccw")}


def turn(frame, how: str | None):
    if not how:
        return frame
    import cv2
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE if how == "cw" else cv2.ROTATE_90_COUNTERCLOCKWISE)


def sample_times(duration: float, every: float) -> list[float]:
    """Times (seconds) to grab, one every `every` seconds, starting at 0."""
    if duration <= 0 or every <= 0:
        return []
    n = int(duration / every + 1e-9) + 1
    return [round(i * every, 3) for i in range(n) if i * every < duration]


def grab(video: Path, every: float, max_seconds: float | None):
    """Yields (time_s, frame) one frame every `every` seconds, decoding sequentially (works for any codec)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if frames else float("inf")
    if max_seconds:
        duration = min(duration, max_seconds)
    want = iter(sample_times(duration, every) if duration != float("inf") else (i * every for i in range(10**9)))
    target = next(want, None)
    i = 0
    while target is not None:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if t + 0.5 / fps >= target:
            yield target, frame
            target = next(want, None)
        i += 1
    cap.release()


def yolo_lines(dets) -> list[str]:
    out = []
    for d in dets:
        if d.cls not in CLASSES:
            continue
        x1, y1, x2, y2 = d.box
        out.append(f"{CLASSES.index(d.cls)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="a folder of videos (searched recursively) or one video")
    ap.add_argument("--labels", help="CSV of video,animal (cat, raccoon, opossum, ..., empty)")
    ap.add_argument("--clips-out", help="write one clip folder per video here, for vote_sim.py clips")
    ap.add_argument("--fps", type=float, default=5.0, help="clip frame rate; use the rate edge/bench.py measures on the Pi 3")
    ap.add_argument("--pool-out", help="write a sparse labeling pool here")
    ap.add_argument("--pool-every", type=float, default=1.0, help="seconds between pool frames")
    ap.add_argument("--prelabel", help="model whose boxes are saved as YOLO .txt next to each pool frame")
    ap.add_argument("--max-seconds", type=float, help="only the first N seconds of each video")
    args = ap.parse_args()
    if not (args.clips_out or args.pool_out):
        ap.error("give --clips-out, --pool-out or both")

    import cv2
    src = Path(args.videos).expanduser()
    videos = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"no videos found in {src}")
    labels = read_labels(args.labels)
    rotations = read_rotations(args.labels)
    det = None
    if args.prelabel:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from edge.vision import Detector
        det = Detector(args.prelabel, imgsz=640, min_conf=0.25)

    unknown = []
    for v in videos:
        animal, name = animal_for(v, labels), slug(v.stem)
        if animal == "unknown":
            unknown.append(v.name)
        n_clip = n_pool = 0
        if args.clips_out:
            d = Path(args.clips_out) / f"{animal}_{name}"
            d.mkdir(parents=True, exist_ok=True)
            for t, frame in grab(v, 1 / args.fps, args.max_seconds):
                cv2.imwrite(str(d / f"frame_{int(round(t * 1000)):07d}ms.jpg"), turn(frame, rotations.get(v.name)))
                n_clip += 1
        if args.pool_out:
            d = Path(args.pool_out)
            d.mkdir(parents=True, exist_ok=True)
            for t, frame in grab(v, args.pool_every, args.max_seconds):
                frame = turn(frame, rotations.get(v.name))
                stem = f"{animal}__{name}__t{int(round(t * 1000)):07d}"
                cv2.imwrite(str(d / f"{stem}.jpg"), frame)
                if det is not None:
                    (d / f"{stem}.txt").write_text("\n".join(yolo_lines(det.detect(frame))))
                n_pool += 1
        print(f"{v.name}: {animal}, {n_clip} clip frames, {n_pool} pool frames")
    if args.pool_out and det is not None:
        (Path(args.pool_out) / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    if unknown:
        print(f"\n{len(unknown)} video(s) have no animal yet ({', '.join(unknown[:5])}{' ...' if len(unknown) > 5 else ''}). "
              "Add them to --labels, or rename the clip folders from unknown_ to the animal before vote_sim.py.")


if __name__ == "__main__":
    main()
