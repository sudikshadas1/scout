"""Draw the trap zone and doorway zone on a frame from the mounted camera and save them to vision_config.json.

Purpose: Plan section 13, Phase 2b step 3. The zones decide "cat past the doorway" vs "cat in the doorway",
         so they must match where the camera actually is. Redraw them whenever the camera moves.
Status:  Written; polygon/config writing tested (tests/test_trap_footage.py). Needs a frame from the real mount.

    python -m edge.draw_zones --image pilot/s1/frame.jpg
    python -m edge.draw_zones --image frame.jpg --trap 0.3,0 1,0 1,1 0.3,1 --door 0,0 0.2,0 0.2,1 0,1   # no GUI

Window: click the corners of the TRAP zone (where a cat counts as fully inside; the bottom of the cat's box
must be in it), press Enter; then click the DOORWAY corners, press Enter. u = undo last point, q = quit
without saving. Coordinates are saved normalized (0..1), so the camera resolution doesn't matter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .vision import CONFIG


def parse_points(pts: list[str]) -> list[list[float]]:
    out = []
    for p in pts:
        x, y = (float(v) for v in p.split(","))
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"point {p} must be normalized 0..1")
        out.append([round(x, 4), round(y, 4)])
    if len(out) < 3:
        raise ValueError("a zone needs at least 3 points")
    return out


def save_zones(trap, door, path: Path = CONFIG) -> None:
    cfg = json.loads(path.read_text())
    cfg["trap_zone"], cfg["doorway_zone"] = trap, door
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def click_polygon(img, title: str, done: list | None = None) -> list[list[float]] | None:
    import cv2
    h, w = img.shape[:2]
    pts: list[tuple[int, int]] = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        view = img.copy()
        if done:
            cv2.polylines(view, [__import__("numpy").array([[int(a * w), int(b * h)] for a, b in done])], True,
                          (0, 200, 255), 2)
        for i, p in enumerate(pts):
            cv2.circle(view, p, 4, (0, 255, 0), -1)
            if i:
                cv2.line(view, pts[i - 1], p, (0, 255, 0), 2)
        cv2.putText(view, f"{title}: click corners, Enter = done, u = undo, q = quit", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(title, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and len(pts) >= 3:
            cv2.destroyWindow(title)
            return [[round(x / w, 4), round(y / h, 4)] for x, y in pts]
        if k == ord("u") and pts:
            pts.pop()
        if k == ord("q"):
            cv2.destroyAllWindows()
            return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="a frame from the mounted trap camera")
    ap.add_argument("--trap", nargs="+", help="x,y points (normalized) instead of clicking")
    ap.add_argument("--door", nargs="+")
    ap.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args()

    if args.trap and args.door:
        trap, door = parse_points(args.trap), parse_points(args.door)
    else:
        import cv2
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read {args.image}")
        trap = click_polygon(img, "TRAP zone")
        if trap is None:
            return print("not saved")
        door = click_polygon(img, "DOORWAY zone", done=trap)
        if door is None:
            return print("not saved")
    save_zones(trap, door, Path(args.config))
    print(f"saved to {args.config}\n  trap_zone    {trap}\n  doorway_zone {door}")
    print("commit edge/vision_config.json so the Pi and everyone else use the same zones")


if __name__ == "__main__":
    main()
