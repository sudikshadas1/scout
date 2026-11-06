"""Reads a run's results.csv: how long it trained, its best epoch, and whether it was still improving at the end.

Purpose: Training longer is the cheapest possible win, but only if the run hadn't levelled off.
Status:  Tested (tests/test_improvement.py). train_yolo.py saves this in summary.json.

    python training/curve.py results/E1b
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

METRIC = "metrics/mAP50-95(B)"


def summarize(rows: list[dict], window: int = 10) -> dict:
    """rows: Ultralytics results.csv. still_improving = the best score came in the last `window` epochs."""
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    scores = [float(r[METRIC]) for r in rows]
    best = max(range(len(scores)), key=scores.__getitem__)
    return {"epochs_run": len(rows), "best_epoch": best + 1, "best_map50_95": round(scores[best], 4),
            "gain_last_window": round(scores[-1] - scores[max(0, len(scores) - 1 - window)], 4),
            "still_improving": best >= len(rows) - window,
            "train_minutes": round(float(rows[-1].get("time", 0)) / 60, 1)}


def read_run(run_dir: Path) -> dict | None:
    f = Path(run_dir) / "results.csv"
    if not f.exists():
        return None
    with open(f, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return summarize(rows) if rows else None


if __name__ == "__main__":
    print(json.dumps(read_run(Path(sys.argv[1])), indent=1))
