"""Picks the cat cutoff and the K-of-N vote: how often would the TRAP (not one frame) report "cat in trap"?

Purpose: The gate acts on the vote, not on single frames. A per-frame recall that looks fine can still give a
         low trap-level recall (60% per frame -> ~34% for 4-of-5), so the cutoff and K have to be chosen together.
Status:  Written; vote logic tested (tests/test_vote.py). Clip mode needs recorded clips (footage pilot).

Two modes:

1. Estimate from a threshold sweep (no clips needed, run it now):
       python training/vote_sim.py sweep results/E1_full_eval/threshold_sweep.csv
   Assumes frames are independent. Real frames of the same cat are correlated, so the truth lies between
   the per-frame number and this estimate. Use it to shortlist settings, not to report final numbers.

2. Measure on real clips (after the footage pilot):
       python training/vote_sim.py clips --model best.pt --clips pilot/clips --full-frame
   --clips holds one subfolder per staged trial, named by what was in it: cat_*, raccoon_*, opossum_*,
   empty_*, ... Each subfolder holds that trial's frames in time order (demo.py --save-dir writes these).
   The model runs once per frame; every cutoff x vote setting is then replayed through the real
   VisionAggregator, so the numbers are what the trap would do. --full-frame ignores the zones (use it until
   the camera is mounted and the zones are drawn).

Writes results/<name>/vote_sweep.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VOTES = [(5, 3), (5, 4), (4, 3), (3, 2), (6, 4)]   # (N, K)
CUTOFFS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def p_at_least(k: int, n: int, p: float) -> float:
    """P(at least k successes in n independent frames, each with probability p)."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def sweep_rows(sweep: list[dict]) -> list[dict]:
    rows = []
    for r in sweep:
        p, q = float(r["cat_recall"]), float(r["noncat_fpr"])
        for n, k in VOTES:
            rows.append({"cutoff": float(r["threshold"]), "N": n, "K": k, "frame_recall": p, "frame_fpr": q,
                         "trap_recall_est": p_at_least(k, n, p), "trap_fpr_est": p_at_least(k, n, q),
                         "decision_s_at_5fps": k / 5})
    return rows


def replay(frames: list[list], cutoff: float, n: int, k: int, cfg: dict, full_frame: bool) -> bool:
    """Run one clip's per-frame detections through the real aggregator. True if cat_in_trap ever fires."""
    from edge.vision import VisionAggregator
    thr = dict(cfg.get("thresholds", {}))
    thr["cat"] = cutoff
    zones = {} if full_frame else {"trap_zone": cfg.get("trap_zone"), "doorway_zone": cfg.get("doorway_zone")}
    v = VisionAggregator(N=n, K=k, thresholds=thr, **zones)
    return any(v.update(dets).cat_in_trap for dets in frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("csv", help="threshold_sweep.csv from training/eval.py")
    s.add_argument("--name", help="results folder (default: next to the csv)")
    c = sub.add_parser("clips")
    c.add_argument("--model", required=True)
    c.add_argument("--clips", required=True)
    c.add_argument("--imgsz", type=int, default=416)
    c.add_argument("--full-frame", action="store_true", help="ignore trap/doorway zones")
    c.add_argument("--name", default="vote_clips")
    args = ap.parse_args()

    if args.mode == "sweep":
        rows = sweep_rows(list(csv.DictReader(open(args.csv))))
        out = Path("results") / args.name if args.name else Path(args.csv).parent
    else:
        import json
        import cv2
        from edge.vision import CONFIG, Detector
        cfg = json.loads(CONFIG.read_text())
        det = Detector(args.model, imgsz=args.imgsz, min_conf=min(CUTOFFS))
        clips = sorted(p for p in Path(args.clips).iterdir() if p.is_dir())
        if not clips:
            sys.exit(f"no clip folders in {args.clips}")
        data = []
        for clip in clips:
            frames = [det.detect(cv2.imread(str(f))) for f in sorted(clip.glob("*.jpg"))]
            data.append((clip.name.split("_")[0].lower() == "cat", clip.name.split("_")[0].lower(), frames))
            print(f"  {clip.name}: {len(frames)} frames")
        rows = []
        for cutoff, (n, k) in product(CUTOFFS, VOTES):
            fired = [(is_cat, kind, replay(fr, cutoff, n, k, cfg, args.full_frame)) for is_cat, kind, fr in data]
            cats = [f for c_, _, f in fired if c_]
            non = [f for c_, _, f in fired if not c_]
            rows.append({"cutoff": cutoff, "N": n, "K": k,
                         "trap_recall": sum(cats) / len(cats) if cats else None,
                         "trap_fpr": sum(non) / len(non) if non else None,
                         "n_cat_clips": len(cats), "n_noncat_clips": len(non),
                         "noncat_fired": ",".join(sorted({kd for c_, kd, f in fired if not c_ and f}))})
        out = Path("results") / args.name

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "vote_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    rk = "trap_recall_est" if args.mode == "sweep" else "trap_recall"
    fk = "trap_fpr_est" if args.mode == "sweep" else "trap_fpr"
    ok = [r for r in rows if r[rk] is not None and r[fk] is not None and r[fk] <= 0.25]
    ok.sort(key=lambda r: (-r[rk], r[fk], r["K"]))
    print(f"\nbest settings with trap false-positive rate <= 25% ({'independent-frame estimate' if args.mode == 'sweep' else 'measured on clips'}):")
    print(f"  {'cutoff':>6} {'vote':>6} {'trap recall':>12} {'trap FPR':>9}")
    for r in ok[:8]:
        print(f"  {r['cutoff']:6.2f} {r['K']}-of-{r['N']:<2} {r[rk]:12.1%} {r[fk]:9.2%}")
    cur = next((r for r in rows if r["cutoff"] == 0.5 and (r["N"], r["K"]) == (5, 4)), None)
    if cur:
        print(f"  (old config, 0.50 and 4-of-5: {cur[rk]:.1%} recall)")
    print(f"wrote {out / 'vote_sweep.csv'}")


if __name__ == "__main__":
    main()
