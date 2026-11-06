"""Picks the cat cutoff and the K-of-N vote: how often would the TRAP (not one frame) report "cat in trap"?

Purpose: The gate acts on the vote, not on single frames. A per-frame recall that looks fine can still give a
         low trap-level recall (60% per frame -> ~34% for 4-of-5), so the cutoff and K have to be chosen together.
Status:  Written; vote logic tested (tests/test_new_tools.py). Clip mode needs recorded clips (footage pilot).
         Goal is under 5% of animals called cats (--max-fpr), so empty pictures and empty_ clips are left out. Sweeps
         made before animal_fpr existed fall back to noncat_fpr. Sweep mode applies it to the single-frame rate: the
         independent-frame trap estimate is far too low (E1b at cutoff 0.05 is 9.9% per frame, 0.8% estimated).
         --fps defaults to 5, but the Pi 3 may run slower: use the rate edge/bench.py measures.

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
   VisionAggregator AND the real GateController (non-cat cooldown, doorway check), so "gate closed on cat
   clips" is what a staged trial would score. --full-frame ignores the zones (use it until
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


def sweep_rows(sweep: list[dict], fps: float = 5.0) -> list[dict]:
    rows = []
    for r in sweep:
        p, q = float(r["cat_recall"]), float(r.get("animal_fpr") or r["noncat_fpr"])
        for n, k in VOTES:
            rows.append({"cutoff": float(r["threshold"]), "N": n, "K": k, "frame_recall": p, "frame_fpr": q,
                         "trap_recall_est": p_at_least(k, n, p), "trap_fpr_est": p_at_least(k, n, q),
                         "decision_s": k / fps})
    return rows


def best_rows(rows: list[dict], recall_key: str, fpr_key: str, max_fpr: float) -> list[dict]:
    """Settings with FPR under max_fpr, best recall first."""
    ok = [r for r in rows if r[recall_key] is not None and r[fpr_key] is not None and r[fpr_key] < max_fpr]
    return sorted(ok, key=lambda r: (-r[recall_key], r[fpr_key], r["K"]))


def replay(frames: list[list], cutoff: float, n: int, k: int, cfg: dict, full_frame: bool,
           gate: bool = False, fps: float = 5.0) -> bool:
    """Run one clip's per-frame detections through the real aggregator (and, with gate=True, the real gate).

    Returns True if cat_in_trap ever fires (gate=False) or if the gate closes (gate=True). The gate adds the
    non-cat cooldown and the doorway check, so it is the number a staged trial actually scores.
    """
    from edge.gate import Command, GateController
    from edge.vision import VisionAggregator
    thr = dict(cfg.get("thresholds", {}))
    thr["cat"] = cutoff
    zones = {} if full_frame else {"trap_zone": cfg.get("trap_zone"), "doorway_zone": cfg.get("doorway_zone")}
    v = VisionAggregator(N=n, K=k, thresholds=thr, same_animal_iou=cfg.get("same_animal_iou", 0.6),
                         same_animal_margin=cfg.get("same_animal_margin", 0.1), **zones)
    if not gate:
        return any(v.update(dets).cat_in_trap for dets in frames)
    t = [0.0]
    g = GateController(clock=lambda: t[0])
    g.arm()
    for dets in frames:
        t[0] += 1.0 / fps
        if g.tick(v.update(dets)).command == Command.CLOSE:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("csv", help="threshold_sweep.csv from training/eval.py")
    s.add_argument("--name", help="results folder (default: next to the csv)")
    s.add_argument("--fps", type=float, default=5.0, help="frame rate measured on the Pi 3")
    s.add_argument("--max-fpr", type=float, default=0.05, help="limit on the single-frame false-positive rate")
    c = sub.add_parser("clips")
    c.add_argument("--model", required=True)
    c.add_argument("--clips", required=True)
    c.add_argument("--imgsz", type=int, default=416)
    c.add_argument("--full-frame", action="store_true", help="ignore trap/doorway zones")
    c.add_argument("--name", default="vote_clips")
    c.add_argument("--fps", type=float, default=5.0, help="frame rate measured on the Pi 3")
    c.add_argument("--max-fpr", type=float, default=0.05, help="limit on the false-positive rate measured on clips")
    args = ap.parse_args()

    if args.mode == "sweep":
        rows = sweep_rows(list(csv.DictReader(open(args.csv))), args.fps)
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
            closed = [(is_cat, kind, replay(fr, cutoff, n, k, cfg, args.full_frame, gate=True, fps=args.fps)) for is_cat, kind, fr in data]
            cats = [f for c_, _, f in fired if c_]
            non = [f for c_, kd, f in fired if not c_ and kd != "empty"]
            gc = [f for c_, _, f in closed if c_]
            gn = [f for c_, kd, f in closed if not c_ and kd != "empty"]
            rows.append({"cutoff": cutoff, "N": n, "K": k,
                         "trap_recall": sum(cats) / len(cats) if cats else None,
                         "trap_fpr": sum(non) / len(non) if non else None,
                         "gate_closed_on_cat": sum(gc) / len(gc) if gc else None,
                         "gate_closed_on_noncat": sum(gn) / len(gn) if gn else None,
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
    ok = best_rows(rows, rk, "frame_fpr" if args.mode == "sweep" else fk, args.max_fpr)
    limit = "single-frame false-positive rate" if args.mode == "sweep" else "trap false-positive rate"
    print(f"\nbest settings with {limit} < {args.max_fpr:.0%} ({'independent-frame estimate' if args.mode == 'sweep' else 'measured on clips'}):")
    print(f"  {'cutoff':>6} {'vote':>6} {'trap recall':>12} {'trap FPR':>9}")
    for r in ok[:8]:
        extra = ""
        if r.get("gate_closed_on_cat") is not None:
            extra = f"   gate closed on {r['gate_closed_on_cat']:.0%} of cat clips"
            if r.get("gate_closed_on_noncat") is not None:
                extra += f", {r['gate_closed_on_noncat']:.0%} of non-cat clips"
        print(f"  {r['cutoff']:6.2f} {r['K']}-of-{r['N']:<2} {r[rk]:12.1%} {r[fk]:9.2%}{extra}")
    cur = next((r for r in rows if r["cutoff"] == 0.5 and (r["N"], r["K"]) == (5, 4)), None)
    if cur:
        print(f"  (old config, 0.50 and 4-of-5: {cur[rk]:.1%} recall)")
    print(f"wrote {out / 'vote_sweep.csv'}")


if __name__ == "__main__":
    main()
