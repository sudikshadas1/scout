"""Plots cat recall by how big the cat is in the picture, for one or more runs, from analyze_errors.py's summary.json.

Purpose: E1b catches more cats the bigger they are (tiny 50%, small 69%, medium 89% at cutoff 0.25). The trap camera
         sees cats up close, so this shows whether a new run (E7 and later) helps the sizes that matter.
Status:  Tested (tests/test_improvement.py). First version by Henry, with the numbers typed in; this one reads them.
Later stages (still to add or change):
  - The validation set has no large cats. Add them when a close-range set exists (prop footage, backyard videos).

    python training/plot_cat_sizes.py results/E1b_analysis/summary.json
    python training/plot_cat_sizes.py results/E1b_analysis416/summary.json results/E7_analysis416/summary.json

Writes results/plots/recall_by_size.png by default (pictures are git-ignored).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SIZES = ["tiny", "small", "medium", "large"]   # analyze_errors.py SIZE_BINS order
TARGET = 75


def run_label(summary_path: Path) -> str:
    """results/E1b_analysis416/summary.json -> E1b (416)"""
    name = summary_path.parent.name
    base, _, tail = name.partition("_analysis")
    return f"{base} ({tail.strip('_')})" if tail.strip("_") else base


def size_rows(summary: dict) -> list[tuple[str, int, int, float]]:
    """[(size, cats caught, cats, recall %)] in size order, leaving out sizes with no cats."""
    by = summary.get("recall_by_size", {})
    rows = []
    for s in SIZES:
        v = by.get(s)
        if v and v.get("n"):
            rows.append((s, v["hit"], v["n"], 100 * v["hit"] / v["n"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", help="summary.json files from training/analyze_errors.py")
    ap.add_argument("--out", default="results/plots/recall_by_size.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for p in map(Path, args.summaries):
        s = json.loads(p.read_text())
        runs.append((run_label(p), s.get("threshold"), size_rows(s)))
    sizes = [s for s in SIZES if any(s == r[0] for _, _, rows in runs for r in rows)]
    if not sizes:
        raise SystemExit("no recall_by_size in these summaries")

    width = 0.8 / len(runs)
    colors = ["#2A9D8F", "#E76F51", "#E9C46A", "#264653"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, thr, rows) in enumerate(runs):
        got = {r[0]: r for r in rows}
        for j, size in enumerate(sizes):
            if size not in got:
                continue
            _, hit, n, pct = got[size]
            x = j + (i - (len(runs) - 1) / 2) * width
            ax.bar(x, pct, width, color=colors[i % len(colors)], label=label if j == 0 or size == sizes[0] else None)
            ax.text(x, pct + 2, f"{pct:.0f}%\n({hit}/{n})", ha="center", fontsize=9)
    ax.axhline(TARGET, color="gray", linestyle="--", label=f"{TARGET}% cat goal (for all cats, not each size)")
    ax.set_xticks(range(len(sizes)), [s.capitalize() for s in sizes])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Cats caught (%)")
    ax.set_xlabel("Cat size in the picture")
    thr = {t for _, t, _ in runs}
    ax.set_title("SCOUT: cats caught by size" + (f", cutoff {thr.pop()}" if len(thr) == 1 else ""))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    for label, _, rows in runs:
        print(f"{label}: " + ", ".join(f"{s} {p:.1f}% ({h}/{n})" for s, h, n, p in rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
