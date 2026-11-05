
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# E1b missed-cat analysis results
sizes = ["Tiny", "Small", "Medium"]
detected = [9, 118, 17]
total = [18, 171, 19]

recall = [100 * d / n for d, n in zip(detected, total)]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    sizes,
    recall,
    color=["#E76F51", "#E9C46A", "#2A9D8F"]
)

for bar, d, n, pct in zip(bars, detected, total, recall):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        pct + 2,
        f"{pct:.1f}%\n({d}/{n})",
        ha="center",
        fontsize=11
    )

ax.axhline(
    75,
    color="gray",
    linestyle="--",
    label="75% overall cat-recall target"
)

ax.set_ylim(0, 110)
ax.set_ylabel("Cat Recall (%)")
ax.set_xlabel("Cat Size")
ax.set_title("SCOUT E1b: Cat Detection Recall by Size")
ax.legend()
ax.grid(axis="y", alpha=0.2)

output = Path("results/halftime")
output.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(output / "e1b_recall_by_size.png", dpi=200)

print(f"Saved {output / 'e1b_recall_by_size.png'}")
