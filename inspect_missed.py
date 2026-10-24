# Superseded by training/analyze_errors.py (groups misses by reason, draws boxes). Kept for reference.

import csv
from pathlib import Path
from PIL import Image, ImageDraw

root = Path.home() / "scratch/scout_data/yolo/images/val"

with open("results/E1_full_eval/per_image.csv") as f:
    rows = list(csv.DictReader(f))

cats = [r for r in rows if r["truth"] == "cat"]

missed = [r for r in cats if float(r["best_cat_conf"]) < 0.05]
detected = [r for r in cats if float(r["best_cat_conf"]) >= 0.25]

selected = [("MISSED", r) for r in missed[:8]]
selected += [("DETECTED", r) for r in detected[:8]]

w, h = 240, 190
sheet = Image.new("RGB", (4 * w, 4 * h), "white")
draw = ImageDraw.Draw(sheet)

for i, (status, row) in enumerate(selected):
    path = root / row["image"]
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((230, 155))

            x = (i % 4) * w
            y = (i // 4) * h

            sheet.paste(im, (x + 5, y + 25))
            draw.text(
                (x + 5, y + 5),
                f"{status}: {float(row['best_cat_conf']):.2f}",
                fill="red" if status == "MISSED" else "green"
            )
    except Exception as e:
        print(f"Error with {path}: {e}")

sheet.save("results/E1_full_eval/cat_comparison.jpg")
print("Saved results/E1_full_eval/cat_comparison.jpg")
