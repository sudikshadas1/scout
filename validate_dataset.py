
from pathlib import Path
from collections import Counter
import math

root = Path.home() / "scratch/scout_data/yolo"
counts = Counter()
invalid = []
total_boxes = 0

for split in ("train", "val", "test"):
    images = list((root / "images" / split).glob("*.jpg"))
    labels = list((root / "labels" / split).glob("*.txt"))

    counts[(split, "images")] = len(images)
    counts[(split, "labels")] = len(labels)

    for path in labels:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue

            try:
                parts = line.split()
                assert len(parts) == 5
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                assert 0 <= cls <= 6
                assert all(math.isfinite(v) for v in (cx, cy, w, h))
                assert 0 <= cx <= 1
                assert 0 <= cy <= 1
                assert 0 < w <= 1
                assert 0 < h <= 1
                assert cx - w/2 >= -1e-5
                assert cx + w/2 <= 1 + 1e-5
                assert cy - h/2 >= -1e-5
                assert cy + h/2 <= 1 + 1e-5

                total_boxes += 1

            except (ValueError, AssertionError):
                invalid.append((path.name, line))

for split in ("train", "val", "test"):
    print(f"{split.upper()}:")
    print(" Images:", counts[(split, "images")])
    print(" Labels:", counts[(split, "labels")])

print("\nTotal valid boxes:", total_boxes)
print("Invalid boxes:", len(invalid))

for example in invalid[:5]:
    print("Invalid example:", example)
