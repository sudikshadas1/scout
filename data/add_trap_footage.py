"""Adds one labeled recording session of OUR trap footage to the dataset (for E5 and the staged trial checks).

Purpose: Plan sections 10 and 12. Staged prop footage checks the camera, zones, vote and gate (never accuracy), so labeled sessions
         need to land in a folder the training and eval scripts already understand.
Status:  Written; class mapping and copy tested (tests/test_new_tools.py). Needs a real labeling export.

Input: a YOLO-format export from the labeling tool (Roboflow "YOLO" export, CVAT "YOLO 1.1", Label Studio
"YOLO"). Images and .txt labels can sit anywhere under the folder; class names come from data.yaml,
obj.names or classes.txt. Names are mapped onto our 7 classes ("cat", "stuffed cat" and "tablet cat" -> cat,
"raccoon toy" -> raccoon, and so on), so label props by the animal they stand for.

Whole sessions go to one split, never single frames (plan section 7): keep 1 or 2 sessions as test only.

    python data/add_trap_footage.py --export ~/Downloads/session1_yolo --session s1_day --split train \
        --data ~/scratch/scout_data
    python data/add_trap_footage.py --export ... --session s4_dark --split test --data ~/scratch/scout_data
    python data/add_trap_footage.py --data ~/scratch/scout_data --summary

Writes <data>/trap/yolo/{images,labels}/<split>/trap__<session>__<name>.jpg|.txt. Then:
    E5 training:   python data/make_train_list.py --data ~/scratch/scout_data --cat-repeat 2 \
                       --extra-dir ~/scratch/scout_data/trap/yolo/images/train --extra-repeat 3 --tag e5
    pass/fail:     python training/eval.py --model <weights> --data ~/scratch/scout_data/trap --split test --name E5_trap
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

from common import CLASS_ID, CLASSES, species_to_class

IMG_EXT = {".jpg", ".jpeg", ".png"}
PROP_WORDS = ("stuffed", "toy", "plush", "tablet", "photo", "printed", "fake", "prop")
NOT_ANIMALS = {"bait", "food", "bowl", "plate", "trap", "door", "doorway", "hand", "empty", "background"}


def read_names(export: Path) -> list[str]:
    for name in ("data.yaml", "data.yml"):
        for f in export.rglob(name):
            text = f.read_text()
            m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
            if m:
                return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
            block = re.search(r"names:\s*\n((?:\s+.*\n?)+)", text)
            if block:
                out = []
                for line in block.group(1).splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("- "):
                        out.append(line[2:].strip().strip("'\""))
                    elif ":" in line:
                        out.append(line.split(":", 1)[1].strip().strip("'\""))
                    else:
                        break
                if out:
                    return out
    for name in ("obj.names", "classes.txt"):
        for f in export.rglob(name):
            return [x.strip() for x in f.read_text().splitlines() if x.strip()]
    raise SystemExit(f"no class names found in {export} (data.yaml, obj.names or classes.txt)")


def map_name(name: str) -> str | None:
    n = name.lower()
    for w in PROP_WORDS:
        n = n.replace(w, " ")
    n = " ".join(n.split())
    if not n or n in NOT_ANIMALS:
        return None
    if n in CLASSES:
        return n
    return species_to_class(n)


def convert_label(text: str, id_map: dict[int, int | None]) -> list[str]:
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        new = id_map.get(int(float(p[0])))
        if new is None:
            continue
        out.append(" ".join([str(new), *p[1:5]]))
    return out


def add_session(export: Path, data: Path, session: str, split: str) -> Counter:
    names = read_names(export)
    id_map = {}
    for i, n in enumerate(names):
        c = map_name(n)
        id_map[i] = CLASS_ID[c] if c else None
        print(f"  label '{n}' -> {c or 'dropped'}")
    labels = {p.stem: p for p in export.rglob("*.txt") if p.name not in ("classes.txt",)}
    root = data / "trap" / "yolo"
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for img in sorted(p for p in export.rglob("*") if p.suffix.lower() in IMG_EXT):
        stem = f"trap__{session}__{img.stem}"
        dst = root / "images" / split / f"{stem}.jpg"
        if img.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy2(img, dst)
        else:
            from PIL import Image
            with Image.open(img) as im:
                im.convert("RGB").save(dst, quality=95)
        lines = convert_label(labels[img.stem].read_text(), id_map) if img.stem in labels else []
        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        ids = {int(line.split()[0]) for line in lines}
        counts["cat" if CLASS_ID["cat"] in ids else (CLASSES[min(ids)] if ids else "empty")] += 1
    return counts


def summary(data: Path) -> None:
    root = data / "trap" / "yolo" / "labels"
    for split in sorted(p.name for p in root.iterdir()) if root.exists() else []:
        sess = Counter()
        cls = Counter()
        for f in (root / split).glob("*.txt"):
            sess[f.name.split("__")[1]] += 1
            ids = {int(x.split()[0]) for x in f.read_text().splitlines() if x.strip()}
            cls["cat" if 0 in ids else (CLASSES[min(ids)] if ids else "empty")] += 1
        print(f"{split:5s} {sum(sess.values()):5d} images  sessions={dict(sess)}  {dict(cls)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--export", help="labeling-tool export folder (YOLO format)")
    ap.add_argument("--session", help="short session name, e.g. s1_day")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    data = Path(args.data).expanduser()
    if args.export:
        if not (args.session and args.split):
            raise SystemExit("--export needs --session and --split")
        c = add_session(Path(args.export).expanduser(), data, args.session, args.split)
        print(f"added session {args.session} to {args.split}: {dict(c)}")
        total = sum(c.values())
        if total and max(c.values()) / total > 0.6:
            print("  warning: one class is over 60% of this session; plan section 7 wants cats, non-cats and the"
                  " empty trap in about equal amounts so the model can't learn 'trap = cat'")
    summary(data)


if __name__ == "__main__":
    main()
