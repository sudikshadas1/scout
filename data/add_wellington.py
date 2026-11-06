"""Adds Wellington Camera Traps (LILA) for E7: many more cats, from many more cameras, plus a new held-out eval set.

Purpose: Our training cats are thin: 2,553 pictures, and 71% of the Caltech ones come from one camera. Only 7 are large
         in the frame, and the trap camera will mostly see large cats. Wellington has 28,599 cat pictures with a usable
         box, from 187 cameras. It also gives a new eval set from unseen cameras, because data/splits.json has only 208
         val cats, from 3 cameras. Scores on it are added next to the 500-picture numbers and never replace them.
Status:  Selection logic tested (tests/test_improvement.py). Metadata and image links checked. Not yet run on PACE.
         3,000 cats by default, because Wellington has no raccoons or opossums. Cats become 2.5x raccoons (now 1.1x);
         E1c raised the cats' share and opossum false positives rose, so watch both animals.
Later stages (still to add or change):
  - Check the contact sheet of selected boxes before the first training run.

Wellington labels whole sequences and has no boxes, so boxes come from LILA's published MegaDetector results, and each
sequence gives its clearest frame. Non-cats and empty pictures come from the same cameras, so "this camera" or "the
tube trap in view" can't mean cat. Whole cameras go to train or val, never both.

    python data/add_wellington.py --data ~/scratch/scout_data
    # -> ~/scratch/scout_data/wct/yolo/{images,labels}/{train,val}
    python data/make_train_list.py --data ~/scratch/scout_data --cat-repeat 1 \
        --extra-dir ~/scratch/scout_data/wct/yolo/images/train --tag e7
    python training/eval.py --model results/E1b/weights/best.pt --data ~/scratch/scout_data/wct --split val --imgsz 416 --name E1b_wct416
"""
from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import CLASS_ID, is_empty, species_to_class

META_URL = "https://storage.googleapis.com/public-datasets-lila/wellingtoncameratraps/wellington_camera_traps.json.zip"
MD_URL = ("https://lila.science/public/lila-md-results/"
          "wellington-camera-traps_images_mdv5a.0.0_results.filtered_rde_0.150_0.850_10_0.200.json.zip")
IMG_URL = "https://storage.googleapis.com/public-datasets-lila/wellington-unzipped/images/{}"
DROP = {"unclassifiable"}


def wellington_class(name: str) -> str | None:
    """Our class for a Wellington label. NZ "possum" is the brushtail, not our opossum, so it is other_animal."""
    n = name.strip().lower()
    if n in DROP or is_empty(n):
        return None
    return "other_animal" if n == "possum" else species_to_class(n)


def usable_box(det: dict, min_conf: float) -> bool:
    """A MegaDetector animal box worth training on. Drops slivers and boxes on the camera's footer strip."""
    if str(det.get("category")) != "1" or det.get("conf", 0) < min_conf:
        return False
    x, y, w, h = det["bbox"]
    if w < 0.02 or h < 0.02:
        return False
    return not (y + h >= 0.96 and h < 0.10)


def split_locations(locations: list[str], val_frac: float, seed: int) -> dict[str, str]:
    locs = sorted(set(locations))
    random.Random(seed).shuffle(locs)
    n_val = round(len(locs) * val_frac)
    return {loc: ("val" if i < n_val else "train") for i, loc in enumerate(locs)}


def clearest_per_sequence(rows: list[dict]) -> list[dict]:
    """One frame per sequence (labels are per sequence, so other frames may not show the animal)."""
    best: dict = {}
    for r in rows:
        if r["seq"] not in best or r["conf"] > best[r["seq"]]["conf"]:
            best[r["seq"]] = r
    return list(best.values())


def capped(rows: list[dict], total: int, per_location: int, rng: random.Random) -> list[dict]:
    """Up to `total` rows at random, at most `per_location` from any one camera."""
    rows = list(rows)
    rng.shuffle(rows)
    out, seen = [], Counter()
    for r in rows:
        if len(out) >= total:
            break
        if seen[r["location"]] < per_location:
            seen[r["location"]] += 1
            out.append(r)
    return out


def build_rows(meta: dict, md: dict, min_conf: float, val_frac: float, seed: int) -> list[dict]:
    """One candidate row per sequence. Sequences with more than one label or more than one animal box are dropped."""
    names = {c["id"]: c["name"] for c in meta["categories"]}
    labels = defaultdict(set)
    for a in meta["annotations"]:
        labels[a["image_id"]].add(names[a["category_id"]])
    dets = {im["file"].lower(): im.get("detections") or [] for im in md["images"]}
    where = split_locations([str(i.get("location")) for i in meta["images"]], val_frac, seed)

    animals, empties = [], []
    for img in meta["images"]:
        found = labels.get(img["id"], set())
        if len(found) != 1:
            continue
        name = next(iter(found))
        loc, ds = str(img.get("location")), dets.get(img["file_name"].lower(), [])
        base = {"file_name": img["file_name"], "seq": img.get("seq_id", img["id"]), "location": loc,
                "split": where[loc], "species": name}
        if is_empty(name):
            if not ds:  # "empty" must really look empty
                empties.append({**base, "cls": None, "conf": 0.0, "bbox": None})
            continue
        cls = wellington_class(name)
        good = [d for d in ds if usable_box(d, min_conf)]
        if cls is None or len(good) != 1 or len(ds) != 1:
            continue
        animals.append({**base, "cls": cls, "conf": good[0]["conf"], "bbox": good[0]["bbox"]})
    return clearest_per_sequence(animals) + clearest_per_sequence(empties)


def select(rows: list[dict], split: str, n_cats: int, per_location: int, bg_frac: float, rng: random.Random) -> list[dict]:
    """Cats, then about as many non-cats spread across species, then empties, all for one split."""
    rows = [r for r in rows if r["split"] == split]
    cats = capped([r for r in rows if r["cls"] == "cat"], n_cats, per_location, rng)
    by_species = defaultdict(list)
    for r in rows:
        if r["cls"] and r["cls"] != "cat":
            by_species[r["species"]].append(r)
    each = max(len(cats) // max(len(by_species), 1), 1)
    others = [x for rs in by_species.values() for x in capped(rs, each, max(per_location // 2, 1), rng)]
    n_bg = int(bg_frac * (len(cats) + len(others)) / max(1 - bg_frac, 1e-6))
    return cats + others + capped([r for r in rows if r["cls"] is None], n_bg, per_location, rng)


def yolo_line(row: dict) -> str:
    """MegaDetector boxes are normalized top-left x, y, width, height."""
    if row["cls"] is None:
        return ""
    x, y, w, h = row["bbox"]
    return f"{CLASS_ID[row['cls']]} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}"


def fetch_json(url: str, dest: Path) -> dict:
    import requests
    if not dest.exists():
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            dest.write_bytes(z.read([n for n in z.namelist() if n.endswith(".json")][0]))
    return json.loads(dest.read_text())


def download(row: dict, root: Path, session) -> bool:
    stem = "wct__" + row["file_name"].rsplit(".", 1)[0]
    img = root / "images" / row["split"] / f"{stem}.jpg"
    if not img.exists():
        r = session.get(IMG_URL.format(row["file_name"]), timeout=60)
        if r.status_code != 200:
            return False
        img.write_bytes(r.content)
    (root / "labels" / row["split"] / f"{stem}.txt").write_text(yolo_line(row))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cats", type=int, default=3000, help="cat pictures for training; more shifts the mix further from raccoons and opossums")
    ap.add_argument("--val-cats", type=int, default=1500, help="cat pictures for the new eval set")
    ap.add_argument("--per-location", type=int, default=300, help="most cat pictures from one camera")
    ap.add_argument("--min-conf", type=float, default=0.7, help="MegaDetector confidence for a usable box")
    ap.add_argument("--val-frac", type=float, default=0.2, help="share of cameras kept for the eval set")
    ap.add_argument("--bg-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=4605)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="print counts, download no images")
    args = ap.parse_args()

    import requests

    data = Path(args.data).expanduser()
    meta_dir = data / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = fetch_json(META_URL, meta_dir / "wct.json")
    md = fetch_json(MD_URL, meta_dir / "wct_md.json")
    rows = build_rows(meta, md, args.min_conf, args.val_frac, args.seed)
    rng = random.Random(args.seed)
    chosen = (select(rows, "train", args.cats, args.per_location, args.bg_frac, rng)
              + select(rows, "val", args.val_cats, args.per_location, args.bg_frac, rng))
    for split in ("train", "val"):
        part = [r for r in chosen if r["split"] == split]
        cats = [r for r in part if r["cls"] == "cat"]
        print(f"{split}: {len(part):,} pictures from {len({r['location'] for r in part})} cameras; "
              f"{len(cats):,} cats from {len({r['location'] for r in cats})} cameras; "
              f"{dict(Counter(r['cls'] or 'background' for r in part))}")
    if args.dry_run:
        return

    root = data / "wct" / "yolo"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    with ThreadPoolExecutor(args.workers) as pool:
        ok = list(pool.map(lambda r: download(r, root, session), chosen))
    print(f"done: {sum(ok):,} ok, {len(ok) - sum(ok):,} failed, in {root}")


if __name__ == "__main__":
    main()
