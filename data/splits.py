"""Freezes location-based train / val / test splits so no camera location is in two splits.

Purpose: Stops the model from memorizing backgrounds: test locations are never seen in training (plan: background-learning safeguards).
Status:  Written and tested on fake data. Not yet run; data/splits.json not yet committed.
Later stages (still to add or change):
  - Phase 1: run once on PACE and commit data/splits.json. Do not change it afterwards.
  - Phase 2a: add a seen-location split (held-out images from training locations) to compare seen vs new locations.
  - Phase 2a: add our own trap footage, split by recording session or day, never by individual frame.

CCT: LILA's recommended split file gives train and val *locations*. We keep LILA train as our
train, and split LILA's val locations in half (seeded) into our val and our frozen test set.
ENA24: split its locations 70/15/15 the same way (falls back to sequence id if an image has
no location field).

Commit the output (data/splits.json) so everyone trains and tests on the same split.

    python data/splits.py --data ~/scratch/scout_data
"""
import argparse
import json
import random
from pathlib import Path

from common import CocoCT


def load_lila_cct_splits(path: Path) -> dict[str, list[str]]:
    d = json.loads(path.read_text())
    d = d.get("splits", d)
    train = d.get("train") or d.get("train_locations")
    val = d.get("val") or d.get("val_locations")
    if not train or not val:
        raise ValueError(f"Unexpected split file layout, keys: {list(d)}")
    return {"train": [str(x) for x in train], "val": [str(x) for x in val]}


def split_groups(groups: list[str], fracs=(0.7, 0.15, 0.15), seed=4605) -> dict[str, list[str]]:
    g = sorted(set(groups))
    random.Random(seed).shuffle(g)
    n_train = round(len(g) * fracs[0])
    n_val = round(len(g) * fracs[1])
    return {
        "train": sorted(g[:n_train]),
        "val": sorted(g[n_train:n_train + n_val]),
        "test": sorted(g[n_train + n_val:]),
    }


def group_key(img: dict) -> str:
    if "location" in img:
        return str(img["location"])
    return "seq:" + str(img.get("seq_id", img["id"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--seed", type=int, default=4605)
    args = ap.parse_args()
    meta = Path(args.data).expanduser() / "meta"

    lila = load_lila_cct_splits(meta / "cct_splits.json")
    val_locs = sorted(lila["val"])
    random.Random(args.seed).shuffle(val_locs)
    half = len(val_locs) // 2
    cct = {"train": sorted(lila["train"]), "val": sorted(val_locs[:half]), "test": sorted(val_locs[half:])}

    # ENA24 has no camera location or sequence metadata.
    # Use all ENA24 images for supplemental training only.
    # Keep the keys consistent with build_manifest.py.
    ena = CocoCT(meta / "ena24.json")

    ena_split = {
        "train": sorted(group_key(img) for img in ena.images.values()),
        "val": [],
        "test": [],
    }

    # ena = CocoCT(meta / "ena24.json")
    # ena_groups = [group_key(img) for img in ena.images.values()]
    # ena_split = split_groups(ena_groups, seed=args.seed)

    out = {"seed": args.seed, "cct": cct, "ena24": ena_split}
    Path(args.out).write_text(json.dumps(out, indent=1))
    for name, s in (("cct", cct), ("ena24", ena_split)):
        print(name, {k: len(v) for k, v in s.items()}, "locations/groups")
    print(f"wrote {args.out}  <- commit this file")


if __name__ == "__main__":
    main()
