"""Shared helpers for the LILA camera-trap datasets (COCO Camera Traps format)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CLASSES = ["cat", "raccoon", "opossum", "squirrel", "dog", "other_animal", "person"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

# Category names that mean "no animal": these images become background (no label file lines).
EMPTY_NAMES = {"empty", "blank", "nothing", "unknown"}
# Category names we drop entirely (not animals, not useful as background).
DROP_NAMES = {"car", "vehicle", "motorcycle", "bicycle"}


def species_to_class(name: str) -> str | None:
    """Map a dataset category name (CCT or ENA24) to one of our 7 classes.

    Returns None for empty images and for categories we drop.
    """
    n = name.strip().lower()
    if n in EMPTY_NAMES or n in DROP_NAMES:
        return None
    if n in ("cat", "domestic cat", "house cat"):
        return "cat"
    if "raccoon" in n:
        return "raccoon"
    if "opossum" in n or "possum" in n:
        return "opossum"
    if "squirrel" in n or "chipmunk" in n:
        return "squirrel"
    if n in ("dog", "domestic dog"):
        return "dog"
    if n in ("human", "person", "people"):
        return "person"
    # bobcat, coyote, fox, skunk, rabbit, deer, birds, etc.
    return "other_animal"


def is_empty(name: str) -> bool:
    return name.strip().lower() in EMPTY_NAMES


class CocoCT:
    """Light wrapper around a COCO Camera Traps json (image-level or bbox file)."""

    def __init__(self, path: str | Path):
        with open(path) as f:
            d = json.load(f)
        self.raw = d
        self.images = {img["id"]: img for img in d["images"]}
        self.cat_name = {c["id"]: c["name"] for c in d["categories"]}
        self.anns_by_image: dict = defaultdict(list)
        for a in d.get("annotations", []):
            self.anns_by_image[a["image_id"]].append(a)

    def species_of(self, image_id) -> set[str]:
        return {self.cat_name[a["category_id"]] for a in self.anns_by_image.get(image_id, [])}

    def location_of(self, image_id) -> str:
        return str(self.images[image_id].get("location", "unknown"))
