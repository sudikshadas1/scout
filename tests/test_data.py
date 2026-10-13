"""Runs the data pipeline end to end on a tiny fake dataset (no network).

Purpose: Catches mapping, split and label bugs before spending PACE time on the real data.
Status:  Passing.
Later stages (still to add or change):
  - Add a test for own-footage ingestion and the seen-location split once they exist.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from common import species_to_class  # noqa: E402
from download_images import yolo_lines  # noqa: E402


def test_species_mapping():
    assert species_to_class("cat") == "cat"
    assert species_to_class("Domestic Cat") == "cat"
    assert species_to_class("Northern Raccoon") == "raccoon"
    assert species_to_class("Virginia Opossum") == "opossum"
    assert species_to_class("Eastern Gray Squirrel") == "squirrel"
    assert species_to_class("bobcat") == "other_animal"
    assert species_to_class("empty") is None
    assert species_to_class("car") is None


def test_yolo_lines_normalize_and_clamp():
    lines = yolo_lines([[0, 100, 50, 200, 100], [1, 900, 0, 200, 100]], 1000, 500)
    assert lines[0] == "0 0.200000 0.200000 0.200000 0.200000"
    cx = float(lines[1].split()[1])
    assert cx <= 1.0


def fake_coco(n_locs=6):
    images, anns = [], []
    cats = [{"id": 0, "name": "empty"}, {"id": 1, "name": "cat"}, {"id": 2, "name": "raccoon"},
            {"id": 3, "name": "opossum"}]
    aid = 0
    for loc in range(n_locs):
        for k in range(12):
            iid = f"L{loc}_{k}"
            images.append({"id": iid, "file_name": f"{iid}.jpg", "location": str(loc),
                           "seq_id": f"{iid}_seq", "width": 1000, "height": 500})
            cat_id = [0, 1, 1, 2, 3, 2][k % 6]
            ann = {"id": aid, "image_id": iid, "category_id": cat_id}
            if cat_id:
                ann["bbox"] = [100, 100, 200, 150]
            anns.append(ann)
            aid += 1
    return {"images": images, "annotations": anns, "categories": cats}


def test_pipeline_on_fake_data(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    coco = fake_coco()
    for name in ("cct_images.json", "cct_bboxes.json", "ena24.json"):
        (meta / name).write_text(json.dumps(coco))
    (meta / "cct_splits.json").write_text(json.dumps(
        {"splits": {"train": ["0", "1", "2", "3"], "val": ["4", "5"]}}))
    splits = tmp_path / "splits.json"

    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, check=True,  # noqa: E731
                                    capture_output=True, text=True).stdout
    run("data/splits.py", "--data", str(tmp_path), "--out", str(splits))
    s = json.loads(splits.read_text())
    cct = s["cct"]
    assert not set(cct["train"]) & set(cct["val"]) and not set(cct["val"]) & set(cct["test"])

    out = run("data/build_manifest.py", "--data", str(tmp_path), "--splits", str(splits))
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert rows, out
    by_split = {r["split"] for r in rows}
    assert "train" in by_split
    # every train image's location must be a train location (no leakage)
    for r in rows:
        if r["dataset"] == "cct":
            loc = r["file_name"].split("_")[0][1:]
            assert loc in cct[r["split"]]
    assert any(r["boxes"] == [] for r in rows), "expected some background images"
    assert all(r["url"].startswith("https://storage.googleapis.com/public-datasets-lila/") for r in rows)
