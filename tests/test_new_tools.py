"""Tests for the vote estimator, seen-location split, trap-footage import, zone tool and E2 scoring.

Purpose: These pick the trap's cutoff and vote, and feed the pass/fail test on our own footage.
Status:  Passing.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "training"))

from add_trap_footage import add_session, convert_label, map_name, read_names  # noqa: E402
from build_seen_split import pick_seen  # noqa: E402
from common import CocoCT  # noqa: E402
from test_data import fake_coco  # noqa: E402
from train_classifier import score  # noqa: E402
from vote_sim import best_rows, p_at_least, replay, sweep_rows  # noqa: E402

from edge.draw_zones import parse_points, save_zones  # noqa: E402
from edge.vision import Detection  # noqa: E402


# ---- vote_sim
def test_p_at_least():
    assert p_at_least(5, 5, 1.0) == 1.0 and p_at_least(1, 5, 0.0) == 0.0
    assert abs(p_at_least(4, 5, 0.6) - 0.33696) < 1e-4   # the "60% per frame -> ~34%" number
    assert p_at_least(3, 5, 0.6) > p_at_least(4, 5, 0.6)


def test_sweep_rows():
    rows = sweep_rows([{"threshold": "0.25", "cat_recall": "0.5", "noncat_fpr": "0.02"}])
    r = next(x for x in rows if (x["N"], x["K"]) == (5, 3))
    assert r["trap_recall_est"] == pytest.approx(0.5) and r["trap_fpr_est"] < 0.001


def test_sweep_prefers_animal_fpr():
    r = sweep_rows([{"threshold": "0.25", "cat_recall": "0.7", "animal_fpr": "0.062", "noncat_fpr": "0.048"}])[0]
    assert r["frame_fpr"] == pytest.approx(0.062)


def test_best_rows_limits_the_single_frame_rate():
    # E1b's numbers: 0.05 is over 5% per frame even though its trap estimate is under 1%
    rows = sweep_rows([{"threshold": "0.05", "cat_recall": "0.731", "noncat_fpr": "0.099"},
                       {"threshold": "0.25", "cat_recall": "0.692", "noncat_fpr": "0.048"}])
    assert best_rows(rows, "trap_recall_est", "trap_fpr_est", 0.05)[0]["cutoff"] == 0.05
    picked = best_rows(rows, "trap_recall_est", "frame_fpr", 0.05)
    assert {r["cutoff"] for r in picked} == {0.25} and (picked[0]["N"], picked[0]["K"]) == (5, 3)


def test_replay_uses_real_aggregator():
    cat = [Detection("cat", 0.3, (0.5, 0.3, 0.8, 0.9))]
    frames = [cat, [], cat, cat, []]
    cfg = {"thresholds": {}}
    assert replay(frames, 0.25, 5, 3, cfg, full_frame=True)
    assert not replay(frames, 0.25, 5, 4, cfg, full_frame=True)
    assert not replay(frames, 0.5, 5, 3, cfg, full_frame=True)  # below the cutoff
    assert replay(frames, 0.25, 5, 3, cfg, full_frame=True, gate=True)
    # a raccoon in the first frame puts the gate in cooldown: the vote fires but the gate stays open
    racc = [[Detection("raccoon", 0.9, (0.1, 0.1, 0.3, 0.3))], cat, cat, cat]
    assert replay(racc, 0.25, 5, 3, cfg, full_frame=True)
    assert not replay(racc, 0.25, 5, 3, cfg, full_frame=True, gate=True)


# ---- seen-location split
def test_seen_split_skips_trained_bursts(tmp_path):
    p = tmp_path / "cct_bboxes.json"
    p.write_text(json.dumps(fake_coco()))
    ds = CocoCT(p)
    splits = {"cct": {"train": ["0", "1"], "val": ["2"], "test": ["3"]}}
    used = {"L0_1.jpg"}
    used_seqs = {"L0_1_seq"}
    import random
    rows = pick_seen(ds, splits, used, used_seqs, per_class=50, per_seq=1, rng=random.Random(0))
    files = {r["file_name"] for r in rows}
    assert rows and all(f.split("_")[0] in ("L0", "L1") for f in files)   # only training locations
    assert "L0_1.jpg" not in files and all(r["split"] == "seen" for r in rows)
    assert all(r["boxes"] for r in rows)                                  # no empties


# ---- trap footage import
def test_map_name():
    assert map_name("cat") == "cat" and map_name("Stuffed Cat") == "cat" and map_name("tablet raccoon") == "raccoon"
    assert map_name("bait") is None and map_name("opossum toy") == "opossum"


def test_read_names_formats(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "data.yaml").write_text("train: x\nnc: 2\nnames: ['cat', 'raccoon']\n")
    assert read_names(tmp_path / "a") == ["cat", "raccoon"]
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "data.yaml").write_text("names:\n  0: stuffed cat\n  1: bait\nnc: 2\n")
    assert read_names(tmp_path / "b") == ["stuffed cat", "bait"]
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "obj.names").write_text("raccoon\ncat\n")
    assert read_names(tmp_path / "c") == ["raccoon", "cat"]


def test_convert_and_add_session(tmp_path):
    assert convert_label("1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1", {0: None, 1: 0}) == ["0 0.5 0.5 0.2 0.2"]
    pytest.importorskip("PIL")
    from PIL import Image
    exp = tmp_path / "export"
    (exp / "train" / "images").mkdir(parents=True)
    (exp / "train" / "labels").mkdir(parents=True)
    (exp / "data.yaml").write_text("names: ['stuffed cat', 'raccoon']\n")
    for name, lbl in (("f1", "0 0.5 0.5 0.3 0.3"), ("f2", "1 0.5 0.5 0.3 0.3"), ("f3", "")):
        Image.new("RGB", (32, 32)).save(exp / "train" / "images" / f"{name}.png")
        (exp / "train" / "labels" / f"{name}.txt").write_text(lbl)
    c = add_session(exp, tmp_path / "data", "s1", "train")
    assert c == {"cat": 1, "raccoon": 1, "empty": 1}
    out = tmp_path / "data" / "trap" / "yolo"
    assert (out / "images" / "train" / "trap__s1__f1.jpg").exists()
    assert (out / "labels" / "train" / "trap__s1__f2.txt").read_text() == "1 0.5 0.5 0.3 0.3"


# ---- zones
def test_zones(tmp_path):
    with pytest.raises(ValueError):
        parse_points(["0,0", "1,1"])
    with pytest.raises(ValueError):
        parse_points(["0,0", "2,0", "1,1"])
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"N": 5, "K": 3, "trap_zone": [], "doorway_zone": []}))
    save_zones(parse_points(["0.3,0", "1,0", "1,1"]), parse_points(["0,0", "0.2,0", "0.2,1"]), cfg)
    d = json.loads(cfg.read_text())
    assert d["trap_zone"][0] == [0.3, 0.0] and d["K"] == 3


# ---- E2 scoring matches eval.py's shape
def test_classifier_score():
    best = [("a", "cat", 0.9), ("b", "cat", 0.2), ("c", "raccoon", 0.7), ("d", "background", 0.1)]
    s = score(best, 0.5)
    assert s["cat_recall"] == 0.5 and s["noncat_fpr"] == 0.5 and s["fpr_by_species"]["raccoon"] == 1.0
