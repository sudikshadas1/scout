import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import video_to_frames as vf  # noqa: E402

cv2 = pytest.importorskip("cv2")


def make_video(path: Path, seconds: float = 2.0, fps: int = 10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(int(seconds * fps)):
        w.write(np.full((48, 64, 3), i * 10 % 255, np.uint8))
    w.release()


def test_sample_times():
    assert vf.sample_times(1.0, 0.2) == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert vf.sample_times(0, 0.2) == []


def test_animal_from_csv_then_name(tmp_path):
    labels = {"IMG_01.mp4": "Raccoon"}
    assert vf.animal_for(Path("IMG_01.mp4"), labels) == "raccoon"
    assert vf.animal_for(Path("cat_night.mp4"), {}) == "cat"
    assert vf.animal_for(Path("other_animal_3.mp4"), {}) == "other_animal"
    assert vf.animal_for(Path("IMG_02.mp4"), {}) == "unknown"


def test_clips_and_pool(tmp_path):
    vids = tmp_path / "v"
    vids.mkdir()
    make_video(vids / "IMG 7.avi", seconds=2.0, fps=10)
    (tmp_path / "labels.csv").write_text("video,animal\nIMG 7.avi,opossum\n")
    script = Path(vf.__file__)
    subprocess.run([sys.executable, str(script), "--videos", str(vids), "--labels", str(tmp_path / "labels.csv"),
                    "--clips-out", str(tmp_path / "clips"), "--pool-out", str(tmp_path / "pool"),
                    "--pool-every", "1"], check=True, capture_output=True)
    clip = tmp_path / "clips" / "opossum_img-7"
    assert len(list(clip.glob("*.jpg"))) == 10          # 2 s at 5 fps
    assert len(list((tmp_path / "pool").glob("*.jpg"))) == 2
    # vote_sim reads the animal from the folder prefix
    assert clip.name.split("_")[0] == "opossum"


def test_yolo_lines():
    class D:
        def __init__(self, cls, box):
            self.cls, self.conf, self.box = cls, 0.9, box
    lines = vf.yolo_lines([D("raccoon", (0.1, 0.2, 0.5, 0.6)), D("bird", (0, 0, 1, 1))])
    assert lines == ["1 0.300000 0.400000 0.400000 0.400000"]
