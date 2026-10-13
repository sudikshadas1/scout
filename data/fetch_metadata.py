"""Downloads only the label/metadata JSONs for Caltech Camera Traps and ENA24 from LILA (about 50 MB, no images).

Purpose: First step of the data pipeline: lets us count species and make splits before downloading any images.
Status:  Written. Not yet run against LILA (needs PACE ICE or a laptop with internet).
Later stages (still to add or change):
  - If a LILA URL has moved, update FILES below.
  - Optional: add the MegaDetector results file here if we use MD pseudo-boxes (build_manifest.py --md).

Run on PACE ICE (or a laptop). No images are downloaded here.

    python data/fetch_metadata.py --out ~/scratch/scout_data
"""
import argparse
import io
import zipfile
from pathlib import Path

import requests

BASE = "https://storage.googleapis.com/public-datasets-lila"
FILES = {
    # Caltech Camera Traps: https://lila.science/datasets/caltech-camera-traps
    "cct_images.json": f"{BASE}/caltechcameratraps/labels/caltech_camera_traps.json.zip",
    "cct_bboxes.json": f"{BASE}/caltechcameratraps/labels/caltech_bboxes_20200316.json",
    "cct_splits.json": f"{BASE}/caltechcameratraps/CaltechCameraTrapsSplits_v0.json",
    # ENA24-detection (non-human images only): https://lila.science/datasets/ena24detection
    "ena24.json": f"{BASE}/ena24/ena24_public.json",
}


def fetch(url: str, dest: Path) -> None:
    print(f"GET {url}")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            inner = [n for n in z.namelist() if n.endswith(".json")][0]
            dest.write_bytes(z.read(inner))
    else:
        dest.write_bytes(r.content)
    print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="data folder, e.g. ~/scratch/scout_data")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).expanduser() / "meta"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = out / name
        if dest.exists() and not args.force:
            print(f"skip {dest} (exists)")
            continue
        fetch(url, dest)


if __name__ == "__main__":
    main()
