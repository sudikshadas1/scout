"""E8: a small cat / not-cat check on each animal the detector finds, looking at just that crop.

Purpose: E1b finds an animal in 92% of cat pictures at cutoff 0.25, but 47 of its 64 misses are wrong labels. This keeps
         the detector for WHERE and adds a second opinion on WHAT, using more pixels than the detector's input. E2 failed
         on whole pictures, where the animal can be tiny; a crop is an easier job. No trap code changes unless it wins.
Status:  Smoke-tested on CPU with a tiny fake dataset; crop logic tested (tests/test_improvement.py). Not yet run on PACE.
Later stages (still to add or change):
  - If it beats E1b: export it, time it on the Pi 3 (E6) and add it to edge/vision.py. Both teams must agree first.

Trains on the boxes in the YOLO labels. Scores on the detections saved by training/score_pipeline.py, so the numbers
sit next to E1b's: a picture counts as a cat if any detected animal's crop scores over the threshold.

    python training/score_pipeline.py --model results/E1b/weights/best.pt --data ~/scratch/scout_data \
        --split val --n 500 --imgsz 416 --name E1b_pipeline416
    python training/train_crop_classifier.py --data ~/scratch/scout_data --extra-data ~/scratch/scout_data/wct \
        --detections results/E1b_pipeline416/detections.jsonl --name E8

Writes results/<name>/metrics.json, threshold_sweep.csv, per_image.csv and model.pt (git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "data"))
sys.path.insert(0, str(HERE.parent))
from analyze_errors import read_labels  # noqa: E402
from eval import THRESHOLDS  # noqa: E402
from train_classifier import score  # noqa: E402


def pad_box(box, pad: float = 0.1):
    """Normalized (x1, y1, x2, y2), grown by `pad` of its size on each side and kept inside the picture."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return max(0.0, x1 - dx), max(0.0, y1 - dy), min(1.0, x2 + dx), min(1.0, y2 + dy)


def crop_pixels(box, width: int, height: int) -> tuple[int, int, int, int]:
    """PIL crop box in pixels, at least 1 pixel each way."""
    x1, y1, x2, y2 = pad_box(box)
    left, top = int(x1 * width), int(y1 * height)
    return left, top, max(int(x2 * width), left + 1), max(int(y2 * height), top + 1)


def training_crops(root: Path, split: str, rng: random.Random) -> list[tuple[Path, tuple, int]]:
    """(image, box, is_cat) for every labelled box, plus one random patch from each empty picture as not-cat."""
    out = []
    for p in sorted((root / "images" / split).glob("*.jpg")):
        boxes = read_labels(root / "labels" / split / f"{p.stem}.txt")
        for cls, box in boxes:
            out.append((p, box, int(cls == "cat")))
        if not boxes:
            w, h = rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5)
            x, y = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
            out.append((p, (x, y, x + w, y + h), 0))
    return out


def image_scores(records: list[dict], probs: dict, det_cutoff: float) -> list[tuple[str, str, float]]:
    """Per picture: the highest cat score over its detections at or above det_cutoff (0 if none)."""
    out = []
    for r in records:
        s = [probs[(r["image"], i)] for i, d in enumerate(r["dets"]) if d[1] >= det_cutoff and (r["image"], i) in probs]
        out.append((r["image"], r["truth"], max(s, default=0.0)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing yolo/")
    ap.add_argument("--extra-data", nargs="*", default=[], help="more folders containing yolo/, e.g. <data>/wct")
    ap.add_argument("--detections", required=True, help="detections.jsonl from score_pipeline.py")
    ap.add_argument("--val-images", help="folder of those pictures (default: <data>/yolo/images/val)")
    ap.add_argument("--name", default="E8")
    ap.add_argument("--det-cutoff", type=float, default=0.1, help="the Pi detector's min_conf")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4605)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--no-pretrained", action="store_true", help="random start (offline smoke tests only)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import models, transforms

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    roots = [Path(d).expanduser() / "yolo" for d in [args.data, *args.extra_data]]
    train = [c for r in roots for c in training_crops(r, "train", rng)]
    n_pos = sum(y for *_, y in train)
    if not n_pos or n_pos == len(train):
        sys.exit("training set needs both cat and non-cat boxes")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = transforms.Compose([transforms.RandomResizedCrop(args.imgsz, scale=(0.7, 1.0), antialias=True),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5),
                                   transforms.RandomGrayscale(0.3),  # night / IR frames are grayscale
                                   transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz), antialias=True), transforms.ToTensor(), norm])

    class Crops(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, box, y = self.items[i]
            with Image.open(p) as im:
                im = im.convert("RGB")
                return self.tf(im.crop(crop_pixels(box, *im.size))), y

    w = [0.5 / n_pos if y else 0.5 / (len(train) - n_pos) for *_, y in train]
    loader = DataLoader(Crops(train, tf_train), batch_size=args.batch, num_workers=args.workers,
                        sampler=WeightedRandomSampler(w, num_samples=len(train), replacement=True))

    weights = None if args.no_pretrained else models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    net = models.mobilenet_v3_small(weights=weights)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss()

    print(f"{len(train):,} training crops ({n_pos:,} cat) on {dev}")
    for ep in range(args.epochs):
        net.train()
        tot, t0 = 0.0, time.time()
        for x, y in loader:
            x, y = x.to(dev), y.float().to(dev)
            opt.zero_grad()
            loss = lossf(net(x).squeeze(1), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        sched.step()
        print(f"epoch {ep + 1}/{args.epochs} loss {tot / len(train):.4f} ({time.time() - t0:.0f}s)")

    # score every detection E1b made on the eval pictures
    records = [json.loads(line) for line in open(args.detections)]
    val_dir = Path(args.val_images).expanduser() if args.val_images else roots[0] / "images" / "val"
    net.eval()
    probs = {}
    with torch.no_grad():
        for r in records:
            todo = [(i, d) for i, d in enumerate(r["dets"]) if d[1] >= args.det_cutoff]
            if not todo:
                continue
            with Image.open(val_dir / r["image"]) as im:
                im = im.convert("RGB")
                x = torch.stack([tf_eval(im.crop(crop_pixels(d[2], *im.size))) for _, d in todo]).to(dev)
            for (i, _), p in zip(todo, torch.sigmoid(net(x).squeeze(1)).tolist()):
                probs[(r["image"], i)] = p
        cpu = net.to("cpu")
        x1 = torch.zeros(1, 3, args.imgsz, args.imgsz)
        for _ in range(3):
            cpu(x1)
        t0 = time.perf_counter()
        for _ in range(20):
            cpu(x1)
        cpu_ms = (time.perf_counter() - t0) / 20 * 1e3

    best = image_scores(records, probs, args.det_cutoff)
    out = Path("results") / args.name
    out.mkdir(parents=True, exist_ok=True)
    m = score(best, args.thr)
    m.update(model="mobilenet_v3_small crop check", detections=args.detections, crop_imgsz=args.imgsz,
             n_images=len(best), n_train_crops=len(train), cpu_ms_per_crop=round(cpu_ms, 1), epochs=args.epochs)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    with open(out / "threshold_sweep.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["threshold", "cat_recall", "noncat_fpr"])
        for t in THRESHOLDS:
            s = score(best, t)
            wr.writerow([t, s["cat_recall"], s["noncat_fpr"]])
    with open(out / "per_image.csv", "w", newline="") as f:
        csv.writer(f).writerows([("image", "truth", "best_crop_cat_prob"), *best])
    torch.save(net.state_dict(), out / "model.pt")
    print(f"{args.name}: {m['cat_recall']:.1%} of cats, {m['noncat_fpr']:.1%} false positives @ {args.thr} "
          f"({cpu_ms:.0f} ms per crop on this CPU)")
    print(f"wrote {out}/metrics.json, threshold_sweep.csv")


if __name__ == "__main__":
    main()
