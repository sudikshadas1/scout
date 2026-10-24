"""E2: MobileNetV3 cat / not-cat classifier, trained only to compare against the YOLO detector.

Purpose: The proposal promises this comparison (plan sections 5 and 9). A classifier says "cat somewhere in the
         picture" but not WHERE, so it can't tell a cat in the doorway from one past it; it is not used on the trap.
Status:  Written; smoke-tested on CPU with a tiny fake dataset. Not yet run on PACE.

Uses the same YOLO folders as the detector: an image is "cat" if its label file has a cat box. Scores use the
same metrics as training/eval.py (cat recall, non-cat false-positive rate by species, threshold sweep), on the
same validation sample (--n 500, same seed), so the numbers sit next to E1 in one table.

    python training/train_classifier.py --data ~/scratch/scout_data --name E2 --epochs 12
    # PACE: sbatch -J scout-E2 training/slurm/train_classifier.sbatch

Writes results/<name>/metrics.json, threshold_sweep.csv, per_image.csv, and model.pt (git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "data"))
sys.path.insert(0, str(HERE.parent))
from eval import THRESHOLDS, sample, true_label  # noqa: E402


def has_cat(label_file: Path) -> int:
    return int(true_label(label_file) == "cat")


def score(best: list[tuple[str, str, float]], thr: float) -> dict:
    """Same output shape as eval.py's score()."""
    n_cat = sum(t == "cat" for _, t, _ in best)
    tp = sum(t == "cat" and c >= thr for _, t, c in best)
    per: dict[str, list[int]] = {}
    for _, t, c in best:
        if t != "cat":
            v = per.setdefault(t, [0, 0])
            v[1] += 1
            v[0] += c >= thr
    n_non = sum(v[1] for v in per.values())
    fp = sum(v[0] for v in per.values())
    return {"threshold": thr, "cat_recall": tp / n_cat if n_cat else None,
            "noncat_fpr": fp / n_non if n_non else None,
            "fpr_by_species": {k: v[0] / v[1] for k, v in sorted(per.items())}, "n_cat": n_cat, "n_noncat": n_non}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="E2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=500, help="val sample, same as eval.py")
    ap.add_argument("--seed", type=int, default=4605)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke tests only)")
    ap.add_argument("--max-train", type=int, default=0, help="cap training images (smoke tests)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import models, transforms

    torch.manual_seed(args.seed)
    root = Path(args.data).expanduser() / "yolo"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = transforms.Compose([transforms.RandomResizedCrop(args.imgsz, scale=(0.6, 1.0)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5),
                                   transforms.RandomGrayscale(0.3),  # night / IR frames are grayscale
                                   transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz)), transforms.ToTensor(), norm])

    class Folder(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, y = self.items[i]
            with Image.open(p) as im:
                return self.tf(im.convert("RGB")), y

    train_imgs = sorted((root / "images" / "train").glob("*.jpg"))
    if args.max_train:
        train_imgs = train_imgs[: args.max_train]
    train = [(p, has_cat(root / "labels" / "train" / f"{p.stem}.txt")) for p in train_imgs]
    n_pos = sum(y for _, y in train)
    if not n_pos or n_pos == len(train):
        sys.exit("training set needs both cat and non-cat images")
    # balance cat / not-cat per batch (cats are the minority)
    w = [0.5 / n_pos if y else 0.5 / (len(train) - n_pos) for _, y in train]
    loader = DataLoader(Folder(train, tf_train), batch_size=args.batch, num_workers=args.workers,
                        sampler=WeightedRandomSampler(w, num_samples=len(train), replacement=True))

    weights = None if args.no_pretrained else models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    net = models.mobilenet_v3_small(weights=weights)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss()

    print(f"{len(train):,} train images ({n_pos:,} cat) on {dev}")
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

    # evaluate on the same validation sample eval.py uses
    items = sample(sorted((root / "images" / "val").glob("*.jpg")), root / "labels" / "val", args.n, args.seed)
    net.eval()
    best = []
    with torch.no_grad():
        for i in range(0, len(items), args.batch):
            chunk = items[i : i + args.batch]
            x = torch.stack([tf_eval(Image.open(p).convert("RGB")) for p, _ in chunk]).to(dev)
            probs = torch.sigmoid(net(x).squeeze(1)).tolist()
            best += [(p.name, t, float(c)) for (p, t), c in zip(chunk, probs)]
        # CPU latency per image, for the comparison table
        cpu = net.to("cpu")
        x1 = torch.zeros(1, 3, args.imgsz, args.imgsz)
        for _ in range(3):
            cpu(x1)
        t0 = time.perf_counter()
        for _ in range(20):
            cpu(x1)
        cpu_ms = (time.perf_counter() - t0) / 20 * 1e3

    out = Path("results") / args.name
    out.mkdir(parents=True, exist_ok=True)
    m = score(best, args.thr)
    m.update(model="mobilenet_v3_small", split="val", imgsz=args.imgsz, n_images=len(best),
             cpu_ms_per_image=round(cpu_ms, 1), epochs=args.epochs)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    with open(out / "threshold_sweep.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["threshold", "cat_recall", "noncat_fpr"])
        for t in THRESHOLDS:
            s = score(best, t)
            wr.writerow([t, s["cat_recall"], s["noncat_fpr"]])
    with open(out / "per_image.csv", "w", newline="") as f:
        csv.writer(f).writerows([("image", "truth", "cat_prob"), *best])
    torch.save(net.state_dict(), out / "model.pt")
    print(f"{args.name}: cat recall {m['cat_recall']:.1%}  non-cat FPR {m['noncat_fpr']:.1%} @ {args.thr}"
          f"  ({cpu_ms:.0f} ms/image on this CPU)")
    print(f"wrote {out}/metrics.json, threshold_sweep.csv")


if __name__ == "__main__":
    main()
