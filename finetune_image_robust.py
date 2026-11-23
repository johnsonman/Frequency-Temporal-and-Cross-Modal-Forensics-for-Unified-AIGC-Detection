"""只使用修正后的 data/image 微调 DeiT，并通过训练增强提升鲁棒性。"""
import io
import json
import hashlib
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T

from train import ImageDetectorViT

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "image"
START_CKPT = ROOT / "checkpoints" / "image_best.pt"
OUT_CKPT = ROOT / "checkpoints" / "image_robust.pt"
HISTORY = ROOT / "checkpoints" / "image_robust_history.json"
SEED = 20261107
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RandomJPEG:
    def __init__(self, p=0.55, quality=(40, 95)):
        self.p, self.quality = p, quality

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=random.randint(*self.quality))
        stream.seek(0)
        with Image.open(stream) as compressed:
            return compressed.convert("RGB").copy()


class ImageFiles(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.samples = []
        for name, label in (("real", 0), ("fake", 1)):
            folder = root / name
            self.samples.extend(
                (p, label) for p in sorted(folder.iterdir())
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
        test_dir = ROOT / "test" / "image"
        test_hashes = {
            hashlib.sha256(p.read_bytes()).digest()
            for p in test_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        } if test_dir.is_dir() else set()
        seen = set()
        clean = []
        for path, label in self.samples:
            digest = hashlib.sha256(path.read_bytes()).digest()
            if digest not in test_hashes and digest not in seen:
                clean.append((path, label))
                seen.add(digest)
        self.samples = clean

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read training image: {path}")
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return self.transform(image), label


def make_stratified_split(samples, val_ratio=0.15, seed=SEED):
    rng = random.Random(seed)
    train_ids, val_ids = [], []
    for label in (0, 1):
        ids = [i for i, (_, y) in enumerate(samples) if y == label]
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_ratio))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    total = correct = 0
    class_total = [0, 0]
    class_correct = [0, 0]
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            for label in (0, 1):
                mask = y == label
                class_total[label] += mask.sum().item()
                class_correct[label] += ((pred == y) & mask).sum().item()
    recalls = [class_correct[i] / max(class_total[i], 1) for i in (0, 1)]
    return loss_sum / max(total, 1), correct / max(total, 1), sum(recalls) / 2


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        RandomJPEG(),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.12),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    eval_tf = T.Compose([
        T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    raw = ImageFiles(DATA, transform=None)
    train_ids, val_ids = make_stratified_split(raw.samples)
    train_ds = ImageFiles(DATA, train_tf)
    val_ds = ImageFiles(DATA, eval_tf)
    train_loader = DataLoader(Subset(train_ds, train_ids), batch_size=48, shuffle=True,
                              num_workers=0, pin_memory=DEVICE == "cuda")
    val_loader = DataLoader(Subset(val_ds, val_ids), batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=DEVICE == "cuda")
    print(f"device={DEVICE}; train={len(train_ids)}; val={len(val_ids)}; counts={sum(y == 0 for _, y in raw.samples)} real / {sum(y == 1 for _, y in raw.samples)} fake", flush=True)

    model = ImageDetectorViT(num_classes=2, pretrained=False).to(DEVICE)
    checkpoint = torch.load(START_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 8e-6},
        {"params": model.classifier.parameters(), "lr": 4e-5},
    ], weight_decay=0.03)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=5e-7,
    )

    best_loss, best_bacc, best_epoch, stale = float("inf"), 0.0, 0, 0
    history = []
    for epoch in range(1, 13):
        start = time.time()
        model.train()
        train_loss_sum = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * y.size(0)
            seen += y.size(0)

        val_loss, val_acc, val_bacc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": train_loss_sum / max(seen, 1),
               "val_loss": val_loss, "val_acc": val_acc, "val_bacc": val_bacc}
        history.append(row)
        print(f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_bacc={val_bacc:.4f} time={time.time()-start:.1f}s", flush=True)

        if val_loss < best_loss:
            best_loss, best_bacc, best_epoch, stale = val_loss, val_bacc, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "val_acc": val_acc, "val_bacc": val_bacc,
                        "seed": SEED, "source": "data/image only"}, OUT_CKPT)
        else:
            stale += 1
            if stale >= 4:
                print("Early stopping.", flush=True)
                break

    HISTORY.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best_epoch={best_epoch} best_val_loss={best_loss:.4f} best_val_bacc={best_bacc:.4f}", flush=True)


if __name__ == "__main__":
    main()
