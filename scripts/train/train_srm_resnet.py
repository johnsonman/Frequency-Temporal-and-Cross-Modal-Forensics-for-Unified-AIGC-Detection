"""
SRM 高频噪声滤波 + ResNet18 双流鉴伪网络 训练脚本
参考 Enhance AI content detectornet.py 架构
"""
import os, math, json, time, sys
import io, random
from pathlib import Path
import numpy as np, cv2
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT/'data'/'image_v3'
CKPT_DIR = PROJECT/'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class RandomJPEGCompression:
    """训练时模拟常见的有损保存，提升对图像传输/平台压缩的适应性。"""
    def __init__(self, p=0.5, quality=(45, 95)):
        self.p, self.quality = p, quality

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=random.randint(*self.quality))
        buf.seek(0)
        with Image.open(buf) as compressed:
            return compressed.convert('RGB').copy()

# ==================== 1. SRM 滤波层 (与原始脚本完全一致) ====================
class SRMFilterLayer(nn.Module):
    """SRM (Spatial Rich Model) 5x5 经典高频噪声提取层 - 不可训练"""
    def __init__(self):
        super().__init__()
        srm_kernel = np.array([
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        ], dtype=np.float32)
        srm_kernel[0] /= 4.0; srm_kernel[1] /= 12.0; srm_kernel[2] /= 2.0
        kt = torch.zeros((3, 3, 5, 5), dtype=torch.float32)
        for i in range(3):
            for c in range(3): kt[i, c] = torch.from_numpy(srm_kernel[i])
        self.srm_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.srm_conv.weight = nn.Parameter(kt, requires_grad=False)
    def forward(self, x):
        return torch.clamp(self.srm_conv(x), -3.0, 3.0)

# ==================== 2. SRM+ResNet18 双流鉴伪网络 ====================
class SRMResNetDetector(nn.Module):
    """6 通道输入: RGB(3ch) + SRM 高频噪声(3ch)"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.srm_layer = SRMFilterLayer()
        # 加载 ImageNet 预训练 ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 修改第一层 conv1 从 3ch → 6ch
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 初始化前 3 通道用原始权重, 后 3 通道小权重
        new_conv.weight.data[:, :3] = old_conv.weight.data
        new_conv.weight.data[:, 3:] = old_conv.weight.data * 0.1
        backbone.conv1 = new_conv
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.single_classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(512, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        noise = self.srm_layer(x)
        combined = torch.cat([x, noise], dim=1)  # [B, 6, H, W]
        feat = self.feature_extractor(combined)
        pooled = torch.flatten(self.avgpool(feat), 1)
        return self.single_classifier(pooled)
    def extract_with_feat(self, x):
        noise = self.srm_layer(x)
        combined = torch.cat([x, noise], dim=1)
        feat = self.feature_extractor(combined)
        pooled = torch.flatten(self.avgpool(feat), 1)
        return feat, self.single_classifier(pooled)

# ==================== 3. 图像数据集 ====================
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.transform = transform
        self.samples = []
        for label, cls in [('real', 0), ('fake', 1)]:
            d = root_dir / label
            if not d.exists(): continue
            files = sorted(d.iterdir())
            if limit: files = files[:limit]
            for f in files:
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                    self.samples.append((str(f), cls))
        print(f"  ImageDataset: {len(self.samples)} samples from {root_dir}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None: img = np.zeros((256,256,3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img, label

# ==================== 4. 训练 ====================
def train():
    train_tf = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(256, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        RandomJPEGCompression(p=0.55, quality=(45, 95)),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = T.Compose([
        T.ToPILImage(), T.Resize((256,256)),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    full_ds = ImageDataset(DATA_DIR, transform=None)
    n_val = int(len(full_ds)*0.15)
    n_train = len(full_ds) - n_val
    train_raw, val_raw = random_split(full_ds, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    class TFSub(Dataset):
        def __init__(self, sub, tf): self.sub, self.tf = sub, tf
        def __len__(self): return len(self.sub)
        def __getitem__(self, i):
            img, lbl = self.sub[i]; return self.tf(img), lbl

    train_ds = TFSub(train_raw, train_tf)
    val_ds = TFSub(val_raw, val_tf)

    bs = 64
    train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, bs, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, BS: {bs}")

    model = SRMResNetDetector(num_classes=2).to(device)
    tp = sum(p.numel() for p in model.parameters())
    print(f"Params: {tp/1e6:.1f}M (ResNet18 pretrained + SRM 6ch)")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs, patience = 40, 8
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {'epoch':[], 'train_loss':[], 'train_acc':[],
               'val_loss':[], 'val_acc':[], 'lr':[]}
    best_val = float('inf')
    best_epoch = 0

    print(f"\n{'Epoch':>6} | {'TLoss':>7} {'TAcc':>7} | {'VLoss':>7} {'VAcc':>7} | {'LR':>10} | {'Time':>5}")
    print('-'*75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train(); tl=0; tc=0; tn=0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tl += loss.item()*xb.size(0)
            tc += (logits.argmax(1)==yb).sum().item(); tn += xb.size(0)
        train_loss, train_acc = tl/tn, tc/tn

        model.eval(); vl=0; vc=0; vn=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vl += loss.item()*xb.size(0)
                vc += (logits.argmax(1)==yb).sum().item(); vn += xb.size(0)
        val_loss, val_acc = vl/vn, vc/vn
        cur_lr = opt.param_groups[0]['lr']
        sched.step(val_loss)

        dt = time.time()-t0
        print(f"{epoch:>6d} | {train_loss:>7.4f} {train_acc:>7.4f} | {val_loss:>7.4f} {val_acc:>7.4f} | {cur_lr:>10.2e} | {dt:>5.0f}s  {'★ BEST' if val_loss<best_val else ''}")

        history['epoch'].append(epoch); history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc); history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc); history['lr'].append(cur_lr)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_acc': val_acc, 'val_loss': val_loss, 'train_acc': train_acc},
                       CKPT_DIR/'srm_resnet_best.pt')
        elif epoch - best_epoch >= patience:
            print(f"Early stopping: validation loss did not improve for {patience} epochs.")
            break

    with open(CKPT_DIR/'srm_resnet_history.json','w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ 训练完成! best val_loss={best_val:.4f} (epoch {best_epoch})")

if __name__ == '__main__':
    train()
