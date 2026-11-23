"""
多模态 AIGC 检测器训练脚本
参考 AIQualityDetector.py 的简洁 CNN 架构

数据:
  图像: data/image/{real, fake}/
  视频: data/video/{real, fake}/*.pt  [16, 3, 224, 224]
  音频: (自动下载)

训练后用 test/ 文件夹测试
"""

import os
import sys
import json
import io
import hashlib
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.resolve()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = PROJECT_ROOT / 'checkpoints'
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 数据加载
# ============================================================

class ImageDataset(Dataset):
    """图像数据集 - 自动从目录加载 real/fake"""
    def __init__(self, root_dir, transform=None, limit=None, extra_real_dirs=None,
                 exclude_dirs=None):
        self.samples = []
        self.transform = transform
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for label_name, label in [('real', 0), ('fake', 1)]:
            label_dir = Path(root_dir) / label_name
            if label_dir.exists():
                files = sorted(p for p in label_dir.iterdir()
                               if p.is_file() and p.suffix.lower() in valid_exts)
                for f in files:
                    self.samples.append((str(f), label))

        for extra_dir in extra_real_dirs or []:
            extra_dir = Path(extra_dir)
            if extra_dir.exists():
                self.samples.extend(
                    (str(p), 0) for p in sorted(extra_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in valid_exts
                )

        excluded_hashes = set()
        for excluded_dir in exclude_dirs or []:
            excluded_dir = Path(excluded_dir)
            if excluded_dir.is_dir():
                for p in excluded_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in valid_exts:
                        h = hashlib.sha256(p.read_bytes()).digest()
                        excluded_hashes.add(h)

        # Remove exact duplicates by file content, including test/train overlap.
        seen_hashes = {}
        filtered = []
        excluded_count = duplicate_count = 0
        for path, label in self.samples:
            h = hashlib.sha256(Path(path).read_bytes()).digest()
            if h in excluded_hashes:
                excluded_count += 1
                continue
            if h in seen_hashes:
                previous_label = seen_hashes[h]
                if previous_label != label:
                    raise ValueError(f"Exact duplicate has conflicting labels: {path}")
                duplicate_count += 1
                continue
            seen_hashes[h] = label
            filtered.append((path, label))
        self.samples = filtered

        if limit:
            self.samples = self.samples[:limit]

        print(f"ImageDataset: {len(self.samples)} unique samples from {root_dir}; "
              f"excluded test overlaps={excluded_count}, duplicate files={duplicate_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path).convert('RGB'))
            img = img[:, :, ::-1]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        if self.transform:
            tensor = self.transform(Image.fromarray(img))
        else:
            tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1]

        return tensor, torch.tensor(label, dtype=torch.long)


class RandomJPEGCompression:
    """Simulate common JPEG re-encoding during training."""
    def __init__(self, probability=0.5, quality=(45, 95)):
        self.probability = probability
        self.quality = quality

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=random.randint(*self.quality))
        stream.seek(0)
        with Image.open(stream) as compressed:
            return compressed.convert('RGB').copy()


class VideoDataset(Dataset):
    """视频数据集 - 加载 .pt 帧序列"""
    def __init__(self, root_dir, limit=None):
        self.samples = []

        for label_name, label in [('real', 0), ('fake', 1)]:
            label_dir = Path(root_dir) / label_name
            if label_dir.exists():
                files = sorted(label_dir.glob('*.pt'))
                for f in files:
                    self.samples.append((str(f), label))

        if limit:
            self.samples = self.samples[:limit]

        random.shuffle(self.samples)
        print(f"VideoDataset: {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = torch.load(path, weights_only=False)  # [T, 3, 224, 224]

        # Normalize
        frames = frames.float()
        if frames.max() > 1.0:
            frames = frames / 255.0
        frames = (frames - 0.5) / 0.5

        return frames, torch.tensor(label, dtype=torch.long)


class MelAudioDataset(Dataset):
    """音频数据集 - 转为 Log-Mel 频谱图"""
    def __init__(self, audio_files, sr=16000, duration=5.0):
        self.samples = audio_files  # [(path, label), ...]
        self.sr = sr
        self.duration = int(duration * sr)
        print(f"MelAudioDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            import librosa
            y, sr = librosa.load(path, sr=self.sr, mono=True)
        except Exception:
            return torch.zeros(1, 128, 256), torch.tensor(label, dtype=torch.long)

        # Trim or pad
        if len(y) > self.duration:
            start = random.randint(0, len(y) - self.duration)
            y = y[start:start + self.duration]
        elif len(y) < self.duration:
            y = np.pad(y, (0, self.duration - len(y)))

        # Log-Mel
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)  # [1, 128, T]
        return tensor, torch.tensor(label, dtype=torch.long)


# ============================================================
# 模型 - Transformer 架构 (替换简单 CNN)
# ============================================================

class ImageDetectorViT(nn.Module):
    """图像真伪检测器 - 预训练 DeiT-Small ViT"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        import timm

        # DeiT-Small: 22M 参数, 快速且准确率好
        self.backbone = timm.create_model(
            'deit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # 去掉分类头
        )
        feat_dim = self.backbone.num_features  # 384

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)  # [B, 384] CLS token
        return self.classifier(feat)


class TemporalAttention(nn.Module):
    """时序自注意力 - Video Transformer 核心"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ffn(x)
        return x


class VideoDetectorViT(nn.Module):
    """视频真伪检测器 - ViT + 时序自注意力 (TimeSformer 风格)"""
    def __init__(self, num_classes=2, pretrained=True, num_frames=16):
        super().__init__()
        import timm

        # 冻结 ViT, 只训练时序层 + 分类头
        self.vit = timm.create_model(
            'deit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0,
        )
        feat_dim = self.vit.num_features  # 384

        # 冻结 ViT 参数 (作为固定特征提取器)
        for p in self.vit.parameters():
            p.requires_grad = False

        # 时序编码器: Transformer 聚合帧特征
        self.temporal_embed = nn.Parameter(torch.randn(1, num_frames, feat_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames, feat_dim) * 0.02)

        self.temporal_attn = nn.Sequential(
            TemporalAttention(feat_dim, num_heads=8),
            TemporalAttention(feat_dim, num_heads=8),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x):
        """x: [B, T, 3, 224, 224]"""
        B, T, C, H, W = x.shape

        # 用 ViT 提取每帧特征 (no_grad 冻结)
        with torch.set_grad_enabled(False):
            flat = x.view(B * T, C, H, W)
            feats = self.vit(flat)  # [B*T, 384]

        feats = feats.view(B, T, -1)  # [B, T, 384]

        # 加时序位置编码
        feats = feats + self.temporal_embed[:, :T] + self.temporal_pos[:, :T]

        # 时序 Transformer 编码
        for layer in self.temporal_attn:
            feats = layer(feats)

        # 取最后一帧 + 全局均值融合
        pooled = feats[:, -1] + feats.mean(dim=1)  # [B, 384]

        return self.classifier(pooled)


class AudioPatchViT(nn.Module):
    """音频真伪检测器 - Patch ViT 处理 Log-Mel 频谱 (AST 风格)"""
    def __init__(self, num_classes=2, embed_dim=192, num_heads=6, depth=6,
                 patch_size=(16, 16), in_chans=1, mel_bins=128):
        super().__init__()

        self.patch_size = patch_size
        patch_h, patch_w = patch_size
        num_patches = (mel_bins // patch_h) * (32 // patch_w)  # 假设时间维约 32 patches

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # 位置编码 + CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=0.15, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        import math
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """x: [B, 1, 128, T] Log-Mel 频谱"""
        # Patch embedding
        patches = self.patch_embed(x)  # [B, D, H', W']
        B, D, Hp, Wp = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # [B, Hp*Wp, D]

        # CLS token + pos embed
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, patches], dim=1)  # [B, 1+N, D]

        # 截断或插值 pos_embed 匹配长度
        N = seq.shape[1]
        if N <= self.pos_embed.shape[1]:
            seq = seq + self.pos_embed[:, :N]
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=N, mode='linear', align_corners=False
            ).transpose(1, 2)
            seq = seq + pos

        # Transformer
        seq = self.encoder(seq)
        cls_out = self.norm(seq[:, 0])

        return self.classifier(cls_out)


# 别名 (保持与 train.py 兼容)
ImageDetectorCNN = ImageDetectorViT
VideoDetectorCNN = VideoDetectorViT
AudioDetectorCNN = AudioPatchViT


# ============================================================
# 训练循环
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    if scheduler:
        scheduler.step()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train_branch(name, model, train_ds, val_ds, epochs=30, batch_size=16, lr=1e-3,
                 early_stopping_patience=3, class_weights=None):
    """训练单个分支"""
    print(f"\n{'='*60}")
    print(f"训练分支: {name}")
    print(f"{'='*60}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(DEVICE)
    if class_weights is not None:
        class_weights = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=0.05 if name == 'image' else 0.0)
    optimizer = AdamW(model.parameters(), lr=lr,
                      weight_decay=0.03 if name == 'image' else 1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    best_val_loss = float('inf')
    best_epoch = 0
    stale_epochs = 0
    history = {'epoch': [], 'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scheduler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        improved = (val_loss < best_val_loss - 1e-4) if name == 'image' else (val_acc > best_val_acc)
        status = "BEST" if improved else "  "
        print(f"  [{name}] Epoch {epoch:2d}/{epochs}  "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}  "
              f"({elapsed:.1f}s) {status}")

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            ckpt_path = OUTPUT_DIR / f'{name}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"    → 保存 best checkpoint: {ckpt_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                print(f"  Early stopping at epoch {epoch}; best epoch={best_epoch}.")
                break

    # 保存 history
    with open(OUTPUT_DIR / f'{name}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[{name}] 训练完成! Best Val Acc: {best_val_acc:.4f} (epoch {best_epoch})")
    return model, best_val_acc


# ============================================================
# 音频数据集准备 (生成高质量合成音频模拟真实/AI差异)
# ============================================================

def prepare_audio_dataset(save_dir, n_per_class=60):
    """
    生成高质量合成音频模拟真实语音 vs AI-TTS 特征差异
    - 真实音频: 自然谐波结构 + 共振峰 + 呼吸噪声 + 基频抖动
    - AI音频: 过度平滑谐波 + 高频集中 + 基频稳定无抖动 + 频谱周期性
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    audio_files = []
    sr = 16000
    duration = 3.0

    print(f"\n生成合成音频数据 ({n_per_class*2} samples)...")

    # 真实音频类 (label=0): 模拟自然语音
    for i in range(n_per_class):
        n_samples = int(duration * sr)
        t = np.linspace(0, duration, n_samples)
        np.random.seed(42 + i * 7)

        # 基频带抖动 (自然语音特征)
        f0 = 150 + np.random.randn() * 30  # Hz
        f0_jitter = f0 + np.random.randn(n_samples) * 2  # 抖动

        # 谐波系列 + 缓慢变化振幅
        y = np.zeros(n_samples)
        for h in range(1, 8):
            amp = 0.5 / h * np.exp(-h * 0.3)
            phase_env = np.sin(2 * np.pi * 2 * t + np.random.randn() * 0.3)
            y += amp * np.sin(2 * np.pi * h * f0_jitter * t) * (0.7 + 0.3 * phase_env)

        # 共振峰滤波 (模拟声道)
        from scipy import signal
        for fcenter, bw, amp in [(800, 150, 1.5), (1500, 200, 1.2), (2500, 300, 0.8), (3500, 400, 0.5)]:
            b, a = signal.butter(2, [fcenter - bw, fcenter + bw], btype='bandpass', fs=sr)
            y_filt = signal.filtfilt(b, a, y)
            y = y + amp * y_filt * 0.3

        # 呼吸噪声 (1/f pink noise)
        pink = np.zeros(n_samples)
        pink[0] = np.random.randn()
        for k in range(1, n_samples):
            pink[k] = (pink[k-1] + np.random.randn() * 0.1) / 1.7
        y += 0.05 * pink

        # 句包络 (能量起伏)
        envelope = np.linspace(0.2, 1.0, n_samples) * np.sin(np.pi * t / duration)
        y = y * envelope

        y = y / (np.max(np.abs(y)) + 1e-8)
        audio_path = save_dir / f"real_{i:03d}.wav"
        import scipy.io.wavfile as wavfile
        wavfile.write(str(audio_path), sr, (y * 32767).astype(np.int16))
        audio_files.append((str(audio_path), 0))

    # AI生成音频类 (label=1): 模拟TTS特征
    for i in range(n_per_class):
        n_samples = int(duration * sr)
        t = np.linspace(0, duration, n_samples)
        np.random.seed(123 + i * 7)

        # 固定基频无抖动 (TTS特征)
        f0 = 180 + np.random.rand() * 40
        f0_stable = np.full(n_samples, f0)

        # 过度平滑的谐波
        y = np.zeros(n_samples)
        for h in range(1, 12):
            amp = 0.4 / h**1.2 * np.exp(-h * 0.2)
            y += amp * np.sin(2 * np.pi * h * f0_stable * t)

        # AI特有的高频谐波能量集中 (周期性频谱伪影)
        for h in range(8, 20):
            amp = 0.15 / h * (0.5 + 0.5 * np.sin(2 * np.pi * 12 * t))  # 周期性调制
            y += amp * np.sin(2 * np.pi * h * f0_stable * t)

        # 干净的带通滤波 (无呼吸噪声)
        from scipy import signal
        b, a = signal.butter(4, [100, 7000], btype='bandpass', fs=sr)
        y = signal.filtfilt(b, a, y)

        # 周期能量包络 (TTS 特性)
        envelope = np.sin(np.pi * t / duration) * (0.9 + 0.1 * np.sin(2 * np.pi * 20 * t))
        y = y * envelope

        y = y / (np.max(np.abs(y)) + 1e-8)
        audio_path = save_dir / f"ai_{i:03d}.wav"
        import scipy.io.wavfile as wavfile
        wavfile.write(str(audio_path), sr, (y * 32767).astype(np.int16))
        audio_files.append((str(audio_path), 1))

    random.shuffle(audio_files)
    print(f"  完成: {len(audio_files)} 个文件 (real={n_per_class}, ai={n_per_class})")
    return audio_files


# ============================================================
# 主函数
# ============================================================

def main():
    global OUTPUT_DIR
    default_init = PROJECT_ROOT / 'checkpoints' / 'image_best.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early-stopping-patience', type=int, default=3)
    parser.add_argument('--skip-audio', action='store_true', help='跳过音频训练')
    parser.add_argument('--skip-video', action='store_true', help='跳过视频训练')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help='checkpoint 和训练历史输出目录（默认 checkpoints）')
    parser.add_argument('--init-checkpoint', type=Path,
                        default=default_init if default_init.exists() else None,
                        help='从现有 image checkpoint 继续微调')
    parser.add_argument('--extra-real-dir', type=Path, action='append', default=None,
                        help='追加真实图像目录；可重复指定多个目录')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"设备: {DEVICE}")
    print(f"训练输出: {OUTPUT_DIR}")

    results = {}

    # ---- 图像训练 ----
    print("\n" + "="*60)
    print("图像数据加载...")
    img_root = PROJECT_ROOT / 'data' / 'image'
    extra_real_dirs = args.extra_real_dir or []
    excluded_dirs = [PROJECT_ROOT / 'test' / 'image']
    img_ds = ImageDataset(img_root, extra_real_dirs=extra_real_dirs,
                          exclude_dirs=excluded_dirs)
    print(f"额外真实图像目录: {extra_real_dirs if extra_real_dirs else '无'}")

    # 训练/验证分割
    n_val = max(10, int(len(img_ds) * 0.2))
    n_train = len(img_ds) - n_val
    train_base, val_base = random_split(img_ds, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(args.seed))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12,
                               saturation=0.08, hue=0.02),
        RandomJPEGCompression(probability=0.5, quality=(45, 95)),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.12),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_ds = ImageDataset(img_root, transform=train_transform,
                            extra_real_dirs=extra_real_dirs, exclude_dirs=excluded_dirs)
    val_ds = ImageDataset(img_root, transform=eval_transform,
                          extra_real_dirs=extra_real_dirs, exclude_dirs=excluded_dirs)
    train_ds.samples = img_ds.samples
    val_ds.samples = img_ds.samples
    train_img = Subset(train_ds, train_base.indices)
    val_img = Subset(val_ds, val_base.indices)
    print(f"图像: train={n_train}, val={n_val}")

    train_labels = [train_ds.samples[i][1] for i in train_base.indices]
    class_counts = np.bincount(train_labels, minlength=2)
    image_class_weights = torch.tensor(
        len(train_labels) / (2.0 * np.maximum(class_counts, 1)), dtype=torch.float32
    )
    print(f"图像类别权重: real={image_class_weights[0]:.3f}, fake={image_class_weights[1]:.3f}")

    img_model = ImageDetectorCNN(num_classes=2, pretrained=args.init_checkpoint is None)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state', checkpoint)
        img_model.load_state_dict(state)
        print(f"从现有检查点继续微调: {args.init_checkpoint}")
    _, img_acc = train_branch('image', img_model, train_img, val_img,
                               epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                               early_stopping_patience=args.early_stopping_patience,
                               class_weights=image_class_weights)
    results['image'] = img_acc

    # ---- 视频训练 ----
    if not args.skip_video:
        print("\n" + "="*60)
        print("视频数据加载...")
        vid_ds = VideoDataset(PROJECT_ROOT / 'data' / 'video')

        n_val_v = max(4, int(len(vid_ds) * 0.25))
        n_train_v = len(vid_ds) - n_val_v
        train_vid, val_vid = random_split(vid_ds, [n_train_v, n_val_v],
                                           generator=torch.Generator().manual_seed(42))
        print(f"视频: train={n_train_v}, val={n_val_v}")

        vid_model = VideoDetectorCNN(num_classes=2)
        _, vid_acc = train_branch('video', vid_model, train_vid, val_vid,
                                   epochs=args.epochs, batch_size=max(4, args.batch_size//2), lr=args.lr)
        results['video'] = vid_acc
    else:
        print("\n跳过视频训练")

    # ---- 音频训练 ----
    if not args.skip_audio:
        print("\n" + "="*60)
        print("音频数据准备...")
        audio_files = prepare_audio_dataset(PROJECT_ROOT / 'data' / 'audio_generated')

        mel_ds = MelAudioDataset(audio_files)
        n_val_a = max(6, int(len(mel_ds) * 0.25))
        n_train_a = len(mel_ds) - n_val_a
        train_mel, val_mel = random_split(mel_ds, [n_train_a, n_val_a],
                                           generator=torch.Generator().manual_seed(42))
        print(f"音频: train={n_train_a}, val={n_val_a}")

        aud_model = AudioDetectorCNN(num_classes=2)
        _, aud_acc = train_branch('audio', aud_model, train_mel, val_mel,
                                   epochs=args.epochs, batch_size=max(8, args.batch_size), lr=args.lr)
        results['audio'] = aud_acc
    else:
        print("\n跳过音频训练")

    # ---- 汇总 ----
    print("\n" + "="*60)
    print("训练完成! 各分支最佳验证准确率:")
    for name, acc in results.items():
        print(f"  {name}: {acc:.4f}")
    print(f"Checkpoint 保存在: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
