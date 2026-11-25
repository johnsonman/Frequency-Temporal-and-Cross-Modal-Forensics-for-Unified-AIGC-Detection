import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
"""
train_v3.py — Real vs AI with STRONG data augmentation
=======================================================
关键改进：训练时强力数据增强抹去数据源特征（Picsum vs others），
让模型专注于"真实 vs AI"的本质特征（纹理、频率域、统计分布）。
"""
import os, sys, glob, json, time, random, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter

class CFG:
    real_dirs = ["data/image_real"]  # 可加更多真实照片目录
    fake_dir = "data/image_v3/fake"
    output_dir = "checkpoints/finetuned"
    device = "cuda"
    seed = 42
    max_per_class = None
    batch_size = 64
    epochs = 15
    lr_heads = 1e-3
    weight_decay = 0.05
    grad_clip = 1.0

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)

# ========== 训练时数据增强 — 强力抹去数据源特征 ==========
# 目标：让模型无法通过"JPEG压缩方式"、"色彩分布"、"边缘锐度"等文件特征区分图片来源
# 只能依靠"真实 vs AI"的本质区别（纹理模式、高频伪影、频率分布等）

def strong_augment(img):
    """对 PIL Image 做强力数据增强"""
    # 1. 随机裁剪 + resize (抹去原图构图/分辨率特征)
    w, h = img.size
    scale = random.uniform(0.6, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w >= 224 and new_h >= 224:
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((256, 256), Image.BILINEAR)
    
    # 2. 随机水平翻转
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 3. 随机颜色抖动 (抹去色彩分布特征)
    img = ImageEnhance.Color(img).enhance(0.3 + random.random() * 1.4)  # 0.3-1.7
    img = ImageEnhance.Brightness(img).enhance(0.6 + random.random() * 0.8)  # 0.6-1.4
    img = ImageEnhance.Contrast(img).enhance(0.5 + random.random() * 1.0)  # 0.5-1.5
    img = ImageEnhance.Sharpness(img).enhance(0.0 + random.random() * 2.0)  # 0-2
    
    # 4. 随机高斯模糊 (抹去边缘锐度特征)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    
    # 5. 中心裁剪到 224 (类似 CLIP 的 CenterCrop)
    img = transforms.CenterCrop(224)(img)
    
    # 6. 随机 JPEG 压缩模拟 (抹去原文件压缩特征)
    if random.random() < 0.5:
        buf = io.BytesIO()
        quality = random.randint(40, 95)
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    
    return img

class RealAIDataset(Dataset):
    def __init__(self, real_dirs, fake_dir, processor, augment=True, max_per_class=None):
        self.proc = processor
        self.augment = augment
        self.samples = []
        
        # 多个真实照片目录
        for rd in real_dirs:
            real_files = sorted([f for f in glob.glob(os.path.join(rd, "*")) 
                               if os.path.isfile(f)])
            if max_per_class:
                real_files = real_files[:max_per_class]
            for f in real_files:
                self.samples.append((f, 0))
        
        fake_files = sorted([f for f in glob.glob(os.path.join(fake_dir, "*")) 
                           if os.path.isfile(f)])
        if max_per_class:
            fake_files = fake_files[:max_per_class]
        for f in fake_files:
            self.samples.append((f, 1))
        
        random.seed(CFG.seed)
        random.shuffle(self.samples)
        
        n_real = sum(1 for _,l in self.samples if l==0)
        n_fake = sum(1 for _,l in self.samples if l==1)
        print(f"  Dataset: {len(self.samples)}  real={n_real} fake={n_fake} augment={augment}")
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.augment:
                img = strong_augment(img)
            pv = self.proc(img, return_tensors="pt").pixel_values.squeeze(0)
        except:
            pv = torch.zeros(3, 224, 224)
        return pv, torch.tensor(label, dtype=torch.long)

def build():
    from models.image_branch import ImageBranch
    ib = ImageBranch(embed_dim=768, num_classes=2, device='cuda')
    ib.cuda()
    for p in ib.clip_model.parameters():
        p.requires_grad = False
    tr = sum(p.numel() for p in ib.parameters() if p.requires_grad)
    to = sum(p.numel() for p in ib.parameters())
    print(f"Heads-only: trainable={tr:,} ({tr/to:.1%}) total={to:,}")
    return ib

def train():
    from transformers import CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    print("=" * 60); print(" Real vs AI v3 (with strong aug)"); print("=" * 60)
    
    # 训练集有增强，验证集无增强
    full_aug = RealAIDataset(CFG.real_dirs, CFG.fake_dir, proc, augment=True, max_per_class=CFG.max_per_class)
    full_clean = RealAIDataset(CFG.real_dirs, CFG.fake_dir, proc, augment=False, max_per_class=CFG.max_per_class)
    
    n_tr = int(len(full_aug) * 0.9); n_va = len(full_aug) - n_tr
    # 用相同的 split，但训练用 aug 版，验证用 clean 版
    indices = list(range(len(full_aug)))
    random.seed(42); random.shuffle(indices)
    tr_idx, va_idx = indices[:n_tr], indices[n_tr:]
    
    tr = torch.utils.data.Subset(full_aug, tr_idx)
    va = torch.utils.data.Subset(full_clean, va_idx)
    print(f"  Train: {len(tr)} (augmented), Val: {len(va)} (clean)")
    
    tr_dl = DataLoader(tr, batch_size=CFG.batch_size, shuffle=True, num_workers=0, drop_last=True)
    va_dl = DataLoader(va, batch_size=CFG.batch_size, shuffle=False, num_workers=0)
    
    ib = build(); ib.train()
    opt = AdamW([p for p in ib.parameters() if p.requires_grad], lr=CFG.lr_heads, weight_decay=CFG.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    
    best = 0.0
    hist = []
    
    for ep in range(CFG.epochs):
        t0 = time.time(); ib.train()
        tl=0; tc=0; tt=0
        for pv, labels in tr_dl:
            pv = pv.cuda(); labels = labels.cuda()
            out = ib(pv)
            loss = bce(out['p_img_logits'], F.one_hot(labels, 2).float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in ib.parameters() if p.requires_grad], CFG.grad_clip)
            opt.step()
            tl += loss.item()*pv.shape[0]
            tc += (out['p_img_logits'].argmax(-1)==labels).sum().item()
            tt += pv.shape[0]
        
        ib.eval(); vl=0; vc=0; vt=0
        with torch.no_grad():
            for pv, labels in va_dl:
                pv = pv.cuda(); labels = labels.cuda()
                out = ib(pv)
                loss = bce(out['p_img_logits'], F.one_hot(labels, 2).float())
                vl += loss.item()*pv.shape[0]
                vc += (out['p_img_logits'].argmax(-1)==labels).sum().item()
                vt += pv.shape[0]
        
        et = time.time()-t0
        tra=tc/max(tt,1);  vaa=vc/max(vt,1)
        trl=tl/max(tt,1);  val=vl/max(vt,1)
        print(f"  Ep {ep+1}/{CFG.epochs} ({et:.1f}s) | tr={trl:.4f}/{tra:.1%} | va={val:.4f}/{vaa:.1%}", flush=True)
        hist.append({"epoch":ep+1,"tr_loss":trl,"tr_acc":tra,"va_loss":val,"va_acc":vaa})
        
        if vaa > best:
            best = vaa
            print(f"  ★ Best → saving", flush=True)
            torch.save({
                "clip_vision": ib.clip_model.state_dict(),
                "adapter": ib.adapter.state_dict(),
                "msff": ib.msff.state_dict(),
                "forensics": ib.forensics.state_dict(),
                "classifier": ib.classifier.state_dict(),
                "global_local_fusion": ib.global_local_fusion.state_dict(),
                "val_acc": vaa, "epoch": ep+1,
                "train_config": {"augment": True, "real_dirs": CFG.real_dirs, "fake_dir": CFG.fake_dir},
            }, os.path.join(CFG.output_dir, "finetuned.pt"))
    
    print(f"\nDone! Best val_acc = {best:.1%}", flush=True)
    with open(os.path.join(CFG.output_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

if __name__ == "__main__":
    train()
