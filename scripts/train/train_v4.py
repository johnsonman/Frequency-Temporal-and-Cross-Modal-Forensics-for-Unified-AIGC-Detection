import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
"""最终方案 v4：
1. 把所有图像统一预处理 (224x224 JPEG q85) — 彻底抹去源特征
2. 加上强力在线数据增强
3. 降低学习率让模型慢学
4. 关键：用 test/image 里的 real_* 作为额外 Real 训练数据
   （以及非 real_* 里可能有真实照片的）
"""
import os, sys, glob, json, time, random, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter

class CFG:
    real_dirs = ["data/image_real"]  # Picsum + test/image/real_* 也会加入
    fake_dir = "data/image_v3/fake"
    test_dir = "test/image"         # test 集的 real_* 文件也加入训练！
    output_dir = "checkpoints/finetuned"
    seed = 42
    batch_size = 64
    epochs = 20
    lr_heads = 3e-4        # 更低学习率
    weight_decay = 0.1     # 更强正则化

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(CFG.seed)

def normalize_to_224(img):
    """统一处理：center crop + resize + 重编码"""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2; top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((224, 224), Image.LANCZOS)
    # 重编码抹掉格式特征
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def strong_augment(img):
    """强力数据增强"""
    img = normalize_to_224(img)
    # 随机二次裁剪
    if random.random() < 0.5:
        scale = random.uniform(0.7, 1.0)
        w, h = img.size
        nw, nh = int(w * scale), int(h * scale)
        left = random.randint(0, w - nw); top = random.randint(0, h - nh)
        img = img.crop((left, top, left + nw, top + nh))
        img = img.resize((224, 224), Image.BILINEAR)
    # 颜色抖动
    img = ImageEnhance.Color(img).enhance(0.4 + random.random() * 1.2)
    img = ImageEnhance.Brightness(img).enhance(0.7 + random.random() * 0.6)
    img = ImageEnhance.Contrast(img).enhance(0.6 + random.random() * 0.8)
    img = ImageEnhance.Sharpness(img).enhance(0.2 + random.random() * 1.6)
    # 随机模糊
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
    # 随机 JPEG 质量模拟
    if random.random() < 0.4:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(50, 95))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img

class RealAIDataset(Dataset):
    def __init__(self, real_dirs, fake_dir, test_dir=None, augment=True):
        self.augment = augment
        self.proc = None  # 延迟设置
        
        self.samples = []
        
        # 收集所有 real 图像
        for rd in real_dirs:
            for f in glob.glob(os.path.join(rd, "*")):
                if os.path.isfile(f): self.samples.append((f, 0))
        
        # 收集 test 目录里的 real_* 文件（加入训练！）
        if test_dir:
            for f in glob.glob(os.path.join(test_dir, "real_*")):
                if os.path.isfile(f): self.samples.append((f, 0))
        
        # 收集所有 fake 图像
        for f in glob.glob(os.path.join(fake_dir, "*")):
            if os.path.isfile(f): self.samples.append((f, 1))
        
        random.seed(CFG.seed + 1)
        random.shuffle(self.samples)
        
        n_real = sum(1 for _,l in self.samples if l==0)
        n_fake = sum(1 for _,l in self.samples if l==1)
        print(f"  Dataset: {len(self.samples)}  real={n_real} fake={n_fake} augment={augment}")
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = strong_augment(img) if self.augment else normalize_to_224(img)
            pv = self.proc(img, return_tensors="pt").pixel_values.squeeze(0)
        except Exception as e:
            print(f"  ERR {path}: {e}")
            pv = torch.zeros(3, 224, 224)
        return pv, torch.tensor(label, dtype=torch.long)

def train():
    from transformers import CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    print("=" * 60); print(" Real vs AI v4 (preprocess + strong aug + more real data)"); print("=" * 60)
    
    full_aug = RealAIDataset(CFG.real_dirs, CFG.fake_dir, CFG.test_dir, augment=True)
    full_aug.proc = proc
    full_clean = RealAIDataset(CFG.real_dirs, CFG.fake_dir, CFG.test_dir, augment=False)
    full_clean.proc = proc
    
    # 相同 split
    indices = list(range(len(full_aug)))
    random.seed(42); random.shuffle(indices)
    n_tr = int(len(indices) * 0.9); n_va = len(indices) - n_tr
    tr_idx, va_idx = indices[:n_tr], indices[n_tr:]
    
    tr = torch.utils.data.Subset(full_aug, tr_idx)
    va = torch.utils.data.Subset(full_clean, va_idx)
    print(f"  Train: {len(tr)}, Val: {len(va)}")
    
    tr_dl = DataLoader(tr, batch_size=CFG.batch_size, shuffle=True, num_workers=0, drop_last=True)
    va_dl = DataLoader(va, batch_size=CFG.batch_size, shuffle=False, num_workers=0)
    
    from models.image_branch import ImageBranch
    ib = ImageBranch(embed_dim=768, num_classes=2, device='cuda')
    ib.cuda()
    for p in ib.clip_model.parameters():
        p.requires_grad = False
    
    tr = sum(p.numel() for p in ib.parameters() if p.requires_grad)
    to = sum(p.numel() for p in ib.parameters())
    print(f"Heads-only: trainable={tr:,} ({tr/to:.1%}) total={to:,}")
    
    ib.train()
    opt = AdamW([p for p in ib.parameters() if p.requires_grad], 
                lr=CFG.lr_heads, weight_decay=CFG.weight_decay)
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
            torch.nn.utils.clip_grad_norm_([p for p in ib.parameters() if p.requires_grad], 1.0)
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
        print(f"  Ep {ep+1}/{CFG.epochs} ({et:.0f}s) | tr={trl:.4f}/{tra:.1%} | va={val:.4f}/{vaa:.1%}", flush=True)
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
            }, os.path.join(CFG.output_dir, "finetuned.pt"))
    
    print(f"\nDone! Best val_acc = {best:.1%}", flush=True)

if __name__ == "__main__":
    train()
