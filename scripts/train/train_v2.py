import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
"""
train_v2.py — 正确的 Real vs AI 训练
=====================================
- Real:  data/image_real/ (下载的真实照片)
- Fake:  data/image_v3/fake/ (8316张 AI 生成图: gengen_edit + hf_ai_art)
"""
import os, sys, glob, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from PIL import Image

class CFG:
    real_dir = "data/image_real"      # ← 真实照片
    fake_dir = "data/image_v3/fake"   # ← AI 生成图
    output_dir = "checkpoints/finetuned"
    device = "cuda"
    seed = 42
    max_per_class = None   # None = 全部
    batch_size = 64
    epochs = 12
    lr_heads = 5e-4        # 降低避免快速饱和
    weight_decay = 0.05    # 加强正则化
    grad_clip = 1.0

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)

class RealAIDataset(Dataset):
    """Real images (label=0) vs AI/Fake images (label=1)"""
    def __init__(self, real_dir, fake_dir, processor, max_per_class=None):
        self.proc = processor
        self.samples = []
        
        # Real (label=0)
        real_files = sorted([f for f in glob.glob(os.path.join(real_dir, "*")) 
                           if os.path.isfile(f)])
        if max_per_class:
            real_files = real_files[:max_per_class]
        for f in real_files:
            self.samples.append((f, 0))
        
        # Fake/AI (label=1)
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
        print(f"  Dataset: {len(self.samples)}  real={n_real} fake={n_fake}")
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            pv = self.proc(img, return_tensors="pt").pixel_values.squeeze(0)
        except:
            pv = torch.zeros(3, 224, 224)
        return pv, torch.tensor(label, dtype=torch.long)

def build():
    from models.image_branch import ImageBranch
    ib = ImageBranch(embed_dim=768, num_classes=2, device='cuda')
    ib.cuda()
    
    # 冻结 CLIP，只训练 heads
    for p in ib.clip_model.parameters():
        p.requires_grad = False
    
    tr = sum(p.numel() for p in ib.parameters() if p.requires_grad)
    to = sum(p.numel() for p in ib.parameters())
    print(f"ImageBranch (heads only): trainable={tr:,} ({tr/to:.1%}) total={to:,}")
    return ib

def train():
    from transformers import CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    print("=" * 60); print(" Real vs AI Finetune (heads-only)"); print("=" * 60)
    
    full = RealAIDataset(CFG.real_dir, CFG.fake_dir, proc, CFG.max_per_class)
    
    n_tr = int(len(full) * 0.9); n_va = len(full) - n_tr
    tr, va = random_split(full, [n_tr, n_va], generator=torch.Generator().manual_seed(42))
    print(f"  Train: {len(tr)}, Val: {len(va)}")
    
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
                "train_config": {"real_dir": CFG.real_dir, "fake_dir": CFG.fake_dir},
            }, os.path.join(CFG.output_dir, "finetuned.pt"))
    
    print(f"\nDone! Best val_acc = {best:.1%}", flush=True)
    with open(os.path.join(CFG.output_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

if __name__ == "__main__":
    train()
