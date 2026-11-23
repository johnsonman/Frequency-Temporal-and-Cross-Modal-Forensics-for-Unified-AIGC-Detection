"""
精简训练脚本 — 只训练 ImageBranch (CLIP + heads)
==================================================
不加载 video/audio 分支，省显存省时间。
训练 image_v3 (8316 fake + 8000 real) → CLIP-ViT-B/16 + 所有 heads
"""
import os, sys, glob, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

class CFG:
    data_dir = "data/image_v3"
    output_dir = "checkpoints/finetuned"
    device = "cuda"
    seed = 42
    batch_size = 64
    num_workers = 0
    epochs = 5
    lr_clip = 1e-4          # CLIP 解冻层 (提)
    lr_heads = 2e-3         # heads (提)
    weight_decay = 0.01
    train_split = 0.9
    grad_clip = 1.0

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)

# ========== Dataset ==========
class ImageV3(Dataset):
    def __init__(self, root, processor):
        self.proc = processor
        self.samples = []
        for label, sub in [(1, "fake"), (0, "real")]:
            for f in glob.glob(os.path.join(root, sub, "*")):
                if os.path.isfile(f): self.samples.append((f, label))
        print(f"  Dataset: {len(self.samples)}  fake={sum(1 for _,l in self.samples if l==1)} real={sum(1 for _,l in self.samples if l==0)}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            pv = self.proc(img, return_tensors="pt").pixel_values.squeeze(0)
        except:
            pv = torch.zeros(3, 224, 224)
        return pv, torch.tensor(label, dtype=torch.long)

# ========== Build (ImageBranch only) ==========
def build():
    from transformers import CLIPVisionModel, CLIPImageProcessor
    from models.image_branch import ImageBranch

    # 直接构建 ImageBranch — 它会自动加载 CLIP
    ib = ImageBranch(embed_dim=768, num_classes=2)
    ib.cuda()

    # 冻结全部 CLIP
    for p in ib.clip_model.parameters():
        p.requires_grad = False

    # 解冻 encoder 所有层 + layernorm
    n = len(ib.clip_model.encoder.layers)
    for i in range(n):
        for p in ib.clip_model.encoder.layers[i].parameters():
            p.requires_grad = True
    if hasattr(ib.clip_model, 'pre_layrnorm'):
        for p in ib.clip_model.pre_layrnorm.parameters():
            p.requires_grad = True
    for p in ib.clip_model.post_layernorm.parameters():
        p.requires_grad = True

    # heads 默认可训练
    tr = sum(p.numel() for p in ib.parameters() if p.requires_grad)
    to = sum(p.numel() for p in ib.parameters())
    print(f"\n{'='*50}")
    print(f"ImageBranch only: trainable={tr:,} ({tr/to:.1%}) total={to:,}")
    print(f"{'='*50}")
    return ib

def param_groups(ib):
    clip_p, head_p = [], []
    clip_ids = set()
    for name, p in ib.named_parameters():
        if p.requires_grad and "clip_model" in name:
            clip_p.append(p); clip_ids.add(id(p))
    for name, p in ib.named_parameters():
        if p.requires_grad and id(p) not in clip_ids:
            head_p.append(p)
    print(f"  CLIP params: {len(clip_p)}, Heads: {len(head_p)}")
    return [
        {"params": clip_p, "lr": CFG.lr_clip, "weight_decay": CFG.weight_decay},
        {"params": head_p, "lr": CFG.lr_heads, "weight_decay": CFG.weight_decay},
    ]

# ========== Train ==========
def train():
    from transformers import CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    print("=" * 60)
    print(" Finetune CLIP-ViT-B/16 + ImageBranch Heads (image_v3)")
    print("=" * 60)

    full = ImageV3(CFG.data_dir, proc)
    n_tr = int(len(full) * CFG.train_split)
    n_va = len(full) - n_tr
    tr, va = random_split(full, [n_tr, n_va], generator=torch.Generator().manual_seed(CFG.seed))
    print(f"  Train: {len(tr)}, Val: {len(va)}")

    tr_dl = DataLoader(tr, batch_size=CFG.batch_size, shuffle=True, num_workers=0, drop_last=True)
    va_dl = DataLoader(va, batch_size=CFG.batch_size, shuffle=False, num_workers=0)

    ib = build()
    ib.train()
    optim = AdamW(param_groups(ib))
    sched = CosineAnnealingLR(optim, T_max=len(tr_dl)*CFG.epochs, eta_min=1e-6)
    bce = nn.BCEWithLogitsLoss()

    best = 0.0
    hist = []

    for ep in range(CFG.epochs):
        t0 = time.time()
        ib.train()
        tl=0; tc=0; tt=0
        for pv, labels in tr_dl:
            pv = pv.cuda(); labels = labels.cuda()
            out = ib(pv)
            logits = out['p_img_logits']
            lab = F.one_hot(labels, 2).float()
            loss = bce(logits, lab)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ib.parameters(), CFG.grad_clip)
            optim.step(); sched.step()
            tl += loss.item()*pv.shape[0]
            tc += (logits.argmax(-1)==labels).sum().item()
            tt += pv.shape[0]

        ib.eval()
        vl=0; vc=0; vt=0
        with torch.no_grad():
            for pv, labels in va_dl:
                pv = pv.cuda(); labels = labels.cuda()
                out = ib(pv)
                logits = out['p_img_logits']
                lab = F.one_hot(labels, 2).float()
                vl += bce(logits, lab).item()*pv.shape[0]
                vc += (logits.argmax(-1)==labels).sum().item()
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
    with open(os.path.join(CFG.output_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

if __name__ == "__main__":
    train()
