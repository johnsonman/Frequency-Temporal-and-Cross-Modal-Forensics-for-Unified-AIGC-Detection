"""极小规模训练 — 确保能快速跑通完整循环"""
import os, sys, glob, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image

class CFG:
    data_dir = "data/image_v3"
    output_dir = "checkpoints/finetuned"
    device = "cuda"
    seed = 42
    max_per_class = 2000   # 只用 2000 fake + 2000 real = 4000 图
    batch_size = 64
    epochs = 10
    lr_heads = 3e-3        # 只训练 heads (CLIP 冻结)
    weight_decay = 0.01
    grad_clip = 1.0

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)

class ImageV3(Dataset):
    def __init__(self, root, processor):
        self.proc = processor
        self.samples = []
        for label, sub in [(1, "fake"), (0, "real")]:
            files = glob.glob(os.path.join(root, sub, "*"))[:CFG.max_per_class]
            for f in files:
                if os.path.isfile(f): self.samples.append((f, label))
        random.shuffle(self.samples)
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

    print("=" * 60); print(" Fast Heads-Only Finetune"); print("=" * 60)
    full = ImageV3(CFG.data_dir, proc)
    n_tr = int(len(full) * 0.9); n_va = len(full) - n_tr
    tr, va = torch.utils.data.random_split(full, [n_tr, n_va], generator=torch.Generator().manual_seed(42))
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
            }, os.path.join(CFG.output_dir, "finetuned.pt"))

    print(f"\nDone! Best val_acc = {best:.1%}", flush=True)
    with open(os.path.join(CFG.output_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

if __name__ == "__main__":
    train()
