"""DF26 视频分支训练 v2: 基于缓存 CLIP 特征, 快速训练 T-Agg + adapter + classifier

特征: data/DF26_feats/manifest.json  (每条 [2,T,768] float16: 0=原图 1=翻转)
切分: 按源视频 group 切分 (无泄漏)
产物: checkpoints/video_df26.pt + _history.json
"""
import os, json, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

T = 8


class Head(nn.Module):
    """与 VideoBranch 内部模块同名的可训练头 (state_dict 键可直接对接 VideoBranch)"""

    def __init__(self, dim=768, num_classes=2):
        super().__init__()
        from models.video_branch import TAggregation
        self.adapter = nn.Linear(dim, dim)
        self.t_agg = TAggregation(embed_dim=dim, num_heads=8, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim // 2, num_classes),
        )

    def forward(self, x):  # x: [B,T,768]
        return self.classifier(self.t_agg(self.adapter(x)))


class FeatDS(Dataset):
    def __init__(self, recs, train=False):
        self.recs = recs
        self.train = train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        f = np.load(r["feat"])  # [2,T,768] fp16
        k = 1 if (self.train and random.random() < 0.5) else 0
        return torch.from_numpy(f[k].astype(np.float32)), r["label"]


def split_by_group(recs, val_ratio=0.2, seed=42):
    groups = sorted({r["group"] for r in recs})
    random.Random(seed).shuffle(groups)
    n_val = max(1, int(len(groups) * val_ratio))
    val_g = set(groups[:n_val])
    return ([r for r in recs if r["group"] not in val_g],
            [r for r in recs if r["group"] in val_g], sorted(val_g))


@torch.no_grad()
def evaluate(head, dl, device, crit):
    head.eval()
    tot = correct = 0
    nr = nrok = nf = nfok = 0
    ls = 0.0
    preds = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = head(x)
        ls += crit(logits, y).item() * y.size(0)
        p = logits.argmax(1)
        preds.append(p.cpu())
        tot += y.size(0); correct += (p == y).sum().item()
        nr += (y == 0).sum().item(); nrok += ((p == 0) & (y == 0)).sum().item()
        nf += (y == 1).sum().item(); nfok += ((p == 1) & (y == 1)).sum().item()
    acc = correct / max(tot, 1)
    rr = nrok / max(nr, 1); rf = nfok / max(nf, 1)
    return {"acc": acc, "bacc": (rr + rf) / 2, "rec_real": rr, "rec_fake": rf,
            "loss": ls / max(tot, 1), "n": tot}, torch.cat(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="checkpoints/video_df26.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("data/DF26_feats/manifest.json", encoding="utf-8") as f:
        recs = json.load(f)
    tr, va, val_g = split_by_group(recs, 0.2, 42)
    print(f"train={len(tr)} val={len(va)} (val groups={len(val_g)}; "
          f"train real={sum(r['label']==0 for r in tr)} fake={sum(r['label']==1 for r in tr)}; "
          f"val real={sum(r['label']==0 for r in va)} fake={sum(r['label']==1 for r in va)})")

    dl_tr = DataLoader(FeatDS(tr, True), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(FeatDS(va, False), batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, persistent_workers=True)

    head = Head().to(device)
    n_par = sum(p.numel() for p in head.parameters())
    print(f"trainable params {n_par/1e6:.2f}M")

    nr = sum(r["label"] == 0 for r in tr); nfa = sum(r["label"] == 1 for r in tr)
    w = torch.tensor([len(tr) / (2 * nr), len(tr) / (2 * nfa)], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    print(f"class weights real={w[0].item():.2f} fake={w[1].item():.2f}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = len(dl_tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    hist, best = [], -1.0
    for ep in range(1, args.epochs + 1):
        head.train()
        t0 = time.time(); run = 0.0; nb = 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(head(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item(); nb += 1
        m, _ = evaluate(head, dl_va, device, crit)
        tl = run / max(nb, 1)
        print(f"[ep{ep}] train_loss={tl:.4f} | val acc={m['acc']*100:.2f}% "
              f"bacc={m['bacc']*100:.2f}% (real={m['rec_real']*100:.1f}% fake={m['rec_fake']*100:.1f}%) "
              f"| {time.time()-t0:.1f}s", flush=True)
        hist.append({"epoch": ep, "train_loss": tl, **m})
        if m["bacc"] > best:
            best = m["bacc"]
            torch.save({"adapter": head.adapter.state_dict(),
                        "t_agg": head.t_agg.state_dict(),
                        "classifier": head.classifier.state_dict(),
                        "num_frames": T, "val_acc": m["acc"], "val_bacc": m["bacc"],
                        "epoch": ep, "val_groups": val_g}, args.out)

    with open(os.path.splitext(args.out)[0] + "_history.json", "w") as f:
        json.dump(hist, f, indent=1)

    # 最佳模型 + 按生成器细分
    ck = torch.load(args.out, map_location="cpu", weights_only=False)
    head.adapter.load_state_dict(ck["adapter"]); head.t_agg.load_state_dict(ck["t_agg"])
    head.classifier.load_state_dict(ck["classifier"])
    _, preds = evaluate(head, dl_va, device, crit)
    print(f"\nbest epoch={ck['epoch']} val_bacc={ck['val_bacc']*100:.2f}% acc={ck['val_acc']*100:.2f}%")
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r, p in zip(va, preds.tolist()):
        if r["label"] != 1:
            continue
        parts = os.path.normpath(r["npy"]).split(os.sep)
        gen = parts[-2] if len(parts) >= 2 else "?"
        agg[gen][0] += int(p == 1); agg[gen][1] += 1
    print("fake 分生成器召回:")
    for g, (ok, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {g:32s} {ok}/{n} = {ok/n*100:.1f}%")
    rr = sum(1 for r, p in zip(va, preds.tolist()) if r["label"] == 0 and p == 0) / sum(1 for r in va if r["label"] == 0)
    print(f"real 召回 = {rr*100:.1f}%")


if __name__ == "__main__":
    main()
