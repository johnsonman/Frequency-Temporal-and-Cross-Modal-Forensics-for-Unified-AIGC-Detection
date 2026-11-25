import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
"""端到端验证: 把 checkpoints/video_df26.pt 载入真实 VideoBranch, 在 DF26 验证集上评估"""
import os, json, random, time
import numpy as np
import torch
from torchvision import transforms

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(MEAN, STD)
device = "cuda" if torch.cuda.is_available() else "cpu"

from models.video_branch import VideoBranch

ck = torch.load("checkpoints/video_df26.pt", map_location="cpu", weights_only=False)
vb = VideoBranch(device=device, num_frames=ck["num_frames"])
vb.adapter.load_state_dict(ck["adapter"])
vb.t_agg.load_state_dict(ck["t_agg"])
vb.classifier.load_state_dict(ck["classifier"])
vb.eval()
print(f"loaded ckpt epoch={ck['epoch']} val_bacc={ck['val_bacc']*100:.2f}%")

with open("data/DF26_feats/manifest.json", encoding="utf-8") as f:
    recs = json.load(f)
groups = sorted({r["group"] for r in recs})
random.Random(42).shuffle(groups)
n_val = max(1, int(len(groups) * 0.2))
val_g = set(groups[:n_val])
va = [r for r in recs if r["group"] in val_g]
print(f"val videos={len(va)}")

t0 = time.time()
tot = cor = nr = nrok = nf = nfok = 0
B = 8
with torch.no_grad():
    for i in range(0, len(va), B):
        batch = va[i:i + B]
        xs = []
        for r in batch:
            arr = np.load(r["npy"])
            x = torch.from_numpy(arr).float().div_(255.).permute(0, 3, 1, 2)
            xs.append(norm(x))
        x = torch.stack(xs).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = vb.classifier(vb.t_agg(vb.extract_frame_features(x)))
        logits = logits.float()
        p = logits.argmax(1).cpu().tolist()
        for r, pp in zip(batch, p):
            y = r["label"]
            tot += 1; cor += int(pp == y)
            if y == 0:
                nr += 1; nrok += int(pp == 0)
            else:
                nf += 1; nfok += int(pp == 1)

acc = cor / tot
rr = nrok / nr
rf = nfok / nf
print(f"[端到端] acc={acc*100:.2f}%  bacc={(rr+rf)/2*100:.2f}%  "
      f"(real={rr*100:.1f}% {nrok}/{nr}, fake={rf*100:.1f}% {nfok}/{nf})  {time.time()-t0:.1f}s")
