import sys, torch, os, glob, time
from PIL import Image
from transformers import CLIPImageProcessor
import numpy as np

print(f"Start ({time.strftime('%H:%M:%S')})", flush=True)

CKPT = "checkpoints/finetuned/finetuned.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"ckpt: ep={ckpt.get('epoch')} va={ckpt.get('val_acc'):.1%}", flush=True)

from models.image_branch import ImageBranch
ib = ImageBranch(embed_dim=768, num_classes=2, device='cuda')
ib.cuda(); ib.eval()
ib.clip_model.load_state_dict(ckpt['clip_vision'], strict=True)
ib.adapter.load_state_dict(ckpt['adapter'])
ib.msff.load_state_dict(ckpt['msff'])
ib.forensics.load_state_dict(ckpt['forensics'])
ib.classifier.load_state_dict(ckpt['classifier'])
ib.global_local_fusion.load_state_dict(ckpt['global_local_fusion'])
print("weights loaded", flush=True)

proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Test set
real_scores, fake_scores = [], []
details = []
for p in sorted(glob.glob('test/image/*')):
    if not os.path.isfile(p): continue
    name = os.path.basename(p)
    gt_real = name.lower().startswith('real_')
    try:
        img = Image.open(p).convert('RGB')
        pv = proc(img, return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
            out = ib(pv)
        pf = torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item()
    except:
        pf = 0.5
    if gt_real: real_scores.append(pf)
    else: fake_scores.append(pf)
    details.append((gt_real, pf, name))

ok = sum(1 for s in real_scores if s < 0.5) + sum(1 for s in fake_scores if s > 0.5)
tot = len(real_scores) + len(fake_scores)
print(f"\nRESULT: thr=0.5 → {ok}/{tot} = {ok/max(tot,1):.1%}", flush=True)
print(f"  Test-Real({len(real_scores)}): p_fake mean={np.mean(real_scores):.4f} min={min(real_scores):.4f} max={max(real_scores):.4f}", flush=True)
print(f"  Test-Fake({len(fake_scores)}): p_fake mean={np.mean(fake_scores):.4f} min={min(fake_scores):.4f} max={max(fake_scores):.4f}", flush=True)

# Best threshold
best_t, best_a = 0.5, 0
for t in np.arange(0.05, 0.95, 0.01):
    acc = (sum(1 for s in real_scores if s < t) + sum(1 for s in fake_scores if s > t)) / tot
    if acc > best_a: best_a, best_t = acc, t
print(f"  Best thr={best_t:.2f} acc={best_a:.1%}", flush=True)

print(f"\nPer-image:", flush=True)
for gt, s, name in sorted(details, key=lambda x: x[1]):
    tl = "Real" if gt else "Fake"
    pl = "AI" if s > 0.5 else "Real"
    c = "✓" if (s > 0.5) != gt else "✗"
    print(f"  {c} p={s:.4f} gt={tl:4s} pred={pl:4s} {name}", flush=True)
