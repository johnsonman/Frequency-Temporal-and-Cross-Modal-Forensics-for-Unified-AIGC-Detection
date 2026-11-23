"""极简验证 — 快速检查测试集"""
import sys, torch, os, glob
sys.path.insert(0, '.')
from PIL import Image
from transformers import CLIPImageProcessor
import numpy as np

log_f = open("_val_result.txt", "w", encoding="utf-8")
def p(msg):
    print(msg, flush=True)
    log_f.write(str(msg) + "\n"); log_f.flush()

CKPT = "checkpoints/finetuned/finetuned.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
p(f"ckpt: ep={ckpt.get('epoch')} va={ckpt.get('val_acc'):.1%}")

from models.image_branch import ImageBranch
ib = ImageBranch(embed_dim=768, num_classes=2, device='cuda')
ib.cuda(); ib.eval()
ib.clip_model.load_state_dict(ckpt['clip_vision'], strict=True)
ib.adapter.load_state_dict(ckpt['adapter'])
ib.msff.load_state_dict(ckpt['msff'])
ib.forensics.load_state_dict(ckpt['forensics'])
ib.classifier.load_state_dict(ckpt['classifier'])
ib.global_local_fusion.load_state_dict(ckpt['global_local_fusion'])
p("weights loaded")

proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Test set
real_scores, fake_scores = [], []
for p_path in sorted(glob.glob('test/image/*')):
    if not os.path.isfile(p_path): continue
    name = os.path.basename(p_path)
    gt_real = name.lower().startswith('real_')
    img = Image.open(p_path).convert('RGB')
    pv = proc(img, return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        out = ib(pv)
    pf = torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item()
    if gt_real: real_scores.append(pf)
    else: fake_scores.append(pf)

ok = sum(1 for s in real_scores if s < 0.5) + sum(1 for s in fake_scores if s > 0.5)
tot = len(real_scores) + len(fake_scores)
p(f"\nRESULT: thr=0.5 → {ok}/{tot} = {ok/max(tot,1):.1%}")
p(f"  Test-Real({len(real_scores)}): p_fake mean={np.mean(real_scores):.4f} min={min(real_scores):.4f} max={max(real_scores):.4f}")
p(f"  Test-Fake({len(fake_scores)}): p_fake mean={np.mean(fake_scores):.4f} min={min(fake_scores):.4f} max={max(fake_scores):.4f}")

# Train Picsum
trs = []
for p_path in sorted(glob.glob('data/image_real/*'))[:50]:
    img = Image.open(p_path).convert('RGB')
    pv = proc(img, return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        out = ib(pv)
    trs.append(torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item())
p(f"  Train-Real(Picsum,50): p_fake mean={np.mean(trs):.4f}")

# Train Fake  
tfs = []
for p_path in sorted(glob.glob('data/image_v3/fake/*'))[:50]:
    img = Image.open(p_path).convert('RGB')
    pv = proc(img, return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        out = ib(pv)
    tfs.append(torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item())
p(f"  Train-Fake(AI,50): p_fake mean={np.mean(tfs):.4f}")

log_f.close()
