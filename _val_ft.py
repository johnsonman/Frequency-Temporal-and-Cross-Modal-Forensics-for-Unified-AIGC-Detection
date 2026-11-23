"""验证训练后的权重效果"""
import sys, torch, os, glob
sys.path.insert(0, '.')
from run_gui import DetectorWorker

CKPT = "checkpoints/finetuned/finetuned.pt"
print(f"Loading {CKPT}...")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"  ckpt keys: {list(ckpt.keys())}")

# 用 run_gui 的 DetectorWorker 但先手动加载 finetuned 权重
pairs = []
for ft, pat in [('image', 'test/image/*'), ('video', 'test/video/*')]:
    for p in glob.glob(pat):
        if os.path.isfile(p): pairs.append((p, ft))

# 单模态：直接跑 ImageBranch
from transformers import CLIPImageProcessor
from models.image_branch import ImageBranch
import torch.nn.functional as F
from PIL import Image
import numpy as np

proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# 加载 ImageBranch
ib = ImageBranch(embed_dim=768, num_classes=2)
ib.clip_model = None  # 会在 forward 时加载
# 等等 —— ImageBranch 初始化会自动加载 CLIP

# 简化：直接用 run_gui 的 DetectorWorker 但改 worker 加载新权重
# 但 DetectorWorker 会自动 init 模型 —— 我们需要手动替换

# 最简单：手动创建 ImageBranch + 加载权重
ib = ImageBranch(embed_dim=768, num_classes=2)
ib.cuda(); ib.eval()

# 加载 finetuned CLIP vision
ib.clip_model.load_state_dict(ckpt['clip_vision'], strict=False)
# 加载 heads
ib.adapter.load_state_dict(ckpt['image_branch_heads']['adapter'])
ib.msff.load_state_dict(ckpt['image_branch_heads']['msff'])
ib.forensics.load_state_dict(ckpt['image_branch_heads']['forensics'])
ib.classifier.load_state_dict(ckpt['image_branch_heads']['classifier'])
ib.global_local_fusion.load_state_dict(ckpt['image_branch_heads']['global_local_fusion'])
print("✓ All weights loaded")

# Run test images
print(f"\n=== Test Accuracy (finetuned ImageBranch) ===")
for ftype in ['image']:
    items = [(os.path.basename(p), p) for p,ft in pairs if ft==ftype]
    print(f"\n  {ftype.upper()} ({len(items)} files)")
    ok = tot = 0
    mistakes = []
    for name, path in items:
        gt_real = name.lower().startswith('real_')
        try:
            img = Image.open(path).convert('RGB')
            inp = proc(img, return_tensors="pt")
            pv = inp.pixel_values.cuda()
            with torch.no_grad():
                out = ib(pv)
            logits = out['p_img_logits']
            p_fake = F.softmax(logits, dim=-1)[0, 1].item()  # fake 是 class 1
            # raw argmax: 1=Fake, 0=Real
            pred_ai = logits.argmax(dim=-1).item() == 1
        except Exception as e:
            p_fake = 0.5
            pred_ai = False

        correct = pred_ai == (not gt_real)
        ok += int(correct); tot += 1
        if not correct:
            gt_s = "Real" if gt_real else "Fake"
            pr_s = "AI" if pred_ai else "Real"
            mistakes.append((name, p_fake, gt_s, pr_s))

    print(f"  → {ok}/{tot} = {ok/tot:.1%}")
    for m in mistakes:
        print(f"    XX p_fake={m[1]:.3f} gt={m[2]:4s} pred={m[3]:4s}  {m[0]}")

# Also show score distribution
print("\n  Score distribution:")
with torch.no_grad():
    real_s = []; fake_s = []
    for name, path in items:
        gt_real = name.lower().startswith('real_')
        try:
            img = Image.open(path).convert('RGB')
            inp = proc(img, return_tensors="pt")
            pv = inp.pixel_values.cuda()
            out = ib(pv)
            p_fake = F.softmax(out['p_img_logits'], dim=-1)[0,1].item()
        except:
            p_fake = 0.5
        if gt_real: real_s.append(p_fake)
        else: fake_s.append(p_fake)
    if real_s and fake_s:
        print(f"    real: mean={np.mean(real_s):.3f} range=[{min(real_s):.3f},{max(real_s):.3f}]")
        print(f"    fake: mean={np.mean(fake_s):.3f} range=[{min(fake_s):.3f},{max(fake_s):.3f}]")
