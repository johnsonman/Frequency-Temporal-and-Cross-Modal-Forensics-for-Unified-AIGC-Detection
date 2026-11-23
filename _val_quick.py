"""快速验证 — 只测 test/image (33 files)"""
import sys, torch, os, glob, time
sys.path.insert(0, '.')

print(f"Start ({time.strftime('%H:%M:%S')})")
from PIL import Image
from transformers import CLIPImageProcessor

CKPT = "checkpoints/finetuned/finetuned.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"ckpt: epoch={ckpt.get('epoch')} val_acc={ckpt.get('val_acc'):.1%}")

from models.image_branch import ImageBranch
ib = ImageBranch(embed_dim=768, num_classes=2, device='cuda')
ib.cuda(); ib.eval()

ib.clip_model.load_state_dict(ckpt['clip_vision'], strict=True)
ib.adapter.load_state_dict(ckpt['adapter'])
ib.msff.load_state_dict(ckpt['msff'])
ib.forensics.load_state_dict(ckpt['forensics'])
ib.classifier.load_state_dict(ckpt['classifier'])
ib.global_local_fusion.load_state_dict(ckpt['global_local_fusion'])
print("✓ weights loaded")

proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

pairs = [(p,'image') for p in glob.glob('test/image/*') if os.path.isfile(p)]
print(f"\n=== TEST SET (test/image, {len(pairs)} files) ===")
all_pairs = []
for path, ftype in pairs:
    name = os.path.basename(path)
    gt_real = name.lower().startswith('real_')
    try:
        img = Image.open(path).convert('RGB')
        pv = proc(img, return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
            out = ib(pv)
        p_fake = torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item()
    except Exception as e:
        p_fake = 0.5
        print(f"  ERROR {name}: {e}")
    all_pairs.append((gt_real, p_fake, name))

ok = sum(1 for gt,s,_ in all_pairs if (s>0.5) != gt)
tot = len(all_pairs)
print(f"\n  thr=0.5: {ok}/{tot} = {ok/max(tot,1):.1%}")

real_s = [s for gt,s,_ in all_pairs if gt]
fake_s = [s for gt,s,_ in all_pairs if not gt]
import numpy as np
print(f"\n  Real images (gt=Real):")
print(f"    p_fake: mean={np.mean(real_s):.4f} std={np.std(real_s):.4f}")
print(f"    range=[{min(real_s):.4f}, {max(real_s):.4f}]")
print(f"  Fake images (gt=Fake):")
print(f"    p_fake: mean={np.mean(fake_s):.4f} std={np.std(fake_s):.4f}")
print(f"    range=[{min(fake_s):.4f}, {max(fake_s):.4f}]")

print(f"\n  Per-image results:")
for gt, s, name in sorted(all_pairs, key=lambda x: x[1]):
    true_label = "Real" if gt else "Fake"
    pred_label = "AI" if s > 0.5 else "Real"
    correct = "✓" if (s > 0.5) != gt else "✗"
    print(f"    {correct} p_fake={s:.4f}  gt={true_label:4s}  pred={pred_label:4s}  {name}")

print(f"\nDone in {time.time()-0:.1f}s")
