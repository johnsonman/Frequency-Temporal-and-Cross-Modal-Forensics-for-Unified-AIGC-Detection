"""诊断模型塌缩 — 分别测试 train Real, train Fake, test Real, test Fake"""
import sys, torch, os, glob
sys.path.insert(0, '.')
from PIL import Image
from transformers import CLIPImageProcessor
import numpy as np

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

def test_dir(path, label_name, limit=50):
    files = sorted([f for f in glob.glob(os.path.join(path, "*")) if os.path.isfile(f)])[:limit]
    scores = []
    for f in files:
        try:
            img = Image.open(f).convert('RGB')
            pv = proc(img, return_tensors="pt").pixel_values.cuda()
            with torch.no_grad():
                out = ib(pv)
            p_fake = torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item()
            scores.append(p_fake)
        except:
            pass
    if scores:
        print(f"\n  {label_name} ({len(scores)} imgs):")
        print(f"    p_fake: mean={np.mean(scores):.4f} std={np.std(scores):.4f} min={min(scores):.4f} max={max(scores):.4f}")
    return scores

print("\n" + "="*60)
print(" TRAINING DATA (subset)")
print("="*60)
test_dir("data/image_real", "Train-Real (Picsum)", limit=50)
test_dir("data/image_v3/fake", "Train-Fake (gengen+hf)", limit=50)

print("\n" + "="*60)
print(" TEST DATA")
print("="*60)
real_test = [(p,'Real') for p in glob.glob('test/image/real_*') if os.path.isfile(p)]
fake_test = [(p,'Fake') for p in glob.glob('test/image/*') if os.path.isfile(p) and not os.path.basename(p).lower().startswith('real_')]

for path, name in real_test[:7]:
    img = Image.open(path).convert('RGB')
    pv = proc(img, return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        out = ib(pv)
    p_fake = torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item()
    print(f"  TEST-REAL  p_fake={p_fake:.4f}  {os.path.basename(path)}")

for path, name in fake_test[:5]:
    img = Image.open(path).convert('RGB')
    pv = proc(img, return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        out = ib(pv)
    p_fake = torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item()
    print(f"  TEST-FAKE  p_fake={p_fake:.4f}  {os.path.basename(path)}")
