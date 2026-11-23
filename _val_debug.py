"""Ultra-minimal validation — debug why it hangs"""
import sys, torch, os, glob, time
from PIL import Image
from transformers import CLIPImageProcessor
import numpy as np

print(f"[{time.strftime('%H:%M:%S')}] Start", flush=True)

CKPT = "checkpoints/finetuned/finetuned.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"[{time.strftime('%H:%M:%S')}] ckpt loaded: ep={ckpt.get('epoch')}", flush=True)

from models.image_branch import ImageBranch
ib = ImageBranch(embed_dim=768, num_classes=2, device='cuda')
ib.cuda(); ib.eval()
ib.clip_model.load_state_dict(ckpt['clip_vision'], strict=True)
ib.adapter.load_state_dict(ckpt['adapter'])
ib.msff.load_state_dict(ckpt['msff'])
ib.forensics.load_state_dict(ckpt['forensics'])
ib.classifier.load_state_dict(ckpt['classifier'])
ib.global_local_fusion.load_state_dict(ckpt['global_local_fusion'])
print(f"[{time.strftime('%H:%M:%S')}] weights on GPU", flush=True)

proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
print(f"[{time.strftime('%H:%M:%S')}] processor ready", flush=True)

# Test single image
test_paths = sorted(glob.glob('test/image/*'))[:3]
for p in test_paths:
    name = os.path.basename(p)
    print(f"  Testing {name}...", flush=True)
    img = Image.open(p).convert('RGB')
    pv = proc(img, return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        out = ib(pv)
    pf = torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item()
    print(f"    → p_fake={pf:.4f}", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] Done!", flush=True)
