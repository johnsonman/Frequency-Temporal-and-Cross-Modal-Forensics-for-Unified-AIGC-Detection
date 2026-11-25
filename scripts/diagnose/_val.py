import sys, torch, os, glob, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

print(f"Start ({time.strftime('%H:%M:%S')})", flush=True); t0=time.time()
import torch, numpy as np
from PIL import Image
from transformers import CLIPImageProcessor
print(f"import done ({time.time()-t0:.1f}s)", flush=True)

CKPT = "checkpoints/finetuned/finetuned.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"ckpt: epoch={ckpt.get('epoch')} val_acc={ckpt.get('val_acc'):.1%}", flush=True)

from models.image_branch import ImageBranch
ib = ImageBranch(embed_dim=768, num_classes=2, device='cuda')
ib.cuda(); ib.eval()

ib.clip_model.load_state_dict(ckpt['clip_vision'], strict=True)
ib.adapter.load_state_dict(ckpt['adapter'])
ib.msff.load_state_dict(ckpt['msff'])
ib.forensics.load_state_dict(ckpt['forensics'])
ib.classifier.load_state_dict(ckpt['classifier'])
ib.global_local_fusion.load_state_dict(ckpt['global_local_fusion'])
print("✓ heads loaded", flush=True)

proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Test set
pairs = [(p,'image') for p in glob.glob('test/image/*') if os.path.isfile(p)]
print(f"\n=== TEST SET (test/image, {len(pairs)} files) ===", flush=True)
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
    except:
        p_fake = 0.5
    all_pairs.append((gt_real, p_fake, name))

ok = sum(1 for gt,s,_ in all_pairs if (s>0.5) != gt)
tot = len(all_pairs)
print(f"  thr=0.5: {ok}/{tot} = {ok/max(tot,1):.1%}")

ba=0; bt=0
for t in np.arange(0.01, 0.99, 0.005):
    ok_t = sum(1 for gt,s,_ in all_pairs if (s>t) != gt)
    if ok_t/max(tot,1) > ba: ba = ok_t/max(tot,1); bt = round(t,3)
print(f"  Best thr={bt} acc={ba:.1%}")

real_s = [s for gt,s,_ in all_pairs if gt]
fake_s = [s for gt,s,_ in all_pairs if not gt]
print(f"  real: mean={np.mean(real_s):.3f} std={np.std(real_s):.3f} range=[{min(real_s):.3f},{max(real_s):.3f}]")
print(f"  fake: mean={np.mean(fake_s):.3f} std={np.std(fake_s):.3f} range=[{min(fake_s):.3f},{max(fake_s):.3f}]")

mistakes = [(n,s) for gt,s,n in all_pairs if (s>bt) != gt]
print(f"  mistakes at thr={bt} ({len(mistakes)}):")
for n,s in mistakes:
    gt = "Real" if os.path.basename(n).lower().startswith('real_') else "Fake"
    pr = "AI" if s > bt else "Real"
    print(f"    XX p={s:.3f} gt={gt:4s} pred={pr:4s} {n}")

# Also test on training set (sanity)
print(f"\n=== TRAINING SET SAMPLE (image_v3) ===", flush=True)
train_pairs = []
for label, sub in [(1, "fake"), (0, "real")]:
    for f in glob.glob(os.path.join("data/image_v3", sub, "*"))[:200]:
        if os.path.isfile(f):
            gt_real = (label == 0)
            try:
                img = Image.open(f).convert('RGB')
                pv = proc(img, return_tensors="pt").pixel_values.cuda()
                with torch.no_grad():
                    out = ib(pv)
                p_fake = torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item()
            except:
                p_fake = 0.5
            train_pairs.append((gt_real, p_fake, os.path.basename(f)))

tk = sum(1 for gt,s,_ in train_pairs if (s>0.5) != gt)
print(f"  thr=0.5: {tk}/{len(train_pairs)} = {tk/max(len(train_pairs),1):.1%}")
print(f"  Done in {time.time()-t0:.1f}s", flush=True)
