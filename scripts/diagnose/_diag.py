import sys, torch, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

print("1. Start...", flush=True); t=time.time()
from models.image_branch import ImageBranch
print(f"2. Import done ({time.time()-t:.1f}s)", flush=True)
ib = ImageBranch(embed_dim=768, num_classes=2, device='cuda')
print(f"3. ImageBranch built ({time.time()-t:.1f}s)", flush=True)

# freeze/unfreeze
for p in ib.clip_model.parameters():
    p.requires_grad = False
n = len(ib.clip_model.encoder.layers)
for i in range(n):
    for p in ib.clip_model.encoder.layers[i].parameters():
        p.requires_grad = True
for p in ib.clip_model.post_layernorm.parameters():
    p.requires_grad = True

tr = sum(p.numel() for p in ib.parameters() if p.requires_grad)
to = sum(p.numel() for p in ib.parameters())
print(f"4. trainable={tr:,}/{to:,} ({tr/to:.1%})", flush=True)

# 一个 dummy forward
x = torch.randn(2,3,224,224, device='cuda')
print("5. Forward...", flush=True)
out = ib(x)
print(f"6. Forward done ({time.time()-t:.1f}s) keys={list(out.keys())}", flush=True)

# backward
bce = torch.nn.BCEWithLogitsLoss()
lab = torch.nn.functional.one_hot(torch.tensor([1,0], device='cuda'), 2).float()
loss = bce(out['p_img_logits'], lab)
loss.backward()
print(f"7. Backward done ({time.time()-t:.1f}s) loss={loss.item():.4f}", flush=True)
print("DONE ✓", flush=True)
