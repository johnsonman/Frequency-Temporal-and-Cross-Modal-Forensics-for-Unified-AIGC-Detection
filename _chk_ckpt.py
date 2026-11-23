import sys, torch
sys.path.insert(0, '.')
from models.image_branch import ImageBranch

ib = ImageBranch(embed_dim=768, num_classes=2)
ib.eval()

ckpt = torch.load("checkpoints/finetuned/finetuned.pt", map_location="cpu", weights_only=False)

ckpt_keys = set(ckpt['clip_vision'].keys())
model_keys = set(ib.clip_model.state_dict().keys())

print(f"\nckpt clip_vision keys: {len(ckpt_keys)}")
print(f"model clip keys: {len(model_keys)}")

missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

print(f"\nMissing in ckpt (model has but ckpt doesn't): {len(missing)}")
for k in sorted(missing)[:10]: print(f"  - {k}")
if len(missing) > 10: print(f"  ... +{len(missing)-10} more")

print(f"\nUnexpected in ckpt (ckpt has but model doesn't): {len(unexpected)}")
for k in sorted(unexpected)[:10]: print(f"  - {k}")
if len(unexpected) > 10: print(f"  ... +{len(unexpected)-10} more")

# Sample model keys
print(f"\nSample model keys: {sorted(model_keys)[:5]}...")
print(f"Sample ckpt keys: {sorted(ckpt_keys)[:5]}...")
