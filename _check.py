import torch, os
import sys

# Check real images
real_count = len([f for f in os.listdir("data/image_real") if os.path.isfile(os.path.join("data/image_real", f))])
fake_count = len([f for f in os.listdir("data/image_v3/fake") if os.path.isfile(os.path.join("data/image_v3/fake", f))])
print(f"Real: {real_count}, Fake: {fake_count}")

# Check checkpoint
ckpt_path = "checkpoints/finetuned/finetuned.pt"
if os.path.exists(ckpt_path):
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint: epoch={c.get('epoch')} val_acc={c.get('val_acc', 0):.1%}")
    if c.get("train_config"):
        print(f"  trained with: {c['train_config']}")
else:
    print("No checkpoint yet")
