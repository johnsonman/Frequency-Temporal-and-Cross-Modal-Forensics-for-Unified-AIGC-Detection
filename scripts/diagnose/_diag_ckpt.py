import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

for name in ['image_best.pt', 'image_finetuned.pt']:
    p = PROJECT_ROOT / 'checkpoints' / name
    if not p.exists():
        print(f"{name}: NOT FOUND")
        continue
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    print("=" * 70)
    print(f"{name}  (top-level type: {type(ckpt).__name__})")
    if isinstance(ckpt, dict):
        print("  top-level keys:", list(ckpt.keys())[:20])
        for k in ['epoch', 'val_acc', 'acc', 'best_acc']:
            if k in ckpt:
                print(f"    {k} = {ckpt[k]}")
        sd = ckpt.get('model_state', ckpt.get('state_dict', None))
        if sd is None:
            # maybe the dict itself is the state_dict
            sd = ckpt if all(hasattr(v, 'shape') for v in list(ckpt.values())[:5]) else None
        if sd is not None and hasattr(sd, 'keys'):
            keys = list(sd.keys())
            print(f"  state_dict entries: {len(keys)}")
            for k in keys[:25]:
                print(f"    {k}  {tuple(sd[k].shape)}")
