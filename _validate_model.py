import sys
from pathlib import Path
import numpy as np, cv2, torch, torch.nn.functional as F
import timm

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("timm:", timm.__version__)

ckpt = torch.load(PROJECT_ROOT / 'checkpoints' / 'image_best.pt', map_location='cpu', weights_only=False)
sd = ckpt['model_state']
print("total keys:", len(sd))
print("has dist_token:", any('dist_token' in k for k in sd))
print("classifier keys:", [k for k in sd if k.startswith('classifier')])
print("last keys:", list(sd.keys())[-6:])

from train import ImageDetectorViT
from PIL import Image
test_dir = PROJECT_ROOT / 'test' / 'image'
images = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])


def eval_ckpt(ckpt_name):
    ck = torch.load(PROJECT_ROOT / 'checkpoints' / ckpt_name, map_location='cpu', weights_only=False)
    s = ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck
    m = ImageDetectorViT(num_classes=2, pretrained=False)
    mi, un = m.load_state_dict(s, strict=False)
    m = m.to(DEVICE).eval()
    print(f"\n##### {ckpt_name}  (missing={len(mi)}, unexpected={len(un)}, "
          f"val_acc={ck.get('val_acc') if isinstance(ck, dict) else 'n/a'})")

    correct = 0
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            img = np.array(Image.open(p).convert('RGB'))[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        t = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        t = ((t - 0.5) / 0.5).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(m(t), dim=1)[0].cpu().numpy()
        ai_p = float(probs[1])
        pred_real = ai_p <= 0.5
        gt_real = p.name.startswith('real_')
        ok = pred_real == gt_real
        correct += ok
        print(f"{'OK ' if ok else 'BAD'} {p.name:22s} ai_prob={ai_p:.4f} "
              f"pred={'REAL' if pred_real else 'AI  '} gt={'REAL' if gt_real else 'AI  '}")
    print(f"  Accuracy: {correct}/{len(images)} = {correct/len(images)*100:.1f}%")


for name in ['image_best.pt', 'image_finetuned.pt']:
    eval_ckpt(name)

