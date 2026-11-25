import sys
from pathlib import Path
import numpy as np, cv2, torch, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from train import ImageDetectorViT
from PIL import Image

test_dir = PROJECT_ROOT / 'test' / 'image'
images = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])


def load_model(name):
    ck = torch.load(PROJECT_ROOT / 'checkpoints' / name, map_location='cpu', weights_only=False)
    s = ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck
    m = ImageDetectorViT(num_classes=2, pretrained=False)
    m.load_state_dict(s, strict=False)
    return m.to(DEVICE).eval()


def read_rgb(p):
    img = cv2.imread(str(p))
    if img is None:
        img = np.array(Image.open(p).convert('RGB'))[:, :, ::-1]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(rgb):
    t = torch.tensor(rgb).permute(2, 0, 1).float() / 255.0
    return ((t - 0.5) / 0.5)


def variants_resize(rgb):
    r = cv2.resize(rgb, (224, 224))
    yield to_tensor(r)
    yield torch.flip(to_tensor(r), dims=[2])


def variants_crop(rgb):
    h, w = rgb.shape[:2]
    s = 256 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    h2, w2 = r.shape[:2]
    y0, x0 = (h2 - 224) // 2, (w2 - 224) // 2
    c = r[y0:y0 + 224, x0:x0 + 224]
    yield to_tensor(c)
    yield torch.flip(to_tensor(c), dims=[2])


@torch.no_grad()
def predict(model, p, variant):
    rgb = read_rgb(p)
    ts = list(variant(rgb))
    batch = torch.stack(ts).to(DEVICE)
    probs = F.softmax(model(batch), dim=1).mean(dim=0).cpu().numpy()
    return float(probs[1])


models = {n: load_model(n) for n in ['image_best.pt', 'image_finetuned.pt']}

configs = {
    'best / resize':            ('image_best.pt', variants_resize),
    'best / crop':              ('image_best.pt', variants_crop),
    'finetuned / resize':       ('image_finetuned.pt', variants_resize),
    'finetuned / crop':         ('image_finetuned.pt', variants_crop),
}

for cfg_name, (mname, variant) in configs.items():
    correct, rows = 0, []
    for p in images:
        ai_p = predict(models[mname], p, variant)
        pred_real = ai_p <= 0.5
        gt_real = p.name.startswith('real_')
        ok = pred_real == gt_real
        correct += ok
        rows.append((p.name, ai_p, ok))
    bad = [r[0] for r in rows if not r[2]]
    print(f"{cfg_name:24s} acc={correct}/{len(images)} ({correct/len(images)*100:5.1f}%)  wrong={bad}")

# ensemble best+finetuned with resize
print()
correct, bad = 0, []
for p in images:
    a = predict(models['image_best.pt'], p, variants_resize)
    b = predict(models['image_finetuned.pt'], p, variants_resize)
    ai_p = (a + b) / 2
    pred_real = ai_p <= 0.5
    gt_real = p.name.startswith('real_')
    ok = pred_real == gt_real
    correct += ok
    if not ok:
        bad.append(p.name)
print(f"{'ensemble best+finetuned/resize':24s} acc={correct}/{len(images)} ({correct/len(images)*100:5.1f}%)  wrong={bad}")
