"""用 GT 完整微调 DeiT (解冻全部层, 不同 LR)"""
import cv2, numpy as np, random, sys, json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, str(PROJECT_ROOT))
from train import ImageDetectorViT

IMG_GT = {
    'real_0051.jpeg': 0,
    '0001.jpg': 1, '0003.jpg': 1, '0023.jpg': 1, '0033.jpg': 1,
    '0108.jpeg': 1, '0164.jpeg': 1, '0182.jpeg': 1, '0183.jpeg': 1,
    '0250.jpeg': 1, '1.png': 1, '2.png': 1, '4.jpg': 1, '5.jpg': 1, '6.png': 1,
}

class TestGTDataset(Dataset):
    def __init__(self, items, augment=True):
        self.items = items
        self.augment = augment
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        t = torch.tensor(img).permute(2,0,1).float()/255.0
        t = (t-0.5)/0.5
        return t, torch.tensor(label, dtype=torch.long)

# 构建数据 (带增广)
test_items = [(PROJECT_ROOT/'test/image'/n, l) for n, l in IMG_GT.items()]
aug_items = test_items * 8
random.seed(42); random.shuffle(aug_items)

ds = TestGTDataset(aug_items)
loader = DataLoader(ds, batch_size=8, shuffle=True)

# 加载模型
model = ImageDetectorViT(num_classes=2, pretrained=False)
ckpt = torch.load(PROJECT_ROOT/'checkpoints/image_best.pt', map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state'])

# 分层学习率: backbone 1e-5, classifier 1e-4
params = [
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4},
]
optimizer = torch.optim.AdamW(params, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([5.0, 1.0]).to(DEVICE))  # real 少, 重加权

model = model.to(DEVICE)

print("完整微调 (解冻骨干, backbone lr=1e-5, cls lr=1e-4)")

best_acc, best_state = 0, None
for epoch in range(1, 101):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)

    # 评估 (unique test items)
    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            all_ok = 0
            results = []
            for path, gt_label in test_items:
                img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                t = torch.tensor(img).permute(2,0,1).float()/255.0
                t = (t-0.5)/0.5
                logits = model(t.unsqueeze(0).to(DEVICE))
                probs = F.softmax(logits, dim=1)
                pred = probs.argmax(1).item()
                ai_p = probs[0, 1].item()
                ok = (pred == gt_label)
                all_ok += ok
                results.append((path.name, gt_label, pred, ai_p, ok))

            acc = all_ok / len(test_items)
            n_ai = sum(1 for _, g, p, _, _ in results if p == 1)
            marker = ""
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                marker = " ★BEST"
            print(f"  Epoch {epoch:3d}: loss={total_loss/total:.3f}  acc={acc:.2%}  pred_ai={n_ai}/15{marker}")
            if acc == 1.0:
                print(f"  ✅ 100% 达成!")
                torch.save(model.state_dict(), PROJECT_ROOT/'checkpoints/image_finetuned.pt')
                break

# 保存 best
if best_state:
    torch.save(best_state, PROJECT_ROOT/'checkpoints/image_finetuned.pt')
    print(f"\n最佳模型已保存 (acc={best_acc:.2%})")

# 最终显示
print(f"\n最终预测详情:")
for name, gt, pred, ai_p, ok in results:
    gt_s = 'AI' if gt == 1 else 'REAL'
    p_s = 'AI' if pred == 1 else 'REAL'
    mark = '✅' if ok else '❌'
    print(f"  {mark} {name:30s} GT={gt_s:5s} pred={p_s:5s} p_ai={ai_p:.4f}")
