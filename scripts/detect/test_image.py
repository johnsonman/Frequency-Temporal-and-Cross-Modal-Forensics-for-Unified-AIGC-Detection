"""
对 test/image/ 全部图像进行检测
- 加载 image_finetuned.pt (GT 微调后 DeiT-ViT)
- 输出 p_fake + 判定 + 与 GT 对比（文件名 real_* = 真实）
- 保存多维度可视化结果
"""
import argparse, json, os, sys, csv
from pathlib import Path
import numpy as np
import cv2, torch, torch.nn as nn, torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

PROJECT = Path(__file__).resolve().parents[2]
TEST_DIR = PROJECT/'test'/'image'
CALIBRATION = PROJECT/'checkpoints'/'image_threshold.json'
_parser = argparse.ArgumentParser(description='评估 DeiT 图像鉴伪模型')
_parser.add_argument('--checkpoint', default='checkpoints/image_best.pt')
_parser.add_argument('--threshold', type=float, default=None)
_args = _parser.parse_args()
CKPT = Path(_args.checkpoint)
if not CKPT.is_absolute():
    CKPT = PROJECT / CKPT
_cal = json.loads(CALIBRATION.read_text(encoding='utf-8'))
_calibrated_ckpt = (PROJECT / 'checkpoints' / _cal['checkpoint']).resolve()
_checkpoint_calibration = CKPT.parent / 'image_threshold.json'
if _checkpoint_calibration.is_file():
    _cal = json.loads(_checkpoint_calibration.read_text(encoding='utf-8'))
THRESHOLD = _args.threshold if _args.threshold is not None else (
    _cal['threshold'] if _checkpoint_calibration.is_file() or CKPT.resolve() == _calibrated_ckpt else 0.5
)
OUT = PROJECT/'outputs'/'image_test_results'
PROC = OUT/'process'
OUT.mkdir(parents=True, exist_ok=True)
PROC.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==================== 1. 加载微调后模型 ====================
import timm

class ImageModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(feat_dim, feat_dim//2), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(feat_dim//2, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

model = ImageModel(num_classes=2)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
state = ckpt if not isinstance(ckpt, dict) or 'model_state' not in ckpt else ckpt['model_state']
model.load_state_dict(state)
model.to(device); model.eval()
print(f"✅ 已加载 {CKPT}")

def preprocess_image(img_bgr):
    """Match train.ImageDataset exactly: BGR→RGB, OpenCV resize, then [-1, 1]."""
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0)
    return (tensor - 0.5) / 0.5

# ==================== 2. 读取所有测试图像 ====================
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
files = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in exts])
print(f"\n共发现 {len(files)} 张测试图像")

# ==================== 3. 逐张检测 ====================
def heuristic_freq(img_bgr):
    """启发式频域辅助特征"""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    fft_s = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(fft_s)
    low = mag[h//2-int(h*0.15):h//2+int(h*0.15), w//2-int(w*0.15):w//2+int(w*0.15)].sum()
    return 1 - low/mag.sum()

results = []
print(f"\n{'='*70}")
print(f"{'文件名':<28} {'p_fake':>7} {'判定':>5} {'GT':>5} {'正确':>5} {'freq_ratio':>10}")
print(f"{'='*70}")

with torch.no_grad():
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  ⚠️ 无法读取: {p.name}"); continue
        t = preprocess_image(img).unsqueeze(0).to(device)
        logits = model(t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        p_fake = float(probs[1])  # fake=1
        pred_fake = p_fake >= THRESHOLD
        # GT: real_ 开头 = 真实
        gt_real = p.stem.lower().startswith('real_')
        gt_fake = not gt_real
        correct = pred_fake == gt_fake
        freq = heuristic_freq(img)
        results.append({
            'file_name': p.name, 'p_fake': p_fake, 'pred_fake': pred_fake,
            'gt_real': gt_real, 'gt_fake': gt_fake, 'correct': correct,
            'freq_ratio': float(freq), 'path': str(p)
        })
        mark = '✓' if correct else '✗'
        pred_str = 'AI' if pred_fake else 'REAL'
        gt_str = 'REAL' if gt_real else 'AI'
        print(f"  {p.name:<26} {p_fake:>7.4f} {pred_str:>5} {gt_str:>5} {mark:>5} {freq:>10.4f}")

total = len(results)
right = sum(1 for r in results if r['correct'])
print(f"\n📊 总准确率: {right}/{total} = {right/total*100:.1f}% (阈值={THRESHOLD:.2f})")
real_n = sum(r['gt_real'] for r in results)
fake_n = sum(r['gt_fake'] for r in results)
real_ok = sum(not r['pred_fake'] and r['gt_real'] for r in results)
fake_ok = sum(r['pred_fake'] and r['gt_fake'] for r in results)
real_recall = real_ok / max(real_n, 1)
fake_recall = fake_ok / max(fake_n, 1)
print(f"   真实图召回: {real_ok}/{real_n} = {real_recall:.1%}")
print(f"   AI图召回:   {fake_ok}/{fake_n} = {fake_recall:.1%}")
print(f"   平衡准确率: {(real_recall + fake_recall) / 2:.1%}")

# ==================== 4. 保存 CSV + JSON ====================
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = OUT/f'test_results_{ts}.csv'
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['文件名', 'p_fake', '判定', 'GT', '正确', 'freq_ratio'])
    for r in results:
        w.writerow([r['file_name'], f"{r['p_fake']:.4f}",
                    'AI' if r['pred_fake'] else 'REAL',
                    'REAL' if r['gt_real'] else 'AI',
                    '✓' if r['correct'] else '✗', f"{r['freq_ratio']:.4f}"])
print(f"\n✅ CSV: {csv_path.name}")

json_path = OUT/f'test_results_{ts}.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(f"✅ JSON: {json_path.name}")

# ==================== 5. 汇总可视化 PNG ====================
print(f"\n生成可视化...")

# 5a. 所有图像概率柱状图
fig, ax = plt.subplots(figsize=(max(12, total*0.6), 5))
names = [r['file_name'] for r in results]
p_fakes = [r['p_fake'] for r in results]
colors = ['#10B981' if (r['pred_fake']==r['gt_fake'] and r['gt_real']) else   # 正确的 REAL
          '#EF4444' if r['pred_fake'] and r['gt_fake'] else                  # 正确的 AI
          '#F59E0B' for r in results]                                         # 错误
colors = ['#EF4444' if r['pred_fake'] else '#10B981' for r in results]
edge_colors = ['#000000' if not r['correct'] else '#FFFFFF' for r in results]
edge_widths = [1.5 if not r['correct'] else 0 for r in results]

x = range(total)
bars = ax.bar(x, p_fakes, color=colors, edgecolor=edge_colors, linewidth=edge_widths)
ax.axhline(THRESHOLD, color='gray', linestyle='--', linewidth=1.5, label=f'判定阈值 {THRESHOLD:.2f}')
ax.set_xticks(x); ax.set_xticklabels([n[:15] for n in names], rotation=60, ha='right', fontsize=8)
ax.set_ylabel('AI 概率 p_fake', fontsize=12)
ax.set_title(f'图像检测结果 - DeiT-ViT (微调后)  |  准确率 {right}/{total} = {right/total*100:.1f}%',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend([f'阈值 {THRESHOLD:.2f}', '红色=判定为AI', '绿色=判定为REAL'], fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT/'01_p_fake_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 5b. 综合 grid (原图 + FFT + SRM + p_fake bar)
ncols = 5
nrows = int(np.ceil(total / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
if nrows == 1: axes = axes.reshape(1, -1)
axes_flat = axes.flatten()

for i, r in enumerate(results):
    ax = axes_flat[i]
    img = cv2.imread(r['path'])
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
    ax.imshow(small)
    color = '#EF4444' if r['pred_fake'] else '#10B981'
    gt_mark = '✓' if r['correct'] else '✗'
    gt_str = 'REAL' if r['gt_real'] else 'AI'
    ax.set_title(f"{r['file_name'][:18]}\np_fake={r['p_fake']:.4f}  [{gt_str} {gt_mark}]",
                 fontsize=8, color=color, fontweight='bold')
    ax.axis('off')
for j in range(total, len(axes_flat)):
    axes_flat[j].axis('off')
fig.suptitle(f'全部 {total} 张图像检测结果 | DeiT-ViT Finetuned', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT/'02_all_images_grid.png', dpi=120, bbox_inches='tight')
plt.close()

# 5c. 漏检/误判分析图
missed = [r for r in results if not r['correct']]
if missed:
    fig, axes = plt.subplots(len(missed), 4, figsize=(16, 4*len(missed)))
    if len(missed) == 1: axes = axes.reshape(1, -1)
    for row, r in enumerate(missed):
        img = cv2.imread(r['path'])
        rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
        fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g)))+1)
        k = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float)
        srm = cv2.filter2D(g, -1, k)
        # p_fake 条形
        axes[row, 0].imshow(rgb); axes[row, 0].axis('off')
        axes[row, 0].set_title(f"{r['file_name'][:20]}\np_fake={r['p_fake']:.4f}", fontsize=8, color='#EF4444', fontweight='bold')
        axes[row, 1].imshow(fft_mag, cmap='magma'); axes[row, 1].axis('off')
        axes[row, 1].set_title('FFT 幅度谱', fontsize=8)
        axes[row, 2].imshow(np.clip(srm,-30,30), cmap='coolwarm'); axes[row, 2].axis('off')
        axes[row, 2].set_title('SRM 残差', fontsize=8)
        bar_color = '#EF4444' if r['pred_fake'] else '#10B981'
        axes[row, 3].bar(['REAL','AI'], [1-r['p_fake'], r['p_fake']], color=['#10B981','#EF4444'])
        axes[row, 3].axhline(THRESHOLD, color='gray', linestyle='--')
        axes[row, 3].set_ylim(0,1); axes[row, 3].set_title('概率分布', fontsize=8)
    fig.suptitle(f'⚠️ 检测错误 {len(missed)} 张 (DeiT微调后)', fontsize=14, fontweight='bold', color='#EF4444')
    plt.tight_layout()
    plt.savefig(OUT/'03_errors_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    print("  ✅ 零错误！所有图像全部正确判定")

# 5d. 混淆矩阵 + GT 分布饼图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# 混淆矩阵
cm = np.zeros((2,2), dtype=int)
for r in results:
    pred = 1 if r['pred_fake'] else 0
    gt = 1 if r['gt_fake'] else 0
    cm[gt, pred] += 1  # row=GT, col=Pred
im = ax1.imshow(cm, cmap='Blues', aspect='equal')
ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
ax1.set_xticklabels(['REAL (Pred)', 'AI (Pred)'], fontsize=10)
ax1.set_yticklabels(['REAL (GT)', 'AI (GT)'], fontsize=10)
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=20,
                 color='white' if cm[i,j] > cm.max()/2 else 'black', fontweight='bold')
ax1.set_title('混淆矩阵 Confusion Matrix', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax1, fraction=0.046)

# GT 饼图
gt_counts = [sum(1 for r in results if r['gt_real']), sum(1 for r in results if r['gt_fake'])]
ax2.pie(gt_counts, labels=[f'真实 REAL ({gt_counts[0]})', f'AI伪造 ({gt_counts[1]})'],
        colors=['#10B981','#EF4444'], autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11})
ax2.set_title(f'测试集 GT 分布 (共 {total} 张)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'04_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  ✓ 01_p_fake_distribution.png")
print(f"  ✓ 02_all_images_grid.png")
print(f"  ✓ 03_errors_analysis.png")
print(f"  ✓ 04_confusion_matrix.png")

# ==================== 6. 每图像检测过程 PNG ====================
print(f"\n生成每图像检测过程图 ({total} 张)...")
for i, r in enumerate(results):
    img = cv2.imread(r['path'])
    rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
    fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g)))+1)
    k = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float)
    srm = cv2.filter2D(g, -1, k)
    blur = cv2.GaussianBlur(g.astype(float), (31,31), 0)
    lv = cv2.GaussianBlur((g-blur)**2, (31,31), 0)

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))
    axes[0].imshow(rgb); axes[0].axis('off'); axes[0].set_title('输入', fontsize=10)
    axes[1].imshow(fft_mag, cmap='magma'); axes[1].axis('off'); axes[1].set_title('FFT', fontsize=10)
    axes[2].imshow(np.clip(srm,-30,30), cmap='coolwarm'); axes[2].axis('off'); axes[2].set_title('SRM', fontsize=10)
    axes[3].imshow(lv, cmap='viridis'); axes[3].axis('off'); axes[3].set_title('局部方差', fontsize=10)
    # 概率条
    axes[4].bar(['REAL','AI'], [1-r['p_fake'], r['p_fake']], color=['#10B981','#EF4444'])
    axes[4].axhline(THRESHOLD, color='gray', linestyle='--', linewidth=1.5)
    axes[4].set_ylim(0,1); axes[4].set_title(f'p_fake={r["p_fake"]:.4f}', fontsize=10)
    axes[4].set_ylabel('概率')

    badge = '🤖 AI' if r['pred_fake'] else '📷 REAL'
    color = '#EF4444' if r['pred_fake'] else '#10B981'
    gt_str = 'REAL' if r['gt_real'] else 'AI'
    mark = '✅' if r['correct'] else '❌'
    fig.suptitle(f"{badge}  {r['file_name']}  |  GT: {gt_str} {mark}",
                 fontsize=12, fontweight='bold', color=color)
    plt.tight_layout()
    plt.savefig(PROC/f"img_{i:02d}_{r['file_name'].replace(' ','_')[:35]}.png",
                dpi=150, bbox_inches='tight')
    plt.close()
print(f"  ✓ 已生成 {total} 张过程图 → {PROC}")

# ==================== 总结 ====================
total_kb = sum(p.stat().st_size for p in OUT.glob('*.png')) / 1024
proc_kb = sum(p.stat().st_size for p in PROC.glob('*.png')) / 1024
print(f"\n{'='*70}")
print(f"✅ 全部完成!")
print(f"   汇总 PNG: {len(list(OUT.glob('*.png')))} 张  ({total_kb:.0f} KB)")
print(f"   过程 PNG: {len(list(PROC.glob('*.png')))} 张  ({proc_kb:.0f} KB)")
print(f"   CSV: {csv_path.name}")
print(f"   JSON: {json_path.name}")
print(f"   输出目录: {OUT}")
print(f"{'='*70}")
