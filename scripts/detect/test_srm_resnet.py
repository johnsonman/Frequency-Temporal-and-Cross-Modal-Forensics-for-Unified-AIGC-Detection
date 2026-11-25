"""
SRM+ResNet18 模型测试脚本
对 test/image/ 全部图像检测 + 保存可视化
"""
import os, json, csv, sys
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Windows 控制台默认编码可能无法输出本脚本中的中文和符号。
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT = Path(__file__).resolve().parents[2]
CKPT = PROJECT/'checkpoints'/'srm_resnet_best.pt'
CKPT_DIR = PROJECT/'checkpoints'
TEST_DIR = PROJECT/'test'/'image'
OUT = PROJECT/'outputs'/'srm_resnet_results'
PROC = OUT/'process'
OUT.mkdir(parents=True, exist_ok=True)
PROC.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==================== 模型定义 (与训练一致) ====================
class SRMFilterLayer(nn.Module):
    def __init__(self):
        super().__init__()
        srm_kernel = np.array([
            [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
            [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],
            [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
        ], dtype=np.float32)
        srm_kernel[0] /= 4.0; srm_kernel[1] /= 12.0; srm_kernel[2] /= 2.0
        kt = torch.zeros((3,3,5,5), dtype=torch.float32)
        for i in range(3):
            for c in range(3): kt[i,c] = torch.from_numpy(srm_kernel[i])
        self.srm_conv = nn.Conv2d(3,3,5,1,2,bias=False)
        self.srm_conv.weight = nn.Parameter(kt, requires_grad=False)
    def forward(self, x):
        return torch.clamp(self.srm_conv(x), -3.0, 3.0)

class SRMResNetDetector(nn.Module):
    """6 通道输入: RGB(3ch) + SRM 高频噪声(3ch)"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.srm_layer = SRMFilterLayer()
        backbone = models.resnet18(weights=None)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        new_conv.weight.data[:, :3] = old_conv.weight.data
        new_conv.weight.data[:, 3:] = old_conv.weight.data * 0.1
        backbone.conv1 = new_conv
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.single_classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(512, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        noise = self.srm_layer(x)
        combined = torch.cat([x, noise], dim=1)
        feat = self.feature_extractor(combined)
        pooled = torch.flatten(self.avgpool(feat), 1)
        return self.single_classifier(pooled)
    def extract_with_feat(self, x):
        noise = self.srm_layer(x)
        combined = torch.cat([x, noise], dim=1)
        feat = self.feature_extractor(combined)
        pooled = torch.flatten(self.avgpool(feat), 1)
        return feat, self.single_classifier(pooled)

# ==================== 加载模型 ====================
if not CKPT.exists():
    print(f"❌ Checkpoint 不存在: {CKPT}")
    print("请先运行 train_srm_resnet.py 训练模型")
    exit(1)

model = SRMResNetDetector(num_classes=2)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
model.load_state_dict(state)
model.to(device); model.eval()
print(f"✅ 已加载 {CKPT.name}")
print(f"   checkpoint val_acc = {ckpt.get('val_acc','N/A')}")

tf = T.Compose([
    T.ToPILImage(), T.Resize((256,256)),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
# 用于 SRM 可视化 (无 normalize)
tf_raw = T.Compose([T.ToPILImage(), T.Resize((256,256)), T.ToTensor()])

# ==================== 读取测试图像 ====================
exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
files = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in exts])
print(f"\n共发现 {len(files)} 张测试图像")

# ==================== 逐张检测 ====================
results = []
print(f"\n{'='*75}")
print(f"{'文件名':<28} {'p_fake':>7} {'判定':>5} {'GT':>5} {'正确':>5}")
print(f"{'='*75}")

with torch.no_grad():
    for p in files:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = tf(img_rgb).unsqueeze(0).to(device)
        feat, logits = model.extract_with_feat(t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        p_fake = float(probs[1])
        pred_fake = p_fake > 0.5
        gt_real = p.stem.lower().startswith('real_')
        gt_fake = not gt_real
        correct = pred_fake == gt_fake
        results.append({
            'file_name': p.name, 'p_fake': p_fake, 'pred_fake': pred_fake,
            'gt_real': gt_real, 'gt_fake': gt_fake, 'correct': correct, 'path': str(p)
        })
        mark = '✓' if correct else '✗'
        print(f"  {p.name:<26} {p_fake:>7.4f} {'AI' if pred_fake else 'REAL':>5} {'REAL' if gt_real else 'AI':>5} {mark:>5}")

total = len(results); right = sum(1 for r in results if r['correct'])
print(f"\n📊 总准确率: {right}/{total} = {right/total*100:.1f}%")

# ==================== 保存 CSV/JSON ====================
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = OUT/f'test_results_{ts}.csv'
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['文件名', 'p_fake', '判定', 'GT', '正确'])
    for r in results:
        w.writerow([r['file_name'], f"{r['p_fake']:.4f}",
                    'AI' if r['pred_fake'] else 'REAL',
                    'REAL' if r['gt_real'] else 'AI',
                    '✓' if r['correct'] else '✗'])
print(f"✅ CSV: {csv_path.name}")

json_path = OUT/f'test_results_{ts}.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

# ==================== 汇总可视化 ====================
print(f"\n生成可视化...")

# 1. p_fake 柱状图
fig, ax = plt.subplots(figsize=(max(12, total*0.6), 5))
names = [r['file_name'] for r in results]
p_fakes = [r['p_fake'] for r in results]
colors = ['#EF4444' if r['pred_fake'] else '#10B981' for r in results]
edges = ['#000000' if not r['correct'] else '#FFFFFF' for r in results]
widths = [1.8 if not r['correct'] else 0 for r in results]
ax.bar(range(total), p_fakes, color=colors, edgecolor=edges, linewidth=widths)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='阈值 0.5')
ax.set_xticks(range(total))
ax.set_xticklabels([n[:15] for n in names], rotation=60, ha='right', fontsize=8)
ax.set_ylabel('AI 概率 p_fake')
ax.set_title(f'SRM+ResNet18 检测结果  |  准确率 {right}/{total} = {right/total*100:.1f}%',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(['阈值 0.5', '红色=AI / 绿色=REAL', '黑边=判定错误'])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT/'01_p_fake_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. 综合 grid
ncols = 5; nrows = int(np.ceil(total/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
if nrows == 1: axes = axes.reshape(1, -1)
axes_flat = axes.flatten()
for i, r in enumerate(results):
    ax = axes_flat[i]
    img = cv2.imread(r['path'])
    rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    color = '#EF4444' if r['pred_fake'] else '#10B981'
    gt = 'REAL' if r['gt_real'] else 'AI'
    mark = '✓' if r['correct'] else '✗'
    ax.set_title(f"{r['file_name'][:18]}\np_fake={r['p_fake']:.4f} [{gt} {mark}]",
                 fontsize=8, color=color, fontweight='bold')
    ax.axis('off')
for j in range(total, len(axes_flat)): axes_flat[j].axis('off')
fig.suptitle(f'SRM+ResNet18 全部 {total} 张图像检测', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'02_all_images_grid.png', dpi=120, bbox_inches='tight')
plt.close()

# 3. 错误分析
missed = [r for r in results if not r['correct']]
if missed:
    fig, axes = plt.subplots(len(missed), 4, figsize=(16, 4*len(missed)))
    if len(missed)==1: axes = axes.reshape(1,-1)
    for row, r in enumerate(missed):
        img = cv2.imread(r['path'])
        rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
        # SRM noise
        inp = tf_raw(img).unsqueeze(0).to(device)
        noise = model.srm_layer(inp).cpu().numpy()[0]
        noise_mag = np.mean(np.abs(noise), axis=0)
        axes[row, 0].imshow(rgb); axes[row, 0].axis('off')
        axes[row, 0].set_title(f"{r['file_name'][:20]}\np_fake={r['p_fake']:.4f}", fontsize=8, color='#EF4444', fontweight='bold')
        axes[row, 1].imshow(noise_mag, cmap='magma'); axes[row, 1].axis('off')
        axes[row, 1].set_title('SRM 噪声幅度', fontsize=8)
        # heatmap
        inp_norm = tf(img).unsqueeze(0).to(device)
        feat, _ = model.extract_with_feat(inp_norm)
        hm = np.mean(feat[0].detach().cpu().numpy(), axis=0)
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        axes[row, 2].imshow(hm, cmap='jet'); axes[row, 2].axis('off')
        axes[row, 2].set_title('特征热力图', fontsize=8)
        # 概率
        axes[row, 3].bar(['REAL','AI'], [1-r['p_fake'], r['p_fake']], color=['#10B981','#EF4444'])
        axes[row, 3].axhline(0.5, color='gray', linestyle='--')
        axes[row, 3].set_ylim(0,1); axes[row, 3].set_title('概率分布', fontsize=8)
    fig.suptitle(f'⚠️ 检测错误 {len(missed)} 张', fontsize=14, fontweight='bold', color='#EF4444')
    plt.tight_layout()
    plt.savefig(OUT/'03_errors_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

# 4. 混淆矩阵 + 训练曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
cm = np.zeros((2,2), dtype=int)
for r in results:
    pred = 1 if r['pred_fake'] else 0
    gt = 1 if r['gt_fake'] else 0
    cm[gt, pred] += 1
im = ax1.imshow(cm, cmap='Blues', aspect='equal')
ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
ax1.set_xticklabels(['REAL (Pred)', 'AI (Pred)']); ax1.set_yticklabels(['REAL (GT)', 'AI (GT)'])
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=20,
                 color='white' if cm[i,j] > cm.max()/2 else 'black', fontweight='bold')
ax1.set_title('混淆矩阵', fontsize=13, fontweight='bold'); plt.colorbar(im, ax=ax1, fraction=0.046)

# 训练曲线
hist_path = CKPT_DIR/'srm_resnet_history.json'
if hist_path.exists():
    h = json.load(open(hist_path))
    ax2.plot(h['epoch'], h['train_loss'], 'b-', alpha=0.7, label='train_loss')
    ax2.plot(h['epoch'], h['val_loss'], 'r-', linewidth=2, label='val_loss')
    ax2r = ax2.twinx()
    ax2r.plot(h['epoch'], h['train_acc'], 'b--', alpha=0.7, label='train_acc')
    ax2r.plot(h['epoch'], h['val_acc'], 'r--', linewidth=2, label='val_acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2r.set_ylabel('Acc')
    ax2.legend(loc='upper left', fontsize=8); ax2r.legend(loc='center right', fontsize=8)
    best = max(h['val_acc'])
    ax2.set_title(f'训练曲线  best val_acc={best:.4f}', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT/'04_confusion_and_history.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. 每图像过程图 (原图 + SRM噪声 + ResNet 特征热力图 + p_fake 条)
print(f"\n生成每图像检测过程图 ({total} 张)...")
for i, r in enumerate(results):
    img = cv2.imread(r['path'])
    rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
    # SRM noise
    inp_raw = tf_raw(img).unsqueeze(0).to(device)
    noise = model.srm_layer(inp_raw).cpu().numpy()[0]
    noise_mag = np.mean(np.abs(noise), axis=0)
    # ResNet heatmap
    inp_norm = tf(img).unsqueeze(0).to(device)
    feat, _ = model.extract_with_feat(inp_norm)
    hm = np.mean(feat[0].detach().cpu().numpy(), axis=0)
    hm = cv2.resize(hm, (224,224))
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))
    axes[0].imshow(rgb); axes[0].axis('off'); axes[0].set_title('输入', fontsize=10)
    axes[1].imshow(noise_mag, cmap='magma'); axes[1].axis('off'); axes[1].set_title('SRM 高频噪声', fontsize=10)
    axes[2].imshow(hm, cmap='jet'); axes[2].axis('off'); axes[2].set_title('ResNet 特征热力图', fontsize=10)
    axes[3].imshow(cv2.applyColorMap(np.uint8(255*hm), cv2.COLORMAP_JET)); axes[3].axis('off')
    axes[3].set_title('热力图叠加', fontsize=10)
    axes[4].bar(['REAL','AI'], [1-r['p_fake'], r['p_fake']], color=['#10B981','#EF4444'])
    axes[4].axhline(0.5, color='gray', linestyle='--', linewidth=1.5)
    axes[4].set_ylim(0,1); axes[4].set_title(f'p_fake={r["p_fake"]:.4f}', fontsize=10)
    axes[4].set_ylabel('概率')

    badge = '🤖 AI' if r['pred_fake'] else '📷 REAL'
    color = '#EF4444' if r['pred_fake'] else '#10B981'
    gt_str = 'REAL' if r['gt_real'] else 'AI'
    mark = '✅' if r['correct'] else '❌'
    fig.suptitle(f"{badge}  {r['file_name']}  |  GT: {gt_str} {mark}",
                 fontsize=12, fontweight='bold', color=color)
    plt.tight_layout()
    plt.savefig(PROC/f"img_{i:02d}_{r['file_name'].replace(' ','_')[:35]}.png", dpi=150, bbox_inches='tight')
    plt.close()

# ==================== 最终总结 ====================
total_kb = sum(p.stat().st_size for p in OUT.glob('*.png')) / 1024
proc_kb = sum(p.stat().st_size for p in PROC.glob('*.png')) / 1024
print(f"\n{'='*75}")
print(f"✅ SRM+ResNet18 测试完成!")
print(f"   准确率: {right}/{total} = {right/total*100:.1f}%")
print(f"   汇总 PNG: {len(list(OUT.glob('*.png')))} 张 ({total_kb:.0f} KB)")
print(f"   过程 PNG: {len(list(PROC.glob('*.png')))} 张 ({proc_kb:.0f} KB)")
print(f"   CSV: {csv_path.name}")
print(f"   输出: {OUT}")
print(f"{'='*75}")
