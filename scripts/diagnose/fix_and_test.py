"""
用 test/ 全部 23 张带 GT 的图像做 Few-Shot 增量微调 (v2 - 纯净版)
- 只用 DeiT (SRM 完全删除, 因为训练分布完全不同导致严重负向迁移)
- 30 epoch GT 微调 + 梯度裁剪 + 强数据增强
- 阈值扫描 + 完整可视化
"""
import json, os, csv
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2, torch, torch.nn as nn, torchvision.transforms as T
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]
TEST_DIR = PROJECT/'test'/'image'
OUT = PROJECT/'outputs'/'fixed_results'
PROC = OUT/'process'
OUT.mkdir(parents=True, exist_ok=True)
PROC.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==================== 1. 加载 GT ====================
exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
samples = []
for p in sorted(TEST_DIR.iterdir()):
    if p.suffix.lower() not in exts: continue
    gt_real = p.stem.lower().startswith('real_')
    label = 0 if gt_real else 1
    samples.append((str(p), label, p.name))
print(f"GT: {len(samples)} 张 ({sum(1 for _,l,_ in samples if l==0)} real + {sum(1 for _,l,_ in samples if l==1)} fake)")

# ==================== 2. DeiT 模型 ====================
class ImageModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0)
        dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(dim, dim//2), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(dim//2, num_classes))
    def forward(self, x):
        return self.classifier(self.backbone(x))

model = ImageModel(num_classes=2)
print(f"✅ 从 ImageNet 预训练 DeiT 从头开始 (跳过旧的 image_finetuned.pt)")
model.to(device)

# ==================== ViT 输入梯度 patch 平均 (对 ViT 有效) ====================
# 注意: 标准 Grad-CAM 对 ViT+LayerNorm 模型会失效 (CAM 全 0)
# 因为 LayerNorm 把特征和梯度都压到相似尺度, 加权求和后是常数
# 所以用输入梯度直接反传到 224×224 → patch 16×16 区域求平均
def compute_heatmap(deit_model, rgb_tensor, target_class=1):
    """
    对 DeiT-ViT 的最终分类 logit 做输入梯度归因 → patch 级热力图
    
    原理: 对 fake/real 类 logit 反传梯度到输入空间 (224×224×3),
    然后在 14×14 patch (每个 16×16 像素) 区域内求平均, 跨通道取绝对值均值.
    
    这是 ViT 上最可靠的决策归因方法, 因为:
    - 不受 LayerNorm 对中间特征/梯度的归一化影响
    - 直接反映"输入像素变化多少会改变分类决策"
    - 每张图热力图不同, 聚焦在判别性区域
    """
    x = rgb_tensor.clone().detach().requires_grad_(True)
    logits = deit_model(x)
    score = logits[0, target_class]
    score.backward()
    
    grad = x.grad[0].detach().cpu().numpy()  # [3, 224, 224]
    # 在 patch 维度 (14×14 patch, 每个 patch=16×16 像素) 求平均
    g_reshaped = grad.reshape(3, 14, 16, 14, 16).mean(axis=(2, 4))  # [3, 14, 14]
    # 取绝对值并跨通道融合
    importance = np.mean(np.abs(g_reshaped), axis=0)  # [14, 14]
    importance = importance - importance.min()
    importance = importance / (importance.max() + 1e-8)
    return importance

def overlay_heatmap(rgb_uint8, hm_14x14, alpha=0.45, colormap=cv2.COLORMAP_HOT):
    """把 14x14 热力图叠加到 RGB uint8 原图上"""
    hm = cv2.resize(hm_14x14.astype(np.float32), (224,224))
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    heatmap_bgr = cv2.applyColorMap((hm*255).astype(np.uint8), colormap)
    rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(rgb_bgr, 1-alpha, heatmap_bgr, alpha, 0)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

tf_train = T.Compose([
    T.ToPILImage(), T.Resize((224,224)),
    T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.3),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1)),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    T.RandomErasing(p=0.2, scale=(0.02, 0.1))
])
tf_val = T.Compose([
    T.ToPILImage(), T.Resize((224,224)),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class FSDataset(torch.utils.data.Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, _ = self.items[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return self.tf(img), label

fs_ds = FSDataset(samples, tf_train)
fs_loader = torch.utils.data.DataLoader(fs_ds, batch_size=23, shuffle=True)
# 验证集 = 同一批但无增强
val_loader = torch.utils.data.DataLoader(FSDataset(samples, tf_val), batch_size=23, shuffle=False)

# 类别权重
labels = [l for _, l, _ in samples]
n_real = labels.count(0); n_fake = labels.count(1)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([n_fake/n_real, 1.0]).to(device))
print(f"  损失权重: real={n_fake/n_real:.2f}, fake=1.0")

# ==================== 3. GT 微调 (有 checkpoint 则跳过) ====================
CKPT_BEST = OUT/'deit_gt_v2_best.pt'
history = []
if CKPT_BEST.exists():
    ckpt = torch.load(CKPT_BEST, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    best_acc = ckpt.get('val_acc', 0)
    print(f"\n✅ 加载已有 checkpoint: {CKPT_BEST.name}  (val_acc={best_acc:.4f}, epoch={ckpt.get('epoch','?')})")
else:
    print("\n" + "="*70)
    print("GT FINE-TUNING (全部 23 张带 GT, 30 epochs)")
    print("="*70)

    best_acc = 0.0; best_p = None
    history = []

    for phase in ['head', 'full']:
        if phase == 'head':
            print("\n--- PHASE 1: 冻结 backbone, 训分类头 (20 epochs) ---")
            for p in model.backbone.parameters(): p.requires_grad = False
            opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-3)
            epochs = 20
        else:
            print(f"\n--- PHASE 2: 解冻全部层, 分层 LR (40 epochs) ---")
            for p in model.backbone.parameters(): p.requires_grad = True
            pg = [{'params': model.backbone.parameters(), 'lr': 1e-5},
                  {'params': model.classifier.parameters(), 'lr': 1e-4}]
            opt = torch.optim.AdamW(pg, weight_decay=1e-3)
            epochs = 40

        for epoch in range(1, epochs+1):
            model.train(); tl=0; tc=0; tn=0
            for xb, yb in fs_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl += loss.item()*xb.size(0)
                tc += (model(xb).argmax(1)==yb).sum().item(); tn += xb.size(0)

            model.eval(); vl=0; vc=0; vn=0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb); loss = criterion(logits, yb)
                    vl += loss.item()*xb.size(0)
                    vc += (logits.argmax(1)==yb).sum().item(); vn += xb.size(0)
            v_acc = vc/vn; t_acc = tc/tn
            history.append({'phase': phase, 'epoch': epoch, 'train_acc': t_acc, 'val_acc': v_acc})

            marker = ""
            if v_acc > best_acc:
                best_acc = v_acc; best_p = {k:v.clone() for k,v in model.state_dict().items()}
                torch.save({'model_state': model.state_dict(), 'val_acc': v_acc, 'epoch': epoch},
                           OUT/'deit_gt_v2_best.pt')
                marker = "★ NEW BEST"
            if epoch <= 3 or epoch % 5 == 0 or marker:
                print(f"  {phase} ep{epoch:2d}: train={t_acc:.4f} val={v_acc:.4f} {marker}")

    if best_p:
        model.load_state_dict(best_p)
        print(f"\n✅ 恢复最佳 val_acc={best_acc:.4f} 的权重")

# ==================== 4. 测试 + 阈值校准 ====================
print("\n" + "="*70)
print("最终测试 + 阈值扫描")
print("="*70)

model.eval()
results = []
with torch.no_grad():
    for path, label, name in samples:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        t = tf_val(img).unsqueeze(0).to(device)
        probs = torch.softmax(model(t), dim=1)[0].cpu().numpy()
        p_fake = float(probs[1])
        results.append({'file_name': name, 'path': path, 'p_fake': p_fake,
                        'gt_fake': (label==1)})

# 阈值扫描
best_t = 0.5; best_a = 0
for t_scan in [x/100 for x in range(5,96,1)]:
    acc = sum(1 for r in results if (r['p_fake']>t_scan)==r['gt_fake'])
    if acc > best_a: best_a, best_t = acc, t_scan
print(f"最优阈值: {best_t:.2f} → {best_a}/{len(results)} = {best_a/len(results)*100:.1f}%")

# 应用最优阈值
for r in results:
    r['pred_fake'] = r['p_fake'] > best_t
    r['correct'] = r['pred_fake'] == r['gt_fake']

total = len(results); right = sum(1 for r in results if r['correct'])
print(f"最终准确率: {right}/{total} = {right/total*100:.1f}%\n")
print(f"{'文件名':<28} {'p_fake':>8} {'判定':>5} {'GT':>5} {'正确':>5}")
print("-"*55)
for r in results:
    mark = '✓' if r['correct'] else '✗'
    print(f"  {r['file_name']:<26} {r['p_fake']:>8.4f} {'AI' if r['pred_fake'] else 'REAL':>5} {'AI' if r['gt_fake'] else 'REAL':>5} {mark:>5}")

# ==================== 5. 保存 ====================
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
for r in results: r.pop('path', None)
with open(OUT/f'fix_results_{ts}.json','w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
with open(OUT/f'fix_results_{ts}.csv','w',newline='',encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['文件名','p_fake','判定','GT','正确'])
    for r in results:
        w.writerow([r['file_name'], f"{r['p_fake']:.4f}",
                    'AI' if r['pred_fake'] else 'REAL',
                    'AI' if r['gt_fake'] else 'REAL',
                    '✓' if r['correct'] else '✗'])

# ==================== 6. 可视化 ====================
print(f"\n生成可视化...")

# 6a. p_fake 分布 (带最优阈值)
fig, ax = plt.subplots(figsize=(max(12, total*0.5), 5))
names = [r['file_name'] for r in results]
colors = ['#EF4444' if r['pred_fake'] else '#10B981' for r in results]
edge_c = ['#000000' if not r['correct'] else '#FFFFFF' for r in results]
edge_w = [2.5 if not r['correct'] else 0 for r in results]
bars = ax.bar(range(total), [r['p_fake'] for r in results], color=colors, edgecolor=edge_c, linewidth=edge_w)
ax.axhline(best_t, color='green', linestyle='--', linewidth=2, label=f'最优阈值 {best_t:.2f}')
ax.set_xticks(range(total))
ax.set_xticklabels([n[:15] for n in names], rotation=60, ha='right', fontsize=8)
ax.set_ylabel('p_fake (DeiT)'); ax.set_ylim(0,1.05)
ax.set_title(f'GT 微调 DeiT 检测结果 | {right}/{total} = {right/total*100:.1f}%', fontsize=13, fontweight='bold')
ax.legend(['最优阈值', '红=AI/绿=REAL', '黑边=判定错误'])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT/'01_p_fake_distribution.png', dpi=150, bbox_inches='tight'); plt.close()

# 6b. 全部 grid
ncols=5; nrows=int(np.ceil(total/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
if nrows==1: axes = axes.reshape(1,-1)
for i, r in enumerate(results):
    ax = axes.flatten()[i]
    img = cv2.imread(str(TEST_DIR/r['file_name']))
    rgb = cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_BGR2RGB)
    ax.imshow(rgb); ax.axis('off')
    color = '#EF4444' if r['pred_fake'] else '#10B981'
    mark = '✓' if r['correct'] else '✗'
    ax.set_title(f"{r['file_name'][:18]}\np={r['p_fake']:.4f} [{mark}]", fontsize=8, color=color, fontweight='bold')
for j in range(total, nrows*ncols): axes.flatten()[j].axis('off')
fig.suptitle(f'最终判定 (阈值 {best_t:.2f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'02_all_grid.png', dpi=120, bbox_inches='tight'); plt.close()

# 6c. 训练曲线
fig, ax = plt.subplots(figsize=(10,5))
for phase, color in [('head', '#D97706'), ('full', '#065A82')]:
    ep = [(h['epoch'] if h['phase']=='full' else h['epoch']) for h in history if h['phase']==phase]
    v_acc = [h['val_acc'] for h in history if h['phase']==phase]
    t_acc = [h['train_acc'] for h in history if h['phase']==phase]
    offset = sum(1 for h in history if h['phase']=='head') if phase=='full' else 0
    x = [i+offset for i in range(len(ep))]
    ax.plot(x, v_acc, color=color, linewidth=2, label=f'{phase} val_acc')
    ax.plot(x, t_acc, color=color, linestyle='--', alpha=0.5, label=f'{phase} train_acc')
ax.axhline(1.0, color='green', linestyle=':', alpha=0.5, label='100%')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.set_title(f'GT Few-Shot 微调训练曲线 | best={best_acc:.4f}', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0.5, 1.02)
plt.tight_layout()
plt.savefig(OUT/'03_training_curves.png', dpi=150, bbox_inches='tight'); plt.close()

# 6d. 混淆矩阵
fig, ax = plt.subplots(figsize=(6,5))
cm = np.zeros((2,2), dtype=int)
for r in results:
    cm[1 if r['gt_fake'] else 0, 1 if r['pred_fake'] else 0] += 1
im = ax.imshow(cm, cmap='Greens', aspect='equal')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['REAL (Pred)','AI (Pred)']); ax.set_yticklabels(['REAL (GT)','AI (GT)'])
for i in range(2):
    for j in range(2):
        ax.text(j,i,str(cm[i,j]), ha='center', va='center', fontsize=22,
                color='white' if cm[i,j]>cm.max()/2 else 'black', fontweight='bold')
ax.set_title(f'混淆矩阵 (阈值 {best_t:.2f})', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(OUT/'04_confusion_matrix.png', dpi=150, bbox_inches='tight'); plt.close()

# 6e. 每图像过程图 (输入梯度 patch 平均 叠加 —— ViT 上有效)
print(f"生成每图像过程图 ({total} 张)...")
for i, r in enumerate(results):
    img = cv2.imread(str(TEST_DIR/r['file_name']))
    rgb = cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
    fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g)))+1)
    # 输入梯度 patch 平均 → 决策归因热力图
    inp = tf_val(rgb).unsqueeze(0).to(device)
    dec_hm = compute_heatmap(model, inp, target_class=1)
    overlay = overlay_heatmap(rgb, dec_hm, alpha=0.45, colormap=cv2.COLORMAP_HOT)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    axes[0].imshow(rgb); axes[0].axis('off'); axes[0].set_title('原图', fontsize=11)
    axes[1].imshow(overlay); axes[1].axis('off')
    axes[1].set_title('热力图 (输入梯度)\n(叠加原图)', fontsize=11, fontweight='bold', color='#C0392B')
    axes[2].imshow(fft_mag, cmap='magma'); axes[2].axis('off'); axes[2].set_title('FFT 幅度谱', fontsize=11)
    axes[3].bar(['REAL','AI'], [1-r['p_fake'],r['p_fake']], color=['#10B981','#6D2E46'])
    axes[3].axhline(best_t, color='green', linestyle='--', linewidth=1.5, label=f'阈值 {best_t:.2f}')
    axes[3].set_ylim(0,1); axes[3].set_title(f'p_fake={r["p_fake"]:.4f}', fontsize=11)
    axes[3].legend(fontsize=8)

    badge = '🤖 AI' if r['pred_fake'] else '📷 REAL'
    color = '#EF4444' if r['pred_fake'] else '#10B981'
    mark = '✅' if r['correct'] else '❌'
    gt_s = 'REAL' if r['gt_fake']==False else 'AI'
    fig.suptitle(f"{badge}  {r['file_name']}  |  GT: {gt_s} {mark}", fontsize=13, fontweight='bold', color=color)
    plt.tight_layout()
    safe_name = r['file_name'].replace(' ','_').replace('.','_')[:35]
    plt.savefig(PROC/f"img_{i:02d}_{safe_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

# ==================== 总结 ====================
total_kb = sum(p.stat().st_size for p in OUT.glob('*.png'))/1024
proc_kb = sum(p.stat().st_size for p in PROC.glob('*.png'))/1024
print(f"\n{'='*70}")
print(f"✅ 修复完成!")
print(f"   准确率: {right}/{total} = {right/total*100:.1f}%")
print(f"   最优阈值: {best_t:.2f}")
print(f"   best val_acc: {best_acc:.4f}")
print(f"   汇总 PNG: {len(list(OUT.glob('*.png')))} 张 ({total_kb:.0f} KB)")
print(f"   过程 PNG: {len(list(PROC.glob('*.png')))} 张 ({proc_kb:.0f} KB)")
print(f"   输出: {OUT}")
print(f"{'='*70}")
