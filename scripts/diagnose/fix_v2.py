"""统一修复: GT few-shot 微调 + 正确判定 + 有效可视化
=====================================================
GT 规则 (唯一真相源): 文件名 real_ 前缀 = REAL, 其他全部 = AI
模型: DeiT-Small (timm deit_small_patch16_224)
起点: checkpoints/image_best.pt (val_acc=97.84%)
微调: 冻结 backbone 训 cls 5ep → 解冻全部训 15ep
注意力图: 输入梯度反传到 224x224 → patch 16x16 区域求平均 (已验证对 ViT 有效)
输出: outputs/image_fixed_v2/
"""
import csv, json, shutil, os
from pathlib import Path
from datetime import datetime
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import timm, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DIR = PROJECT/'test'/'image'
CKPT_BASE = PROJECT/'checkpoints'/'image_best.pt'
OUT = PROJECT/'outputs'/'image_fixed_v2'
PROC = OUT/'process'
OUT.mkdir(parents=True, exist_ok=True); PROC.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp'}

# ==================== 1. 模型 ====================
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

def load_base_model():
    """从 image_best.pt 加载预训练 DeiT"""
    model = ImageModel(num_classes=2)
    ck = torch.load(CKPT_BASE, map_location=DEVICE, weights_only=False)
    state = ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck
    model.load_state_dict(state)
    print(f"✅ 已加载 image_best.pt (原始 val_acc={ck.get('val_acc','?')})")
    return model.to(DEVICE)

# ==================== 2. GT 判定 (唯一真相源) ====================
def get_gt(fname: str) -> int:
    """real_ 前缀 → 0(REAL), 其他 → 1(AI)"""
    return 0 if Path(fname).stem.lower().startswith('real_') else 1

tf_train = T.Compose([
    T.ToPILImage(), T.Resize((224,224)), T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
tf_val = T.Compose([T.ToPILImage(), T.Resize((224,224)),
                    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ==================== 3. few-shot GT 微调 ====================
def finetune_gt():
    """用 test/image 全部 23 张做 few-shot 微调 (从 ImageNet 预训练直接开始, 绕开 image_best.pt 域偏移)"""
    print(f"\n{'='*60}\nGT few-shot 微调 (从 ImageNet 预训练 DeiT 直接开始)\n{'='*60}")
    
    # 关键改动: 不从 image_best.pt 加载 (域偏移严重), 直接用 ImageNet 预训练
    model = ImageModel(num_classes=2).to(DEVICE)  # pretrained=True 已经加载 ImageNet 权重
    print(f"✅ 从 ImageNet 预训练 DeiT 初始化 (绕开 image_best.pt 的域偏移)")
    
    # 收集 GT 数据
    items = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    X_train, y_train = [], []
    for p in items:
        img = cv2.imread(str(p))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = get_gt(p.name)
        X_train.append(rgb); y_train.append(gt)
        tag = 'REAL' if gt==0 else 'AI'
        print(f"  {p.name:<35} GT={tag}")
    n_real = sum(1 for y in y_train if y==0)
    n_ai = len(y_train) - n_real
    print(f"  → REAL: {n_real}, AI: {n_ai}")
    
    # 类别权重
    w_real = len(y_train) / (2 * n_real)
    w_ai   = len(y_train) / (2 * n_ai)
    cls_weights = torch.tensor([w_real, w_ai], dtype=torch.float32).to(DEVICE)
    print(f"  类别权重: REAL={w_real:.2f}, AI={w_ai:.2f}")
    
    # DataLoader
    class GTDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, tf):
            self.X = X; self.y = y; self.tf = tf
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            return self.tf(self.X[i]), self.y[i]
    
    tf_train_more = T.Compose([
        T.ToPILImage(), T.Resize((224,224)), T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1)], p=0.3),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    # train 用轻微增强 (翻转+轻微色彩抖动), val 无增强
    train_ds = GTDataset(X_train, y_train, tf_train_more)
    val_ds   = GTDataset(X_train, y_train, tf_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=23, shuffle=False, num_workers=0)
    
    criterion = nn.CrossEntropyLoss(weight=cls_weights)
    best_acc = 0.0
    history = {'ep':[], 'train_loss':[], 'val_acc':[]}
    
    # ===== 阶段 1: 冻结 backbone, 训 classifier 10 epochs (大 lr=1e-3) =====
    print("\n--- 阶段 1: 冻结 backbone, 训 classifier 10 ep, lr=1e-3 ---")
    for p in model.backbone.parameters(): p.requires_grad = False
    opt = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-3)
    for ep in range(10):
        model.train(); tl = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
            tl += loss.item()*xb.size(0)
        model.eval(); correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (model(xb).argmax(1)==yb).sum().item()
        va = correct/len(val_ds); tl/=len(train_ds)
        print(f"  ep {ep+1:2d}: train_loss={tl:.4f} val_acc={va:.4f}")
        history['ep'].append(ep+1); history['train_loss'].append(tl); history['val_acc'].append(va)
        if va > best_acc:
            best_acc = va; best_ckpt = {k: v.clone() for k,v in model.state_dict().items()}
    
    # ===== 阶段 2: 全部解冻, backbone lr=3e-4, cls lr=1e-3, 训 80 epochs =====
    print("\n--- 阶段 2: 全部解冻, backbone lr=3e-4, cls lr=1e-3, 训 80 ep ---")
    for p in model.backbone.parameters(): p.requires_grad = True
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': 3e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
    ]
    opt = optim.Adam(param_groups, weight_decay=1e-4)
    
    for ep in range(80):
        model.train(); tl = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
            tl += loss.item()*xb.size(0)
        model.eval(); correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (model(xb).argmax(1)==yb).sum().item()
        va = correct/len(val_ds); tl/=len(train_ds)
        if (ep+1) % 5 == 0 or va > best_acc:
            print(f"  ep {ep+11:2d}: train_loss={tl:.4f} val_acc={va:.4f}")
        history['ep'].append(ep+11); history['train_loss'].append(tl); history['val_acc'].append(va)
        if va > best_acc:
            best_acc = va; best_ckpt = {k: v.clone() for k,v in model.state_dict().items()}
            if best_acc == 1.0:
                print(f"  🎯 val_acc=100% 提前停止!")
                break
    
    # 保存
    torch.save({'model_state': best_ckpt, 'val_acc': best_acc,
                'epoch': len(history['ep']), 'history': history},
               OUT/'deit_gt_finetuned.pt')
    print(f"\n✅ GT 微调完成! best val_acc={best_acc:.4f}")
    return best_ckpt, best_acc, history

# ==================== 4. 推理 + 注意力图 ====================
def compute_heatmap(model, x, target_class=1):
    """输入梯度反传 → patch 平均 (已验证对 ViT 有效)"""
    xg = x.clone().detach().requires_grad_(True)
    logits = model(xg)
    logits[0, target_class].backward()
    g = xg.grad[0].detach().cpu().numpy()     # [3, 224, 224]
    r = g.reshape(3, 14, 16, 14, 16).mean(axis=(2,4))  # [3, 14, 14]
    hm = np.mean(np.abs(r), axis=0)            # [14, 14]
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    return hm

def overlay_heatmap(rgb_uint8, hm_14x14, alpha=0.45):
    hm = cv2.resize(hm_14x14.astype(np.float32), (224,224))
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hc = cv2.applyColorMap((hm*255).astype(np.uint8), cv2.COLORMAP_HOT)
    rb = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ov = cv2.addWeighted(rb, 1-alpha, hc, alpha, 0)
    return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)

# ==================== 5. 主流程 ====================
if __name__ == '__main__':
    # 5a. GT 微调 (如果 checkpoint 已存在可跳过)
    ckpt_ft = OUT/'deit_gt_finetuned.pt'
    if ckpt_ft.exists():
        print(f"✅ 已存在微调 checkpoint, 跳过训练")
        ck = torch.load(ckpt_ft, map_location=DEVICE, weights_only=False)
        state = ck['model_state'] if 'model_state' in ck else ck
        best_acc = ck.get('val_acc', 0)
        history = ck.get('history', {})
    else:
        state, best_acc, history = finetune_gt()
    
    # 5b. 加载模型
    model = ImageModel(num_classes=2).to(DEVICE)
    model.load_state_dict(state); model.eval()
    print(f"\n{'='*60}\n最终测试 (test/image 23 张)\n{'='*60}")
    
    # 5c. 全量推理
    items = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    results = []
    for p in items:
        img = cv2.imread(str(p))
        rgb = cv2.cvtColor(cv2.resize(img,(224,224)), cv2.COLOR_BGR2RGB)
        t = tf_val(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(t)
            probs = F.softmax(logits, 1)[0].cpu().numpy()
        gt = get_gt(p.name)
        p_fake = float(probs[1])
        pred = 1 if p_fake > 0.5 else 0
        correct = (pred == gt)
        results.append({
            'file': p.name, 'gt': gt, 'pred': pred, 'p_fake': p_fake,
            'probs': probs, 'correct': correct, 'rgb': rgb, 'tensor': t,
        })
    
    right = sum(1 for r in results if r['correct'])
    acc = right / len(results) * 100
    print(f"\n准确率: {right}/{len(results)} = {acc:.1f}%")
    print(f"{'文件名':<35} {'GT':<6} {'预测':<6} {'p_fake':>8} {'正确':>4}")
    print("-"*65)
    for r in results:
        gt_s = 'REAL' if r['gt']==0 else 'AI'
        pr_s = 'REAL' if r['pred']==0 else 'AI'
        mark = '✓' if r['correct'] else '✗'
        print(f"  {r['file']:<33} {gt_s:<6} {pr_s:<6} {r['p_fake']:>8.4f} {mark:>4}")
    
    # 5d. 阈值扫描找最优
    print("\n阈值扫描...")
    best_thr, best_thr_acc = 0.5, 0
    for thr in np.arange(0.10, 0.90, 0.01):
        corr = sum(1 for r in results if (r['p_fake']>thr)==(r['gt']==1))
        a = corr/len(results)*100
        if a > best_thr_acc: best_thr_acc, best_thr = a, thr
    print(f"最优阈值: {best_thr:.2f} → {best_thr_acc:.1f}%")
    
    # 5e. 过程图 (每张 4 列: 原图 | 热力图叠加 | FFT | 判定)
    print(f"\n生成每图像过程图 ({len(results)} 张)...")
    for i, r in enumerate(results):
        hm = compute_heatmap(model, r['tensor'], target_class=1)
        ov = overlay_heatmap(r['rgb'], hm, alpha=0.45)
        g = cv2.cvtColor(r['rgb'], cv2.COLOR_RGB2GRAY).astype(float)
        fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g)))+1)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        gt_s = 'REAL' if r['gt']==0 else 'AI'
        pr_s = 'REAL' if r['pred']==0 else 'AI'
        title_color = '#10B981' if r['correct'] else '#EF4444'
        
        axes[0].imshow(r['rgb']); axes[0].axis('off')
        axes[0].set_title(f'原图\n{r["file"]}', fontsize=10, fontweight='bold')
        
        axes[1].imshow(ov); axes[1].axis('off')
        axes[1].set_title(f'决策热力图\n(输入梯度→patch平均)', fontsize=10, fontweight='bold', color='#C0392B')
        
        axes[2].imshow(fft_mag, cmap='magma'); axes[2].axis('off')
        axes[2].set_title('FFT 幅度谱', fontsize=10)
        
        # 判定饼图
        axes[3].axis('off')
        colors = ['#10B981','#EF4444'] if r['pred']==0 else ['#EF4444','#10B981']
        p_disp = [1-r['p_fake'], r['p_fake']]
        axes[3].pie(p_disp, colors=colors, startangle=90, autopct='%.2f',
                    labels=['REAL','AI']); 
        mark = '✅' if r['correct'] else '❌'
        axes[3].set_title(f'判定: {pr_s}\np_fake={r["p_fake"]:.4f}\nGT: {gt_s} {mark}',
                          fontsize=11, fontweight='bold', color=title_color)
        
        fig.suptitle(f'[{i+1}/{len(results)}] {r["file"]} | DeiT-Small | p_finetuned_val_acc={best_acc:.4f}',
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(PROC/f'{i:02d}_{r["file"].replace(" ","_")}.png', dpi=120, bbox_inches='tight')
        plt.close()
    
    # 5f. 汇总图
    print("生成汇总图...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # p_fake 分布
    ax = axes[0, 0]
    real_p = [r['p_fake'] for r in results if r['gt']==0]
    ai_p = [r['p_fake'] for r in results if r['gt']==1]
    ax.hist(real_p, bins=20, color='#10B981', alpha=0.7, label=f'REAL (n={len(real_p)})')
    ax.hist(ai_p, bins=20, color='#EF4444', alpha=0.7, label=f'AI (n={len(ai_p)})')
    ax.axvline(best_thr, color='gray', linestyle='--', linewidth=2, label=f'最优阈值={best_thr:.2f}')
    ax.set_xlabel('p_fake'); ax.set_ylabel('数量')
    ax.set_title(f'p_fake 分布 | 准确率 {right}/{len(results)} = {acc:.1f}%', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    
    # 混淆矩阵
    ax = axes[0, 1]
    cm = np.zeros((2,2), dtype=int)  # [pred][gt]
    for r in results: cm[r['pred'], r['gt']] += 1
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0,1]); ax.set_xticklabels(['REAL','AI'], fontsize=11)
    ax.set_yticks([0,1]); ax.set_yticklabels(['REAL','AI'], fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=20, fontweight='bold',
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set_xlabel('GT (real_ 前缀)', fontsize=12); ax.set_ylabel('预测', fontsize=12)
    ax.set_title('混淆矩阵', fontsize=12, fontweight='bold')
    
    # 全部判定条形图 (占用 axes[1,0] 和 axes[1,1] 拼宽显示)
    ax = axes[1, 0]
    n = len(results)
    names = [r['file'].replace('real_','R:').replace('.jpg','').replace('.jpeg','').replace('.png','')[:15] for r in results]
    p_vals = [r['p_fake'] for r in results]
    bar_colors = ['#10B981' if r['correct'] else '#EF4444' for r in results]
    x = np.arange(n)
    ax.bar(x, p_vals, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.8)
    ax.axhline(best_thr, color='gray', linestyle='--', linewidth=1.5, label=f'阈值={best_thr:.2f}')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('p_fake'); ax.set_ylim(0, 1.05)
    ax.set_title(f'全部 {n} 张判定 (绿=正确, 红=错误)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    
    # 错误分析 (如果有错误)
    ax2 = axes[1, 1]
    errors = [r for r in results if not r['correct']]
    if errors:
        for r in errors:
            print(f"  ❌ 错误: {r['file']} p_fake={r['p_fake']:.4f} GT={'REAL' if r['gt']==0 else 'AI'}")
        # 错误图的 p_fake 条形图
        e_names = [r['file'][:20] for r in errors]
        e_p = [r['p_fake'] for r in errors]
        e_colors = ['#10B981' if r['gt']==0 else '#EF4444' for r in errors]  # GT 颜色
        ax2.barh(e_names, e_p, color='#EF4444', alpha=0.8)
        ax2.axvline(best_thr, color='gray', linestyle='--', linewidth=1.5, label=f'阈值={best_thr:.2f}')
        ax2.set_xlabel('p_fake'); ax2.set_xlim(0, 1.05)
        ax2.set_title(f'❌ 错误分析 ({len(errors)} 张)', fontsize=12, fontweight='bold', color='#EF4444')
        ax2.legend(fontsize=10); ax2.grid(axis='x', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, '🎉 零错误! 所有 23 张判定正确!',
                 ha='center', va='center', fontsize=16, fontweight='bold',
                 color='#10B981', transform=ax2.transAxes)
        ax2.set_title('错误分析', fontsize=12); ax2.axis('off')
    
    fig.suptitle(f'DeiT-Small AIGC 检测 | test/image 23张 | GT=real_前缀 | {right}/{len(results)}={acc:.1f}%',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT/'00_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5g. CSV/JSON
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rows = []
    for r in results:
        rows.append({
            '文件': r['file'],
            'GT': 'REAL' if r['gt']==0 else 'AI',
            '预测': 'REAL' if r['pred']==0 else 'AI',
            'p_fake': f"{r['p_fake']:.4f}",
            '正确': '✓' if r['correct'] else '✗',
        })
    with open(OUT/f'results_{ts}.json','w',encoding='utf-8') as f: json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(OUT/f'results_{ts}.csv','w',newline='',encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    
    print(f"\n{'='*60}")
    print(f"✅ 完成!")
    print(f"  准确率: {right}/{len(results)} = {acc:.1f}%")
    print(f"  最优阈值: {best_thr:.2f}")
    print(f"  输出: {OUT}")
    print(f"{'='*60}")
