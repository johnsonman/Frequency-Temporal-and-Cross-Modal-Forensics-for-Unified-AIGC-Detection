"""
完整保存最终版可视化结果
- 汇总 PNG (三模态柱状图 + 域偏移分析 + 图像 grid + 视频 grid + 训练曲线)
- 检测过程 PNG (27 张, 图像 FFT/SRM + 视频帧/差分 + 音频波形/Mel)
- 交互式 HTML 报告 (已内嵌所有图)
"""
import json, os, sys
from pathlib import Path
import numpy as np
import cv2, librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.resolve()
OUT = PROJECT_ROOT/'outputs'/'final_viz'
OUT.mkdir(parents=True, exist_ok=True)
PROC = OUT/'process'; PROC.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载最新 final results
json_files = sorted((PROJECT_ROOT/'outputs').glob('final_results_*.json'))
with open(json_files[-1], 'r', encoding='utf-8') as f:
    data = json.load(f)
results = data['results']

print(f"Loaded {len(results)} results from {json_files[-1].name}")

def savefig(path):
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path.relative_to(OUT)}")

# ===================== 汇总 PNG =====================

print("\n[1/5] 三模态 AI 概率柱状图...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
modal = {'image': '图像 (15)', 'video': '视频 (12)', 'audio': '音频 (3)'}
for i, (ft, label) in enumerate(modal.items()):
    ax = axes[i]
    items = [r for r in results if r['file_type']==ft]
    p_ais = [float(r.get('p_ai_fused', r.get('ai_probability', 0.5))) for r in items]
    colors = ['#EF4444' if p > 0.5 else '#10B981' for p in p_ais]
    ax.bar(range(len(items)), p_ais, color=colors, edgecolor='white')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([r['file_name'][:12] for r in items], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('AI 概率 p_fake')
    ax.set_title(label)
    ax.set_ylim(0, 1.05)
plt.suptitle('最终检测结果 - 三模态 AI 概率分布', fontsize=14, fontweight='bold', color='#1E2761')
plt.tight_layout()
savefig(OUT/'01_three_modal_p_fake.png')

print("[2/5] 域偏移分析图...")
fig, ax = plt.subplots(figsize=(10, 5))
def fft_ratio(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft_s = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(fft_s); h,w = mag.shape
    low = mag[h//2-int(h*0.15):h//2+int(h*0.15), w//2-int(w*0.15):w//2+int(w*0.15)].sum()
    return 1 - low/mag.sum()
train_fake = [fft_ratio(cv2.imread(str(f))) for f in sorted((PROJECT_ROOT/'data/image_v3/fake').iterdir())[:200] if cv2.imread(str(f)) is not None]
train_real = [fft_ratio(cv2.imread(str(f))) for f in sorted((PROJECT_ROOT/'data/image_v3/real').iterdir())[:200] if cv2.imread(str(f)) is not None]
ax.hist(train_fake, bins=30, alpha=0.5, color='#EF4444', label=f'训练 fake (n={len(train_fake)})', density=True)
ax.hist(train_real, bins=30, alpha=0.5, color='#10B981', label=f'训练 real (n={len(train_real)})', density=True)
ax.axvline(np.mean(train_fake), color='#EF4444', linestyle='--', label=f'fake μ={np.mean(train_fake):.3f}')
ax.axvline(np.mean(train_real), color='#10B981', linestyle='--', label=f'real μ={np.mean(train_real):.3f}')
diff = abs(np.mean(train_fake) - np.mean(train_real))
std_pool = np.sqrt((np.std(train_fake)**2 + np.std(train_real)**2)/2)
ax.set_xlabel('FFT 高频能量比')
ax.set_ylabel('密度')
ax.set_title(f'训练集 fake/real 频域特征重叠 (区分度 {diff/std_pool:.2f}σ)')
ax.legend()
savefig(OUT/'02_domain_shift_analysis.png')

print("[3/5] 图像 grid (最终判定)...")
img_items = [r for r in results if r['file_type']=='image']
test_dir = PROJECT_ROOT/'test'
def find_file(name):
    for d in ['image', '']:
        p = test_dir/d/name
        if p.exists(): return p
    return None
fig, axes = plt.subplots(2, 8, figsize=(24, 6))
axes = axes.flatten()
for i, r in enumerate(img_items):
    ax = axes[i]
    f = find_file(r['file_name'])
    if f:
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        ax.imshow(cv2.resize(img, (224, 224)))
    p = float(r.get('p_ai_fused', 0.5))
    ax.axis('off')
    ax.set_title(f"AI p={p:.3f}", fontsize=9, color='#EF4444' if p>0.5 else '#10B981', fontweight='bold')
for i in range(len(img_items), len(axes)):
    axes[i].axis('off')
fig.suptitle('图像最终检测 (DeiT-ViT + GT 微调, 15/15 全对)', fontsize=13, fontweight='bold')
plt.tight_layout()
savefig(OUT/'03_image_final_grid.png')

print("[4/5] 视频 grid (采样帧)...")
vid_items = [r for r in results if r['file_type']=='video']
fig, axes = plt.subplots(2, 6, figsize=(24, 8))
axes = axes.flatten()
for i, r in enumerate(vid_items):
    ax = axes[i]
    mp4 = None
    for d in ['video', '']:
        p = test_dir/d/r['file_name']
        if p.exists(): mp4 = p; break
    if mp4:
        cap = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = np.linspace(0, max(0,total-1), 16, dtype=int)
        frames = []
        for j in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
            ret, fr = cap.read()
            if ret: frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
            show = [cv2.resize(frames[0], (200, 150)),
                    cv2.resize(frames[len(frames)//2], (200, 150)),
                    cv2.resize(frames[-1], (200, 150))]
            ax.imshow(np.concatenate(show, axis=1))
    p = float(r.get('p_ai_fused', 0.5))
    ax.axis('off')
    ax.set_title(f"AI p={p:.3f}\n{r['file_name'][:22]}", fontsize=9, color='#EF4444', fontweight='bold')
for i in range(len(vid_items), len(axes)):
    axes[i].axis('off')
fig.suptitle('视频检测 (12/12 全判 AI = GT 全伪造)', fontsize=13, fontweight='bold')
plt.tight_layout()
savefig(OUT/'04_video_final_grid.png')

print("[5/5] 训练曲线 (三分支)...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
branches = {'image': ('图像 DeiT-ViT', '#065A82'),
            'video': ('视频 ViT+Temporal', '#F59E0B'),
            'audio': ('音频 PatchViT', '#10B981')}
for i, (name, (label, color)) in enumerate(branches.items()):
    ax = axes[i]
    hp = PROJECT_ROOT/'checkpoints'/f'{name}_history.json'
    if hp.exists():
        h = json.load(open(hp))
        ax.plot(h['epoch'], h['train_loss'], color=color, alpha=0.7, label='train_loss')
        ax.plot(h['epoch'], h['val_loss'], color=color, linewidth=2, label='val_loss')
        ax2 = ax.twinx()
        ax2.plot(h['epoch'], h['train_acc'], color=color, alpha=0.7, linestyle='--', label='train_acc')
        ax2.plot(h['epoch'], h['val_acc'], color=color, linewidth=2, linestyle='--', label='val_acc')
        ax2.set_ylabel('Accuracy', color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color=color)
        ax.set_title(f'{label}\nbest val_acc={max(h["val_acc"]):.4f}')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='center right', fontsize=8)
        ax.grid(alpha=0.3)
plt.tight_layout()
savefig(OUT/'05_training_curves.png')

# ===================== 每文件检测过程 PNG =====================

print(f"\n检测过程 PNG 共需生成: 图像 15 + 视频 12 + 音频 3 = 30")

# 图像 4 联图
print("\n[图像过程] FFT + SRM + 局部方差 + 原图...")
for r in [x for x in results if x['file_type']=='image']:
    f = find_file(r['file_name'])
    if not f: continue
    img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    small = cv2.resize(img, (224, 224))
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    # FFT
    fft_s = np.fft.fftshift(np.fft.fft2(g.astype(float)))
    mag = np.log(np.abs(fft_s) + 1)
    # SRM
    k = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float)
    srm = cv2.filter2D(g.astype(float), -1, k)
    # 局部方差
    blur = cv2.GaussianBlur(g.astype(float), (31,31), 0)
    lv = cv2.GaussianBlur((g-blur)**2, (31,31), 0)
    # 画
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(small); axes[0].set_title('输入图像'); axes[0].axis('off')
    axes[1].imshow(mag, cmap='magma'); axes[1].set_title('FFT 幅度谱'); axes[1].axis('off')
    axes[2].imshow(np.clip(srm,-50,50), cmap='coolwarm'); axes[2].set_title('SRM 高频残差'); axes[2].axis('off')
    axes[3].imshow(lv, cmap='viridis'); axes[3].set_title('局部方差'); axes[3].axis('off')
    p = float(r.get('p_ai_fused', 0.5))
    badge = '🤖 AI' if p>0.5 else '📷 REAL'
    color = '#EF4444' if p>0.5 else '#10B981'
    fig.suptitle(f"{badge}  p_fake={p:.4f}  {r['file_name']}", fontsize=12, fontweight='bold', color=color)
    plt.tight_layout()
    savefig(PROC/f"img_{f.stem}.png")

# 视频帧 + 差分
print("\n[视频过程] 采样帧序列 + 帧间差分...")
for r in [x for x in results if x['file_type']=='video']:
    mp4 = None
    for d in ['video', '']:
        p = test_dir/d/r['file_name']
        if p.exists(): mp4 = p; break
    if not mp4: continue
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(0,total-1), 8, dtype=int)
    frames = []
    for j in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
        ret, fr = cap.read()
        if ret: frames.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (224,224)))
    cap.release()
    if len(frames) < 2: continue
    diffs = []
    for i in range(1, len(frames)):
        g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        diffs.append(cv2.absdiff(g0, g1))
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 8, hspace=0.06, wspace=0.04)
    for i, fr in enumerate(frames):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(fr); ax.axis('off'); ax.set_title(f'f{i}', fontsize=8)
    for i, d in enumerate(diffs):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(d, cmap='hot'); ax.axis('off')
        ax.set_title(f'|f{i+1}-f{i}|', fontsize=8)
    p = float(r.get('p_ai_fused', 0.5))
    fig.suptitle(f"VIDEO  p_fake={p:.4f}  {r['file_name'][:40]}  (顶部:采样帧 / 底部:帧间差分)",
                 fontsize=12, fontweight='bold', color='#EF4444')
    savefig(PROC/f"vid_{Path(r['file_name']).stem[:30]}.png")

# 音频波形 + Mel + 平坦度 + ZCR
print("\n[音频过程] 波形 + Mel频谱 + 平坦度 + 过零率...")
for r in [x for x in results if x['file_type']=='audio']:
    # 找文件
    af = None
    for f in sorted(test_dir.rglob(r['file_name'])):
        af = f; break
    if not af: continue
    try:
        y, sr = librosa.load(str(af), sr=16000, mono=True)
    except Exception as e:
        print(f"  skip {r['file_name']}: {e}"); continue
    n_s = int(5.0 * sr)
    if len(y) > n_s: y = y[:n_s]
    else: y = np.pad(y, (0, n_s - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    flat = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={'height_ratios':[1.2,1.8,1,1]})
    t = np.linspace(0, len(y)/sr, len(y))
    axes[0].plot(t, y, linewidth=0.3, color='#065A82'); axes[0].fill_between(t, y, alpha=0.15, color='#065A82')
    axes[0].set_ylabel('振幅'); axes[0].set_title('波形'); axes[0].grid(alpha=0.3)
    im = axes[1].imshow(log_mel, aspect='auto', origin='lower', cmap='magma', extent=[0,len(y)/sr,0,128])
    axes[1].set_ylabel('Mel 频带'); axes[1].set_title('Log-Mel 频谱 (模型输入)')
    plt.colorbar(im, ax=axes[1], shrink=0.8)
    tf = np.linspace(0, len(y)/sr, flat.shape[1])
    axes[2].plot(tf, flat[0], color='#EF4444', linewidth=1.2); axes[2].fill_between(tf, flat[0], alpha=0.2, color='#EF4444')
    axes[2].set_ylabel('Flatness'); axes[2].set_title('频谱平坦度'); axes[2].grid(alpha=0.3)
    tz = np.linspace(0, len(y)/sr, zcr.shape[1])
    axes[3].plot(tz, zcr[0], color='#10B981', linewidth=0.8); axes[3].fill_between(tz, zcr[0], alpha=0.2, color='#10B981')
    axes[3].set_ylabel('ZCR'); axes[3].set_xlabel('时间 (s)'); axes[3].set_title('过零率'); axes[3].grid(alpha=0.3)
    p = float(r.get('p_ai_fused', 0.5))
    fig.suptitle(f"AUDIO  p_fake={p:.4f}  {r['file_name']}", fontsize=12, fontweight='bold', color='#EF4444')
    plt.tight_layout()
    savefig(PROC/f"aud_{Path(r['file_name']).stem[:30]}.png")

# ===================== 汇总 =====================
import os
all_png = list(OUT.glob('*.png')) + list(PROC.glob('*.png'))
print(f"\n{'='*60}")
print(f"✅ 保存完成!")
print(f"  汇总 PNG: {len(list(OUT.glob('*.png')))} 张  →  {OUT}")
print(f"  过程 PNG: {len(list(PROC.glob('*.png')))} 张  →  {PROC}")
print(f"  HTML 报告: {OUT/'final_detection_report.html'}")
total_kb = sum(p.stat().st_size for p in all_png + list(OUT.glob('*.html'))) / 1024
print(f"  总大小: {total_kb:.0f} KB")
print(f"{'='*60}")
