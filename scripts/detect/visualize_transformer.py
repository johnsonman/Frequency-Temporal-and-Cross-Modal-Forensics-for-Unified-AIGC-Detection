"""
Transformer 检测结果可视化 — 包含检测过程
生成: 汇总图表 PNG + 各文件检测过程 PNG + 交互式 HTML 报告
"""

import os
import json
import sys
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'transformer_viz'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = OUTPUT_DIR / 'process'
PROC_DIR.mkdir(exist_ok=True)

# 找最新的 transformer 结果
json_files = sorted(OUTPUT_DIR.parent.glob('transformer_results_*.json'))
if not json_files:
    print("错误: 找不到 transformer_results_*.json")
    sys.exit(1)
JSON_PATH = json_files[-1]
print(f"使用结果: {JSON_PATH}")

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    results = json.load(f)

# Matplotlib CJK
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 汇总图表
# ============================================================

def plot_summary():
    """生成汇总图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('多模态 AIGC 检测汇总  (Transformer, DeiT-ViT 图像 Val Acc 97.8%)',
                 fontsize=16, fontweight='bold', color='#1E2761')

    # --- (1) 各模态 AI 概率分布 ---
    ax = axes[0, 0]
    modal_colors = {'image': '#065A82', 'video': '#F59E0B', 'audio': '#34D399'}
    modal_labels = {'image': '图像 (15)', 'video': '视频 (12)', 'audio': '音频 (3)'}
    for ft in ['image', 'video', 'audio']:
        probs = [r['ai_probability'] for r in results if r['file_type'] == ft]
        if probs:
            ax.hist(probs, bins=10, alpha=0.7, label=modal_labels[ft],
                    color=modal_colors[ft], edgecolor='white')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='判定阈值 0.5')
    ax.set_xlabel('AI 概率 (p_fake)')
    ax.set_ylabel('文件数')
    ax.set_title('各模态 AI 概率分布')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    # --- (2) 判定饼图 ---
    ax = axes[0, 1]
    modal_stats = {}
    for r in results:
        ft = r['file_type']
        if ft not in modal_stats:
            modal_stats[ft] = {'ai': 0, 'real': 0}
        if r['is_ai']:
            modal_stats[ft]['ai'] += 1
        else:
            modal_stats[ft]['real'] += 1

    # 子图嵌套: 一个大图 + 两个小图
    sizes = [sum(s['ai'] for s in modal_stats.values()),
             sum(s['real'] for s in modal_stats.values())]
    ax.pie(sizes, labels=[f'AI ({sizes[0]})', f'真实 ({sizes[1]})'],
           colors=['#EF4444', '#10B981'], autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title('全局判定分布')

    # --- (3) 置信度分布 ---
    ax = axes[0, 2]
    for ft in ['image', 'video', 'audio']:
        confs = [r['confidence'] for r in results if r['file_type'] == ft]
        if confs:
            ax.hist(confs, bins=10, alpha=0.7, label=modal_labels[ft],
                    color=modal_colors[ft], edgecolor='white')
    ax.axvline(0.7, color='#F96167', linestyle='--', alpha=0.7, label='高置信度 0.7')
    ax.set_xlabel('置信度')
    ax.set_ylabel('文件数')
    ax.set_title('预测置信度分布')
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, 1.01)

    # --- (4) 按文件的 AI 概率柱状图 (图像 only, 最可信) ---
    ax = axes[1, 0]
    img_results = [r for r in results if r['file_type'] == 'image']
    img_names = [os.path.splitext(r['file_name'])[0][:12] for r in img_results]
    img_probs = [r['ai_probability'] for r in img_results]
    img_colors = ['#EF4444' if p > 0.5 else '#10B981' for p in img_probs]
    bars = ax.bar(range(len(img_results)), img_probs, color=img_colors, edgecolor='white')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(img_results)))
    ax.set_xticklabels(img_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('p_fake')
    ax.set_title('图像分支 AI 概率 (⭐可信)')
    ax.set_ylim(0, 1.05)

    # --- (5) 视频 AI 概率 ---
    ax = axes[1, 1]
    vid_results = [r for r in results if r['file_type'] == 'video']
    vid_names = [r['file_name'][:25] for r in vid_results]
    vid_probs = [r['ai_probability'] for r in vid_results]
    vid_colors = ['#EF4444' if p > 0.5 else '#10B981' for p in vid_probs]
    ax.bar(range(len(vid_results)), vid_probs, color=vid_colors, edgecolor='white')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(vid_results)))
    ax.set_xticklabels(range(len(vid_results)), fontsize=8)
    ax.set_ylabel('p_fake')
    ax.set_title('视频分支 AI 概率 (⚠过拟合)')
    ax.set_ylim(0, 1)

    # --- (6) 音频 AI 概率 ---
    ax = axes[1, 2]
    aud_results = [r for r in results if r['file_type'] == 'audio']
    aud_names = [r['file_name'][:20] for r in aud_results]
    aud_probs = [r['ai_probability'] for r in aud_results]
    aud_colors = ['#EF4444' if p > 0.5 else '#10B981' for p in aud_probs]
    ax.bar(range(len(aud_results)), aud_probs, color=aud_colors, edgecolor='white')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(aud_results)))
    ax.set_xticklabels(aud_names, rotation=30, fontsize=8)
    ax.set_ylabel('p_fake')
    ax.set_title('音频分支 AI 概率')
    ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUTPUT_DIR / 'summary_overview.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


def plot_per_file_scores():
    """所有文件分数列表图"""
    fig, ax = plt.subplots(figsize=(16, 10))

    # 按模态分组 + 按文件排序
    ft_order = {'image': 0, 'video': 1, 'audio': 2}
    sorted_results = sorted(results, key=lambda r: (ft_order.get(r['file_type'], 9), r['file_name']))

    y_pos = np.arange(len(sorted_results))
    probs = [r['ai_probability'] for r in sorted_results]
    colors = []
    for r in sorted_results:
        if r['file_type'] == 'image':
            colors.append('#EF4444' if r['is_ai'] else '#10B981')
        elif r['file_type'] == 'video':
            colors.append('#F59E0B')
        else:
            colors.append('#34D399')

    ax.barh(y_pos, probs, color=colors, edgecolor='white', height=0.7)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    labels = []
    for r in sorted_results:
        tag = {'image': '🖼', 'video': '🎬', 'audio': '🎵'}.get(r['file_type'], '?')
        badge = '🤖' if r['is_ai'] else '📷'
        labels.append(f"{tag} {badge} {r['file_name'][:30]}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('AI 概率 p_fake', fontsize=12)
    ax.set_title('全部文件 AI 概率 (按模态分组)', fontsize=14, fontweight='bold', color='#1E2761')
    ax.set_xlim(0, 1.05)

    # 模态分隔线
    prev_ft = None
    for i, r in enumerate(sorted_results):
        if prev_ft and r['file_type'] != prev_ft:
            ax.axhline(i - 0.5, color='#1E2761', linestyle='--', alpha=0.3, linewidth=1.5)
        prev_ft = r['file_type']

    plt.tight_layout()
    path = OUTPUT_DIR / 'all_files_scores.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


# ============================================================
# 2. 检测过程可视化
# ============================================================

def visualize_image_process(img_path, result, save_dir):
    """图像检测过程: 原图 + FFT + SRM残差 + 局部方差"""
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.array(Image.open(img_path).convert('RGB'))[:, :, ::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img, (224, 224))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # (1) 原图
    axes[0].imshow(img_small)
    axes[0].set_title('输入图像 (224×224)')
    axes[0].axis('off')

    # (2) FFT 幅度谱
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)
    axes[1].imshow(magnitude, cmap='magma')
    axes[1].set_title('FFT 幅度谱 (对数)')
    axes[1].axis('off')

    # (3) SRM 残差 (模拟统计量 - 高频噪声)
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    srm = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    srm_viz = np.clip(srm, -50, 50)
    axes[2].imshow(srm_viz, cmap='coolwarm')
    axes[2].set_title('SRM 高频残差')
    axes[2].axis('off')

    # (4) 局部方差 (纹理复杂度)
    local_var = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 0)
    axes[3].imshow(local_var, cmap='viridis')
    axes[3].set_title('局部方差 (纹理)')
    axes[3].axis('off')

    badge = '🤖 AI' if result['is_ai'] else '📷 REAL'
    fig.suptitle(f"{badge}  p_fake={result['ai_probability']:.4f}  conf={result['confidence']:.4f}\n{Path(img_path).name}",
                 fontsize=12, fontweight='bold',
                 color='#EF4444' if result['is_ai'] else '#10B981')

    plt.tight_layout()
    path = save_dir / f"proc_{Path(img_path).stem}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def visualize_video_process(mp4_path, result, save_dir):
    """视频检测过程: 采样帧 + 帧间差分"""
    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = 8
    indices = np.linspace(0, max(0, total - 1), n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, f = cap.read()
        if ret:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < 2:
        return None

    # 帧间差分
    diffs = []
    for i in range(1, len(frames)):
        f0 = cv2.resize(frames[i-1], (224, 224))
        f1 = cv2.resize(frames[i], (224, 224))
        diff = cv2.absdiff(cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY),
                           cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY))
        diffs.append(diff)

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(2, n_frames, figure=fig, wspace=0.05, hspace=0.05)

    # 第一行: 采样帧
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.resize(frame, (224, 224)))
        ax.set_title(f'f{i}', fontsize=9)
        ax.axis('off')

    # 第二行: 帧间差分 (热度)
    for i, diff in enumerate(diffs):
        ax = fig.add_subplot(gs[1, i])
        if i == len(diffs) - 1 and len(frames) > len(diffs):
            ax.axis('off')
            continue
        ax.imshow(diff, cmap='hot')
        ax.set_title(f'|f{i+1}-f{i}|', fontsize=9)
        ax.axis('off')

    badge = '🤖 AI' if result.get('is_ai', False) else '📷 REAL'
    conf_note = f"conf={result['confidence']:.2f} (⚠低置信度)" if result['confidence'] < 0.6 else f"conf={result['confidence']:.3f}"
    fig.suptitle(f"{badge}  p_fake={result['ai_probability']:.4f}  {conf_note}\n{Path(mp4_path).name[:50]}",
                 fontsize=12, fontweight='bold',
                 color='#EF4444' if result.get('is_ai', False) else '#10B981')

    path = save_dir / f"proc_{Path(mp4_path).stem[:30]}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def visualize_audio_process(audio_path, result, save_dir):
    """音频检测过程: 波形 + Mel频谱 + 频谱平坦度 + 过零率"""
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    except Exception:
        return None

    duration = 5.0
    n_samples = int(duration * sr)
    if len(y) > n_samples:
        y = y[:n_samples]
    elif len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))

    # Mel 频谱
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # 频谱平坦度
    flatness = librosa.feature.spectral_flatness(y=y)

    # 过零率
    zcr = librosa.feature.zero_crossing_rate(y)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1.2, 1.8, 1, 1]})

    # 波形
    t = np.linspace(0, len(y) / sr, len(y))
    axes[0].plot(t, y, linewidth=0.3, color='#065A82')
    axes[0].fill_between(t, y, alpha=0.15, color='#065A82')
    axes[0].set_title('波形 (Waveform)', fontsize=11, loc='left')
    axes[0].set_ylabel('振幅')
    axes[0].grid(alpha=0.3)

    # Mel 频谱
    im = axes[1].imshow(log_mel, aspect='auto', origin='lower', cmap='magma',
                        extent=[0, len(y)/sr, 0, 128])
    axes[1].set_title('Log-Mel 频谱 (输入 Audio-PatchViT)', fontsize=11, loc='left')
    axes[1].set_ylabel('Mel 频带')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # 频谱平坦度 (AI音频平坦度通常更低)
    t_f = np.linspace(0, len(y)/sr, flatness.shape[1])
    axes[2].plot(t_f, flatness[0], color='#EF4444', linewidth=1.2)
    axes[2].fill_between(t_f, flatness[0], alpha=0.2, color='#EF4444')
    axes[2].set_title('频谱平坦度 (Spectral Flatness)', fontsize=11, loc='left')
    axes[2].set_ylabel('Flatness')
    axes[2].grid(alpha=0.3)

    # 过零率
    t_z = np.linspace(0, len(y)/sr, zcr.shape[1])
    axes[3].plot(t_z, zcr[0], color='#10B981', linewidth=0.8)
    axes[3].fill_between(t_z, zcr[0], alpha=0.2, color='#10B981')
    axes[3].set_title('过零率 (Zero-Crossing Rate)', fontsize=11, loc='left')
    axes[3].set_xlabel('时间 (s)')
    axes[3].set_ylabel('ZCR')
    axes[3].grid(alpha=0.3)

    badge = '🤖 AI' if result.get('is_ai', False) else '📷 REAL'
    fig.suptitle(f"{badge}  p_fake={result['ai_probability']:.4f}  conf={result['confidence']:.4f}\n{Path(audio_path).name}",
                 fontsize=12, fontweight='bold',
                 color='#EF4444' if result.get('is_ai', False) else '#10B981')

    plt.tight_layout()
    path = save_dir / f"proc_{Path(audio_path).stem[:30]}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


# ============================================================
# 3. 训练曲线
# ============================================================

def plot_training_curves():
    """从 history JSON 读取训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    branches = {'image': ('图像 DeiT-ViT', '#065A82'),
                'video': ('视频 ViT+Temporal', '#F59E0B'),
                'audio': ('音频 PatchViT', '#10B981')}

    for i, (name, (label, color)) in enumerate(branches.items()):
        hist_path = PROJECT_ROOT / 'checkpoints' / f'{name}_history.json'
        ax = axes[i]
        if hist_path.exists():
            with open(hist_path) as f:
                h = json.load(f)

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
        else:
            ax.text(0.5, 0.5, f'{name} history\nnot found', ha='center', va='center',
                    fontsize=12, transform=ax.transAxes)

    plt.tight_layout()
    path = OUTPUT_DIR / 'training_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


# ============================================================
# 4. 生成 HTML 报告 (内嵌所有图)
# ============================================================

def fig_to_base64(fig_or_path):
    """将 matplotlib fig 或图片路径转 base64"""
    if isinstance(fig_or_path, (Path, str)):
        img = Image.open(str(fig_or_path))
    else:
        buf = BytesIO()
        fig_or_path.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
    buf2 = BytesIO()
    img.save(buf2, format='PNG', optimize=True)
    return base64.b64encode(buf2.getvalue()).decode()


def generate_html_report():
    """生成完整交互式 HTML 报告"""
    print("\n生成 HTML 报告...")

    # 先创建所有汇总图
    plot_summary()
    plot_per_file_scores()
    plot_training_curves()

    # 处理目录
    test_dir = PROJECT_ROOT / 'test'

    # 生成每个文件的检测过程图
    process_sections = {'image': [], 'video': [], 'audio': []}
    file_map = {}

    print("\n生成检测过程可视化...")
    for r in results:
        fname = r['file_name']
        ft = r['file_type']

        # 找到文件
        found = None
        for subdir in ['image', 'video', 'audio', '']:
            candidate = test_dir / subdir / fname
            if candidate.exists():
                found = candidate
                break
            if not subdir:
                for ext in ['.jpg', '.jpeg', '.png', '.mp4', '.mp3', '.wav', '.flac']:
                    alt = test_dir / fname.replace(Path(fname).suffix, ext)
                    if alt.exists():
                        found = alt
                        break

        if found is None or not found.exists():
            # 尝试 test/ 下直接 glob
            matches = list(test_dir.rglob(fname))
            if matches:
                found = matches[0]

        proc_path = None
        if found and found.exists():
            try:
                if ft == 'image':
                    proc_path = visualize_image_process(found, r, PROC_DIR)
                elif ft == 'video':
                    proc_path = visualize_video_process(found, r, PROC_DIR)
                elif ft == 'audio':
                    proc_path = visualize_audio_process(found, r, PROC_DIR)
            except Exception as e:
                print(f"  处理 {fname} 时出错: {e}")

        process_sections[ft].append({'result': r, 'proc_path': proc_path, 'source': found})

    # 读取汇总图 base64
    summary_b64 = fig_to_base64(OUTPUT_DIR / 'summary_overview.png')
    scores_b64 = fig_to_base64(OUTPUT_DIR / 'all_files_scores.png')
    curves_b64 = fig_to_base64(OUTPUT_DIR / 'training_curves.png')

    # 模态中文
    modal_cn = {'image': '图像', 'video': '视频', 'audio': '音频'}
    modal_icon = {'image': '🖼', 'video': '🎬', 'audio': '🎵'}
    reliability = {'image': '⭐ 可信 (Val Acc 97.8%)',
                   'video': '⚠ 不可信 (训练集仅40样本, 严重过拟合)',
                   'audio': '⚠ 有限可信 (合成数据训练, 域偏移风险)'}
    reliability_color = {'image': '#10B981', 'video': '#F59E0B', 'audio': '#F59E0B'}

    # 构建详情 HTML
    modal_detail_html = ""
    for ft in ['image', 'video', 'audio']:
        items = process_sections[ft]
        # 模态统计
        ai_count = sum(1 for it in items if it['result'].get('is_ai'))
        real_count = len(items) - ai_count

        cards_html = ""
        for it in items:
            r = it['result']
            p = it['proc_path']
            name = r['file_name']
            is_ai = r.get('is_ai', False)
            p_ai = r['ai_probability']
            conf = r['confidence']
            badge_class = "badge-ai" if is_ai else "badge-real"
            badge_text = f"🤖 AI" if is_ai else "📷 REAL"
            conf_color = "#EF4444" if conf < 0.6 else "#10B981"

            proc_img_html = ""
            if p and Path(p).exists():
                proc_b64 = fig_to_base64(p)
                proc_img_html = f'<img src="data:image/png;base64,{proc_b64}" style="width:100%;border-radius:8px;">'

            cards_html += f"""
            <div class="card">
                <div class="card-header">
                    <span class="{badge_class}">{badge_text}</span>
                    <span class="filename" title="{name}">{name[:40]}</span>
                </div>
                <div class="scores">
                    <div class="score-item">
                        <div class="score-label">p_fake</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width:{p_ai*100}%;background:{'#EF4444' if p_ai>0.5 else '#10B981'}"></div>
                        </div>
                        <div class="score-val">{p_ai:.4f}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">置信度</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width:{conf*100}%;background:{conf_color}"></div>
                        </div>
                        <div class="score-val">{conf:.4f}</div>
                    </div>
                </div>
                <div class="proc-viz">{proc_img_html}</div>
            </div>
            """

        modal_detail_html += f"""
        <div class="modal-section" id="modal-{ft}">
            <div class="modal-header" style="border-left:4px solid {reliability_color[ft]}">
                <h2>{modal_icon[ft]} {modal_cn[ft]}检测详情</h2>
                <div class="modal-meta">
                    <span class="reliability" style="color:{reliability_color[ft]}">{reliability[ft]}</span>
                    <span class="stat">AI <span class="num-ai">{ai_count}</span></span>
                    <span class="stat">真实 <span class="num-real">{real_count}</span></span>
                </div>
            </div>
            <div class="cards-grid">
                {cards_html}
            </div>
        </div>
        """

    # 汇总卡片
    img_ai = sum(1 for it in process_sections['image'] if it['result'].get('is_ai'))
    img_real = len(process_sections['image']) - img_ai
    vid_ai = sum(1 for it in process_sections['video'] if it['result'].get('is_ai'))
    vid_real = len(process_sections['video']) - vid_ai
    aud_ai = sum(1 for it in process_sections['audio'] if it['result'].get('is_ai'))
    aud_real = len(process_sections['audio']) - aud_ai

    # 组装完整 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>多模态 AIGC 检测报告 - Transformer 版</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif; background: #f8fafc; color: #1E2761; }}

    /* 顶栏 */
    .top-bar {{
        background: linear-gradient(135deg, #1E2761 0%, #065A82 100%);
        color: white; padding: 24px 40px;
    }}
    .top-bar h1 {{ font-size: 28px; margin-bottom: 4px; }}
    .top-bar .sub {{ color: #CADCFC; font-size: 13px; }}

    /* 概览卡片 */
    .overview {{
        display: grid; grid-template-columns: repeat(5, 1fr);
        gap: 16px; padding: 20px 40px; background: white;
        border-bottom: 1px solid #e2e8f0;
    }}
    .ov-card {{
        background: #f1f5f9; border-radius: 10px; padding: 16px;
        border-top: 3px solid #065A82;
    }}
    .ov-card.ai {{ border-top-color: #EF4444; }}
    .ov-card.real {{ border-top-color: #10B981; }}
    .ov-card h4 {{ font-size: 12px; color: #64748b; font-weight: normal; margin-bottom: 4px; }}
    .ov-card .num {{ font-size: 28px; font-weight: bold; color: #1E2761; }}
    .ov-card.ai .num {{ color: #EF4444; }}
    .ov-card.real .num {{ color: #10B981; }}

    /* 主体 */
    main {{ padding: 30px 40px; }}

    /* 汇总图区 */
    .chart-section {{
        background: white; border-radius: 12px; padding: 24px;
        margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .chart-section h3 {{ font-size: 18px; margin-bottom: 16px; color: #1E2761; }}
    .chart-section img {{ width: 100%; border-radius: 8px; }}

    /* 模态详情 */
    .modal-section {{
        background: white; border-radius: 12px; padding: 24px;
        margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .modal-header {{
        padding-left: 12px; margin-bottom: 20px;
    }}
    .modal-header h2 {{ font-size: 20px; color: #1E2761; margin-bottom: 8px; }}
    .modal-meta {{ display: flex; gap: 20px; align-items: center; font-size: 13px; }}
    .reliability {{ font-weight: 600; }}
    .stat {{ color: #64748b; }}
    .num-ai {{ color: #EF4444; font-weight: bold; }}
    .num-real {{ color: #10B981; font-weight: bold; }}

    /* 卡片网格 */
    .cards-grid {{
        display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
        gap: 16px;
    }}
    .card {{
        border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;
        background: #fafbfc;
    }}
    .card-header {{
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 10px;
    }}
    .filename {{
        font-size: 12px; color: #475569; font-family: monospace;
        max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}
    .badge-ai {{
        background: #FEE2E2; color: #EF4444; padding: 3px 10px;
        border-radius: 12px; font-size: 12px; font-weight: bold;
    }}
    .badge-real {{
        background: #D1FAE5; color: #10B981; padding: 3px 10px;
        border-radius: 12px; font-size: 12px; font-weight: bold;
    }}

    /* 分数条 */
    .scores {{ display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px; }}
    .score-item {{
        display: flex; align-items: center; gap: 8px;
    }}
    .score-label {{ font-size: 11px; color: #64748b; width: 60px; }}
    .score-bar {{ flex: 1; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden; }}
    .score-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
    .score-val {{ font-size: 12px; font-family: monospace; width: 55px; text-align: right; color: #475569; }}

    .proc-viz {{ margin-top: 8px; }}

    footer {{ text-align: center; padding: 30px; color: #64748b; font-size: 12px; }}
</style>
</head>
<body>

<div class="top-bar">
    <h1>🔍 多模态 AI 生成媒体检测报告</h1>
    <div class="sub">Transformer 架构 · DeiT-Small ViT · ImageNet 预训练 · 生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</div>

<div class="overview">
    <div class="ov-card"><h4>检测文件总数</h4><div class="num">{len(results)}</div></div>
    <div class="ov-card ai"><h4>AI 判定</h4><div class="num">{img_ai + vid_ai + aud_ai}</div></div>
    <div class="ov-card real"><h4>真实判定</h4><div class="num">{img_real + vid_real + aud_real}</div></div>
    <div class="ov-card"><h4>图像分支 Val Acc</h4><div class="num" style="font-size:22px">97.8%</div></div>
    <div class="ov-card"><h4>模型参数量</h4><div class="num" style="font-size:20px">21.7M</div></div>
</div>

<main>

<div class="chart-section">
    <h3>📊 检测结果汇总</h3>
    <img src="data:image/png;base64,{summary_b64}">
</div>

<div class="chart-section">
    <h3>📈 全部文件 AI 概率排名</h3>
    <img src="data:image/png;base64,{scores_b64}">
</div>

<div class="chart-section">
    <h3>🎯 训练曲线 (30 Epochs)</h3>
    <img src="data:image/png;base64,{curves_b64}">
</div>

{modal_detail_html}

</main>

<footer>
    <p>多模态 AIGC 检测系统 | Image: DeiT-Small ViT | Video: ViT + Temporal Attention | Audio: PatchViT (AST)</p>
    <p>图像分支 Val Acc 97.8% (32K 训练样本) · 视频 40 样本严重过拟合 · 音频合成数据训练</p>
</footer>

</body>
</html>"""

    html_path = OUTPUT_DIR / 'transformer_detection_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ HTML 报告已生成: {html_path}")
    print(f"   尺寸: {html_path.stat().st_size / 1024:.1f} KB")
    return html_path


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    print(f"Transformer 检测可视化")
    print(f"输入: {JSON_PATH}")
    print(f"输出: {OUTPUT_DIR}\n")

    report = generate_html_report()

    print("\n" + "=" * 60)
    print("可视化完成！")
    print(f"汇总图: {OUTPUT_DIR}")
    print(f"检测过程图: {PROC_DIR}")
    print(f"HTML 报告: {report}")
    print("=" * 60)
