"""
多模态 AIGC 检测结果可视化

生成:
  1. 汇总图表 (分数分布、模态对比、判定饼图)
  2. 分支对比 (NN vs 启发式散点图)
  3. 检测过程可视化 (FFT频谱图、SRM残差图、帧间差异、Mel频谱图等)
  4. HTML 交互式检测报告
"""

import os
import sys
import json
import argparse
import base64
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from PIL import Image

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import cv2
import librosa

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 工具函数
# ============================================================

def fig_to_base64(fig) -> str:
    """matplotlib figure 转 base64 PNG"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# 1. 结果汇总图表
# ============================================================

def plot_fused_scores_bar(results: list, out_dir: str):
    """所有文件融合分数柱状图，按模态着色"""
    fig, ax = plt.subplots(figsize=(16, 7))

    colors = {'image': '#3b82f6', 'video': '#f59e0b', 'audio': '#10b981'}
    labels = [r['file_name'] for r in results]
    scores = [r['fused_score'] for r in results]
    bar_colors = [colors.get(r['file_type'], '#888') for r in results]

    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color=bar_colors, edgecolor='white', linewidth=0.5, alpha=0.9)

    # 阈值线
    ax.axhline(0.5, color='#ef4444', linestyle='--', linewidth=2, label='AI阈值 (0.5)')

    # 在每个柱子上标注分数
    for bar, score in zip(bars, scores):
        ypos = bar.get_height() if score > 0.5 else bar.get_height() + 0.01
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{score:.3f}', ha='center', va='bottom' if score > 0.5 else 'top',
                fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('AI生成概率 (融合分数)', fontsize=12)
    ax.set_title('多模态 AIGC 检测 — 融合分数分布', fontsize=14, fontweight='bold')
    ax.set_ylim(0.3, 0.7)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in colors.items()]
    ax.legend(handles=legend_elements + [ax.get_lines()[0]],
              labels=['图像', '视频', '音频', 'AI阈值'], loc='upper right', fontsize=10)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, '01_fused_scores_bar.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_score_distribution(results: list, out_dir: str):
    """分数分布直方图 + 核密度估计"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    types = [('image', '图像', '#3b82f6'), ('video', '视频', '#f59e0b'), ('audio', '音频', '#10b981')]

    for ax, (ftype, label, color) in zip(axes, types):
        scores = [r['fused_score'] for r in results if r['file_type'] == ftype]
        if not scores:
            ax.set_visible(False)
            continue

        ax.hist(scores, bins=8, range=(0.35, 0.65), color=color, alpha=0.7, edgecolor='white', density=True)

        # KDE
        from scipy.stats import gaussian_kde
        if len(scores) > 1:
            kde = gaussian_kde(scores)
            x_range = np.linspace(0.35, 0.65, 200)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2)

        ax.axvline(0.5, color='#ef4444', linestyle='--', linewidth=1.5, label='AI阈值')
        ax.axvline(np.mean(scores), color='#333', linestyle=':', linewidth=1.5, label=f'均值={np.mean(scores):.3f}')

        ax.set_title(f'{label} ({len(scores)}个)', fontsize=13, fontweight='bold')
        ax.set_xlabel('融合AI概率')
        ax.set_ylabel('密度')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('各模态检测分数分布', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(out_dir, '02_score_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_judgment_pie(results: list, out_dir: str):
    """AI/真实判定饼图 + 模态分组柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 饼图
    ai_count = sum(1 for r in results if r['is_ai'])
    real_count = len(results) - ai_count
    sizes = [real_count, ai_count]
    labels_pie = [f'真实\n{real_count}个', f'AI生成\n{ai_count}个']
    colors_pie = ['#10b981', '#ef4444']

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels_pie, colors=colors_pie,
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 12})
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax1.set_title('整体判定分布', fontsize=14, fontweight='bold')

    # 分组柱状图
    types = ['image', 'video', 'audio']
    ai_counts = [sum(1 for r in results if r['file_type'] == t and r['is_ai']) for t in types]
    real_counts = [sum(1 for r in results if r['file_type'] == t and not r['is_ai']) for t in types]
    totals = [ai + real for ai, real in zip(ai_counts, real_counts)]

    x = np.arange(3)
    w = 0.35
    ax2.bar(x - w/2, real_counts, w, label='真实', color='#10b981', edgecolor='white')
    ax2.bar(x + w/2, ai_counts, w, label='AI生成', color='#ef4444', edgecolor='white')

    for i, (ai, real, total) in enumerate(zip(ai_counts, real_counts, totals)):
        ax2.text(i - w/2, real + 0.05, f'{real}', ha='center', fontweight='bold', fontsize=11)
        ax2.text(i + w/2, ai + 0.05, f'{ai}', ha='center', fontweight='bold', fontsize=11)
        if total > 0:
            ax2.text(i, max(ai, real) + 0.4, f'Total: {total}', ha='center', fontsize=10, color='#666')

    ax2.set_xticks(x)
    ax2.set_xticklabels(['图像', '视频', '音频'], fontsize=12)
    ax2.set_ylabel('文件数量', fontsize=12)
    ax2.set_title('各模态判定分布', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, '03_judgment_pie.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {path}")


# ============================================================
# 2. 分支对比
# ============================================================

def plot_branch_comparison(results: list, out_dir: str):
    """NN分数 vs 启发式分数 散点图 + 融合权重"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'image': '#3b82f6', 'video': '#f59e0b', 'audio': '#10b981'}

    # 散点图: NN vs Heuristic
    for ftype, color in colors.items():
        subset = [r for r in results if r['file_type'] == ftype]
        if not subset:
            continue
        nn_scores = [r['nn_score'] for r in subset]
        h_scores = [r['heuristic_score'] for r in subset]
        ax1.scatter(nn_scores, h_scores, c=color, s=80, alpha=0.8, edgecolors='white',
                    linewidths=1, label={'image': '图像', 'video': '视频', 'audio': '音频'}[ftype])

    lims = [0.35, 0.65]
    ax1.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y=x')
    ax1.axhline(0.5, color='#ef4444', linestyle=':', alpha=0.5)
    ax1.axvline(0.5, color='#ef4444', linestyle=':', alpha=0.5)
    ax1.set_xlabel('神经网络分数 (NN)', fontsize=12)
    ax1.set_ylabel('启发式分数 (Heuristic)', fontsize=12)
    ax1.set_title('分支分数对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.grid(alpha=0.3)

    # 融合前后对比
    labels = [r['file_name'][:12] for r in results]
    x = np.arange(len(results))
    width = 0.25

    nn_vals = [r['nn_score'] for r in results]
    h_vals = [r['heuristic_score'] for r in results]
    fused_vals = [r['fused_score'] for r in results]

    ax2.bar(x - width, nn_vals, width, label='NN分支', color='#6366f1', alpha=0.7)
    ax2.bar(x, h_vals, width, label='启发式分支', color='#f97316', alpha=0.7)
    ax2.bar(x + width, fused_vals, width, label='融合结果', color='#22c55e', alpha=0.9)

    ax2.axhline(0.5, color='#ef4444', linestyle='--', linewidth=1, alpha=0.7, label='AI阈值')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax2.set_ylabel('AI概率分数', fontsize=12)
    ax2.set_title('三分支分数对比', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.35, 0.65)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, '04_branch_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {path}")


# ============================================================
# 3. 检测过程可视化 — 图像
# ============================================================

def visualize_image_process(image_path: str, out_dir: str):
    """
    图像检测过程可视化: 原图 + FFT频谱 + SRM残差 + 颜色直方图
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 5, figure=fig, wspace=0.3)

    # 1. 原图
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_rgb)
    ax1.set_title('原图', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. FFT 频谱
    ax2 = fig.add_subplot(gs[1])
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1.0)
    im2 = ax2.imshow(magnitude, cmap='inferno')
    ax2.set_title('FFT 幅度谱', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    # 3. SRM 残差
    ax3 = fig.add_subplot(gs[2])
    kernel = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32)
    residual = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    im3 = ax3.imshow(residual, cmap='RdBu_r', vmin=-30, vmax=30)
    ax3.set_title('SRM 残差', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, shrink=0.7)

    # 4. 局部方差热力图
    ax4 = fig.add_subplot(gs[3])
    blur = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 10)
    variance_map = (gray.astype(np.float32) - blur) ** 2
    im4 = ax4.imshow(variance_map, cmap='hot')
    ax4.set_title('局部方差', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, shrink=0.7)

    # 5. 颜色直方图
    ax5 = fig.add_subplot(gs[4])
    colors = ['#ef4444', '#22c55e', '#3b82f6']
    for i, (name, color) in enumerate(zip(['R', 'G', 'B'], colors)):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax5.plot(hist, color=color, linewidth=1.2, label=name)
    ax5.set_title('颜色直方图', fontsize=12, fontweight='bold')
    ax5.set_xlabel('像素值'); ax5.set_ylabel('频数')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    filename = os.path.basename(image_path)
    fig.suptitle(f'图像检测过程: {filename}', fontsize=14, fontweight='bold', y=1.02)

    return fig


# ============================================================
# 4. 检测过程可视化 — 视频
# ============================================================

def visualize_video_process(video_path: str, out_dir: str, num_frames: int = 8):
    """
    视频检测过程可视化: 采样帧序列 + 帧间差异曲线 + 光流可视化
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < 2:
        return None

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(3, num_frames, figure=fig, hspace=0.35, wspace=0.15)

    # Row 1: 采样帧
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frame)
        ax.set_title(f'帧{i+1}', fontsize=9)
        ax.axis('off')
    fig.text(0.5, 0.72, '采样帧序列', ha='center', fontsize=12, fontweight='bold')

    # Row 2: 帧间差异热力图
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    diffs = []
    for i in range(len(grays) - 1):
        diff = cv2.absdiff(grays[i], grays[i+1])
        diffs.append(diff)

    for i, diff in enumerate(diffs):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(diff, cmap='hot', vmin=0, vmax=50)
        ax.set_title(f'Δ帧{i+1}-{i+2}', fontsize=8)
        ax.axis('off')

    fig.text(0.5, 0.48, '帧间差异热力图', ha='center', fontsize=12, fontweight='bold')

    # Row 3: 光流可视化
    for i in range(min(len(grays) - 1, num_frames)):
        ax = fig.add_subplot(gs[2, i])
        flow = cv2.calcOpticalFlowFarneback(grays[i], grays[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 下采样光流并绘制
        step = 10
        h, w = flow.shape[:2]
        Y, X = np.mgrid[step//2:h:step, step//2:w:step]
        U = flow[Y, X, 0]
        V = flow[Y, X, 1]
        mag = np.sqrt(U**2 + V**2)
        ax.quiver(X, Y, U, V, mag, cmap='plasma', scale=50, width=0.003)
        ax.set_title(f'光流{i+1}', fontsize=8)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    fig.text(0.5, 0.24, '光流场可视化', ha='center', fontsize=12, fontweight='bold')

    filename = os.path.basename(video_path)
    fig.suptitle(f'视频检测过程: {filename}', fontsize=14, fontweight='bold', y=1.005)

    return fig


# ============================================================
# 5. 检测过程可视化 — 音频
# ============================================================

def visualize_audio_process(audio_path: str, out_dir: str, sr_target: int = 16000):
    """
    音频检测过程可视化: 波形 + Mel频谱图 + 频谱平坦度 + MFCC
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    except Exception:
        return None

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # 1. 波形
    ax1 = fig.add_subplot(gs[0, :])
    t = np.linspace(0, len(y)/sr, len(y))
    ax1.plot(t, y, color='#3b82f6', linewidth=0.8, alpha=0.7)
    ax1.fill_between(t, y, alpha=0.2, color='#3b82f6')
    ax1.set_title('波形 (Waveform)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时间 (s)'); ax1.set_ylabel('振幅')
    ax1.grid(alpha=0.3)

    # 2. Mel频谱图
    ax2 = fig.add_subplot(gs[1, 0])
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    im2 = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
    ax2.set_title('Mel 频谱图', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, format='%+2.0f dB')

    # 3. 频谱平坦度 + 质心
    ax3 = fig.add_subplot(gs[1, 1])
    flatness = librosa.feature.spectral_flatness(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    t_flat = np.linspace(0, len(y)/sr, flatness.shape[1])
    ax3.plot(t_flat, flatness[0], color='#8b5cf6', linewidth=2, label='频谱平坦度')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t_flat, centroid[0], color='#f59e0b', linewidth=2, label='频谱质心')
    ax3.set_title('频谱特征时序', fontsize=12, fontweight='bold')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('平坦度')
    ax3_twin.set_ylabel('质心 (Hz)')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax3.grid(alpha=0.3)

    # 4. MFCC
    ax4 = fig.add_subplot(gs[2, 0])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    im4 = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax4, cmap='viridis')
    ax4.set_title('MFCC (20维)', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax4)

    # 5. 过零率
    ax5 = fig.add_subplot(gs[2, 1])
    zcr = librosa.feature.zero_crossing_rate(y)
    t_zcr = np.linspace(0, len(y)/sr, zcr.shape[1])
    ax5.fill_between(t_zcr, zcr[0], alpha=0.4, color='#ec4899')
    ax5.plot(t_zcr, zcr[0], color='#ec4899', linewidth=1.5)
    ax5.set_title('过零率 (ZCR)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('时间 (s)'); ax5.set_ylabel('ZCR')
    ax5.grid(alpha=0.3)

    filename = os.path.basename(audio_path)
    duration = len(y) / sr
    fig.suptitle(f'音频检测过程: {filename}  |  时长: {duration:.2f}s', fontsize=14, fontweight='bold', y=1.005)

    return fig


# ============================================================
# 6. 特征热力图汇总
# ============================================================

def plot_feature_heatmap(results: list, out_dir: str):
    """启发式特征热力图 — 跨文件对比"""
    # 收集所有启发式特征名
    all_keys = []
    seen = set()
    for r in results:
        if r.get('heuristics'):
            for k in r['heuristics'].keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

    if not all_keys:
        return

    # 构建矩阵
    short_names = {
        'freq_ratio': 'FFT高频比', 'srm_std_ratio': 'SRM残差比',
        'local_var': '局部方差', 'blockiness': '块效应', 'color_entropy': '颜色熵',
        'temporal_consistency': '时序一致性', 'motion_score': '运动分数', 'frame_variance': '帧方差比',
        'spectral_flatness': '频谱平坦度', 'zcr_mean': 'ZCR均值', 'zcr_std': 'ZCR标准差',
        'mfcc_mean': 'MFCC均值', 'mfcc_std': 'MFCC标准差',
        'spectral_centroid': '频谱质心', 'spectral_rolloff': '频谱滚降', 'spectral_bandwidth': '频谱带宽',
    }

    data = []
    labels_short = []
    for r in results:
        row = []
        h = r.get('heuristics', {})
        for k in all_keys:
            row.append(h.get(k, np.nan))
        data.append(row)
        labels_short.append(r['file_name'][:15])

    matrix = np.array(data, dtype=np.float64)

    # 对每列归一化到 [0,1]
    norms = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            col_min, col_max = valid.min(), valid.max()
            if col_max - col_min > 1e-8:
                norms[:, j] = (col - col_min) / (col_max - col_min)
            else:
                norms[:, j] = 0.5
        else:
            norms[:, j] = 0.5

    # 按融合分数排序
    fused = [r['fused_score'] for r in results]
    sort_idx = np.argsort(fused)[::-1]
    norms = norms[sort_idx]
    labels_short = [labels_short[i] for i in sort_idx]
    results_sorted = [results[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(max(12, len(all_keys) * 1.2), max(8, len(results) * 0.4)))

    im = ax.imshow(norms, aspect='auto', cmap='RdYlGn', interpolation='nearest')

    # 标记AI生成/真实
    for i, r in enumerate(results_sorted):
        marker = '🤖' if r['is_ai'] else '📷'
        color = '#ef4444' if r['is_ai'] else '#10b981'
        ax.text(-0.5, i, marker, ha='center', va='center', fontsize=10, fontweight='bold')
        # 分数标注
        ax.text(len(all_keys), i, f"{r['fused_score']:.3f}", ha='left', va='center', fontsize=8,
                color=color, fontweight='bold')

    ax.set_xticks(np.arange(len(all_keys)))
    ax.set_xticklabels([short_names.get(k, k) for k in all_keys], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(labels_short)))
    ax.set_yticklabels(labels_short, fontsize=8)
    ax.set_xlabel('启发式特征', fontsize=11)
    ax.set_ylabel('检测文件 (按融合分数排序)', fontsize=11)
    ax.set_title('检测特征热力图 (归一化, 红=低, 绿=高)', fontsize=13, fontweight='bold')

    # 右侧分数列标题
    ax.text(len(all_keys), -1.5, 'AI概率', ha='center', fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8, label='归一化特征值')

    plt.tight_layout()
    path = os.path.join(out_dir, '05_feature_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {path}")


# ============================================================
# 7. 运行所有过程可视化
# ============================================================

def generate_process_visualizations(results: list, out_dir: str):
    """为每个文件生成检测过程可视化"""
    process_dir = os.path.join(out_dir, 'process')
    image_dir = os.path.join(process_dir, 'image')
    video_dir = os.path.join(process_dir, 'video')
    audio_dir = os.path.join(process_dir, 'audio')
    ensure_dir(image_dir); ensure_dir(video_dir); ensure_dir(audio_dir)

    print("\n生成检测过程可视化...")
    b64_images = {'image': {}, 'video': {}, 'audio': {}}

    for r in results:
        file_path = r['file_path']
        ftype = r['file_type']
        fname = r['file_name']
        print(f"  处理: {fname}")

        try:
            if ftype == 'image':
                fig = visualize_image_process(file_path, image_dir)
                if fig:
                    save_path = os.path.join(image_dir, fname.replace('.', '_') + '_process.png')
                    fig.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
                    b64_images['image'][file_path] = fig_to_base64(fig)
                    print(f"    ✓ {save_path}")

            elif ftype == 'video':
                fig = visualize_video_process(file_path, video_dir)
                if fig:
                    save_path = os.path.join(video_dir, fname.replace('.', '_') + '_process.png')
                    fig.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
                    b64_images['video'][file_path] = fig_to_base64(fig)
                    print(f"    ✓ {save_path}")

            elif ftype == 'audio':
                fig = visualize_audio_process(file_path, audio_dir)
                if fig:
                    save_path = os.path.join(audio_dir, fname.replace('.', '_') + '_process.png')
                    fig.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
                    b64_images['audio'][file_path] = fig_to_base64(fig)
                    print(f"    ✓ {save_path}")
        except Exception as e:
            print(f"    ✗ 失败: {e}")

    return b64_images


# ============================================================
# 8. 生成 HTML 报告
# ============================================================

def generate_html_report(results: list, b64_summary: dict, b64_process: dict, out_path: str):
    """生成 HTML 交互式检测报告"""

    # 统计
    ai_count = sum(1 for r in results if r['is_ai'])
    real_count = len(results) - ai_count

    types_count = {}
    for r in results:
        t = r['file_type']
        types_count.setdefault(t, {'total': 0, 'ai': 0})
        types_count[t]['total'] += 1
        if r['is_ai']:
            types_count[t]['ai'] += 1

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>多模态 AIGC 检测报告</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
    padding: 20px;
  }}
  .container {{ max-width: 1600px; margin: 0 auto; }}

  h1 {{
    text-align: center;
    font-size: 2.2em;
    margin: 30px 0;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
  }}

  h2 {{
    color: #60a5fa;
    margin: 30px 0 15px;
    font-size: 1.5em;
    border-left: 4px solid #60a5fa;
    padding-left: 12px;
  }}

  /* 概览卡片 */
  .overview {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 30px;
  }}
  .card {{
    background: rgba(30, 41, 59, 0.8);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(96, 165, 250, 0.2);
    backdrop-filter: blur(10px);
    text-align: center;
  }}
  .card .num {{ font-size: 2.5em; font-weight: 800; margin: 5px 0; }}
  .card .label {{ font-size: 0.95em; color: #94a3b8; }}
  .card.green .num {{ color: #34d399; }}
  .card.red .num {{ color: #f87171; }}
  .card.blue .num {{ color: #60a5fa; }}
  .card.yellow .num {{ color: #fbbf24; }}

  /* 图表区域 */
  .chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
  }}
  .chart-card {{
    background: rgba(30, 41, 59, 0.6);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(96, 165, 250, 0.15);
  }}
  .chart-card img {{ width: 100%; border-radius: 8px; }}

  /* 按模态分组 */
  .modal-section {{ margin-top: 40px; }}
  .file-section {{
    background: rgba(30, 41, 59, 0.5);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border-left: 5px solid #60a5fa;
  }}
  .file-section.ai {{ border-left-color: #f87171; }}
  .file-section.real {{ border-left-color: #34d399; }}

  .file-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    flex-wrap: wrap;
    gap: 10px;
  }}
  .file-name {{ font-size: 1.15em; font-weight: 600; color: #cbd5e1; }}
  .badge {{
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9em;
  }}
  .badge.ai {{ background: rgba(248, 113, 113, 0.2); color: #f87171; }}
  .badge.real {{ background: rgba(52, 211, 153, 0.2); color: #34d399; }}

  .scores {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 12px;
  }}
  .score-item {{
    background: rgba(15, 23, 42, 0.6);
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.9em;
  }}
  .score-item .val {{ font-weight: 700; font-size: 1.1em; }}

  .heuristics {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
  }}
  .heuristic-tag {{
    background: rgba(96, 165, 250, 0.15);
    color: #93c5fd;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.82em;
  }}

  .process-img img {{
    width: 100%;
    border-radius: 8px;
    margin-top: 10px;
  }}

  .tabs {{ display: flex; gap: 8px; margin-bottom: 20px; }}
  .tab {{
    padding: 10px 24px;
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-radius: 8px;
    cursor: pointer;
    font-size: 1em;
    color: #94a3b8;
    transition: all 0.2s;
  }}
  .tab.active {{
    background: rgba(96, 165, 250, 0.2);
    color: #60a5fa;
    border-color: #60a5fa;
  }}

  .timestamp {{ text-align: center; color: #64748b; margin-top: 40px; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container">
  <h1>🎨 多模态 AIGC 检测报告</h1>

  <div class="overview">
    <div class="card blue">
      <div class="label">检测文件总数</div>
      <div class="num">{len(results)}</div>
    </div>
    <div class="card red">
      <div class="label">判定为 AI 生成</div>
      <div class="num">{ai_count}</div>
    </div>
    <div class="card green">
      <div class="label">判定为 真实</div>
      <div class="num">{real_count}</div>
    </div>
    <div class="card yellow">
      <div class="label">AI生成占比</div>
      <div class="num">{ai_count/len(results)*100:.1f}%</div>
    </div>
  </div>

  <h2>📊 结果汇总图表</h2>
  <div class="chart-grid">
"""

    for key, title in [
        ('01', '融合分数柱状图'),
        ('02', '各模态分数分布'),
        ('03', '判定饼图'),
        ('04', '分支分数对比'),
        ('05', '特征热力图'),
    ]:
        if key in b64_summary:
            html += f"""    <div class="chart-card">
      <h3 style="color:#cbd5e1;margin-bottom:10px;">{title}</h3>
      <img src="data:image/png;base64,{b64_summary[key]}" alt="{title}">
    </div>
"""

    # 按模态分组
    modal_names = {'image': '📷 图像', 'video': '🎬 视频', 'audio': '🎵 音频'}
    modal_colors = {'image': 'blue', 'video': 'yellow', 'audio': 'green'}

    for ftype in ['image', 'video', 'audio']:
        subset = [r for r in results if r['file_type'] == ftype]
        if not subset:
            continue

        html += f"""
  <h2>{modal_names[ftype]} 检测详情 ({len(subset)}个)</h2>
"""

        for r in sorted(subset, key=lambda x: -x['fused_score']):
            is_ai = r['is_ai']
            result_class = 'ai' if is_ai else 'real'
            badge_text = '🤖 AI生成' if is_ai else '📷 真实'
            fname = r['file_name']

            heur_html = ""
            if r.get('heuristics'):
                short_names = {
                    'freq_ratio': 'FFT高频比', 'srm_std_ratio': 'SRM残差比',
                    'local_var': '局部方差', 'blockiness': '块效应', 'color_entropy': '颜色熵',
                    'temporal_consistency': '时序一致性', 'motion_score': '运动分数', 'frame_variance': '帧方差比',
                    'spectral_flatness': '频谱平坦度', 'zcr_mean': 'ZCR均值',
                    'mfcc_mean': 'MFCC均值', 'spectral_centroid': '频谱质心',
                    'spectral_rolloff': '频谱滚降', 'spectral_bandwidth': '频谱带宽',
                }
                heur_html = '<div class="heuristics">'
                for k, v in r['heuristics'].items():
                    heur_html += f'<span class="heuristic-tag">{short_names.get(k, k)}: {v:.4f}</span>'
                heur_html += '</div>'

            process_img = ""
            b64_key = r['file_path']
            if ftype in b64_process and b64_key in b64_process[ftype]:
                process_img = f'<div class="process-img"><img src="data:image/png;base64,{b64_process[ftype][b64_key]}" alt="检测过程"></div>'

            html += f"""  <div class="file-section {result_class}">
    <div class="file-header">
      <span class="file-name">{fname}</span>
      <span class="badge {result_class}">{badge_text} (p={r['fused_score']:.4f})</span>
    </div>
    <div class="scores">
      <div class="score-item">融合: <span class="val">{r['fused_score']:.4f}</span></div>
      <div class="score-item">NN: <span class="val">{r['nn_score']:.4f}</span></div>
      <div class="score-item">启发式: <span class="val">{r['heuristic_score']:.4f}</span></div>
      <div class="score-item">置信度: <span class="val">{r['confidence']:.4f}</span></div>
    </div>
    {heur_html}
    {process_img}
  </div>
"""

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html += f"""
  <p class="timestamp">检测报告生成时间: {timestamp} | 融合策略: 70%启发式 + 30%预训练骨干特征</p>
</div>
</body>
</html>
"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✓ HTML 报告已生成: {out_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="多模态 AIGC 检测结果可视化")
    parser.add_argument("--json", type=str, default=None, help="融合结果JSON路径")
    parser.add_argument("--out-dir", type=str, default="outputs/visualization", help="输出目录")
    args = parser.parse_args()

    out_dir = str(PROJECT_ROOT / args.out_dir)
    ensure_dir(out_dir)

    # 加载最新的融合结果
    if args.json:
        json_path = args.json
    else:
        # 自动找最新的
        json_files = sorted(
            [f for f in os.listdir(str(PROJECT_ROOT / 'outputs'))
             if f.startswith('fused_results') and f.endswith('.json')]
        )
        if not json_files:
            print("错误: 找不到 fused_results_*.json，请先运行 test_fused.py")
            return
        json_path = str(PROJECT_ROOT / 'outputs' / json_files[-1])

    print(f"加载结果: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"共 {len(results)} 个检测结果")

    # 生成汇总图表
    print("\n生成汇总图表...")
    b64_summary = {}

    plot_fused_scores_bar(results, out_dir)
    fig = plt.gcf(); plt.close()
    # 重新生成并保存b64
    import matplotlib.pyplot as plt_mpl
    # 我们需要重新渲染来获取b64, 这里直接从文件读取
    for name, func in [
        ('01', plot_fused_scores_bar),
        ('02', plot_score_distribution),
        ('03', plot_judgment_pie),
        ('04', plot_branch_comparison),
        ('05', plot_feature_heatmap),
    ]:
        path = os.path.join(out_dir, f'{name}_*.png')
        # 该函数已保存文件, 读取文件作为b64
        import glob
        matches = glob.glob(path)
        if matches:
            latest = max(matches, key=os.path.getmtime)
            with open(latest, 'rb') as pf:
                b64_summary[name] = base64.b64encode(pf.read()).decode()

    # 生成检测过程可视化
    b64_process = generate_process_visualizations(results, out_dir)

    # 生成HTML报告
    html_path = os.path.join(out_dir, 'aigc_detection_report.html')
    generate_html_report(results, b64_summary, b64_process, html_path)

    print(f"\n{'='*60}")
    print(f"可视化完成!")
    print(f"  输出目录: {out_dir}")
    print(f"  HTML报告: {html_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
