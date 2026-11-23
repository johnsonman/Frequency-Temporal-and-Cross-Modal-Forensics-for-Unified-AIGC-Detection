"""
多模态 AIGC 检测架构图生成器
============================
基于原始 PNG 风格，用 matplotlib 绘制：
  1. multimodal_full.png    — 完整多模态架构总览
  2. image_branch.png       — 图像分支 (CLIP-ViT + MSFF + SRM/FFT)
  3. video_branch.png       — 视频分支 (帧采样 + 共享编码器 + T-Agg)
  4. audio_branch.png       — 音频分支 (Wav2Vec2 + Log-Mel + 注意力池化)
  5. gated_fusion.png       — 门控融合 G-Fuse
  6. cross_modal_consistency.png — 跨模态一致性 X-Cons
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm

OUT_DIR = Path(__file__).parent / 'docs' / 'architecture'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体 (Windows 自带黑体 SimHei)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DengXian',
                                     'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 样式工具
# ============================================================

def rounded_box(ax, xy, w, h, fc='#E3F2FD', ec='#1565C0', lw=1.2, alpha=0.95, radius=0.03):
    """圆角矩形框"""
    box = FancyBboxPatch(
        xy, w, h, boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
        zorder=2
    )
    ax.add_patch(box)
    return box


def pill_label(ax, cx, cy, w, h, text, fc='#1565C0', ec=None, tc='white', fontsize=9, fontweight='bold'):
    """胶囊标签 (MSFF / T-Agg / X-Cons / G-Fuse 等)"""
    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h, boxstyle=f"round,pad={h/2}",
        facecolor=fc, edgecolor=ec or fc, linewidth=1, alpha=0.95, zorder=4
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center', color=tc, fontsize=fontsize,
            fontweight=fontweight, zorder=5)
    return box


def arrow(ax, x1, y1, x2, y2, color='#757575', lw=1.3, style='->', shrinkA=3, shrinkB=3):
    """箭头连接"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle='arc3,rad=0',
                                shrinkA=shrinkA, shrinkB=shrinkB),
                zorder=6)


def text_center(ax, cx, cy, text, fontsize=10, color='#212121', fontweight='normal', fontstyle='normal', ha='center', va='center'):
    ax.text(cx, cy, text, ha=ha, va=va, fontsize=fontsize,
            color=color, fontweight=fontweight, fontstyle=fontstyle, zorder=3)


def title(ax, title_str, subtitle_str=None):
    """图标题"""
    ax.set_title(title_str, fontsize=16, fontweight='bold', color='#1A237E', pad=14)
    if subtitle_str:
        ax.text(0.5, -0.04, subtitle_str, transform=ax.transAxes,
                ha='center', va='top', fontsize=10, color='#546E7A')
    ax.axis('off')


# ============================================================
# 1. 完整多模态架构总览  (multimodal_full.png)
# ============================================================

def draw_full_architecture():
    fig = plt.figure(figsize=(18, 11), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)

    # ========== 图像分支 ==========
    rounded_box(ax, (0.3, 7.2), 7.5, 3.4, fc='#E3F2FD', ec='#1565C0')
    text_center(ax, 0.7, 10.3, '图像分支  Φ_img', fontsize=12, fontweight='bold', color='#0D47A1')

    # 输入
    rounded_box(ax, (0.5, 8.5), 0.9, 0.9, fc='#212121', ec='#212121')
    text_center(ax, 0.95, 8.95, 'IMG', fontsize=8, color='white', fontweight='bold')

    # 预处理
    rounded_box(ax, (1.7, 8.4), 1.2, 1.1, fc='#BBDEFB', ec='#1565C0')
    text_center(ax, 2.3, 9.3, '图像\n预处理', fontsize=9, color='#0D47A1', fontweight='bold')
    text_center(ax, 2.3, 8.65, 'Resize\nNormalize', fontsize=6.5, color='#5D4037')

    # CLIP 编码器
    rounded_box(ax, (3.2, 7.8), 1.8, 1.9, fc='#90CAF9', ec='#1565C0')
    text_center(ax, 4.1, 9.4, 'CLIP/ViT 视觉编码器', fontsize=9.5, fontweight='bold', color='#0D47A1')
    text_center(ax, 4.1, 8.85, '(ViT-B/16)', fontsize=7.5, color='#1A237E')
    text_center(ax, 4.1, 8.4, 'Layer 3  (浅层)', fontsize=7, color='#37474F')
    text_center(ax, 4.1, 8.05, 'Layer 6  (中层)', fontsize=7, color='#37474F')

    # MSFF
    rounded_box(ax, (5.3, 8.4), 2.0, 2.0, fc='#64B5F6', ec='#1565C0')
    pill_label(ax, 6.3, 10.1, 1.1, 0.4, 'MSFF', fc='#1976D2', fontsize=10)
    text_center(ax, 6.3, 9.35, '多尺度特征融合', fontsize=9, fontweight='bold', color='#0D47A1')
    text_center(ax, 6.3, 8.85, '特征投影', fontsize=7, color='#0D47A1')
    text_center(ax, 6.3, 8.55, '层级注意力加权', fontsize=7, color='#0D47A1')
    text_center(ax, 6.3, 8.25, '残差融合', fontsize=7, color='#0D47A1')

    # SRM/FFT 支路
    rounded_box(ax, (5.3, 7.4), 2.0, 0.9, fc='#FFE0B2', ec='#F57C00')
    pill_label(ax, 6.3, 7.95, 1.5, 0.35, '局部取证支路', fc='#F57C00', fontsize=8)
    text_center(ax, 6.3, 7.6, 'SRM 残差 / FFT 频谱', fontsize=7.5, color='#E65100', fontweight='bold')

    # 图像表征
    rounded_box(ax, (7.6, 8.55), 0.6, 1.2, fc='#1976D2', ec='#0D47A1')
    text_center(ax, 7.9, 9.5, '图像\n分类头', fontsize=7.5, color='white', fontweight='bold')

    # 图像特征向量输出
    rounded_box(ax, (7.6, 7.45), 0.6, 0.85, fc='#42A5F5', ec='#0D47A1')
    text_center(ax, 7.9, 7.88, 'f_img', fontsize=8, color='white', fontweight='bold')

    # ========== 视频分支 ==========
    rounded_box(ax, (0.3, 3.9), 7.5, 3.1, fc='#E0F7FA', ec='#00695C')
    text_center(ax, 0.7, 6.7, '视频分支  Φ_vid', fontsize=12, fontweight='bold', color='#00695C')

    rounded_box(ax, (0.5, 4.9), 0.9, 0.9, fc='#212121', ec='#212121')
    text_center(ax, 0.95, 5.35, 'VID', fontsize=8, color='white', fontweight='bold')

    rounded_box(ax, (1.7, 4.8), 1.3, 1.1, fc='#B2EBF2', ec='#00695C')
    text_center(ax, 2.35, 5.75, '视频帧序列\n预处理', fontsize=9, color='#004D40', fontweight='bold')
    text_center(ax, 2.35, 5.2, '帧采样 / 归一化', fontsize=7, color='#37474F')

    # 共享编码器
    rounded_box(ax, (3.3, 4.4), 1.6, 2.0, fc='#80DEEA', ec='#00695C')
    pill_label(ax, 4.1, 6.15, 1.4, 0.38, '共享 CLIP/ViT', fc='#00838F', fontsize=9)
    text_center(ax, 4.1, 5.6, '帧级特征序列', fontsize=8, fontweight='bold', color='#004D40')
    text_center(ax, 4.1, 5.15, '(参数共享)', fontsize=7, color='#006064')
    text_center(ax, 4.1, 4.8, 'T × D', fontsize=7, color='#37474F', fontstyle='italic')

    # T-Agg
    rounded_box(ax, (5.2, 4.7), 2.2, 1.6, fc='#4DD0E1', ec='#00695C')
    pill_label(ax, 6.3, 6.1, 1.0, 0.4, 'T-Agg', fc='#00ACC1', fontsize=10)
    text_center(ax, 6.3, 5.55, '时序聚合', fontsize=9, fontweight='bold', color='#004D40')
    text_center(ax, 6.3, 5.1, 'Manba / Transformer', fontsize=7, color='#004D40')
    text_center(ax, 6.3, 4.82, '帧级注意力加权', fontsize=7, color='#37474F')

    rounded_box(ax, (7.6, 5.25), 0.6, 1.0, fc='#00ACC1', ec='#00695C')
    text_center(ax, 7.9, 6.1, 'p_vid', fontsize=8, color='white', fontweight='bold')
    rounded_box(ax, (7.6, 4.35), 0.6, 0.75, fc='#26C6DA', ec='#00695C')
    text_center(ax, 7.9, 4.72, 'f_vid', fontsize=8, color='white', fontweight='bold')

    # ========== 音频分支 ==========
    rounded_box(ax, (0.3, 0.6), 7.5, 3.0, fc='#F3E5F5', ec='#6A1B9A')
    text_center(ax, 0.7, 3.3, '音频分支  Φ_aud', fontsize=12, fontweight='bold', color='#4A148C')

    rounded_box(ax, (0.5, 1.6), 0.9, 0.9, fc='#212121', ec='#212121')
    text_center(ax, 0.95, 2.05, 'AUD', fontsize=8, color='white', fontweight='bold')

    rounded_box(ax, (1.7, 1.5), 1.3, 1.1, fc='#E1BEE7', ec='#6A1B9A')
    text_center(ax, 2.35, 2.45, '音频波形\n预处理', fontsize=9, color='#4A148C', fontweight='bold')
    text_center(ax, 2.35, 1.9, '重采样 / 归一化', fontsize=7, color='#37474F')

    rounded_box(ax, (3.3, 1.1), 1.6, 1.9, fc='#CE93D8', ec='#6A1B9A')
    pill_label(ax, 4.1, 2.85, 1.3, 0.38, 'Wav2Vec2', fc='#8E24AA', fontsize=9)
    text_center(ax, 4.1, 2.3, 'SSL 编码器', fontsize=8, fontweight='bold', color='#4A148C')
    text_center(ax, 4.1, 1.85, 'Transformer Layers', fontsize=7, color='#4A148C')
    text_center(ax, 4.1, 1.45, 'H_a ∈ R^{T×d}', fontsize=7, color='#37474F', fontstyle='italic')

    # Log-Mel + CNN
    rounded_box(ax, (5.2, 1.25), 2.2, 1.75, fc='#BA68C8', ec='#6A1B9A')
    pill_label(ax, 6.3, 2.85, 1.3, 0.38, '频谱特征分支', fc='#8E24AA', fontsize=9)
    text_center(ax, 6.3, 2.35, 'Log-Mel + 2D CNN', fontsize=8, fontweight='bold', color='#4A148C')
    text_center(ax, 6.3, 2.0, '多尺度时间卷积', fontsize=7, color='#4A148C')
    text_center(ax, 6.3, 1.65, '注意力池化 / GAP', fontsize=7, color='#37474F')

    rounded_box(ax, (7.6, 1.65), 0.6, 1.0, fc='#8E24AA', ec='#6A1B9A')
    text_center(ax, 7.9, 2.5, 'p_aud', fontsize=8, color='white', fontweight='bold')
    rounded_box(ax, (7.6, 0.85), 0.6, 0.7, fc='#AB47BC', ec='#6A1B9A')
    text_center(ax, 7.9, 1.2, 'f_aud', fontsize=8, color='white', fontweight='bold')

    # ========== X-Cons 跨模态一致性 ==========
    rounded_box(ax, (8.3, 5.8), 4.0, 3.0, fc='#E8F5E9', ec='#2E7D32')
    pill_label(ax, 10.3, 8.45, 1.6, 0.45, 'X-Cons  跨模态一致性', fc='#2E7D32', fontsize=10)
    text_center(ax, 10.3, 7.85, '同步对比学习', fontsize=9, fontweight='bold', color='#1B5E20')
    text_center(ax, 10.3, 7.4, '音画对齐建模', fontsize=8, color='#37474F')
    text_center(ax, 10.3, 6.95, 'f_sync', fontsize=9, color='#1B5E20', fontweight='bold', fontstyle='italic')
    text_center(ax, 10.3, 6.5, '异步对齐建模', fontsize=8, color='#37474F')
    text_center(ax, 10.3, 6.05, 'f_async', fontsize=9, color='#1B5E20', fontweight='bold', fontstyle='italic')

    # ========== G-Fuse 门控融合 ==========
    rounded_box(ax, (13.0, 3.0), 4.7, 5.8, fc='#FFF3E0', ec='#E65100')
    pill_label(ax, 15.35, 8.45, 2.0, 0.5, 'G-Fuse  门控融合', fc='#E65100', fontsize=11)
    text_center(ax, 15.35, 7.95, '特征对齐', fontsize=9, color='#BF360C', fontweight='bold')
    text_center(ax, 15.35, 7.5, '可靠性评估', fontsize=9, color='#BF360C', fontweight='bold')
    text_center(ax, 15.35, 7.05, '门控加权融合', fontsize=9, color='#BF360C', fontweight='bold')
    text_center(ax, 15.35, 6.6, '模态掩码 m_k (缺失模态)', fontsize=8, color='#37474F')
    text_center(ax, 15.35, 6.15, '全连接分类器', fontsize=9, color='#BF360C', fontweight='bold')
    text_center(ax, 15.35, 5.5, 'p_fake', fontsize=12, color='#BF360C', fontweight='bold', fontstyle='italic')
    text_center(ax, 15.35, 5.05, 'AI 生成 / 真实', fontsize=9, color='#37474F')

    # ========== 箭头连接 ==========
    # 图像分支箭头
    arrow(ax, 1.4, 8.95, 1.7, 8.95)
    arrow(ax, 2.9, 8.95, 3.2, 8.75)
    arrow(ax, 5.0, 8.75, 5.3, 9.4)
    arrow(ax, 5.0, 8.35, 5.3, 7.85)
    arrow(ax, 7.3, 9.4, 7.6, 9.15)
    arrow(ax, 7.3, 7.85, 7.6, 7.88)

    # 视频分支箭头
    arrow(ax, 1.4, 5.35, 1.7, 5.35)
    arrow(ax, 3.0, 5.35, 3.3, 5.4)
    arrow(ax, 4.9, 5.35, 5.2, 5.5)
    arrow(ax, 7.4, 5.5, 7.6, 5.75)
    arrow(ax, 7.4, 4.9, 7.6, 4.73)

    # 音频分支箭头
    arrow(ax, 1.4, 2.05, 1.7, 2.05)
    arrow(ax, 3.0, 2.05, 3.3, 2.05)
    arrow(ax, 4.9, 2.05, 5.2, 2.13)
    arrow(ax, 7.4, 2.13, 7.6, 2.15)
    arrow(ax, 7.4, 1.6, 7.6, 1.2)

    # 三个分支 → X-Cons
    arrow(ax, 7.9, 7.88, 9.3, 7.8, color='#4CAF50', lw=1.5)  # f_img
    arrow(ax, 7.9, 4.72, 9.3, 7.0, color='#4CAF50', lw=1.5)  # f_vid
    arrow(ax, 7.9, 1.2, 9.3, 6.4, color='#4CAF50', lw=1.5)  # f_aud

    # 三个分支 → G-Fuse
    arrow(ax, 8.2, 7.88, 13.0, 7.88, color='#FF9800', lw=1.5)
    arrow(ax, 8.2, 4.72, 13.0, 4.72, color='#FF9800', lw=1.5)
    arrow(ax, 8.2, 1.2, 13.0, 1.2, color='#FF9800', lw=1.5)

    # X-Cons → G-Fuse
    arrow(ax, 12.3, 7.3, 13.0, 7.3, color='#2E7D32', lw=1.8)

    # 输出
    rounded_box(ax, (13.8, 2.3), 3.1, 0.5, fc='#D84315', ec='#BF360C')
    text_center(ax, 15.35, 2.55, 'p_fake  (最终 AI 生成概率)', fontsize=10, color='white', fontweight='bold')
    arrow(ax, 15.35, 3.0, 15.35, 2.8)

    # ========== 图例（底部） ==========
    ax.text(0.5, 0.15, '数据流', ha='center', va='center', fontsize=7,
            color='#757575', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#BDBDBD', alpha=0.7))
    ax.text(0.15, 0.12, '→  主数据流', fontsize=8, color='#212121', transform=ax.transAxes)
    ax.text(0.42, 0.12, '>>  特征融合/辅助流', fontsize=8, color='#4CAF50', transform=ax.transAxes)
    ax.text(0.75, 0.12, '◎  分类头 / 输出', fontsize=8, color='#FF9800', transform=ax.transAxes)

    title(ax, '多模态 AI 生成媒体检测 — 完整架构',
          'Model: CLIP-ViT (Image+Video) + Wav2Vec2 (Audio) + MSFF + T-Agg + X-Cons + G-Fuse')

    fig.tight_layout()
    fig.savefig(OUT_DIR / '01_multimodal_full.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ 01_multimodal_full.png")


# ============================================================
# 2. 图像分支  (image_branch.png)
# ============================================================

def draw_image_branch():
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)

    # 输入
    rounded_box(ax, (0.3, 4.0), 1.3, 1.3, fc='#212121', ec='#212121')
    text_center(ax, 0.95, 4.8, 'IMG', fontsize=10, color='white', fontweight='bold')
    text_center(ax, 0.95, 4.0, '图像输入', fontsize=8, color='white', fontweight='bold')

    # 预处理
    rounded_box(ax, (2.0, 4.0), 1.8, 1.3, fc='#BBDEFB', ec='#1565C0')
    text_center(ax, 2.9, 5.0, '图像预处理', fontsize=10, color='#0D47A1', fontweight='bold')
    text_center(ax, 2.9, 4.5, 'Resize 224×224\nNormalize (mean/std)', fontsize=7.5, color='#37474F')
    text_center(ax, 2.9, 4.0, '数据增强 (训练时)', fontsize=7, color='#5D4037')

    arrow(ax, 1.6, 4.65, 2.0, 4.65)

    # CLIP ViT
    rounded_box(ax, (4.3, 1.5), 3.5, 6.3, fc='#90CAF9', ec='#1565C0')
    pill_label(ax, 6.05, 7.4, 1.8, 0.45, 'CLIP-ViT-B/16', fc='#1565C0', fontsize=10)
    text_center(ax, 6.05, 6.85, 'ViT 视觉编码器', fontsize=11, color='#0D47A1', fontweight='bold')

    # Patch Embedding
    rounded_box(ax, (4.6, 5.9), 2.9, 0.7, fc='#E3F2FD', ec='#1565C0')
    text_center(ax, 6.05, 6.25, 'Patch Embedding (14×14 patches)', fontsize=8, color='#0D47A1', fontweight='bold')

    # Layer 3, 6, 12 标注
    for layer, label, y in [(3, 'Layer 3 (浅层)', 5.1), (6, 'Layer 6 (中层)', 4.4), (12, 'Layer 12 (深层)', 3.7)]:
        rounded_box(ax, (4.6, y - 0.3), 2.9, 0.55, fc='white', ec='#1976D2', lw=1)
        text_center(ax, 6.05, y, label, fontsize=8.5, color='#1565C0', fontweight='bold')
        text_center(ax, 7.8, y, f'hidden_states[{layer}]\n{layer}层 ViT Encoder', fontsize=7, color='#37474F', ha='left')

    # CLS token & patch tokens
    rounded_box(ax, (4.6, 2.75), 2.9, 0.7, fc='white', ec='#1976D2', lw=1)
    text_center(ax, 6.05, 3.1, 'CLS Token + Patch Tokens', fontsize=8, color='#1565C0', fontweight='bold')

    # Pooler
    rounded_box(ax, (4.6, 1.7), 2.9, 0.85, fc='white', ec='#1976D2', lw=1)
    text_center(ax, 6.05, 2.12, 'Post-Layernorm → 特征向量 f_img', fontsize=8, color='#1565C0', fontweight='bold')

    arrow(ax, 3.8, 4.65, 4.3, 4.65)

    # MSFF 多尺度融合（右侧大框）
    rounded_box(ax, (8.3, 4.8), 4.2, 3.8, fc='#64B5F6', ec='#1565C0')
    pill_label(ax, 10.4, 8.25, 1.8, 0.5, 'MSFF  多尺度特征融合', fc='#1565C0', fontsize=11)

    # 三路输入
    text_center(ax, 10.4, 7.5, '三路 ViT 隐藏态', fontsize=9, color='#0D47A1', fontweight='bold')

    rounded_box(ax, (8.7, 6.8), 3.4, 0.55, fc='white', ec='#1976D2')
    text_center(ax, 10.4, 7.08, 'Layer 3: h_s  (浅层语义)', fontsize=7.5, color='#1565C0')
    rounded_box(ax, (8.7, 6.15), 3.4, 0.55, fc='white', ec='#1976D2')
    text_center(ax, 10.4, 6.43, 'Layer 6: h_m  (中层语义)', fontsize=7.5, color='#1565C0')
    rounded_box(ax, (8.7, 5.5), 3.4, 0.55, fc='white', ec='#1976D2')
    text_center(ax, 10.4, 5.78, 'Layer 12: h_d (深层语义)', fontsize=7.5, color='#1565C0')

    # 融合步骤
    text_center(ax, 10.4, 4.95, '→ 特征投影 → 层级注意力加权 → 残差融合',
                fontsize=7.5, color='#0D47A1', fontstyle='italic')

    # 输出图像表征
    rounded_box(ax, (12.9, 5.45), 1.1, 1.0, fc='#1976D2', ec='#0D47A1')
    text_center(ax, 13.45, 5.95, 'f_img', fontsize=10, color='white', fontweight='bold', fontstyle='italic')

    # 箭头: CLIP → MSFF
    arrow(ax, 7.8, 5.1, 8.3, 6.0, color='#1565C0', lw=1.5)
    arrow(ax, 7.8, 4.4, 8.3, 6.7, color='#1565C0', lw=1.5)
    arrow(ax, 7.8, 3.7, 8.3, 7.3, color='#1565C0', lw=1.5)
    arrow(ax, 12.5, 6.7, 12.9, 5.95)

    # SRM / FFT 局部取证支路（右侧下框）
    rounded_box(ax, (8.3, 2.1), 5.7, 2.3, fc='#FFE0B2', ec='#F57C00')
    pill_label(ax, 10.4, 4.05, 1.8, 0.45, 'SRM / FFT  局部取证支路', fc='#F57C00', fontsize=10)

    # 三路输入
    text_center(ax, 11.15, 3.35, '从原图直接计算（不经 ViT）', fontsize=8, color='#E65100', fontstyle='italic')

    # SRM
    rounded_box(ax, (8.7, 2.55), 1.5, 1.15, fc='white', ec='#F57C00')
    text_center(ax, 9.45, 3.3, 'SRM', fontsize=9, color='#E65100', fontweight='bold')
    text_center(ax, 9.45, 2.85, '3 核残差滤波器\n高通噪声残差', fontsize=6.5, color='#37474F')

    # FFT
    rounded_box(ax, (10.55, 2.55), 1.5, 1.15, fc='white', ec='#F57C00')
    text_center(ax, 11.3, 3.3, 'FFT', fontsize=9, color='#E65100', fontweight='bold')
    text_center(ax, 11.3, 2.85, '频域幅度谱\n高频能量占比', fontsize=6.5, color='#37474F')

    # 融合输出
    rounded_box(ax, (12.4, 2.55), 1.3, 1.15, fc='#FFB74D', ec='#F57C00')
    text_center(ax, 13.05, 3.3, 'f_forg', fontsize=9, color='white', fontweight='bold')
    text_center(ax, 13.05, 2.85, '取证特征', fontsize=6.5, color='white')

    # 图像分类头
    rounded_box(ax, (14.1, 5.45), 1.6, 1.0, fc='#0D47A1', ec='#0D47A1')
    text_center(ax, 14.9, 5.95, 'p_img\n分类头', fontsize=8, color='white', fontweight='bold')
    arrow(ax, 14.0, 5.95, 14.1, 5.95)

    # 原图 → SRM/FFT
    arrow(ax, 0.95, 5.3, 0.95, 4.4, color='#F57C00', lw=1.2, style='-')
    arrow(ax, 0.95, 4.4, 9.45, 3.7, color='#F57C00', lw=1.2)
    arrow(ax, 0.95, 4.4, 11.3, 3.7, color='#F57C00', lw=1.2)
    arrow(ax, 10.2, 3.1, 12.4, 3.1)
    arrow(ax, 12.05, 3.1, 12.4, 3.1)
    # f_img + f_forg → 分类头
    arrow(ax, 13.6, 5.95, 14.1, 5.95, color='#1565C0', lw=1.3)
    arrow(ax, 13.05, 3.7, 13.05, 5.45, color='#F57C00', lw=1.3)
    arrow(ax, 13.05, 5.45, 14.1, 5.95, color='#F57C00', lw=1.3)

    title(ax, '图像分支 Image Branch — MSFF 多尺度 + SRM/FFT 局部取证',
          'Input → Preprocess → CLIP-ViT → MSFF + Forensics Branch → f_img / f_forg')

    fig.tight_layout()
    fig.savefig(OUT_DIR / '02_image_branch.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ 02_image_branch.png")


# ============================================================
# 3. 视频分支  (video_branch.png)
# ============================================================

def draw_video_branch():
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)

    rounded_box(ax, (0.3, 4.0), 1.3, 1.3, fc='#212121', ec='#212121')
    text_center(ax, 0.95, 4.8, 'VID', fontsize=10, color='white', fontweight='bold')
    text_center(ax, 0.95, 4.0, '视频输入', fontsize=8, color='white', fontweight='bold')

    rounded_box(ax, (2.0, 4.0), 2.0, 1.3, fc='#B2EBF2', ec='#00695C')
    pill_label(ax, 3.0, 5.0, 1.2, 0.38, '预处理', fc='#00838F', fontsize=8)
    text_center(ax, 3.0, 4.45, '帧采样 T=8~16\nResize / Normalize', fontsize=7.5, color='#004D40')
    arrow(ax, 1.6, 4.65, 2.0, 4.65)

    # 关键帧可视化
    rounded_box(ax, (4.4, 4.2), 2.5, 0.9, fc='white', ec='#00ACC1', lw=1.2)
    text_center(ax, 5.65, 4.65, '[F1 F2 F3 F4 … Fn]', fontsize=9, color='#00695C', fontweight='bold')
    arrow(ax, 4.0, 4.65, 4.4, 4.65)

    # 共享 CLIP/ViT（每帧过一次）
    rounded_box(ax, (7.3, 1.2), 3.2, 6.5, fc='#80DEEA', ec='#00695C')
    pill_label(ax, 8.9, 7.3, 1.6, 0.45, '共享 CLIP-ViT', fc='#00838F', fontsize=10)
    text_center(ax, 8.9, 6.75, '逐帧特征提取', fontsize=10, color='#004D40', fontweight='bold')
    text_center(ax, 8.9, 6.3, '(参数共享)', fontsize=8, color='#006064', fontstyle='italic')

    rounded_box(ax, (7.6, 5.75), 2.6, 0.6, fc='white', ec='#00ACC1')
    text_center(ax, 8.9, 6.05, '每帧 → 768-d 向量', fontsize=8, color='#004D40')

    rounded_box(ax, (7.6, 4.75), 2.6, 0.6, fc='white', ec='#00ACC1')
    text_center(ax, 8.9, 5.05, 'Layer 3 / 6 / 12 隐藏态', fontsize=8, color='#004D40')

    rounded_box(ax, (7.6, 3.75), 2.6, 0.6, fc='white', ec='#00ACC1')
    text_center(ax, 8.9, 4.05, 'CLS token 聚合', fontsize=8, color='#004D40')

    rounded_box(ax, (7.6, 2.55), 2.6, 1.0, fc='#00ACC1', ec='#00695C')
    text_center(ax, 8.9, 3.05, '帧级特征序列\nH ∈ R^{T×D}', fontsize=8.5, color='white', fontweight='bold')

    arrow(ax, 6.9, 4.65, 7.3, 4.65)

    # T-Agg 时序聚合
    rounded_box(ax, (11.0, 1.2), 4.7, 6.5, fc='#4DD0E1', ec='#00695C')
    pill_label(ax, 13.35, 7.3, 1.3, 0.48, 'T-Agg', fc='#00ACC1', fontsize=11)
    text_center(ax, 13.35, 6.75, '时序聚合模块', fontsize=10.5, color='#004D40', fontweight='bold')

    text_center(ax, 13.35, 6.1, '输入: H ∈ R^{T×D}', fontsize=8, color='#004D40')

    # 四个子块
    rounded_box(ax, (11.3, 5.25), 2.0, 0.7, fc='white', ec='#00ACC1')
    text_center(ax, 12.3, 5.6, 'Manba Attention', fontsize=7.5, color='#004D40', fontweight='bold')

    rounded_box(ax, (13.6, 5.25), 1.9, 0.7, fc='white', ec='#00ACC1')
    text_center(ax, 14.55, 5.6, 'Transformer\nEncoder', fontsize=7.5, color='#004D40', fontweight='bold')

    rounded_box(ax, (11.3, 4.1), 2.0, 0.7, fc='white', ec='#00ACC1')
    text_center(ax, 12.3, 4.45, '帧注意力加权', fontsize=7.5, color='#004D40', fontweight='bold')

    rounded_box(ax, (13.6, 4.1), 1.9, 0.7, fc='white', ec='#00ACC1')
    text_center(ax, 14.55, 4.45, '时序建模\nLSTM / GRU', fontsize=7.5, color='#004D40', fontweight='bold')

    text_center(ax, 13.35, 3.25, '→ 全局池化 → 视频表征 f_vid',
                fontsize=8.5, color='#004D40', fontweight='bold', fontstyle='italic')

    # 输出
    rounded_box(ax, (12.5, 2.0), 1.0, 0.9, fc='#00ACC1', ec='#00695C')
    text_center(ax, 13.0, 2.45, 'f_vid', fontsize=10, color='white', fontweight='bold', fontstyle='italic')
    rounded_box(ax, (13.8, 2.0), 1.0, 0.9, fc='#00695C', ec='#004D40')
    text_center(ax, 14.3, 2.45, 'p_vid', fontsize=10, color='white', fontweight='bold')

    # 箭头
    arrow(ax, 10.5, 3.05, 11.0, 4.6, color='#00ACC1', lw=1.5)
    arrow(ax, 12.3, 5.95, 13.35, 5.95)
    arrow(ax, 13.35, 5.95, 14.55, 5.95)
    arrow(ax, 12.3, 4.8, 13.35, 4.8)
    arrow(ax, 13.35, 4.8, 14.55, 4.8)
    arrow(ax, 13.35, 3.8, 13.0, 2.9)
    arrow(ax, 13.0, 2.0, 13.8, 2.0)

    title(ax, '视频分支 Video Branch — 帧采样 + 共享 CLIP-ViT + T-Agg 时序聚合',
          'Input → Frame Sampling → Shared ViT → Frame Features → T-Agg → f_vid / p_vid')

    fig.tight_layout()
    fig.savefig(OUT_DIR / '03_video_branch.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ 03_video_branch.png")


# ============================================================
# 4. 音频分支  (audio_branch.png)
# ============================================================

def draw_audio_branch():
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)

    rounded_box(ax, (0.3, 4.0), 1.3, 1.3, fc='#212121', ec='#212121')
    text_center(ax, 0.95, 4.8, 'AUD', fontsize=10, color='white', fontweight='bold')
    text_center(ax, 0.95, 4.0, '音频输入', fontsize=8, color='white', fontweight='bold')

    rounded_box(ax, (2.0, 4.0), 2.0, 1.3, fc='#E1BEE7', ec='#6A1B9A')
    pill_label(ax, 3.0, 5.0, 1.2, 0.38, '预处理', fc='#8E24AA', fontsize=8)
    text_center(ax, 3.0, 4.45, '重采样 16kHz\n单声道转换', fontsize=8, color='#4A148C')
    arrow(ax, 1.6, 4.65, 2.0, 4.65)

    # Wav2Vec2 / SSL
    rounded_box(ax, (4.6, 1.4), 3.3, 6.3, fc='#CE93D8', ec='#6A1B9A')
    pill_label(ax, 6.25, 7.2, 2.0, 0.48, 'Wav2Vec2-Base  SSL 编码器', fc='#8E24AA', fontsize=10)
    text_center(ax, 6.25, 6.65, '自监督预训练 (LibriSpeech)', fontsize=9, color='#4A148C', fontweight='bold')

    rounded_box(ax, (4.9, 5.9), 2.7, 0.6, fc='white', ec='#8E24AA')
    text_center(ax, 6.25, 6.2, 'Transformer Layers × 12', fontsize=8, color='#4A148C')

    rounded_box(ax, (4.9, 4.9), 2.7, 0.6, fc='white', ec='#8E24AA')
    text_center(ax, 6.25, 5.2, '帧级隐藏态 (20ms/帧)', fontsize=8, color='#4A148C')

    rounded_box(ax, (4.9, 3.9), 2.7, 0.6, fc='white', ec='#8E24AA')
    text_center(ax, 6.25, 4.2, '平均池化 → 时间序列特征', fontsize=8, color='#4A148C')

    rounded_box(ax, (4.9, 2.3), 2.7, 1.2, fc='#8E24AA', ec='#6A1B9A')
    text_center(ax, 6.25, 2.95, 'SSL 特征\nH_a ∈ R^{T×768}', fontsize=9, color='white', fontweight='bold')

    arrow(ax, 4.0, 4.65, 4.6, 4.65)

    # Log-Mel + 2D CNN 支路
    rounded_box(ax, (8.3, 4.4), 3.2, 3.3, fc='#BA68C8', ec='#6A1B9A')
    pill_label(ax, 9.9, 7.35, 2.0, 0.48, 'Log-Mel + 2D CNN', fc='#6A1B9A', fontsize=10)

    rounded_box(ax, (8.6, 6.45), 2.6, 0.7, fc='white', ec='#8E24AA')
    text_center(ax, 9.9, 6.8, 'Log-Mel 频谱图\n128 mel bands, 256 hop', fontsize=7.5, color='#4A148C', fontweight='bold')

    rounded_box(ax, (8.6, 5.45), 2.6, 0.7, fc='white', ec='#8E24AA')
    text_center(ax, 9.9, 5.8, '2D CNN (3 层)\nConv2d + BN + ReLU', fontsize=7.5, color='#4A148C', fontweight='bold')

    rounded_box(ax, (8.6, 4.65), 2.6, 0.6, fc='#7B1FA2', ec='#6A1B9A')
    text_center(ax, 9.9, 4.95, '空间特征 → 时间池化', fontsize=7.5, color='white', fontweight='bold')

    # 融合
    rounded_box(ax, (8.3, 1.4), 3.2, 2.7, fc='#AB47BC', ec='#6A1B9A')
    pill_label(ax, 9.9, 3.95, 2.0, 0.48, '融合 + 注意力池化', fc='#6A1B9A', fontsize=10)

    rounded_box(ax, (8.6, 3.0), 2.6, 0.7, fc='white', ec='#8E24AA')
    text_center(ax, 9.9, 3.35, 'SSL 特征 ⊕ 频谱特征', fontsize=8, color='#4A148C', fontweight='bold')

    rounded_box(ax, (8.6, 2.1), 2.6, 0.7, fc='white', ec='#8E24AA')
    text_center(ax, 9.9, 2.45, '多头注意力池化\n(学习帧权重)', fontsize=7.5, color='#4A148C')

    rounded_box(ax, (8.6, 1.55), 2.6, 0.4, fc='#6A1B9A', ec='#4A148C')
    text_center(ax, 9.9, 1.75, '→ f_aud 音频表征向量', fontsize=7.5, color='white', fontweight='bold')

    # 输出
    rounded_box(ax, (12.2, 3.0), 1.3, 1.0, fc='#6A1B9A', ec='#4A148C')
    text_center(ax, 12.85, 3.5, 'f_aud', fontsize=10, color='white', fontweight='bold', fontstyle='italic')
    rounded_box(ax, (12.2, 1.5), 1.3, 1.0, fc='#4A148C', ec='#4A148C')
    text_center(ax, 12.85, 2.0, 'p_aud', fontsize=10, color='white', fontweight='bold')

    # 箭头
    arrow(ax, 7.6, 2.9, 8.3, 2.7, color='#8E24AA', lw=1.5)
    arrow(ax, 3.0, 5.3, 3.0, 6.1, color='#8E24AA', lw=1.2)  # 预处理 → Log-Mel
    arrow(ax, 3.0, 6.1, 9.9, 7.15, color='#8E24AA', lw=1.2)
    arrow(ax, 11.5, 5.8, 11.7, 3.8)
    arrow(ax, 11.5, 3.35, 12.2, 3.5)
    arrow(ax, 12.2, 2.8, 12.2, 2.5)

    title(ax, '音频分支 Audio Branch — Wav2Vec2 SSL + Log-Mel CNN + 注意力池化',
          'Input → Preprocess → Wav2Vec2 / Log-Mel → Feature Fusion → Attn Pool → f_aud / p_aud')

    fig.tight_layout()
    fig.savefig(OUT_DIR / '04_audio_branch.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ 04_audio_branch.png")


# ============================================================
# 5. G-Fuse 门控融合  (gated_fusion.png)
# ============================================================

def draw_gated_fusion():
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)

    # 五个输入
    inputs = [
        ('f_img',    0.4, 7.8, '#1565C0', '#E3F2FD'),
        ('f_vid',    0.4, 5.8, '#00695C', '#E0F7FA'),
        ('f_aud',    0.4, 3.8, '#6A1B9A', '#F3E5F5'),
        ('f_sync',   0.4, 1.8, '#2E7D32', '#E8F5E9'),
        ('f_async',  0.4, 0.3, '#2E7D32', '#E8F5E9'),
    ]
    for label, x, y, ec, fc in inputs:
        rounded_box(ax, (x, y - 0.55), 1.2, 1.1, fc=fc, ec=ec)
        text_center(ax, x + 0.6, y, label, fontsize=10, color=ec, fontweight='bold', fontstyle='italic')
        arrow(ax, x + 1.6, y, 3.0, y, color=ec, lw=1.4)

    # 特征对齐
    rounded_box(ax, (3.0, 0.3), 2.3, 8.2, fc='#FFF3E0', ec='#E65100')
    pill_label(ax, 4.15, 8.0, 1.4, 0.45, 'G-Fuse  门控融合', fc='#E65100', fontsize=11)

    rounded_box(ax, (3.3, 6.6), 1.7, 1.2, fc='white', ec='#E65100')
    text_center(ax, 4.15, 7.2, '特征对齐', fontsize=9.5, color='#BF360C', fontweight='bold')
    text_center(ax, 4.15, 6.85, 'MLP 投影到\n共同嵌入空间', fontsize=7, color='#37474F')

    rounded_box(ax, (3.3, 4.5), 1.7, 1.7, fc='white', ec='#E65100')
    pill_label(ax, 4.15, 5.9, 1.1, 0.38, '门控权重 g_k', fc='#FF6D00', fontsize=9)
    text_center(ax, 4.15, 5.2, '可靠性评估', fontsize=9, color='#BF360C', fontweight='bold')
    text_center(ax, 4.15, 4.8, '对每个模态预测\n可靠性权重 g_k ∈ [0,1]', fontsize=7, color='#37474F')

    rounded_box(ax, (3.3, 2.7), 1.7, 1.5, fc='white', ec='#E65100')
    pill_label(ax, 4.15, 3.85, 1.1, 0.38, '模态掩码 m_k', fc='#FF6D00', fontsize=9)
    text_center(ax, 4.15, 3.25, '缺失模态处理', fontsize=8.5, color='#BF360C', fontweight='bold')
    text_center(ax, 4.15, 2.9, '有模态 → m_k=1\n缺失 → m_k=0', fontsize=7, color='#37474F')

    rounded_box(ax, (3.3, 0.6), 1.7, 1.8, fc='#FFE0B2', ec='#E65100')
    pill_label(ax, 4.15, 2.0, 1.4, 0.4, '加权融合', fc='#E65100', fontsize=9)
    text_center(ax, 4.15, 1.45, 'f = Σ m_k · g_k · f_k', fontsize=8, color='#BF360C', fontweight='bold', fontstyle='italic')
    text_center(ax, 4.15, 1.0, '归一化: g_k / Σ g_k', fontsize=7, color='#37474F')

    # 连接箭头（内部）
    arrow(ax, 4.15, 6.05, 4.15, 6.2)
    arrow(ax, 4.15, 5.4, 4.15, 4.2)
    arrow(ax, 4.15, 4.1, 4.15, 3.2)
    arrow(ax, 4.15, 2.4, 4.15, 2.4)
    arrow(ax, 4.15, 1.6, 4.15, 1.6)

    # 全连接分类器
    rounded_box(ax, (6.0, 1.6), 2.2, 5.2, fc='#FFCC80', ec='#E65100')
    pill_label(ax, 7.1, 6.4, 1.2, 0.4, '分类头', fc='#E65100', fontsize=10)
    text_center(ax, 7.1, 5.9, 'MLP Classifier', fontsize=9, color='#BF360C', fontweight='bold')

    rounded_box(ax, (6.3, 5.2), 1.6, 0.7, fc='white', ec='#E65100')
    text_center(ax, 7.1, 5.55, 'Linear + ReLU + Dropout', fontsize=7.5, color='#BF360C')

    rounded_box(ax, (6.3, 4.1), 1.6, 0.7, fc='white', ec='#E65100')
    text_center(ax, 7.1, 4.45, 'Linear (2 classes)', fontsize=7.5, color='#BF360C')

    rounded_box(ax, (6.3, 3.0), 1.6, 0.7, fc='white', ec='#E65100')
    text_center(ax, 7.1, 3.35, 'Softmax', fontsize=8, color='#BF360C', fontweight='bold')

    rounded_box(ax, (6.3, 1.9), 1.6, 0.7, fc='#BF360C', ec='#E65100')
    text_center(ax, 7.1, 2.25, '[p_real, p_fake]', fontsize=7.5, color='white', fontweight='bold')

    arrow(ax, 5.0, 1.5, 6.0, 5.5)
    arrow(ax, 7.1, 4.85, 7.1, 4.1)
    arrow(ax, 7.1, 3.8, 7.1, 3.0)
    arrow(ax, 7.1, 2.7, 7.1, 1.9)

    # 最终输出
    rounded_box(ax, (9.0, 2.9), 2.5, 1.2, fc='#BF360C', ec='#8D6E63')
    text_center(ax, 10.25, 3.5, 'p_fake', fontsize=14, color='white', fontweight='bold', fontstyle='italic')
    text_center(ax, 10.25, 3.15, 'AI 生成概率', fontsize=8, color='white')
    arrow(ax, 7.9, 2.25, 9.0, 3.5)

    rounded_box(ax, (12.0, 2.9), 2.5, 1.2, fc='#E65100', ec='#BF360C')
    text_center(ax, 13.25, 3.5, '最终裁决', fontsize=10, color='white', fontweight='bold')
    text_center(ax, 13.25, 3.15, 'Real  /  Fake', fontsize=9, color='white', fontstyle='italic')
    arrow(ax, 11.5, 3.5, 12.0, 3.5)

    title(ax, '门控融合 G-Fuse — 可靠性加权 + 缺失模态处理',
          'f_img, f_vid, f_aud, f_sync, f_async → Align → Gate → Weighted Sum → MLP → p_fake')

    fig.tight_layout()
    fig.savefig(OUT_DIR / '05_gated_fusion.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ 05_gated_fusion.png")


# ============================================================
# 6. 跨模态一致性  (cross_modal_consistency.png)
# ============================================================

def draw_cross_modal_consistency():
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)

    # 两路输入（音画 + 图像）
    rounded_box(ax, (0.3, 5.6), 1.8, 1.3, fc='#E0F7FA', ec='#00695C')
    pill_label(ax, 1.2, 6.55, 1.3, 0.38, '视频分支', fc='#00838F', fontsize=9)
    text_center(ax, 1.2, 6.05, 'f_vid', fontsize=11, color='#004D40', fontweight='bold', fontstyle='italic')
    arrow(ax, 2.1, 6.25, 3.0, 6.25)

    rounded_box(ax, (0.3, 1.9), 1.8, 1.3, fc='#F3E5F5', ec='#6A1B9A')
    pill_label(ax, 1.2, 2.85, 1.3, 0.38, '音频分支', fc='#8E24AA', fontsize=9)
    text_center(ax, 1.2, 2.35, 'f_aud', fontsize=11, color='#4A148C', fontweight='bold', fontstyle='italic')
    arrow(ax, 2.1, 2.55, 3.0, 2.55)

    # ===== X-Cons 大框 =====
    rounded_box(ax, (3.0, 0.5), 8.2, 8.0, fc='#E8F5E9', ec='#2E7D32')
    pill_label(ax, 7.1, 7.95, 2.0, 0.5, 'X-Cons  跨模态一致性', fc='#2E7D32', fontsize=11)
    text_center(ax, 7.1, 7.4, '同步 + 异步 音画对齐建模', fontsize=9, color='#1B5E20', fontweight='bold', fontstyle='italic')

    # 投影层
    rounded_box(ax, (3.3, 6.0), 2.0, 0.9, fc='white', ec='#2E7D32')
    text_center(ax, 4.3, 6.45, '投影头 (视频)', fontsize=8, color='#1B5E20', fontweight='bold')
    text_center(ax, 4.3, 6.1, 'MLP → z_v', fontsize=7, color='#37474F')
    rounded_box(ax, (3.3, 2.3), 2.0, 0.9, fc='white', ec='#2E7D32')
    text_center(ax, 4.3, 2.75, '投影头 (音频)', fontsize=8, color='#1B5E20', fontweight='bold')
    text_center(ax, 4.3, 2.4, 'MLP → z_a', fontsize=7, color='#37474F')
    arrow(ax, 3.0, 6.25, 3.3, 6.45)
    arrow(ax, 3.0, 2.55, 3.3, 2.75)

    # ===== 同步对比学习 =====
    rounded_box(ax, (5.8, 4.5), 4.9, 3.2, fc='#C8E6C9', ec='#2E7D32')
    pill_label(ax, 8.25, 7.45, 1.8, 0.45, '同步对比学习', fc='#2E7D32', fontsize=10)

    rounded_box(ax, (6.1, 6.65), 1.5, 0.55, fc='white', ec='#2E7D32')
    text_center(ax, 6.85, 6.92, '(匹配正样本)', fontsize=7, color='#1B5E20', fontweight='bold')
    rounded_box(ax, (6.1, 5.2), 1.5, 0.85, fc='white', ec='#2E7D32')
    text_center(ax, 6.85, 5.75, '(不匹配负样本)', fontsize=7, color='#546E7A', fontweight='bold')

    rounded_box(ax, (8.0, 5.85), 2.3, 0.6, fc='white', ec='#2E7D32')
    text_center(ax, 9.15, 6.15, 'InfoNCE / Triplet Loss', fontsize=7.5, color='#1B5E20', fontweight='bold')

    rounded_box(ax, (8.0, 4.95), 2.3, 0.6, fc='white', ec='#2E7D32')
    text_center(ax, 9.15, 5.25, '拉近 / 推远特征空间', fontsize=7.5, color='#546E7A')

    arrow(ax, 5.3, 6.45, 6.1, 6.92)
    arrow(ax, 5.3, 2.75, 6.1, 5.62)
    arrow(ax, 7.6, 6.92, 8.0, 6.15)
    arrow(ax, 7.6, 5.62, 8.0, 5.25)

    rounded_box(ax, (8.3, 4.55), 1.7, 0.35, fc='#2E7D32', ec='#2E7D32')
    text_center(ax, 9.15, 4.72, 'f_sync', fontsize=8, color='white', fontweight='bold', fontstyle='italic')

    # ===== 异步对齐建模 =====
    rounded_box(ax, (5.8, 1.0), 4.9, 3.2, fc='#DCEDC8', ec='#558B2F')
    pill_label(ax, 8.25, 3.95, 1.8, 0.45, '异步对齐建模', fc='#558B2F', fontsize=10)

    rounded_box(ax, (6.1, 3.1), 4.4, 0.75, fc='white', ec='#558B2F')
    text_center(ax, 8.3, 3.48, '时间偏移相关性 (±50ms ~ ±500ms)', fontsize=7.5, color='#33691E')

    rounded_box(ax, (6.1, 2.05), 4.4, 0.75, fc='white', ec='#558B2F')
    text_center(ax, 8.3, 2.43, '跨时间帧注意力机制 / DTW 对齐', fontsize=7.5, color='#33691E')

    rounded_box(ax, (8.3, 1.25), 1.7, 0.45, fc='#558B2F', ec='#558B2F')
    text_center(ax, 9.15, 1.48, 'f_async', fontsize=8.5, color='white', fontweight='bold', fontstyle='italic')

    arrow(ax, 5.3, 6.45, 6.1, 3.48, color='#33691E', lw=1.0)
    arrow(ax, 5.3, 2.75, 6.1, 2.43, color='#33691E', lw=1.0)
    arrow(ax, 10.5, 3.48, 10.0, 1.48, color='#558B2F', lw=1.0)
    arrow(ax, 10.5, 2.43, 10.0, 1.48, color='#558B2F', lw=1.0)

    # 输出
    text_center(ax, 13.25, 6.25, '→ f_sync', fontsize=10, color='#2E7D32', fontweight='bold', fontstyle='italic')
    text_center(ax, 13.25, 2.45, '→ f_async', fontsize=10, color='#558B2F', fontweight='bold', fontstyle='italic')

    arrow(ax, 10.0, 4.72, 12.5, 6.25, color='#2E7D32', lw=1.5)
    arrow(ax, 10.0, 1.48, 12.5, 2.45, color='#558B2F', lw=1.5)

    # 右侧说明
    rounded_box(ax, (11.5, 4.4), 3.5, 1.3, fc='white', ec='#9E9E9E')
    text_center(ax, 13.25, 5.0, '同步一致性 f_sync', fontsize=9, color='#2E7D32', fontweight='bold')
    text_center(ax, 13.25, 4.65, '同一时刻音画是否匹配\n(真实视频自然对齐)', fontsize=7, color='#546E7A')

    rounded_box(ax, (11.5, 0.6), 3.5, 1.3, fc='white', ec='#9E9E9E')
    text_center(ax, 13.25, 1.2, '异步一致性 f_async', fontsize=9, color='#558B2F', fontweight='bold')
    text_center(ax, 13.25, 0.85, '相邻时刻音画是否协调\n(AI 视频易有错位)', fontsize=7, color='#546E7A')

    title(ax, '跨模态一致性 X-Cons — 同步 + 异步 音画对齐学习',
          'f_vid, f_aud → Projection Heads → Contrastive Learning → f_sync / f_async')

    fig.tight_layout()
    fig.savefig(OUT_DIR / '06_cross_modal_consistency.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ 06_cross_modal_consistency.png")


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  多模态 AIGC 检测架构图生成")
    print(f"  输出目录: {OUT_DIR}")
    print("=" * 70)

    draw_full_architecture()
    draw_image_branch()
    draw_video_branch()
    draw_audio_branch()
    draw_gated_fusion()
    draw_cross_modal_consistency()

    print("\n" + "=" * 70)
    print(f"  ✅ 全部 6 张架构图已生成到:")
    print(f"     {OUT_DIR}")
    print("=" * 70)
