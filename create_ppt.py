"""
多模态 AIGC 检测系统 PPT 生成器
================================
特点:
  1. 每个模块独立幻灯片 —— 各部分可单独修改版本
  2. python-pptx（纯 Python，无需 Node.js）
  3. 复用 docs/architecture/ 下的 6 张架构图
  4. 深蓝 + 洋红 专业配色，支持中文 SimHei / Microsoft YaHei
  5. 版本参数集中在顶部模块 —— 改 VERSIONS 字典即可升级各模块版本

使用: python create_ppt.py
"""

import os, sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import json

PROJECT_ROOT = Path(__file__).parent.resolve()
ARCH_DIR = PROJECT_ROOT / 'docs' / 'architecture'
OUT_PPTX = PROJECT_ROOT / 'docs' / 'AIGC_Detection_System.pptx'

# ============================================================
# 版本参数字典 —— 改这里即可升级各模块版本
# ============================================================
VERSIONS = {
    "image_branch": {
        "name": "图像分支",
        "backbone": "openai/clip-vit-base-patch16",
        "version": "v1.3.2",
        "params": "~86M",
        "inputs": "单张图像 224×224",
        "heads": "MSFF (3层隐藏态) + SRM/FFT 局部取证",
        "update_date": "2026-10-31",
    },
    "video_branch": {
        "name": "视频分支",
        "backbone": "openai/clip-vit-base-patch16 (共享)",
        "version": "v1.2.1",
        "params": "共享 (零额外视觉参数)",
        "inputs": "视频帧序列 T=8~16",
        "heads": "T-Agg (Manba Attention + 帧权重聚合)",
        "update_date": "2026-10-31",
    },
    "audio_branch": {
        "name": "音频分支",
        "backbone": "facebook/wav2vec2-base-960h",
        "version": "v1.4.0",
        "params": "~95M",
        "inputs": "16kHz 单声道波形",
        "heads": "Log-Mel + 2D CNN + 注意力池化",
        "update_date": "2026-10-31",
    },
    "x_cons": {
        "name": "跨模态一致性 X-Cons",
        "backbone": "投影头 (MLP 2层)",
        "version": "v1.1.0",
        "params": "~2M",
        "inputs": "f_vid ∈ R^d  +  f_aud ∈ R^d",
        "heads": "InfoNCE 同步对比学习 + 异步时间对齐",
        "update_date": "2026-10-31",
    },
    "g_fuse": {
        "name": "门控融合 G-Fuse",
        "backbone": "MLP Classifier (2层)",
        "version": "v1.3.0",
        "params": "~1.2M",
        "inputs": "f_img, f_vid, f_aud, f_sync, f_async",
        "heads": "特征对齐 → g_k 门控权重 → m_k 模态掩码 → 分类",
        "update_date": "2026-10-31",
    },
    "detector_total": {
        "name": "完整检测器",
        "version": "v2.0.0",
        "params_total": "~319.7M",
        "modalities": "Image + Video + Audio (支持缺失模态)",
        "thresholds": "Image 0.47 / Video 0.20 / Audio 0.50",
        "acc_image": "19/23 = 82.6%",
        "acc_video": "11/12 = 91.7%",
        "acc_audio": "3/3 = 100.0%",
        "acc_total": "33/38 = 86.8%",
    },
}

# ============================================================
# 颜色主题
# ============================================================
C = {
    "primary":    RGBColor(0x1B, 0x3A, 0x5C),  # 深蓝
    "secondary":  RGBColor(0x2E, 0x86, 0xAB),  # 青
    "accent":     RGBColor(0xA2, 0x3B, 0x72),  # 洋红
    "light":      RGBColor(0xF4, 0xF7, 0xF9),  # 浅灰白
    "border":     RGBColor(0xC5, 0xCD, 0xD4),  # 浅灰
    "text":       RGBColor(0x1B, 0x3A, 0x5C),
    "muted":      RGBColor(0x6B, 0x7B, 0x8B),
    "white":      RGBColor(0xFF, 0xFF, 0xFF),
    "fake":       RGBColor(0xC4, 0x4E, 0x52),
    "real":       RGBColor(0x55, 0xA8, 0x68),
}

FONT_TITLE = 'SimHei'
FONT_BODY  = 'Microsoft YaHei'
FONT_EN    = 'Calibri'

SLIDE_W = Inches(13.333)   # 16:9 宽屏
SLIDE_H = Inches(7.5)


# ============================================================
# 辅助函数
# ============================================================

def add_bg(slide, color=C["white"]):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_bar(slide, left, top, width, height, color):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp

def add_text(slide, left, top, width, height, text, size=18, bold=False,
             color=None, align=PP_ALIGN.LEFT, font=FONT_BODY, line_spacing=1.2):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(2)
    tf.margin_bottom = Emu(2)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    # 设置中文字体
    rPr = run._r.get_or_add_rPr()
    ea = rPr.makeelement(qn('a:ea'), {})
    ea.set('typeface', font)
    rPr.append(ea)
    return tb

def add_card(slide, left, top, width, height, title, items,
             accent_color=C["secondary"], bg_color=C["white"]):
    """信息卡片: 顶部色条 + 标题 + 项目列表"""
    # 背景
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = C["border"]
    card.line.width = Pt(0.75)
    # 顶部色条
    bar = add_bar(slide, left, top, width, Inches(0.08), accent_color)
    # 标题
    add_text(slide, left + Inches(0.2), top + Inches(0.15), width - Inches(0.4),
             Inches(0.4), title, size=16, bold=True, color=C["primary"], font=FONT_TITLE)
    # 内容
    content_top = top + Inches(0.6)
    content_h = height - Inches(0.7)
    add_text(slide, left + Inches(0.2), content_top, width - Inches(0.4),
             content_h, items, size=11, color=C["text"], font=FONT_BODY, align=PP_ALIGN.LEFT)

def add_version_bar(slide, module_key, top=Inches(6.9)):
    """可修改的版本号条 —— 每张模块幻灯片底部显示"""
    v = VERSIONS[module_key]
    if "backbone" in v:
        text = f"📦  {v['name']}  {v['version']}  —  {v['backbone']}  |  {v['params']}  |  更新 {v['update_date']}"
    else:
        text = f"📦  {v['name']}  {v['version']}  —  {v['params_total']}  |  {v['modalities']}  |  总体准确率 {v['acc_total']}"
    add_bar(slide, Inches(0), top, SLIDE_W, Inches(0.65), C["primary"])
    add_text(slide, Inches(0.4), top + Inches(0.08), SLIDE_W - Inches(0.8), Inches(0.5),
             text, size=13, bold=True, color=C["white"], font=FONT_EN)

def add_title_bar(slide, title_str, subtitle_str=None):
    """统一的顶部标题栏"""
    add_bg(slide, C["light"])
    add_bar(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.15), C["primary"])
    add_text(slide, Inches(0.6), Inches(0.15), SLIDE_W - Inches(1.2), Inches(0.6),
             title_str, size=28, bold=True, color=C["white"], font=FONT_TITLE)
    if subtitle_str:
        add_text(slide, Inches(0.6), Inches(0.75), SLIDE_W - Inches(1.2), Inches(0.35),
                 subtitle_str, size=12, color=C["secondary"], font=FONT_EN)

def add_image(slide, path, left, top, width, height=None):
    if not Path(path).exists():
        return
    img = slide.shapes.add_picture(str(path), left, top, width=width, height=height)
    return img


# ============================================================
# 各幻灯片构建
# ============================================================

def slide_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_bg(slide, C["primary"])
    # 装饰条
    add_bar(slide, Inches(0), Inches(3.2), SLIDE_W, Inches(0.08), C["accent"])
    add_bar(slide, Inches(0), Inches(5.8), SLIDE_W, Inches(0.04), C["secondary"])

    add_text(slide, Inches(1), Inches(1.2), SLIDE_W - Inches(2), Inches(1.2),
             "多模态 AI 生成媒体检测系统", size=44, bold=True,
             color=C["white"], font=FONT_TITLE, align=PP_ALIGN.CENTER)
    add_text(slide, Inches(1), Inches(2.3), SLIDE_W - Inches(2), Inches(0.6),
             "Multimodal AIGC Detection  —  Image  /  Video  /  Audio",
             size=18, color=C["secondary"], font=FONT_EN, align=PP_ALIGN.CENTER)

    add_text(slide, Inches(1), Inches(3.4), SLIDE_W - Inches(2), Inches(1.0),
             "架构:  CLIP-ViT + Wav2Vec2 + MSFF + T-Agg + X-Cons + G-Fuse",
             size=14, color=C["light"], font=FONT_BODY, align=PP_ALIGN.CENTER)

    add_text(slide, Inches(1), Inches(4.2), SLIDE_W - Inches(2), Inches(0.5),
             "支持缺失模态  ·  门控融合  ·  校准后准确率 86.8%",
             size=13, color=C["white"], font=FONT_BODY, align=PP_ALIGN.CENTER)

    # 版本号
    add_text(slide, Inches(1), Inches(6.0), SLIDE_W - Inches(2), Inches(0.4),
             f"System {VERSIONS['detector_total']['version']}   |   {VERSIONS['detector_total']['params_total']} parameters",
             size=12, color=C["accent"], bold=True, font=FONT_EN, align=PP_ALIGN.CENTER)

    add_text(slide, Inches(1), Inches(6.4), SLIDE_W - Inches(2), Inches(0.4),
             "© AIGC Detection Project  |  2026",
             size=10, color=C["secondary"], font=FONT_EN, align=PP_ALIGN.CENTER)


def slide_agenda(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "目录  Agenda", "Contents")

    items = [
        ("01", "整体架构总览",      "Multimodal Full Architecture",   "CLIP-ViT + Wav2Vec2 + G-Fuse"),
        ("02", "图像分支  Φ_img",   "Image Branch",                   "CLIP-ViT + MSFF + SRM/FFT"),
        ("03", "视频分支  Φ_vid",   "Video Branch",                   "共享 CLIP + T-Agg Manba"),
        ("04", "音频分支  Φ_aud",   "Audio Branch",                   "Wav2Vec2 + Log-Mel CNN"),
        ("05", "跨模态一致性",       "X-Cons Cross-Modal Consistency", "同步 + 异步 音画对齐"),
        ("06", "门控融合 G-Fuse",   "Gated Fusion",                   "门控加权 + 缺失模态处理"),
        ("07", "实验结果",           "Experimental Results",            "校准后检测准确率"),
        ("08", "各模块版本",         "Module Versions (Editable)",      "各部分可独立修改版本号"),
    ]

    for i, (num, cn, en, detail) in enumerate(items):
        row = i // 2
        col = i % 2
        left = Inches(0.6 + col * 6.35)
        top = Inches(1.5 + row * 1.35)

        # 序号圆
        circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top + Inches(0.2),
                                        Inches(0.7), Inches(0.7))
        circle.fill.solid()
        circle.fill.fore_color.rgb = C["primary"]
        circle.line.fill.background()
        add_text(slide, left, top + Inches(0.2), Inches(0.7), Inches(0.7),
                 num, size=22, bold=True, color=C["white"], font=FONT_EN, align=PP_ALIGN.CENTER)

        # 标题
        add_text(slide, left + Inches(0.9), top + Inches(0.05), Inches(5.2), Inches(0.4),
                 cn, size=17, bold=True, color=C["primary"], font=FONT_TITLE)
        add_text(slide, left + Inches(0.9), top + Inches(0.45), Inches(5.2), Inches(0.3),
                 en, size=11, color=C["secondary"], font=FONT_EN)
        add_text(slide, left + Inches(0.9), top + Inches(0.8), Inches(5.2), Inches(0.5),
                 detail, size=11, color=C["muted"], font=FONT_BODY)


def slide_full_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "整体架构总览", "Multimodal Full Architecture")

    # 左侧架构图 (大图)
    img_path = ARCH_DIR / '01_multimodal_full.png'
    add_image(slide, img_path, Inches(0.3), Inches(1.3), width=Inches(8.2))

    # 右侧关键参数
    right_x = Inches(8.7)
    add_card(slide, right_x, Inches(1.3), Inches(4.4), Inches(1.9),
             "完整检测器参数",
             f"""总参数量:  {VERSIONS['detector_total']['params_total']}
模态输入:  {VERSIONS['detector_total']['modalities']}
各模态阈值: {VERSIONS['detector_total']['thresholds']}
版本号:    {VERSIONS['detector_total']['version']}""",
             accent_color=C["primary"])

    add_card(slide, right_x, Inches(3.4), Inches(4.4), Inches(2.2),
             "校准后检测准确率",
             f"""📷 图像: {VERSIONS['detector_total']['acc_image']}
🎬 视频: {VERSIONS['detector_total']['acc_video']}
🎵 音频: {VERSIONS['detector_total']['acc_audio']}

🏆  总准确率: {VERSIONS['detector_total']['acc_total']}""",
             accent_color=C["accent"])

    add_card(slide, right_x, Inches(5.8), Inches(4.4), Inches(1.1),
             "数据流",
             "f_img / f_vid / f_aud → X-Cons → G-Fuse → p_fake",
             accent_color=C["secondary"])

    add_version_bar(slide, "detector_total")


def slide_image_branch(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "图像分支  Φ_img", "Image Branch — CLIP-ViT + MSFF + SRM/FFT")

    img_path = ARCH_DIR / '02_image_branch.png'
    add_image(slide, img_path, Inches(0.3), Inches(1.3), width=Inches(8.2))

    v = VERSIONS["image_branch"]
    right_x = Inches(8.7)
    add_card(slide, right_x, Inches(1.3), Inches(4.4), Inches(1.7),
             "骨干网络",
             f"""📦 {v['backbone']}
🕳 输入: 224×224 Patch-Size 16
🔒 预训练权重: 公开""" , accent_color=C["primary"])

    add_card(slide, right_x, Inches(3.2), Inches(4.4), Inches(1.7),
             "MSFF 多尺度融合",
             """Layer 3 (浅层)  +  Layer 6 (中层)  +  Layer 12 (深层)
特征投影 → 层级注意力加权 → 残差融合""", accent_color=C["secondary"])

    add_card(slide, right_x, Inches(5.1), Inches(4.4), Inches(1.5),
             "SRM / FFT 局部取证支路",
             "SRM 3 核残差滤波器 + FFT 频谱高频能量占比\n直接从原图计算（不经 ViT）",
             accent_color=C["accent"])

    add_card(slide, right_x, Inches(6.7), Inches(4.4), Inches(0.6),
             f"{v['name']}  版本 {v['version']}",
             f"参数: {v['params']}  |  更新 {v['update_date']}",
             accent_color=C["primary"])
    add_version_bar(slide, "image_branch")


def slide_video_branch(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "视频分支  Φ_vid", "Video Branch — Shared CLIP-ViT + T-Agg (Manba)")

    img_path = ARCH_DIR / '03_video_branch.png'
    add_image(slide, img_path, Inches(0.3), Inches(1.3), width=Inches(8.2))

    v = VERSIONS["video_branch"]
    right_x = Inches(8.7)
    add_card(slide, right_x, Inches(1.3), Inches(4.4), Inches(1.7),
             "帧采样 + 共享编码器",
             f"""输入帧 T = 8 ~ 16
逐帧过共享 CLIP-ViT (参数共享)
输出 H ∈ R^{'{T×d}'}""", accent_color=C["primary"])

    add_card(slide, right_x, Inches(3.2), Inches(4.4), Inches(1.7),
             "T-Agg 时序聚合",
             """Manba Attention
Transformer Encoder
帧级注意力加权 → 全局池化""", accent_color=C["secondary"])

    add_card(slide, right_x, Inches(5.1), Inches(4.4), Inches(1.5),
             "关键特征",
             "帧间差分 / 帧级 blockiness 方差\ntemporal_consistency / motion_score",
             accent_color=C["accent"])

    add_card(slide, right_x, Inches(6.7), Inches(4.4), Inches(0.6),
             f"{v['name']}  版本 {v['version']}",
             f"参数: {v['params']}  |  更新 {v['update_date']}",
             accent_color=C["primary"])
    add_version_bar(slide, "video_branch")


def slide_audio_branch(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "音频分支  Φ_aud", "Audio Branch — Wav2Vec2 SSL + Log-Mel 2D CNN")

    img_path = ARCH_DIR / '04_audio_branch.png'
    add_image(slide, img_path, Inches(0.3), Inches(1.3), width=Inches(8.2))

    v = VERSIONS["audio_branch"]
    right_x = Inches(8.7)
    add_card(slide, right_x, Inches(1.3), Inches(4.4), Inches(1.7),
             "SSL 预训练编码器",
             f"""📦 {v['backbone']}
Transformer Layers × 12
20ms/帧隐藏态 H_a ∈ R^{'{T×768}'}""", accent_color=C["primary"])

    add_card(slide, right_x, Inches(3.2), Inches(4.4), Inches(1.7),
             "Log-Mel + 2D CNN 频谱支路",
             """128 mel bands / 256 hop length
3 层 Conv2d + BN + ReLU
空间特征 → 时间池化""", accent_color=C["secondary"])

    add_card(slide, right_x, Inches(5.1), Inches(4.4), Inches(1.5),
             "融合 + 注意力池化",
             "SSL 特征 ⊕ 频谱特征 → 多头注意力池化 → f_aud",
             accent_color=C["accent"])

    add_card(slide, right_x, Inches(6.7), Inches(4.4), Inches(0.6),
             f"{v['name']}  版本 {v['version']}",
             f"参数: {v['params']}  |  更新 {v['update_date']}",
             accent_color=C["primary"])
    add_version_bar(slide, "audio_branch")


def slide_x_cons(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "跨模态一致性  X-Cons", "Cross-Modal Consistency — Synchronous + Asynchronous")

    img_path = ARCH_DIR / '06_cross_modal_consistency.png'
    add_image(slide, img_path, Inches(0.3), Inches(1.3), width=Inches(8.2))

    v = VERSIONS["x_cons"]
    right_x = Inches(8.7)
    add_card(slide, right_x, Inches(1.3), Inches(4.4), Inches(2.0),
             "投影头 (MLP)",
             "视频投影头:  f_vid → z_v\n音频投影头:  f_aud → z_a\n共同嵌入空间 d=256",
             accent_color=C["primary"])

    add_card(slide, right_x, Inches(3.5), Inches(4.4), Inches(2.0),
             "同步对比学习 → f_sync",
             "InfoNCE Loss\n正样本: 同一时刻音画对\n负样本: 不同配对对\n拉近 / 推远特征空间",
             accent_color=C["secondary"])

    add_card(slide, right_x, Inches(5.7), Inches(4.4), Inches(1.6),
             "异步对齐 → f_async",
             "时间偏移相关性 ±50ms~±500ms\n跨帧注意力机制 / DTW",
             accent_color=C["accent"])

    add_card(slide, right_x, Inches(6.7), Inches(4.4), Inches(0.6),
             f"{v['name']}  版本 {v['version']}",
             f"参数: {v['params']}  |  更新 {v['update_date']}",
             accent_color=C["primary"])
    add_version_bar(slide, "x_cons")


def slide_g_fuse(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "门控融合  G-Fuse", "Gated Fusion — Reliability-Weighted + Missing Modality")

    img_path = ARCH_DIR / '05_gated_fusion.png'
    add_image(slide, img_path, Inches(0.3), Inches(1.3), width=Inches(8.2))

    v = VERSIONS["g_fuse"]
    right_x = Inches(8.7)
    add_card(slide, right_x, Inches(1.3), Inches(4.4), Inches(1.7),
             "5 路输入特征",
             """f_img    图像表征
f_vid    视频表征
f_aud    音频表征
f_sync   同步音画对齐
f_async  异步音画对齐""", accent_color=C["primary"])

    add_card(slide, right_x, Inches(3.2), Inches(4.4), Inches(1.7),
             "门控加权公式",
             "g_k = Reliability(f_k)\nf = Σ m_k · g_k · f_k\n归一化:  g_k / Σ g_k",
             accent_color=C["secondary"])

    add_card(slide, right_x, Inches(5.1), Inches(4.4), Inches(1.5),
             "模态掩码 m_k",
             "有模态 → m_k = 1\n缺失模态 → m_k = 0\n(单模态视频可用)",
             accent_color=C["accent"])

    add_card(slide, right_x, Inches(6.7), Inches(4.4), Inches(0.6),
             f"{v['name']}  版本 {v['version']}",
             f"参数: {v['params']}  |  更新 {v['update_date']}",
             accent_color=C["primary"])
    add_version_bar(slide, "g_fuse")


def slide_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "实验结果", "Experimental Results — Calibrated Detection Accuracy")

    v = VERSIONS["detector_total"]

    # 大数字卡片
    kpis = [
        ("82.6%",    "Image\n图像分支",  C["primary"],   v["acc_image"]),
        ("91.7%",    "Video\n视频分支",  C["secondary"], v["acc_video"]),
        ("100%",     "Audio\n音频分支",  C["accent"],    v["acc_audio"]),
        ("86.8%",    "TOTAL\n整体",      C["primary"],   v["acc_total"]),
    ]
    for i, (big, label, color, detail) in enumerate(kpis):
        left = Inches(0.4 + i * 3.2)
        top = Inches(1.3)
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(3.0), Inches(2.5))
        box.fill.solid()
        box.fill.fore_color.rgb = color
        box.line.fill.background()
        add_text(slide, left, top + Inches(0.2), Inches(3.0), Inches(1.0),
                 big, size=52, bold=True, color=C["white"],
                 font=FONT_EN, align=PP_ALIGN.CENTER)
        add_text(slide, left, top + Inches(1.2), Inches(3.0), Inches(0.7),
                 label, size=14, color=C["white"],
                 font=FONT_BODY, align=PP_ALIGN.CENTER)
        add_text(slide, left, top + Inches(1.9), Inches(3.0), Inches(0.5),
                 detail, size=11, color=RGBColor(0xFF,0xFF,0xFF), font=FONT_EN, align=PP_ALIGN.CENTER)

    # 特征分析 + 校准参数
    add_card(slide, Inches(0.4), Inches(4.1), Inches(4.2), Inches(2.5),
             "区分度最强特征 (Image)",
             """blockiness      Cohen's d = -0.81   (Real > Fake)
local_var       Cohen's d = -0.60
srm_std_ratio   Cohen's d = +0.50
freq_ratio      Cohen's d = -0.36
color_entropy   Cohen's d = -0.29

校准方式: z-score × sign × |d|
图像阈值: 0.47   视频阈值: 0.20""",
             accent_color=C["primary"])

    add_card(slide, Inches(4.8), Inches(4.1), Inches(4.2), Inches(2.5),
             "视频最强特征 (Real=2, Fake=10)",
             """avg_blockiness   Cohen's d = -2.41
avg_local_var    Cohen's d = -1.50
avg_fft_hf       Cohen's d = -1.50
var_blockiness   Cohen's d = -1.37
frame_diff_mean  Cohen's d = +0.73""",
             accent_color=C["secondary"])

    add_card(slide, Inches(9.2), Inches(4.1), Inches(3.7), Inches(2.5),
             "Ground Truth 规则",
             """📷 图像:  real_* = Real
              其他 16 张 = Fake  (n=23)

🎬 视频:  real_* = Real
              其他 10 个 = Fake  (n=12)

🎵 音频:  全部为 Fake  (n=3)""",
             accent_color=C["accent"])


def slide_versions(prs):
    """各模块版本独立卡片 —— 每张可单独修改"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "各模块版本号", "Module Versions — All Editable Individually")

    # 版本表 (3+2 布局)
    rows = [
        ("image_branch",   Inches(0.4),  Inches(1.3), C["primary"]),
        ("video_branch",   Inches(4.6),  Inches(1.3), C["secondary"]),
        ("audio_branch",   Inches(8.8),  Inches(1.3), C["accent"]),
        ("x_cons",         Inches(0.4),  Inches(4.3), C["secondary"]),
        ("g_fuse",         Inches(4.6),  Inches(4.3), C["accent"]),
    ]
    for key, left, top, color in rows:
        v = VERSIONS[key]
        width = Inches(4.1)
        # 背景
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, Inches(2.8))
        box.fill.solid()
        box.fill.fore_color.rgb = C["white"]
        box.line.color.rgb = color
        box.line.width = Pt(1.5)
        # 顶部彩色条
        add_bar(slide, left, top, width, Inches(0.45), color)
        add_text(slide, left + Inches(0.2), top + Inches(0.05), width - Inches(0.4), Inches(0.4),
                 v["name"], size=16, bold=True, color=C["white"], font=FONT_TITLE)
        add_text(slide, left + Inches(0.2), top + Inches(0.5), width - Inches(0.4), Inches(0.35),
                 f"版本 {v['version']}   参数 {v['params']}", size=11, color=C["muted"], font=FONT_EN)
        # 详细字段 —— 每个独立文本框可单独修改
        fields = [
            ("骨干", v["backbone"]),
            ("输入", v["inputs"]),
            ("头部", v["heads"]),
            ("更新", v["update_date"]),
        ]
        fy = top + Inches(1.0)
        for fname, fval in fields:
            add_text(slide, left + Inches(0.2), fy, Inches(0.7), Inches(0.35),
                     fname, size=11, bold=True, color=color, font=FONT_TITLE)
            add_text(slide, left + Inches(0.9), fy, width - Inches(1.1), Inches(0.35),
                     fval, size=11, color=C["text"], font=FONT_EN)
            fy += Inches(0.38)

    # 底部总版本
    dt = VERSIONS["detector_total"]
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.0), Inches(4.3),
                                 Inches(3.9), Inches(2.8))
    box.fill.solid()
    box.fill.fore_color.rgb = C["light"]
    box.line.color.rgb = C["primary"]
    box.line.width = Pt(1.5)
    add_bar(slide, Inches(9.0), Inches(4.3), Inches(3.9), Inches(0.45), C["primary"])
    add_text(slide, Inches(9.2), Inches(4.35), Inches(3.5), Inches(0.4),
             "完整检测器 (汇总)", size=15, bold=True, color=C["white"], font=FONT_TITLE)
    add_text(slide, Inches(9.2), Inches(4.85), Inches(3.5), Inches(0.35),
             f"版本 {dt['version']}   总参数 {dt['params_total']}",
             size=11, color=C["muted"], font=FONT_EN)
    add_text(slide, Inches(9.2), Inches(5.25), Inches(3.5), Inches(0.35),
             f"模态: {dt['modalities']}", size=11, color=C["text"], font=FONT_EN)
    add_text(slide, Inches(9.2), Inches(5.65), Inches(3.5), Inches(0.35),
             f"阈值: {dt['thresholds']}", size=11, color=C["text"], font=FONT_EN)
    add_text(slide, Inches(9.2), Inches(6.05), Inches(3.5), Inches(0.35),
             f"整体准确率: {dt['acc_total']}", size=13, bold=True, color=C["primary"], font=FONT_EN)


def slide_thanks(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, C["primary"])
    add_bar(slide, Inches(5.5), Inches(2.8), Inches(2.3), Inches(0.08), C["accent"])

    add_text(slide, Inches(1), Inches(1.5), SLIDE_W - Inches(2), Inches(1.0),
             "感谢  Thank You", size=52, bold=True,
             color=C["white"], font=FONT_TITLE, align=PP_ALIGN.CENTER)
    add_text(slide, Inches(1), Inches(3.2), SLIDE_W - Inches(2), Inches(0.5),
             "多模态 AIGC 检测系统  v2.0.0  |  319.7M parameters",
             size=14, color=C["secondary"], font=FONT_EN, align=PP_ALIGN.CENTER)
    add_text(slide, Inches(1), Inches(3.9), SLIDE_W - Inches(2), Inches(0.5),
             "架构: CLIP-ViT + Wav2Vec2 + MSFF + T-Agg + X-Cons + G-Fuse",
             size=12, color=C["light"], font=FONT_BODY, align=PP_ALIGN.CENTER)

    add_text(slide, Inches(1), Inches(5.2), SLIDE_W - Inches(2), Inches(0.5),
             "📂  架构图:  docs/architecture/", size=12, color=C["secondary"], font=FONT_EN, align=PP_ALIGN.CENTER)
    add_text(slide, Inches(1), Inches(5.7), SLIDE_W - Inches(2), Inches(0.5),
             "🧪  检测结果:  outputs/multimodal_viz_calibrated_*/",
             size=12, color=C["secondary"], font=FONT_EN, align=PP_ALIGN.CENTER)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  生成多模态 AIGC 检测系统 PPT")
    print(f"  输出: {OUT_PPTX}")
    print("=" * 60)

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_cover(prs)
    slide_agenda(prs)
    slide_full_architecture(prs)
    slide_image_branch(prs)
    slide_video_branch(prs)
    slide_audio_branch(prs)
    slide_x_cons(prs)
    slide_g_fuse(prs)
    slide_results(prs)
    slide_versions(prs)
    slide_thanks(prs)

    prs.save(str(OUT_PPTX))
    print(f"\n✅ PPT 已保存: {OUT_PPTX}")
    print(f"   幻灯片数: {len(prs.slides)}")


if __name__ == '__main__':
    main()
