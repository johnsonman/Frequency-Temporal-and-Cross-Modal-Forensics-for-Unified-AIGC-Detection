"""
将 AIGC 检测结果转为 PPTX 演示文稿
使用 python-pptx + 已生成的汇总图表和检测过程可视化
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.resolve()

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn


# ============================================================
# 主题配置 — Midnight Executive + Ocean Gradient 混合
# ============================================================
COLORS = {
    'navy':      RGBColor(0x1E, 0x27, 0x61),   # 深蓝主色 (标题深色背景)
    'teal':      RGBColor(0x06, 0x5A, 0x82),   # 深青蓝 (主色调)
    'cyan':      RGBColor(0x1C, 0x72, 0x93),   # 青色 (次级)
    'midnight':  RGBColor(0x21, 0x29, 0x5C),   # 午夜蓝 (强调)
    'ice':       RGBColor(0xCA, 0xDC, 0xFC),   # 冰蓝 (浅色强调)
    'light':     RGBColor(0xE8, 0xF4, 0xF8),   # 极浅背景
    'white':     RGBColor(0xFF, 0xFF, 0xFF),
    'offwhite':  RGBColor(0xF8, 0xFA, 0xFC),
    'dark':      RGBColor(0x1A, 0x1F, 0x3C),
    'text':      RGBColor(0x1E, 0x27, 0x61),
    'text_gray': RGBColor(0x47, 0x55, 0x69),
    'ai_red':    RGBColor(0xEF, 0x44, 0x44),   # AI 红色
    'real_green':RGBColor(0x10, 0xB9, 0x81),   # 真实绿色
    'gold':      RGBColor(0xF5, 0x9E, 0x0B),   # 视频橙金
    'mint':      RGBColor(0x34, 0xD3, 0x99),   # 音频薄荷
}


def set_cjk_font(run, font_name='Microsoft YaHei'):
    """设置中文字体"""
    try:
        run.font.name = font_name
        rPr = run._r.get_or_add_rPr()
        rPr.set(qn('w:eastAsia'), font_name)
    except Exception:
        pass


def add_title_bar(slide, title, subtitle=None, dark=True):
    """为幻灯片添加左侧竖条装饰 + 标题"""
    bar_color = COLORS['navy'] if dark else COLORS['teal']

    # 左侧装饰条
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.35), Inches(5.625)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = bar_color
    shape.line.fill.background()

    # 标题文本框
    txBox = slide.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(9.1), Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = title
    run.font.size = Pt(26)
    run.font.bold = True
    run.font.color.rgb = COLORS['navy']
    set_cjk_font(run)

    if subtitle:
        p2 = tf.add_paragraph()
        run2 = p2.add_run()
        run2.text = subtitle
        run2.font.size = Pt(13)
        run2.font.color.rgb = COLORS['text_gray']
        set_cjk_font(run2)


def add_footer(slide, text="多模态 AI 生成媒体检测报告"):
    """添加页脚"""
    txBox = slide.shapes.add_textbox(Inches(0.6), Inches(5.25), Inches(9.0), Inches(0.35))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(9)
    run.font.color.rgb = COLORS['text_gray']
    set_cjk_font(run)
    p.alignment = PP_ALIGN.RIGHT


def add_card(slide, left, top, width, height, fill_color=None, border_color=None):
    """添加圆角卡片容器"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color if fill_color else COLORS['offwhite']
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_text_block(slide, left, top, width, height, text, font_size=14,
                   bold=False, color=None, align=PP_ALIGN.LEFT):
    """添加文本块"""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(4)
    tf.margin_right = Pt(4)
    tf.margin_top = Pt(2)
    tf.margin_bottom = Pt(2)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color if color else COLORS['text']
    set_cjk_font(run)
    return txBox


def add_bullet_list(slide, left, top, width, height, items, font_size=13, color=None):
    """添加项目符号列表"""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(4)
    tf.margin_top = Pt(2)

    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.level = 0
        run = p.add_run()
        run.text = f"• {item}"
        run.font.size = Pt(font_size)
        run.font.color.rgb = color if color else COLORS['text']
        set_cjk_font(run)

    return txBox


# ============================================================
# 幻灯片生成
# ============================================================

def create_cover_slide(prs):
    """封面页 (深色)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局

    # 背景渐变 - 用矩形块模拟
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(10), Inches(5.625))
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLORS['navy']
    bg.line.fill.background()

    # 装饰斜线条纹
    for i in range(-3, 8):
        stripe = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(i * 1.8 - 2), Inches(4.5),
            Inches(0.8), Inches(1.5)
        )
        stripe.fill.solid()
        stripe.fill.fore_color.rgb = COLORS['cyan']
        stripe.line.fill.background()
        stripe.rotation = 30
        stripe.fill.transparency = 0.7

    # 左侧主装饰条
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.8), Inches(0.08), Inches(2.2)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = COLORS['cyan']
    accent.line.fill.background()

    # 主标题
    txBox = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(8.5), Inches(1.2))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "多模态 AI 生成媒体"
    run.font.size = Pt(40)
    run.font.bold = True
    run.font.color.rgb = COLORS['white']
    set_cjk_font(run)

    p2 = tf.add_paragraph()
    run2 = p2.add_run()
    run2.text = "真伪检测系统  ·  测试报告"
    run2.font.size = Pt(28)
    run2.font.bold = True
    run2.font.color.rgb = COLORS['cyan']
    set_cjk_font(run2)

    # 副标题说明
    txBox2 = slide.shapes.add_textbox(Inches(1.0), Inches(3.2), Inches(8.5), Inches(0.6))
    tf2 = txBox2.text_frame
    p3 = tf2.paragraphs[0]
    run3 = p3.add_run()
    run3.text = "图像  ·  视频  ·  音频  多模态融合检测框架"
    run3.font.size = Pt(14)
    run3.font.color.rgb = COLORS['ice']
    set_cjk_font(run3)

    # 底部日期
    date_str = datetime.now().strftime('%Y年 %m月 %d日')
    txBox3 = slide.shapes.add_textbox(Inches(1.0), Inches(4.9), Inches(8.5), Inches(0.4))
    tf3 = txBox3.text_frame
    p4 = tf3.paragraphs[0]
    p4.alignment = PP_ALIGN.LEFT
    run4 = p4.add_run()
    run4.text = f"报告生成  |  {date_str}"
    run4.font.size = Pt(12)
    run4.font.color.rgb = COLORS['ice']
    set_cjk_font(run4)

    return slide


def create_toc_slide(prs):
    """目录页"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "目录  CONTENTS")
    add_footer(slide)

    sections = [
        ("01", "项目概述",          "背景、目标与核心问题"),
        ("02", "架构设计",          "多模态检测框架与核心模块"),
        ("03", "测试配置",          "环境、模型与测试数据集"),
        ("04", "检测结果汇总",      "30 文件融合检测结果"),
        ("05", "各模态检测详情",    "图像 / 视频 / 音频 逐文件分析"),
        ("06", "局限性与展望",      "当前方案的不足与改进方向"),
    ]

    start_y = Inches(1.4)
    for i, (num, title, desc) in enumerate(sections):
        y = start_y + Inches(i * 0.68)

        # 编号圆
        circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(0.7), y, Inches(0.45), Inches(0.45)
        )
        circle.fill.solid()
        circle.fill.fore_color.rgb = COLORS['navy']
        circle.line.fill.background()

        # 编号文本
        num_box = slide.shapes.add_textbox(Inches(0.7), y, Inches(0.45), Inches(0.45))
        num_p = num_box.text_frame.paragraphs[0]
        num_p.alignment = PP_ALIGN.CENTER
        nr = num_p.add_run()
        nr.text = num
        nr.font.size = Pt(13)
        nr.font.bold = True
        nr.font.color.rgb = COLORS['white']
        set_cjk_font(nr)

        # 标题
        tBox = slide.shapes.add_textbox(Inches(1.3), y, Inches(3.0), Inches(0.3))
        tr = tBox.text_frame.paragraphs[0].add_run()
        tr.text = title
        tr.font.size = Pt(17)
        tr.font.bold = True
        tr.font.color.rgb = COLORS['navy']
        set_cjk_font(tr)

        # 描述
        dBox = slide.shapes.add_textbox(Inches(1.3), y + Inches(0.32), Inches(8.0), Inches(0.3))
        dr = dBox.text_frame.paragraphs[0].add_run()
        dr.text = desc
        dr.font.size = Pt(11)
        dr.font.color.rgb = COLORS['text_gray']
        set_cjk_font(dr)

    return slide


def create_overview_slide(prs, results):
    """项目概述页"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "01  项目概述", "背景与目标")
    add_footer(slide)

    ai_count = sum(1 for r in results if r['is_ai'])
    real_count = len(results) - ai_count

    # 三个统计卡片
    cards = [
        ("检测文件", f"{len(results)}", "覆盖图像/视频/音频三类", COLORS['navy']),
        ("AI 判定",   f"{ai_count}",  f"占比 {ai_count/len(results)*100:.0f}%", COLORS['ai_red']),
        ("真实判定", f"{real_count}", f"占比 {real_count/len(results)*100:.0f}%", COLORS['real_green']),
    ]

    card_w = Inches(2.9)
    card_h = Inches(1.8)
    gap = Inches(0.15)
    total_w = card_w * 3 + gap * 2
    start_x = Inches(0.7) + (Inches(10) - Inches(0.7) - total_w) / 2

    for i, (label, value, sub, color) in enumerate(cards):
        x = start_x + (card_w + gap) * i
        card = add_card(slide, x, Inches(1.5), card_w, card_h,
                        fill_color=COLORS['white'], border_color=color)
        # 左侧彩色条
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, x, Inches(1.5), Inches(0.08), card_h
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        # 标签
        add_text_block(slide, x + Inches(0.25), Inches(1.6), Inches(2.5), Inches(0.4),
                       label, font_size=11, color=COLORS['text_gray'])
        # 数字
        add_text_block(slide, x + Inches(0.25), Inches(1.95), Inches(2.5), Inches(0.8),
                       value, font_size=38, bold=True, color=color)
        # 子说明
        add_text_block(slide, x + Inches(0.25), Inches(2.8), Inches(2.5), Inches(0.4),
                       sub, font_size=10, color=COLORS['text_gray'])

    # 目标说明
    add_text_block(slide, Inches(0.7), Inches(3.6), Inches(8.5), Inches(0.4),
                   "核心目标", font_size=15, bold=True, color=COLORS['navy'])

    add_bullet_list(slide, Inches(0.7), Inches(4.05), Inches(4.2), Inches(1.3), [
        "搭建多模态 AIGC 检测框架",
        "覆盖图像 / 视频 / 音频三种模态",
        "验证启发式特征 + 预训练骨干的有效性",
    ])

    add_bullet_list(slide, Inches(5.2), Inches(4.05), Inches(4.2), Inches(1.3), [
        "融合分数 = 70% 启发式 + 30% 预训练",
        "无标注数据场景下的零样本检测",
        "生成完整的检测过程可视化报告",
    ])

    return slide


def create_architecture_slide(prs):
    """架构设计页"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "02  架构设计", "多模态检测框架核心模块")
    add_footer(slide)

    modules = [
        ("Φ_img  图像分支", [
            "CLIP-ViT-B/16 视觉编码器",
            "MSFF 多尺度特征融合 (Layer 3/6/12)",
            "SRM + FFT 局部取证支路",
        ], COLORS['teal']),
        ("Φ_vid  视频分支", [
            "帧采样 + 共享 CLIP 编码器",
            "T-Agg 时序聚合 (Manba + Transformer)",
            "跨帧注意力 + 帧级加权融合",
        ], COLORS['gold']),
        ("Φ_aud  音频分支", [
            "Wav2Vec2-base 自监督编码器",
            "Log-Mel + 2D CNN 频谱支路",
            "多头注意力池化 + MFCC 统计",
        ], COLORS['mint']),
        ("G-Fuse  门控融合", [
            "X-Cons 跨模态一致性建模",
            "可靠性评估 + 门控加权",
            "模态掩码处理缺失模态",
        ], COLORS['navy']),
    ]

    for i, (title, items, color) in enumerate(modules):
        row = i // 2
        col = i % 2
        card_x = Inches(0.7) + col * Inches(4.6)
        card_y = Inches(1.5) + row * Inches(1.95)
        card_w = Inches(4.3)
        card_h = Inches(1.75)

        # 卡片
        card = add_card(slide, card_x, card_y, card_w, card_h,
                        fill_color=COLORS['light'], border_color=color)
        # 顶部彩色横条
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, card_x, card_y, card_w, Inches(0.06)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        add_text_block(slide, card_x + Inches(0.2), card_y + Inches(0.2),
                       card_w - Inches(0.4), Inches(0.4),
                       title, font_size=14, bold=True, color=color)
        add_bullet_list(slide, card_x + Inches(0.2), card_y + Inches(0.6),
                        card_w - Inches(0.4), Inches(1.1),
                        items, font_size=10, color=COLORS['text'])

    return slide


def create_config_slide(prs):
    """测试配置页"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "03  测试配置", "环境与数据集")
    add_footer(slide)

    # 环境信息
    env_card = add_card(slide, Inches(0.7), Inches(1.5), Inches(4.3), Inches(3.7))
    add_text_block(slide, Inches(0.9), Inches(1.65), Inches(4.0), Inches(0.35),
                   "运行环境", font_size=14, bold=True, color=COLORS['navy'])

    env_info = [
        "操作系统:   Windows 11",
        "Python:     3.13.14",
        "PyTorch:    2.14.0 (CUDA 13.0)",
        "GPU:        NVIDIA CUDA 可用",
        "模型参数:   319,679,717",
    ]
    add_bullet_list(slide, Inches(0.9), Inches(2.1), Inches(4.0), Inches(3.0),
                    env_info, font_size=11, color=COLORS['text'])

    # 模型配置
    model_card = add_card(slide, Inches(5.0), Inches(1.5), Inches(4.3), Inches(3.7))
    add_text_block(slide, Inches(5.2), Inches(1.65), Inches(4.0), Inches(0.35),
                   "预训练模型", font_size=14, bold=True, color=COLORS['navy'])

    model_info = [
        "图像编码器:   openai/clip-vit-base-patch16",
        "视频编码器:   共享 CLIP ViT",
        "音频编码器:   facebook/wav2vec2-base-960h",
        "嵌入维度:     768",
        "融合策略:     70% 启发式 + 30% NN",
    ]
    add_bullet_list(slide, Inches(5.2), Inches(2.1), Inches(4.0), Inches(3.0),
                    model_info, font_size=11, color=COLORS['text'])

    # 数据集
    ds_card = add_card(slide, Inches(0.7), Inches(3.9), Inches(8.6), Inches(1.3))
    add_text_block(slide, Inches(0.9), Inches(4.0), Inches(8.2), Inches(0.35),
                   "测试数据集", font_size=14, bold=True, color=COLORS['navy'])

    ds_info = "图像 15 张 (.jpg/.jpeg/.png)  ·  视频 12 段 (.mp4)  ·  音频 3 段 (.mp3)"
    add_text_block(slide, Inches(0.9), Inches(4.4), Inches(8.2), Inches(0.6),
                   ds_info, font_size=12, color=COLORS['text'])

    return slide


def create_summary_slide(prs, chart_dir):
    """检测结果汇总页 - 插入柱状图"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "04  检测结果汇总", "融合分数分布 (30 个文件)")
    add_footer(slide)

    chart_path = str(chart_dir / '01_fused_scores_bar.png')
    if os.path.exists(chart_path):
        pic = slide.shapes.add_picture(
            chart_path, Inches(0.7), Inches(1.4),
            width=Inches(9.0), height=Inches(3.8)
        )

    return slide


def create_distribution_slide(prs, chart_dir):
    """分布与分支对比"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "04  检测结果汇总", "分数分布 · 分支对比 · 判定分布")
    add_footer(slide)

    charts = [
        ('03_judgment_pie.png',    Inches(0.7),  Inches(1.4), Inches(4.3), Inches(4.0)),
        ('04_branch_comparison.png', Inches(5.0), Inches(1.4), Inches(4.3), Inches(4.0)),
    ]

    for fname, x, y, w, h in charts:
        path = str(chart_dir / fname)
        if os.path.exists(path):
            slide.shapes.add_picture(path, x, y, width=w, height=h)

    return slide


def create_heatmap_slide(prs, chart_dir):
    """特征热力图"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "04  检测结果汇总", "启发式特征热力图 (按融合分数排序)")
    add_footer(slide)

    chart_path = str(chart_dir / '05_feature_heatmap.png')
    if os.path.exists(chart_path):
        slide.shapes.add_picture(
            chart_path, Inches(0.5), Inches(1.4),
            width=Inches(9.3), height=Inches(3.8)
        )

    return slide


def create_detail_slide(prs, results, ftype, chart_dir, process_dir):
    """模态检测详情页 — 每页展示3个文件"""
    subset = [r for r in results if r['file_type'] == ftype]
    modal_label = {'image': '图像', 'video': '视频', 'audio': '音频'}[ftype]

    slides = []
    per_slide = 3

    for page_start in range(0, len(subset), per_slide):
        page_items = subset[page_start:page_start + per_slide]
        page_num = page_start // per_slide + 1
        total_pages = (len(subset) + per_slide - 1) // per_slide

        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_title_bar(slide,
                      f"05  检测详情 · {modal_label}",
                      f"第 {page_num} / {total_pages} 页")
        add_footer(slide)

        for i, result in enumerate(page_items):
            y = Inches(1.4) + Inches(i * 1.38)
            row_h = Inches(1.25)

            # 行卡片
            row = add_card(slide, Inches(0.7), y, Inches(8.6), row_h)

            fname = result['file_name'][:30]
            is_ai = result['is_ai']
            badge_color = COLORS['ai_red'] if is_ai else COLORS['real_green']
            badge_text = "AI" if is_ai else "真实"

            # 左竖色条
            bar = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Inches(0.7), y, Inches(0.08), row_h
            )
            bar.fill.solid()
            bar.fill.fore_color.rgb = badge_color
            bar.line.fill.background()

            # 文件名
            add_text_block(slide, Inches(0.95), y + Inches(0.1),
                           Inches(4.5), Inches(0.35),
                           fname, font_size=13, bold=True, color=COLORS['navy'])

            # 徽章
            badge = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.95), y + Inches(0.48),
                Inches(1.0), Inches(0.42)
            )
            badge.fill.solid()
            badge.fill.fore_color.rgb = badge_color
            badge.line.fill.background()
            btx = slide.shapes.add_textbox(Inches(0.95), y + Inches(0.48), Inches(1.0), Inches(0.42))
            bp = btx.text_frame.paragraphs[0]
            bp.alignment = PP_ALIGN.CENTER
            br = bp.add_run()
            br.text = f"🤖 {badge_text}" if is_ai else f"📷 {badge_text}"
            br.font.size = Pt(11)
            br.font.bold = True
            br.font.color.rgb = COLORS['white']
            set_cjk_font(br)

            # 分数信息 (右侧)
            scores_text = (
                f"融合 p_fake={result['fused_score']:.4f}   "
                f"NN={result['nn_score']:.4f}   "
                f"启发式={result['heuristic_score']:.4f}   "
                f"置信度={result['confidence']:.4f}"
            )
            add_text_block(slide, Inches(2.15), y + Inches(0.48),
                           Inches(7.0), Inches(0.42),
                           scores_text, font_size=11, color=COLORS['text_gray'])

            # 启发式特征摘要
            if result.get('heuristics'):
                short = {
                    'freq_ratio': 'FFT比', 'srm_std_ratio': 'SRM比',
                    'local_var': '局部方差', 'blockiness': '块效应',
                    'color_entropy': '颜色熵',
                    'temporal_consistency': '时序一致',
                    'motion_score': '运动分', 'frame_variance': '帧方差',
                    'spectral_flatness': '频谱平坦',
                }
                tags = []
                for k, v in list(result['heuristics'].items())[:3]:
                    if k in short:
                        tags.append(f"{short[k]}={v:.3f}")
                if tags:
                    add_text_block(slide, Inches(2.15), y + Inches(0.88),
                                   Inches(7.0), Inches(0.3),
                                   "  |  ".join(tags), font_size=10, color=COLORS['text_gray'])

        slides.append(slide)

    return slides


def create_conclusion_slide(prs, results):
    """局限性与展望"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "06  局限性与展望", "Limitation & Outlook")
    add_footer(slide)

    left_items = [
        "分类头为随机初始化，未在 AIGC 标注数据上训练",
        "融合权重 (70/30) 为启发式设定，未校准",
        "单音频/单图像检测不支持跨模态一致性",
        "缺乏领域特定 AIGC 生成器的先验知识",
    ]
    right_items = [
        "引入预训练 AIGC 检测器作为分类头",
        "在大规模标注数据集上微调分类头",
        "加入生成器特异性检测 (SD / Sora / VALL-E)",
        "探索多模态联合检测的门控权重学习",
    ]

    # 左 - 局限性
    left_card = add_card(slide, Inches(0.7), Inches(1.5), Inches(4.3), Inches(3.5),
                         fill_color=COLORS['light'], border_color=COLORS['ai_red'])
    add_text_block(slide, Inches(0.9), Inches(1.65), Inches(4.0), Inches(0.35),
                   "⚠  当前局限性", font_size=14, bold=True, color=COLORS['ai_red'])
    add_bullet_list(slide, Inches(0.9), Inches(2.1), Inches(4.0), Inches(2.8),
                    left_items, font_size=11, color=COLORS['text'])

    # 右 - 展望
    right_card = add_card(slide, Inches(5.0), Inches(1.5), Inches(4.3), Inches(3.5),
                          fill_color=COLORS['light'], border_color=COLORS['real_green'])
    add_text_block(slide, Inches(5.2), Inches(1.65), Inches(4.0), Inches(0.35),
                   "✦  未来改进方向", font_size=14, bold=True, color=COLORS['real_green'])
    add_bullet_list(slide, Inches(5.2), Inches(2.1), Inches(4.0), Inches(2.8),
                    right_items, font_size=11, color=COLORS['text'])

    return slide


def create_end_slide(prs):
    """结束页"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(10), Inches(5.625))
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLORS['navy']
    bg.line.fill.background()

    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(4.6), Inches(2.8), Inches(0.8), Inches(0.06)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = COLORS['cyan']
    accent.line.fill.background()

    txBox = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(8.5), Inches(1.0))
    p = txBox.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "THANK YOU"
    run.font.size = Pt(48)
    run.font.bold = True
    run.font.color.rgb = COLORS['white']
    set_cjk_font(run)

    txBox2 = slide.shapes.add_textbox(Inches(1.0), Inches(3.1), Inches(8.5), Inches(0.6))
    p2 = txBox2.text_frame.paragraphs[0]
    p2.alignment = PP_ALIGN.CENTER
    run2 = p2.add_run()
    run2.text = "多模态 AI 生成媒体真伪检测系统"
    run2.font.size = Pt(16)
    run2.font.color.rgb = COLORS['ice']
    set_cjk_font(run2)

    return slide


# ============================================================
# 主入口
# ============================================================

def main():
    # 加载检测结果
    json_path = None
    json_files = sorted(
        [f for f in os.listdir(str(PROJECT_ROOT / 'outputs'))
         if f.startswith('fused_results') and f.endswith('.json')]
    )
    if json_files:
        json_path = str(PROJECT_ROOT / 'outputs' / json_files[-1])
        print(f"加载结果: {json_path}")

    if not json_path:
        print("错误: 找不到 fused_results_*.json")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    chart_dir = PROJECT_ROOT / 'outputs' / 'visualization'
    process_dir = chart_dir / 'process'

    # 创建 PPT
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)

    print("\n生成幻灯片...")
    print("  [1/12] 封面")
    create_cover_slide(prs)

    print("  [2/12] 目录")
    create_toc_slide(prs)

    print("  [3/12] 项目概述")
    create_overview_slide(prs, results)

    print("  [4/12] 架构设计")
    create_architecture_slide(prs)

    print("  [5/12] 测试配置")
    create_config_slide(prs)

    print("  [6/12] 汇总 - 融合分数柱状图")
    create_summary_slide(prs, chart_dir)

    print("  [7/12] 汇总 - 分布与分支对比")
    create_distribution_slide(prs, chart_dir)

    print("  [8/12] 汇总 - 特征热力图")
    create_heatmap_slide(prs, chart_dir)

    print("  [9/12] 图像检测详情")
    create_detail_slide(prs, results, 'image', chart_dir, process_dir)

    print("  [10/12] 视频检测详情")
    create_detail_slide(prs, results, 'video', chart_dir, process_dir)

    print("  [11/12] 音频检测详情")
    create_detail_slide(prs, results, 'audio', chart_dir, process_dir)

    print("  [12/12] 局限性与展望")
    create_conclusion_slide(prs, results)

    print("  [13/12] 结束页")
    create_end_slide(prs)

    # 保存
    out_path = str(PROJECT_ROOT / 'outputs' / 'visualization' / 'aigc_detection_report.pptx')
    prs.save(out_path)
    print(f"\n✓ PPTX 已生成: {out_path}")
    print(f"  共 {len(prs.slides)} 页")


if __name__ == "__main__":
    main()
