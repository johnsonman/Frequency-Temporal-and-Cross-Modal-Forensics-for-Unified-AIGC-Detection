"""
多模态 AIGC 检测 Transformer 架构 - PPT 框架图生成
使用 python-pptx 原生 shapes 构建可编辑架构图
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs', 'architecture_diagram.pptx')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ===== 配色方案 =====
C = {
    # 主分支颜色
    'img':     RGBColor(0x06, 0x5A, 0x82),   # 深海军蓝 - 图像
    'img_l':   RGBColor(0xC8, 0xE5, 0xF1),   # 浅蓝背景
    'img_m':   RGBColor(0x1C, 0x72, 0x93),   # 中蓝
    'vid':     RGBColor(0xD9, 0x77, 0x06),   # 橙色 - 视频
    'vid_l':   RGBColor(0xFE, 0xF3, 0xC7),   # 浅橙背景
    'vid_m':   RGBColor(0xEA, 0x9E, 0x0C),   # 中橙
    'aud':     RGBColor(0x02, 0x80, 0x90),   # 青色 - 音频
    'aud_l':   RGBColor(0xA5, 0xF3, 0xE0),   # 浅青背景
    'aud_m':   RGBColor(0x00, 0xA8, 0x96),   # 中青
    'xcons':   RGBColor(0xB9, 0x1C, 0x1C),   # 深红 - 跨模态
    'xcons_l': RGBColor(0xFE, 0xE2, 0xE2),   # 浅红背景
    'fuse':    RGBColor(0x6D, 0x2E, 0x46),   # 深紫红 - 门控融合
    'fuse_l':  RGBColor(0xF3, 0xE8, 0xEC),   # 浅紫背景
    # 通用
    'dark':    RGBColor(0x21, 0x29, 0x36),
    'gray':    RGBColor(0x64, 0x74, 0x8B),
    'light':   RGBColor(0xF1, 0xF5, 0xF9),
    'white':   RGBColor(0xFF, 0xFF, 0xFF),
    'line':    RGBColor(0x94, 0xA3, 0xB8),
    'black':   RGBColor(0x00, 0x00, 0x00),
}

# ===== 辅助函数 =====
def add_rounded_rect(slide, x, y, w, h, fill, line=None, line_w=0.75, radius=0.08):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.shadow.inherit = False
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_w)
    # 圆角
    shape.adjustments[0] = radius * 10  # python-pptx adjustments 0-10 范围
    return shape

def add_rect(slide, x, y, w, h, fill, line=None, line_w=0.75):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.shadow.inherit = False
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_w)
    return shape

def add_oval(slide, x, y, w, h, fill, line=None, line_w=0.75):
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.shadow.inherit = False
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_w)
    return shape

def add_text(shape_or_slide, text, x=None, y=None, w=None, h=None,
             font_size=10, bold=False, color=None, align=PP_ALIGN.CENTER,
             font_face='Calibri'):
    # Check if it's a slide (has .shapes) vs a shape
    if not hasattr(shape_or_slide, 'shapes'):
        # Already a textbox shape
        txBox = shape_or_slide
    elif x is not None:
        txBox = shape_or_slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    else:
        txBox = shape_or_slide
    tf = txBox.text_frame
    tf.word_wrap = True
    if x is not None and hasattr(shape_or_slide, 'shapes'):
        tf.margin_left = Pt(3); tf.margin_right = Pt(3)
        tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.name = font_face
    if color:
        run.font.color.rgb = color
    return txBox

def add_text_multi(slide, lines, x, y, w, h, font_size=9, color=None,
                   bold=False, align=PP_ALIGN.LEFT, line_spacing=1.25, font_face='Calibri'):
    txBox = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(4); tf.margin_right = Pt(4)
    tf.margin_top = Pt(3); tf.margin_bottom = Pt(3)
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(0)
        p.space_before = Pt(0)
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.name = font_face
        if color:
            run.font.color.rgb = color
    return txBox

def add_arrow(slide, x1, y1, x2, y2, color=None, width=1.25):
    """添加带箭头的连接线"""
    line_shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
        Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    if color is None:
        color = C['line']
    line_shape.line.color.rgb = color
    line_shape.line.width = Pt(width)
    # 尝试添加箭头 - python-pptx connector 默认带末端箭头
    from pptx.oxml.ns import qn
    line = line_shape.line._get_or_add_ln()
    # 设置末端箭头
    tailEnd = line.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'})
    line.append(tailEnd)
    # 设置始端无箭头
    headEnd = line.makeelement(qn('a:headEnd'), {'type': 'none'})
    line.append(headEnd)
    return line_shape

def add_dashed_arrow(slide, x1, y1, x2, y2, color=None, width=1.25):
    line_shape = add_arrow(slide, x1, y1, x2, y2, color, width)
    line_shape.line.dash_style = 4  # dash
    return line_shape

def add_curved_arrow(slide, x1, y1, x2, y2, color=None, width=1.25):
    """使用 curve 连接器"""
    line_shape = slide.shapes.add_connector(MSO_CONNECTOR.CURVE,
        Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    if color is None:
        color = C['line']
    line_shape.line.color.rgb = color
    line_shape.line.width = Pt(width)
    from pptx.oxml.ns import qn
    line = line_shape.line._get_or_add_ln()
    tailEnd = line.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'})
    line.append(tailEnd)
    return line_shape

def set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


# ========================== 创建演示文稿 ==========================
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]  # blank

# ================================================================
# SLIDE 1: 封面
# ================================================================
s1 = prs.slides.add_slide(blank)
set_bg(s1, C['dark'])

# 顶部装饰条
add_rect(s1, 0, 0, 13.333, 0.08, C['img'])
add_rect(s1, 0, 7.42, 13.333, 0.08, C['fuse'])

# 大标题
tb = s1.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(1.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
run = p.add_run()
run.text = "多模态 AIGC 生成内容真伪检测"
run.font.size = Pt(48); run.font.bold = True; run.font.name = 'Georgia'
run.font.color.rgb = C['white']; run.font.character_spacing = Pt(2)

# 副标题
tb2 = s1.shapes.add_textbox(Inches(0.8), Inches(3.5), Inches(11.7), Inches(1.0))
tf2 = tb2.text_frame; tf2.word_wrap = True
p2 = tf2.paragraphs[0]
run2 = p2.add_run()
run2.text = "Transformer-Based Multimodal Framework"
run2.font.size = Pt(26); run2.font.bold = False; run2.font.name = 'Georgia'
run2.font.color.rgb = C['img_m']

# 三分支彩色指示条
bar_y = 4.8
add_rect(s1, 3.0, bar_y, 1.3, 0.05, C['img'])
add_rect(s1, 4.5, bar_y, 1.3, 0.05, C['vid'])
add_rect(s1, 6.0, bar_y, 1.3, 0.05, C['aud'])
add_rect(s1, 7.5, bar_y, 1.3, 0.05, C['xcons'])
add_rect(s1, 9.0, bar_y, 1.3, 0.05, C['fuse'])

# 底部信息
footer = s1.shapes.add_textbox(Inches(0.8), Inches(5.2), Inches(11.7), Inches(0.5))
tf_f = footer.text_frame
tf_f.paragraphs[0].alignment = PP_ALIGN.CENTER
r = tf_f.paragraphs[0].add_run()
r.text = "CLIP/VIT + WavLM + Temporal Attention + Gated Fusion"
r.font.size = Pt(16); r.font.name = 'Calibri'; r.font.color.rgb = C['gray']

footer2 = s1.shapes.add_textbox(Inches(0.8), Inches(6.5), Inches(11.7), Inches(0.5))
tf_f2 = footer2.text_frame
tf_f2.paragraphs[0].alignment = PP_ALIGN.CENTER
r2 = tf_f2.paragraphs[0].add_run()
r2.text = "模型框架图 · Model Architecture Diagram"
r2.font.size = Pt(12); r2.font.name = 'Calibri'; r2.font.color.rgb = C['gray']

print("  ✓ Slide 1 封面")


# ================================================================
# SLIDE 2: 目录
# ================================================================
s2 = prs.slides.add_slide(blank)
set_bg(s2, C['white'])

add_rect(s2, 0, 0, 13.333, 0.7, C['dark'])
title2 = s2.shapes.add_textbox(Inches(0.6), Inches(0.12), Inches(12), Inches(0.5))
tf_t2 = title2.text_frame
p = tf_t2.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
run = p.add_run(); run.text = "目录   CONTENTS"
run.font.size = Pt(28); run.font.bold = True; run.font.name = 'Georgia'
run.font.color.rgb = C['white']

sections = [
    ("01", "整体架构概览",      "Overview & System Architecture",      C['img']),
    ("02", "图像分支 Φ_img",    "Image Branch: CLIP-VIT + MSFF",       C['img_m']),
    ("03", "视频分支 Φ_vid",    "Video Branch: Temporal Aggregation",   C['vid']),
    ("04", "音频分支 Φ_aud",    "Audio Branch: WavLM + Semantic",       C['aud']),
    ("05", "跨模态一致性",      "Cross-Modal Consistency X-Cons",      C['xcons']),
    ("06", "门控融合 G-Fuse",   "Gated Fusion & Final Classification", C['fuse']),
    ("07", "关键模块 & 输出",   "Key Modules & Final Output",          C['dark']),
]

for i, (num, cn, en, col) in enumerate(sections):
    y = 1.0 + i * 0.82
    # 序号圆圈
    num_oval = add_oval(s2, 0.8, y, 0.55, 0.55, col)
    tf_n = num_oval.text_frame
    tf_n.word_wrap = False
    tf_n.margin_left = Pt(0); tf_n.margin_right = Pt(0)
    tf_n.margin_top = Pt(0); tf_n.margin_bottom = Pt(0)
    tf_n.vertical_anchor = MSO_ANCHOR.MIDDLE
    pn = tf_n.paragraphs[0]; pn.alignment = PP_ALIGN.CENTER
    rn = pn.add_run(); rn.text = num
    rn.font.size = Pt(14); rn.font.bold = True; rn.font.color.rgb = C['white']; rn.font.name = 'Calibri'
    # 中文标题
    tb3 = s2.shapes.add_textbox(Inches(1.5), Inches(y + 0.02), Inches(5.0), Inches(0.4))
    tf3 = tb3.text_frame
    p3 = tf3.paragraphs[0]
    r3 = p3.add_run(); r3.text = cn
    r3.font.size = Pt(18); r3.font.bold = True; r3.font.color.rgb = C['dark']; r3.font.name = 'Georgia'
    # 英文副标题
    tb4 = s2.shapes.add_textbox(Inches(6.6), Inches(y + 0.02), Inches(6.0), Inches(0.4))
    tf4 = tb4.text_frame
    p4 = tf4.paragraphs[0]
    r4 = p4.add_run(); r4.text = en
    r4.font.size = Pt(13); r4.font.color.rgb = C['gray']; r4.font.italic = True; r4.font.name = 'Calibri'

print("  ✓ Slide 2 目录")


# ================================================================
# SLIDE 3: 整体架构概览 (主框架图)
# ================================================================
s3 = prs.slides.add_slide(blank)
set_bg(s3, C['white'])

# 标题
add_rect(s3, 0, 0, 13.333, 0.65, C['dark'])
title = s3.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.5))
tf = title.text_frame
p = tf.paragraphs[0]
run = p.add_run(); run.text = "整体架构概览   Overall Architecture"
run.font.size = Pt(24); run.font.bold = True; run.font.name = 'Georgia'
run.font.color.rgb = C['white']

# ===== 三分支区域背景 =====
# 图像分支 (左上)
img_bg = add_rect(s3, 0.3, 0.85, 5.0, 3.8, C['img_l'], line=C['img'], line_w=1.5)
# 视频分支 (左下)  
vid_bg = add_rect(s3, 0.3, 4.75, 5.0, 2.6, C['vid_l'], line=C['vid'], line_w=1.5)
# 音频分支 (右下)
aud_bg = add_rect(s3, 5.6, 4.75, 4.0, 2.6, C['aud_l'], line=C['aud'], line_w=1.5)
# 跨模态一致性 (中)
xcons_bg = add_rect(s3, 5.6, 0.85, 4.0, 3.8, C['xcons_l'], line=C['xcons'], line_w=1.5)
# 门控融合 (右)
fuse_bg = add_rect(s3, 9.9, 0.85, 3.2, 6.5, C['fuse_l'], line=C['fuse'], line_w=1.5)

# ===== 区域标题 =====
def section_title(slide, text, x, y, w, color):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.38))
    tf = tb.text_frame
    tf.margin_left = Pt(4); tf.margin_right = Pt(4)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = text
    r.font.size = Pt(14); r.font.bold = True; r.font.name = 'Georgia'
    r.font.color.rgb = color
    return tb

section_title(s3, "图像分支  Φ_img", 0.4, 0.9, 4.8, C['img'])
section_title(s3, "视频分支  Φ_vid", 0.4, 4.8, 4.8, C['vid'])
section_title(s3, "音频分支  Φ_aud", 5.7, 4.8, 3.8, C['aud'])
section_title(s3, "跨模态一致性  X-Cons", 5.7, 0.9, 3.8, C['xcons'])
section_title(s3, "门控融合  G-Fuse", 10.0, 0.9, 3.0, C['fuse'])


# ===== 图像分支内部模块 =====
y_img = 1.35
add_rounded_rect(s3, 0.5, y_img, 1.3, 0.6, C['img'], radius=0.12)
add_text_multi(s3, ["图像输入", "(单张图像)"], 0.5, y_img, 1.3, 0.6, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

add_rounded_rect(s3, 2.0, y_img, 1.3, 0.6, C['img_m'], radius=0.12)
add_text_multi(s3, ["图像预处理", "Resize/Normalize"], 2.0, y_img, 1.3, 0.6, font_size=8, color=C['white'], align=PP_ALIGN.CENTER)

add_arrow(s3, 1.8, y_img+0.3, 2.0, y_img+0.3, C['img'])

add_rounded_rect(s3, 3.5, y_img-0.3, 1.6, 1.2, C['img_m'], radius=0.12)
add_text_multi(s3, ["CLIP / ViT", "视觉编码器", "", "ViT-B/16 或 ViT-L/14"], 3.5, y_img-0.3, 1.6, 1.2, font_size=7, color=C['white'], align=PP_ALIGN.CENTER)
add_arrow(s3, 3.3, y_img+0.3, 3.5, y_img+0.3, C['img'])

# ViT 三层标注
add_text(s3, "Layer 3 (浅层)", 3.7, y_img+0.95, 1.2, 0.25, font_size=7, color=C['img'], bold=True)
add_text(s3, "Layer 6 (中层)",  3.7, y_img+1.18, 1.2, 0.25, font_size=7, color=C['img'], bold=True)
add_text(s3, "Layer 12 (深层)", 3.7, y_img+1.41, 1.4, 0.25, font_size=7, color=C['img'], bold=True)

# MSFF 多尺度特征融合
y_msff = y_img + 1.9
add_rounded_rect(s3, 0.5, y_msff, 2.2, 0.85, C['img'], radius=0.12)
add_text_multi(s3, ["MSFF 多尺度特征融合", "", "· 特征投影  · 层级注意力加权  · 残差融合"], 0.5, y_msff, 2.2, 0.85, font_size=7.5, color=C['white'], align=PP_ALIGN.LEFT)
add_arrow(s3, 3.8, y_img+0.9, 1.6, y_msff+0.85, C['img'])

# SRM/FFT 局部取证支路
add_rounded_rect(s3, 2.9, y_msff, 2.2, 0.85, C['img_m'], radius=0.12)
add_text_multi(s3, ["局部取证支路 SRM/FFT", "", "高频残差 / 频域特征"], 2.9, y_msff, 2.2, 0.85, font_size=7.5, color=C['white'], align=PP_ALIGN.LEFT)

# 图像表征 + 分类
y_img_repr = y_msff + 1.0
add_rounded_rect(s3, 0.5, y_img_repr, 2.0, 0.6, C['dark'], radius=0.12)
add_text_multi(s3, ["图像表征 f_img"], 0.5, y_img_repr, 2.0, 0.6, font_size=9, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s3, 1.6, y_msff+0.85, 1.5, y_img_repr, C['img'])
add_arrow(s3, 4.0, y_msff+0.85, 2.6, y_img_repr+0.3, C['img'])

# 图像分类头
add_rounded_rect(s3, 2.7, y_img_repr, 1.8, 0.6, C['fuse'], radius=0.12)
add_text_multi(s3, ["图像分类头", "p_img"], 2.7, y_img_repr, 1.8, 0.6, font_size=9, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s3, 2.5, y_img_repr+0.3, 2.7, y_img_repr+0.3, C['fuse'])


# ===== 视频分支内部模块 =====
y_vid = 5.05
add_rounded_rect(s3, 0.5, y_vid, 1.1, 0.55, C['vid'], radius=0.12)
add_text_multi(s3, ["视频输入", "(多帧)"], 0.5, y_vid, 1.1, 0.55, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

add_rounded_rect(s3, 1.7, y_vid, 1.4, 0.55, C['vid_m'], radius=0.12)
add_text_multi(s3, ["视频帧序列", "帧采样/归一化"], 1.7, y_vid, 1.4, 0.55, font_size=7.5, color=C['white'], align=PP_ALIGN.CENTER)
add_arrow(s3, 1.6, y_vid+0.275, 1.7, y_vid+0.275, C['vid'])

add_rounded_rect(s3, 3.2, y_vid-0.05, 2.0, 0.65, C['vid_m'], radius=0.12)
add_text_multi(s3, ["共享图像编码器 CLIP/VIT", "(参数共享)"], 3.2, y_vid-0.05, 2.0, 0.65, font_size=7.5, color=C['white'], align=PP_ALIGN.CENTER)
add_arrow(s3, 3.1, y_vid+0.275, 3.2, y_vid+0.275, C['vid'])

# 帧特征序列 + T-Agg
y_vid2 = y_vid + 0.75
add_rounded_rect(s3, 0.5, y_vid2, 1.5, 0.6, C['vid_m'], radius=0.12)
add_text_multi(s3, ["帧特征序列", "T×D"], 0.5, y_vid2, 1.5, 0.6, font_size=7.5, color=C['white'], align=PP_ALIGN.CENTER)
add_arrow(s3, 4.2, y_vid+0.6, 1.25, y_vid2, C['vid'])

add_rounded_rect(s3, 2.1, y_vid2, 1.5, 0.6, C['vid'], radius=0.12)
add_text_multi(s3, ["T-Agg 时序聚合", "Mamba / Transformer", "时序建模 + 帧级特征融合"], 2.1, y_vid2, 1.5, 0.6, font_size=6.5, color=C['white'], align=PP_ALIGN.CENTER)
add_arrow(s3, 2.0, y_vid2+0.3, 2.1, y_vid2+0.3, C['vid'])

add_rounded_rect(s3, 3.7, y_vid2, 1.5, 0.6, C['dark'], radius=0.12)
add_text_multi(s3, ["视频表征 f_vid"], 3.7, y_vid2, 1.5, 0.6, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s3, 3.6, y_vid2+0.3, 3.7, y_vid2+0.3, C['vid'])


# ===== 音频分支内部模块 =====
y_aud = 5.05
add_rounded_rect(s3, 5.8, y_aud, 1.1, 0.55, C['aud'], radius=0.12)
add_text_multi(s3, ["音频输入", "(波形)"], 5.8, y_aud, 1.1, 0.55, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

add_rounded_rect(s3, 7.0, y_aud, 1.5, 0.55, C['aud_m'], radius=0.12)
add_text_multi(s3, ["音频预处理", "重采样/归一化"], 7.0, y_aud, 1.5, 0.55, font_size=7.5, color=C['white'], align=PP_ALIGN.CENTER)
add_arrow(s3, 6.9, y_aud+0.275, 7.0, y_aud+0.275, C['aud'])

add_rounded_rect(s3, 5.8, y_aud+0.75, 1.5, 0.6, C['aud_m'], radius=0.12)
add_text_multi(s3, ["WavLM / XLS-R", "Transformer Layers", "自监督预训练"], 5.8, y_aud+0.75, 1.5, 0.6, font_size=6.5, color=C['white'], align=PP_ALIGN.CENTER)
add_arrow(s3, 7.75, y_aud+0.55, 6.55, y_aud+0.75, C['aud'])

add_rounded_rect(s3, 7.5, y_aud+0.75, 2.0, 0.6, C['aud'], radius=0.12)
add_text_multi(s3, ["时频特征分支 f_tf", "Log-Mel + 2D CNN"], 7.5, y_aud+0.75, 2.0, 0.6, font_size=7, color=C['white'], align=PP_ALIGN.CENTER)

# 音频表征
add_rounded_rect(s3, 5.8, y_aud+1.45, 1.7, 0.55, C['dark'], radius=0.12)
add_text_multi(s3, ["音频表征 f_aud"], 5.8, y_aud+1.45, 1.7, 0.55, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s3, 6.55, y_aud+1.35, 6.65, y_aud+1.45, C['aud'])

# 声学语义分支标注
add_text(s3, "声学语义分支 f_sem", 7.5, y_aud+1.45, 2.0, 0.3, font_size=7, color=C['aud'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s3, 8.5, y_aud+1.35, 8.5, y_aud+1.45, C['aud'])


# ===== 跨模态一致性 X-Cons =====
y_xc = 1.35
add_rounded_rect(s3, 5.8, y_xc, 1.8, 1.0, C['xcons'], radius=0.12)
add_text_multi(s3, ["同步对比学习", "f_img  ↔  f_vid  ↔  f_aud", "拉齐真实/伪造样本分布"], 5.8, y_xc, 1.8, 1.0, font_size=7.5, color=C['white'], align=PP_ALIGN.CENTER)

add_rounded_rect(s3, 5.8, y_xc+1.15, 1.8, 1.0, C['xcons'], radius=0.12)
add_text_multi(s3, ["跨对相关建模", "f_sync  /  f_async", "多模态特征交互"], 5.8, y_xc+1.15, 1.8, 1.0, font_size=7.5, color=C['white'], align=PP_ALIGN.CENTER)

# 从 f_img/f_vid/f_aud 到 X-Cons 的虚线
add_dashed_arrow(s3, 2.5, y_img_repr+0.3, 5.8, y_xc+0.3, C['xcons'])
add_dashed_arrow(s3, 4.45, y_vid2+0.6, 5.8, y_xc+1.4, C['xcons'])
add_dashed_arrow(s3, 7.5, y_aud+1.725, 7.6, y_xc+1.4, C['xcons'])


# ===== 门控融合 G-Fuse =====
# 输入特征
fuse_x = 10.05
add_rounded_rect(s3, fuse_x, 1.3, 1.3, 0.4, C['img'], radius=0.1)
add_text(s3, "f_img", fuse_x, 1.3, 1.3, 0.4, font_size=9, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

add_rounded_rect(s3, fuse_x, 1.8, 1.3, 0.4, C['vid'], radius=0.1)
add_text(s3, "f_vid", fuse_x, 1.8, 1.3, 0.4, font_size=9, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

add_rounded_rect(s3, fuse_x, 2.3, 1.3, 0.4, C['aud'], radius=0.1)
add_text(s3, "f_aud", fuse_x, 2.3, 1.3, 0.4, font_size=9, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

add_rounded_rect(s3, fuse_x, 2.8, 1.3, 0.4, C['xcons'], radius=0.1)
add_text(s3, "f_sync", fuse_x, 2.8, 1.3, 0.4, font_size=9, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

# 特征对齐
add_rounded_rect(s3, fuse_x-0.05, 3.4, 1.4, 0.55, C['fuse'], radius=0.1)
add_text_multi(s3, ["特征对齐", "Projection"], fuse_x-0.05, 3.4, 1.4, 0.55, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

# 箭头
for yy in [1.7, 2.2, 2.7, 3.2]:
    add_arrow(s3, fuse_x+0.65, yy, fuse_x+0.65, yy+0.15, C['fuse'])

# 可靠性评估
add_rounded_rect(s3, fuse_x-0.05, 4.1, 1.4, 0.55, C['fuse'], radius=0.1)
add_text_multi(s3, ["可靠性评估", "Confidence"], fuse_x-0.05, 4.1, 1.4, 0.55, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s3, fuse_x+0.65, 3.95, fuse_x+0.65, 4.1, C['fuse'])

# 门控加权融合
add_rounded_rect(s3, fuse_x-0.05, 4.8, 1.4, 0.55, C['fuse'], radius=0.1)
add_text_multi(s3, ["门控加权融合", "Gated Attention"], fuse_x-0.05, 4.8, 1.4, 0.55, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s3, fuse_x+0.65, 4.65, fuse_x+0.65, 4.8, C['fuse'])

# 全连接分类器
add_rounded_rect(s3, fuse_x-0.05, 5.5, 1.4, 0.55, C['dark'], radius=0.1)
add_text_multi(s3, ["全连接分类器", "FC Classifier"], fuse_x-0.05, 5.5, 1.4, 0.55, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s3, fuse_x+0.65, 5.35, fuse_x+0.65, 5.5, C['fuse'])

# 最终判定 p_fake
add_oval(s3, fuse_x+0.1, 6.2, 1.2, 0.8, C['fuse'])
add_text_multi(s3, ["p_fake", "最终判定 (AI/真实)"], fuse_x+0.1, 6.2, 1.2, 0.8, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s3, fuse_x+0.65, 6.05, fuse_x+0.7, 6.2, C['fuse'])


# ===== 图例 =====
legend_y = 6.4
add_rect(s3, 0.3, legend_y, 1.0, 0.35, C['dark'])
add_text_multi(s3, ["图例 Legend"], 0.3, legend_y, 1.0, 0.35, font_size=9, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

legends = [
    (1.4,  C['img'],    "图像分支"),
    (2.5,  C['vid'],    "视频分支"),
    (3.6,  C['aud'],    "音频分支"),
    (4.7,  C['xcons'],  "跨模态一致性"),
    (6.0,  C['fuse'],   "门控融合"),
    (7.2,  C['dark'],   "特征表征"),
]
for lx, col, label in legends:
    add_rounded_rect(s3, lx, legend_y+0.075, 0.3, 0.2, col, radius=0.05)
    add_text(s3, label, lx+0.35, legend_y, 0.9, 0.35, font_size=9, color=C['dark'], align=PP_ALIGN.LEFT)

print("  ✓ Slide 3 整体架构")


# ================================================================
# SLIDE 4: 图像分支详解
# ================================================================
s4 = prs.slides.add_slide(blank)
set_bg(s4, C['white'])

add_rect(s4, 0, 0, 13.333, 0.65, C['img'])
title4 = s4.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.5))
tf = title4.text_frame; p = tf.paragraphs[0]
r = p.add_run(); r.text = "图像分支  Φ_img   Image Branch"
r.font.size = Pt(26); r.font.bold = True; r.font.name = 'Georgia'; r.font.color.rgb = C['white']

# 左侧架构流
stream_x = 0.5
nodes = [
    ("图像输入", "(B, 3, H, W)", C['img']),
    ("图像预处理", "Resize 224×224 + Normalize + 数据增强", C['img_m']),
    ("CLIP-ViT 视觉编码器", "ViT-B/16 或 ViT-L/14\n预训练权重", C['img_m']),
    ("三层特征提取", "Layer 3 (浅) → 纹理\nLayer 6 (中) → 结构\nLayer 12 (深) → 语义", C['img']),
    ("MSFF 多尺度特征融合", "特征投影 + 层级注意力加权\n残差融合", C['img']),
    ("SRM/FFT 局部取证支路", "高频残差噪声 + 频域特征\n捕获生成器痕迹", C['img_m']),
    ("图像表征 f_img", "拼接融合 → [B, D]", C['dark']),
    ("图像分类头 (辅助)", "MLP → p_img", C['fuse']),
]

for i, (name, desc, col) in enumerate(nodes):
    y = 0.85 + i * 0.72
    box = add_rounded_rect(s4, stream_x, y, 3.5, 0.62, col, radius=0.1)
    add_text_multi(s4, [name, desc], stream_x, y, 3.5, 0.62, font_size=8, color=C['white'], align=PP_ALIGN.LEFT)
    if i < len(nodes) - 1:
        add_arrow(s4, stream_x + 1.75, y + 0.62, stream_x + 1.75, y + 0.72, C['img'])

# 右侧 MSFF 模块详解
right_x = 4.2
add_rect(s4, right_x, 0.85, 8.8, 3.4, C['img_l'], line=C['img'], line_w=1)
add_text(s4, "MSFF - 多尺度特征融合模块详解", right_x+0.15, 0.9, 8.5, 0.35, font_size=13, bold=True, color=C['img'])

# ViT 三层
vlbs = [("Layer 3", "浅层特征\n纹理/边缘", 1.35),
        ("Layer 6", "中层特征\n结构/形状", 2.25),
        ("Layer 12", "深层特征\n语义/内容", 3.15)]
for vl, vd, vy in vlbs:
    add_rounded_rect(s4, right_x+0.2, vy, 1.6, 0.55, C['img'], radius=0.1)
    add_text_multi(s4, [vl, vd], right_x+0.2, vy, 1.6, 0.55, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

# 特征投影
add_rounded_rect(s4, right_x+2.1, 1.75, 1.5, 1.2, C['img_m'], radius=0.1)
add_text_multi(s4, ["特征投影", "Linear", "将三层映射", "到同一空间"], right_x+2.1, 1.75, 1.5, 1.2, font_size=7.5, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s4, right_x+1.8, 1.4, right_x+2.1, 1.95, C['img'])
add_arrow(s4, right_x+1.8, 2.3, right_x+2.1, 2.35, C['img'])
add_arrow(s4, right_x+1.8, 3.2, right_x+2.1, 2.85, C['img'])

# 层级注意力
add_rounded_rect(s4, right_x+3.8, 1.75, 1.5, 1.2, C['img_m'], radius=0.1)
add_text_multi(s4, ["层级注意力", "Cross-Attention", "学习层间权重", "融合"], right_x+3.8, 1.75, 1.5, 1.2, font_size=7.5, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s4, right_x+3.6, 2.35, right_x+3.8, 2.35, C['img'])

# 残差融合
add_rounded_rect(s4, right_x+5.5, 1.75, 1.5, 1.2, C['img'], radius=0.1)
add_text_multi(s4, ["残差融合", "+ Residual", "保留各层", "原始信息"], right_x+5.5, 1.75, 1.5, 1.2, font_size=7.5, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s4, right_x+5.3, 2.35, right_x+5.5, 2.35, C['img'])

# 输出 f_img
add_rounded_rect(s4, right_x+7.2, 2.05, 1.3, 0.6, C['dark'], radius=0.1)
add_text_multi(s4, ["f_img"], right_x+7.2, 2.05, 1.3, 0.6, font_size=12, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s4, right_x+7.0, 2.35, right_x+7.2, 2.35, C['img'])

# SRM/FFT 下方
add_rect(s4, right_x, 4.35, 8.8, 2.95, C['img_l'], line=C['img'], line_w=1)
add_text(s4, "SRM/FFT 局部取证支路 - 捕获生成器痕迹", right_x+0.15, 4.4, 8.5, 0.35, font_size=13, bold=True, color=C['img'])

# SRM 模块
add_rounded_rect(s4, right_x+0.3, 4.9, 2.8, 2.0, C['img_m'], radius=0.1)
add_text_multi(s4, ["SRM 噪声残差", "", "· 高通滤波器提取图像残差", "· 真实照片 vs AI图像", "  在高频噪声结构上有差异", "· 用于检测 GAN/SD 生成痕迹"], right_x+0.3, 4.9, 2.8, 2.0, font_size=8, color=C['white'], align=PP_ALIGN.LEFT)

# FFT 模块
add_rounded_rect(s4, right_x+3.3, 4.9, 2.8, 2.0, C['img_m'], radius=0.1)
add_text_multi(s4, ["FFT 频域分析", "", "· 将图像转换到频域", "· 周期性伪影检测", "  (Diffusion 模型特有)", "· 高频能量比特征"], right_x+3.3, 4.9, 2.8, 2.0, font_size=8, color=C['white'], align=PP_ALIGN.LEFT)

# 与 MSFF 汇合
add_rounded_rect(s4, right_x+6.3, 5.5, 2.3, 0.8, C['dark'], radius=0.1)
add_text_multi(s4, ["与 f_img 拼接融合", "Concat + Projection"], right_x+6.3, 5.5, 2.3, 0.8, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s4, right_x+6.1, 5.9, right_x+6.3, 5.9, C['img'])
add_arrow(s4, right_x+4.7, 5.9, right_x+6.3, 5.7, C['img'])

print("  ✓ Slide 4 图像分支")


# ================================================================
# SLIDE 5: 视频 + 音频分支详解
# ================================================================
s5 = prs.slides.add_slide(blank)
set_bg(s5, C['white'])

# 左 - 视频
add_rect(s5, 0, 0, 6.5, 0.65, C['vid'])
title5a = s5.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(6), Inches(0.5))
tf = title5a.text_frame; p = tf.paragraphs[0]
r = p.add_run(); r.text = "视频分支  Φ_vid"
r.font.size = Pt(22); r.font.bold = True; r.font.name = 'Georgia'; r.font.color.rgb = C['white']

vid_nodes = [
    ("视频输入", "(B, T, 3, H, W)", C['vid']),
    ("帧采样 + 归一化", "均匀采样 T=16 帧", C['vid_m']),
    ("共享图像编码器 CLIP-ViT", "(与图像分支参数共享)", C['vid_m']),
    ("帧特征序列", "[B, T, D]", C['vid_m']),
    ("T-Agg 时序聚合", "Mamba / Transformer\n建模帧间依赖关系", C['vid']),
    ("帧级特征融合", "MeanPool / CrossAttn\n聚合时序信息", C['vid']),
    ("视频表征 f_vid", "[B, D]", C['dark']),
]
for i, (name, desc, col) in enumerate(vid_nodes):
    y = 0.85 + i * 0.88
    if y + 0.65 > 7.5: break
    add_rounded_rect(s5, 0.3, y, 2.8, 0.72, col, radius=0.1)
    add_text_multi(s5, [name, desc], 0.3, y, 2.8, 0.72, font_size=8.5, color=C['white'], align=PP_ALIGN.LEFT)
    if i < len(vid_nodes) - 1:
        add_arrow(s5, 1.7, y+0.72, 1.7, y+0.88, C['vid'])

# T-Agg 时序聚合详解
add_rect(s5, 3.3, 0.85, 3.0, 4.2, C['vid_l'], line=C['vid'], line_w=1)
add_text(s5, "T-Agg 时序聚合详解", 3.45, 0.9, 2.7, 0.3, font_size=11, bold=True, color=C['vid'])

add_rounded_rect(s5, 3.45, 1.35, 2.7, 1.0, C['vid'], radius=0.1)
add_text_multi(s5, ["Mamba", "选择性状态空间模型\n线性复杂度 O(T)\n捕捉长程时序依赖"], 3.45, 1.35, 2.7, 1.0, font_size=7.5, color=C['white'], align=PP_ALIGN.CENTER)

add_rounded_rect(s5, 3.45, 2.5, 2.7, 1.0, C['vid_m'], radius=0.1)
add_text_multi(s5, ["Transformer Encoder", "Self-Attention\n全对全帧间交互\nO(T²) 复杂度"], 3.45, 2.5, 2.7, 1.0, font_size=7.5, color=C['white'], align=PP_ALIGN.CENTER)

add_rounded_rect(s5, 3.45, 3.65, 2.7, 1.2, C['dark'], radius=0.1)
add_text_multi(s5, ["输出: 视频表征 f_vid", "", "· 全局平均池化", "· 或 CLS token 聚合", "· 维度对齐到 f_img/f_aud"], 3.45, 3.65, 2.7, 1.2, font_size=7.5, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

# ============ 右 - 音频 ============
add_rect(s5, 6.833, 0, 6.5, 0.65, C['aud'])
title5b = s5.shapes.add_textbox(Inches(6.833), Inches(0.1), Inches(6), Inches(0.5))
tf = title5b.text_frame; p = tf.paragraphs[0]
r = p.add_run(); r.text = "音频分支  Φ_aud"
r.font.size = Pt(22); r.font.bold = True; r.font.name = 'Georgia'; r.font.color.rgb = C['white']

aud_nodes = [
    ("音频输入", "原始波形 (16kHz)", C['aud']),
    ("音频预处理", "重采样 + 归一化 + 分段", C['aud_m']),
    ("WavLM / XLS-R 编码器", "Transformer Layers (自监督预训练)", C['aud_m']),
    ("时频特征分支 f_tf", "Log-Mel + 2D CNN\n频谱图卷积特征", C['aud']),
    ("声学语义分支 f_sem", "Attention Pooling + Mean\n全局语义表示", C['aud']),
    ("音频表征 f_aud", "[B, D] 拼接融合", C['dark']),
]
for i, (name, desc, col) in enumerate(aud_nodes):
    y = 0.85 + i * 0.98
    if y + 0.65 > 7.5: break
    add_rounded_rect(s5, 6.833, y, 2.8, 0.82, col, radius=0.1)
    add_text_multi(s5, [name, desc], 6.833, y, 2.8, 0.82, font_size=8, color=C['white'], align=PP_ALIGN.LEFT)
    if i < len(aud_nodes) - 1:
        add_arrow(s5, 8.233, y+0.82, 8.233, y+0.98, C['aud'])

# 音频双分支详解
add_rect(s5, 9.8, 0.85, 3.2, 4.8, C['aud_l'], line=C['aud'], line_w=1)
add_text(s5, "双分支特征详解", 9.95, 0.9, 2.9, 0.3, font_size=11, bold=True, color=C['aud'])

add_rounded_rect(s5, 9.95, 1.35, 2.9, 1.6, C['aud'], radius=0.1)
add_text_multi(s5, ["时频特征 f_tf", "Time-Frequency Branch", "", "· Log-Mel 频谱图 (128 mel, 1024 FFT)", "· 2D CNN 提取局部时频纹理", "· 捕获 AI-TTS 高频伪影"], 9.95, 1.35, 2.9, 1.6, font_size=7.5, color=C['white'], align=PP_ALIGN.LEFT)

add_rounded_rect(s5, 9.95, 3.1, 2.9, 1.6, C['aud_m'], radius=0.1)
add_text_multi(s5, ["声学语义 f_sem", "Semantic Branch", "", "· WavLM 隐层表示 H_a ∈ R^(T×d)", "· 注意力池化 + 均值池化", "· 建模全局声学语义信息"], 9.95, 3.1, 2.9, 1.6, font_size=7.5, color=C['white'], align=PP_ALIGN.LEFT)

add_rounded_rect(s5, 9.95, 4.9, 2.9, 0.65, C['dark'], radius=0.1)
add_text_multi(s5, ["融合 f_aud = Concat(f_tf, f_sem)"], 9.95, 4.9, 2.9, 0.65, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s5, 11.4, 2.95, 11.4, 4.9, C['aud'])
add_arrow(s5, 11.4, 4.7, 11.4, 4.9, C['aud'])

print("  ✓ Slide 5 视频+音频分支")


# ================================================================
# SLIDE 6: 跨模态一致性 + 门控融合
# ================================================================
s6 = prs.slides.add_slide(blank)
set_bg(s6, C['white'])

# 上 - X-Cons
add_rect(s6, 0, 0, 13.333, 0.65, C['xcons'])
title6 = s6.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.5))
tf = title6.text_frame; p = tf.paragraphs[0]
r = p.add_run(); r.text = "跨模态一致性 & 门控融合   X-Cons + G-Fuse"
r.font.size = Pt(24); r.font.bold = True; r.font.name = 'Georgia'; r.font.color.rgb = C['white']

# X-Cons 左大框
add_rect(s6, 0.3, 0.85, 7.8, 3.0, C['xcons_l'], line=C['xcons'], line_w=1.5)
add_text(s6, "X-Cons 跨模态一致性", 0.45, 0.9, 7.5, 0.35, font_size=14, bold=True, color=C['xcons'])

# 三个输入特征
for i, (name, col, x_pos) in enumerate([("f_img", C['img'], 0.5),
                                         ("f_vid", C['vid'], 2.3),
                                         ("f_aud", C['aud'], 4.1)]):
    add_rounded_rect(s6, x_pos, 1.4, 1.5, 0.5, col, radius=0.1)
    add_text(s6, name, x_pos, 1.4, 1.5, 0.5, font_size=11, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

# 同步对比学习
add_rounded_rect(s6, 0.5, 2.05, 3.5, 1.6, C['xcons'], radius=0.1)
add_text_multi(s6, ["同步对比学习 Sync Contrastive Learning", "", "目标：拉齐同一样本不同模态特征", "", "· 正样本: 同一 (图像, 视频, 音频) 三元组", "· 负样本: 跨样本不同模态组合", "· Loss = InfoNCE / SupCon 变体"], 0.5, 2.05, 3.5, 1.6, font_size=7.5, color=C['white'], align=PP_ALIGN.LEFT)
for xi in [1.25, 3.05, 4.85]:
    add_dashed_arrow(s6, xi, 1.9, 2.25, 2.05, C['xcons'])

# 跨对相关建模
add_rounded_rect(s6, 4.2, 2.05, 3.7, 1.6, C['xcons'], radius=0.1)
add_text_multi(s6, ["跨对相关建模 Cross-Pair Relation", "", "捕获模态间依赖关系", "", "· 同步特征 f_sync: 真实/伪造样本的对齐表示", "· 异步特征 f_async: 捕捉跨模态不一致性", "· 交互建模 → 辅助判别依据"], 4.2, 2.05, 3.7, 1.6, font_size=7.5, color=C['white'], align=PP_ALIGN.LEFT)
add_dashed_arrow(s6, 4.0, 2.85, 4.2, 2.85, C['xcons'])

# G-Fuse 右大框
add_rect(s6, 8.3, 0.85, 4.7, 6.2, C['fuse_l'], line=C['fuse'], line_w=1.5)
add_text(s6, "G-Fuse 门控融合", 8.45, 0.9, 4.4, 0.35, font_size=14, bold=True, color=C['fuse'])

# G-Fuse 输入特征
gf_inputs = [("f_img", C['img'], 1.3), ("f_vid", C['vid'], 1.95),
             ("f_aud", C['aud'], 2.6), ("f_sync", C['xcons'], 3.25)]
for name, col, yp in gf_inputs:
    add_rounded_rect(s6, 8.5, yp, 1.3, 0.45, col, radius=0.08)
    add_text(s6, name, 8.5, yp, 1.3, 0.45, font_size=10, color=C['white'], bold=True, align=PP_ALIGN.CENTER)

# 特征对齐
add_rounded_rect(s6, 10.0, 1.3, 2.8, 0.6, C['fuse'], radius=0.1)
add_text_multi(s6, ["特征对齐 Feature Alignment", "Projection Matrix"], 10.0, 1.3, 2.8, 0.6, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
for yp in [1.75, 2.4, 3.05, 3.7]:
    add_arrow(s6, 9.8, yp+0.225, 10.0, yp+0.225, C['fuse'])

# 可靠性评估
add_rounded_rect(s6, 10.0, 2.05, 2.8, 0.6, C['fuse'], radius=0.1)
add_text_multi(s6, ["可靠性评估 Reliability Estimation", "Uncertainty / Confidence Head"], 10.0, 2.05, 2.8, 0.6, font_size=8, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s6, 11.4, 1.9, 11.4, 2.05, C['fuse'])

# 门控加权融合
add_rounded_rect(s6, 10.0, 2.8, 2.8, 0.8, C['fuse'], radius=0.1)
add_text_multi(s6, ["门控加权融合 Gated Fusion", "", "α_i = softmax(W·f_i + b)", "f_fused = Σ α_i · f_i"], 10.0, 2.8, 2.8, 0.8, font_size=7.5, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s6, 11.4, 2.65, 11.4, 2.8, C['fuse'])
add_arrow(s6, 11.4, 2.65, 11.4, 3.6, C['fuse'])  # skip - no

# 全连接分类器
add_rounded_rect(s6, 10.0, 3.8, 2.8, 0.6, C['dark'], radius=0.1)
add_text_multi(s6, ["全连接分类器", "FC Classifier"], 10.0, 3.8, 2.8, 0.6, font_size=9, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s6, 11.4, 3.6, 11.4, 3.8, C['fuse'])

# 最终输出
add_oval(s6, 10.6, 4.5, 1.6, 1.0, C['fuse'])
add_text_multi(s6, ["p_fake", "最终概率", "(0=真实 / 1=AI)"], 10.6, 4.5, 1.6, 1.0, font_size=9, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
add_arrow(s6, 11.4, 4.4, 11.4, 4.5, C['fuse'])

# 下方: G-Fuse 机制图解
add_rect(s6, 0.3, 4.0, 7.8, 3.25, C['fuse_l'], line=C['fuse'], line_w=1)
add_text(s6, "门控融合机制图解", 0.45, 4.05, 7.5, 0.3, font_size=13, bold=True, color=C['fuse'])

# 融合权重示意
weights = [("f_img", 0.35, C['img']), ("f_vid", 0.28, C['vid']),
           ("f_aud", 0.25, C['aud']), ("f_sync", 0.12, C['xcons'])]
bar_x = 0.5; bar_y = 4.5; bar_h = 0.4; bar_w_max = 6.5
for i, (name, w, col) in enumerate(weights):
    yy = bar_y + i * 0.5
    add_text(s6, name, bar_x, yy, 1.0, bar_h, font_size=9, bold=True, color=C['dark'], align=PP_ALIGN.RIGHT)
    bar_w = bar_w_max * w
    add_rounded_rect(s6, bar_x + 1.05, yy+0.05, bar_w, bar_h-0.1, col, radius=0.05)
    add_text(s6, f"α = {w:.2f}", bar_x + 1.05 + bar_w + 0.1, yy, 1.2, bar_h, font_size=9, color=C['dark'], align=PP_ALIGN.LEFT)

# 公式
add_text_multi(s6, ["f_fused = α₁·f_img + α₂·f_vid + α₃·f_aud + α₄·f_sync", "α = softmax( Reliability(x) )  ← 自适应权重", "当某模态缺失/不可靠时，门控自动降权"],
               0.5, 6.2, 7.5, 0.9, font_size=9, color=C['fuse'], bold=True)

print("  ✓ Slide 6 跨模态一致性+门控融合")


# ================================================================
# SLIDE 7: 关键模块总结 + 输出说明
# ================================================================
s7 = prs.slides.add_slide(blank)
set_bg(s7, C['white'])

add_rect(s7, 0, 0, 13.333, 0.65, C['dark'])
title7 = s7.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.5))
tf = title7.text_frame; p = tf.paragraphs[0]
r = p.add_run(); r.text = "关键模块总结 & 最终输出   Key Modules & Output"
r.font.size = Pt(24); r.font.bold = True; r.font.name = 'Georgia'; r.font.color.rgb = C['white']

# 左 - 模块卡片
modules = [
    ("MSFF",     "多尺度特征融合",          "图像分支", "CLIP-ViT 三层特征 + 层级注意力 + 残差", C['img']),
    ("T-Agg",    "时序聚合模块",            "视频分支", "Mamba / Transformer 建模帧间依赖", C['vid']),
    ("X-Cons",   "跨模态一致性学习",        "跨模态",   "同步对比学习 + 跨对相关建模", C['xcons']),
    ("G-Fuse",   "门控融合模块",            "融合层",   "可靠性评估 + 自适应加权融合 + FC分类", C['fuse']),
    ("SRM/FFT",  "局部取证支路",            "图像辅助", "高频噪声 + 频域特征检测AI痕迹", C['img_m']),
    ("Shared Encoder", "参数共享图像编码器", "图像/视频", "CLIP-ViT 参数共享, 减少冗余", C['vid_m']),
]

for i, (code, name, branch, desc, col) in enumerate(modules):
    col_idx = i % 3; row_idx = i // 3
    x = 0.3 + col_idx * 4.3; y = 0.85 + row_idx * 2.85
    # 背景
    add_rect(s7, x, y, 4.1, 2.6, C['light'], line=col, line_w=1.25)
    # 顶部色条
    add_rect(s7, x, y, 4.1, 0.08, col)
    # 代码标识
    code_box = add_rounded_rect(s7, x + 0.15, y + 0.15, 1.2, 0.4, col, radius=0.08)
    add_text(s7, code, x+0.15, y+0.15, 1.2, 0.4, font_size=10, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
    # 模块名
    add_text_multi(s7, [name], x+1.45, y+0.15, 2.5, 0.4, font_size=12, color=C['dark'], bold=True, align=PP_ALIGN.LEFT)
    # 所属分支
    branch_box = add_rounded_rect(s7, x + 0.15, y + 0.6, 1.0, 0.32, col, radius=0.05)
    add_text(s7, branch, x+0.15, y+0.6, 1.0, 0.32, font_size=7.5, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
    # 描述
    add_text_multi(s7, [desc], x + 0.15, y + 1.0, 3.8, 1.4, font_size=10, color=C['gray'], align=PP_ALIGN.LEFT)

# 右列 - 输出规范
out_x = 9.0
out_y = 0.85
add_rect(s7, out_x-0.1, out_y, 4.2, 2.85, C['fuse_l'], line=C['fuse'], line_w=1.25)
add_text(s7, "最终输出规范", out_x, out_y+0.05, 4.0, 0.35, font_size=13, bold=True, color=C['fuse'])

outputs = [
    ("p_fake",  "最终 AI 生成概率",         "[0, 1] 范围, >0.5 → AI"),
    ("p_img",   "图像分支辅助概率",         "图像模态独立判定"),
    ("p_vid",   "视频分支辅助概率",         "视频模态独立判定"),
    ("p_aud",   "音频分支辅助概率",         "音频模态独立判定"),
]
for i, (name, desc, range_v) in enumerate(outputs):
    oy = out_y + 0.5 + i * 0.5
    add_rounded_rect(s7, out_x, oy, 1.3, 0.38, C['fuse'], radius=0.08)
    add_text(s7, name, out_x, oy, 1.3, 0.38, font_size=9, color=C['white'], bold=True, align=PP_ALIGN.CENTER)
    add_text_multi(s7, [desc, range_v], out_x+1.4, oy-0.02, 2.7, 0.42, font_size=8.5, color=C['dark'], align=PP_ALIGN.LEFT)

# 模型特点 (底部)
add_rect(s7, 0.3, 6.65, 12.7, 0.7, C['dark'])
feats = [
    ("✓", "共享视觉编码器，减少冗余参数", C['img']),
    ("✓", "多尺度特征融合，兼顾全局语义与局部伪造痕迹", C['vid']),
    ("✓", "时序建模 + 音频双分支，细粒度检测能力", C['aud']),
    ("✓", "跨模态一致性学习，模态间互补增强鲁棒性", C['xcons']),
    ("✓", "门控融合，自适应处理模态缺失/不可靠", C['fuse']),
]
for i, (mark, text, col) in enumerate(feats):
    fx = 0.5 + i * 2.55
    add_text(s7, mark, fx, 6.7, 0.25, 0.3, font_size=11, bold=True, color=col)
    add_text_multi(s7, [text], fx+0.3, 6.7, 2.2, 0.3, font_size=8.5, color=C['white'])

print("  ✓ Slide 7 总结+输出")


# ================================================================
# SLIDE 8: 感谢页
# ================================================================
s8 = prs.slides.add_slide(blank)
set_bg(s8, C['dark'])
add_rect(s8, 0, 0, 13.333, 0.08, C['img'])
add_rect(s8, 0, 7.42, 13.333, 0.08, C['fuse'])

thanks = s8.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(1.2))
tf = thanks.text_frame; tf.word_wrap = True; tf.paragraphs[0].alignment = PP_ALIGN.CENTER
r = tf.paragraphs[0].add_run(); r.text = "THANK YOU"
r.font.size = Pt(60); r.font.bold = True; r.font.name = 'Georgia'
r.font.color.rgb = C['white']; r.font.character_spacing = Pt(4)

thanks2 = s8.shapes.add_textbox(Inches(0.8), Inches(3.5), Inches(11.7), Inches(1.0))
tf2 = thanks2.text_frame; tf2.word_wrap = True; tf2.paragraphs[0].alignment = PP_ALIGN.CENTER
r2 = tf2.paragraphs[0].add_run(); r2.text = "多模态 AIGC 真伪检测 Transformer 架构"
r2.font.size = Pt(22); r2.font.name = 'Georgia'
r2.font.color.rgb = C['img_m']

thanks3 = s8.shapes.add_textbox(Inches(0.8), Inches(4.6), Inches(11.7), Inches(0.8))
tf3 = thanks3.text_frame; tf3.paragraphs[0].alignment = PP_ALIGN.CENTER
r3 = tf3.paragraphs[0].add_run(); r3.text = "CLIP-ViT  ·  WavLM  ·  Temporal Transformer  ·  Gated Fusion"
r3.font.size = Pt(14); r3.font.name = 'Calibri'; r3.font.color.rgb = C['gray']

print("  ✓ Slide 8 感谢页")


# ===== 保存 =====
prs.save(OUT)
print(f"\n{'='*60}")
print(f"✅ PPT 框架图已保存!")
print(f"   路径: {OUT}")
print(f"   页数: 8 页")
print(f"{'='*60}")
