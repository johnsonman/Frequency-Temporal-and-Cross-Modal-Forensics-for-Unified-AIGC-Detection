# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 20:26:32 2026

@author: joe
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 0. 目录初始化
# ==========================================
OUTPUT_DIR = "output_features"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "video_frame_detections")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ==========================================
# 1. AI 视频帧真伪检测与特征提取网络
# ==========================================
class AIVideoFrameDetector(nn.Module):
    """
    仿真 AI 鉴伪网络：
    1. 提取空间/频域伪影特征图 (Feature Map)
    2. 计算当前帧为 AI 生成 (Fake) 的概率得分
    """
    def __init__(self):
        super().__init__()
        # 特征提取网络层
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 真伪分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        fake_prob = self.classifier(feat)
        return feat, fake_prob

# ==========================================
# 2. 视频逐帧真伪检测与可视化主函数
# ==========================================
def detect_and_visualize_video(video_path, output_video_name="video_forgery_detection.mp4", frame_stride=1):
    """
    :param video_path: 本地视频文件路径
    :param output_video_name: 保存的综合检测视频文件名
    :param frame_stride: 采样步长（1 表示逐帧检测，2 表示隔帧检测）
    """
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 -> {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法读取视频 -> {video_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"开始逐帧真伪检测: {video_path}")
    print(f"视频分辨率: {width}x{height} | 总帧数: {total_frames} | FPS: {fps}")

    # 初始化模型
    model = AIVideoFrameDetector()
    model.eval()

    # 初始化合成画布尺寸 (3 画面并排: 原始帧 | 特征热力图 | 真伪诊断分析)
    canvas_w = width * 3
    canvas_h = height
    output_video_path = os.path.join(OUTPUT_DIR, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps / frame_stride, (canvas_w, canvas_h))

    frame_idx = 0
    processed_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # 1. 图像预处理
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 2. 模型前向推理（提取特征图与真伪概率）
        with torch.no_grad():
            feat_maps, fake_prob_tensor = model(input_tensor)
            fake_prob = fake_prob_tensor.item()  # 0.0 ~ 1.0 之间的伪造概率

        # 3. 提取特征图并渲染为彩色热力图
        feat_numpy = feat_maps[0].numpy()
        heatmap = np.mean(feat_numpy, axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        if heatmap_colored_bgr.shape[:2] != (height, width):
            heatmap_colored_bgr = cv2.resize(heatmap_colored_bgr, (width, height))

        # 4. 构建真伪分析诊断板 (Diagnostic Panel)
        overlay_bgr = cv2.addWeighted(frame_bgr, 0.5, heatmap_colored_bgr, 0.5, 0)
        
        # 判断真伪标签与颜色 (红色代表 AI/Fake，绿色代表 Real)
        is_fake = fake_prob >= 0.5
        label_str = "FAKE (AI Generated)" if is_fake else "REAL (Authentic)"
        theme_color = (0, 0, 255) if is_fake else (0, 255, 0)

        # 绘制 UI 文字标注与概率进度条
        cv2.putText(overlay_bgr, f"Result: {label_str}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, theme_color, 2, cv2.LINE_AA)
        cv2.putText(overlay_bgr, f"Fake Prob: {fake_prob * 100:.2f}%", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 伪造概率进度条可视化
        bar_x, bar_y, bar_w, bar_h = 20, 90, int(width * 0.6), 15
        cv2.rectangle(overlay_bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        fill_w = int(bar_w * fake_prob)
        cv2.rectangle(overlay_bgr, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), theme_color, -1)

        # 5. 横向拼接三视图 (原图 | 伪影特征图 | 真伪分析诊断图)
        canvas = np.hstack((frame_bgr, heatmap_colored_bgr, overlay_bgr))

        # 顶部全局帧数标注
        cv2.putText(canvas, f"Frame: {frame_idx}/{total_frames}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 写入视频流与保存帧文件
        writer.write(canvas)
        frame_save_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}_detection.png")
        cv2.imwrite(frame_save_path, canvas)

        processed_count += 1
        frame_idx += 1

        if processed_count % 10 == 0 or frame_idx == total_frames:
            print(f"进度: 已完成 {processed_count} 帧的真伪诊断...")

    cap.release()
    writer.release()

    print("\n视频逐帧真伪检测与可视化处理全部完成！")
    print(f"1. 包含真伪标记与特征图的合成视频已保存至: {output_video_path}")
    print(f"2. 各帧单张分析图已保存至文件夹: {FRAMES_DIR}")

# ==========================================
# 3. 运行检测
# ==========================================
if __name__ == "__main__":
    # 指定本地待检测视频路径
    target_video = "test/video/10216928771641495851.mp4"
    
    detect_and_visualize_video(
        video_path=target_video,
        output_video_name="video_forgery_detection_result.mp4",
        frame_stride=1  # 逐帧检测（若希望提速可设为 2 或 3）
    )