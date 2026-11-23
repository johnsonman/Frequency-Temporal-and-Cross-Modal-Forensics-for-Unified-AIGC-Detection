# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 20:23:45 2026

@author: joe
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ==========================================
# 0. 目录初始化
# ==========================================
OUTPUT_DIR = "output_features"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "video_frames_features")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ==========================================
# 1. 简易特征提取网络
# ==========================================
class SimpleFeatureDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        f1 = self.relu(self.conv1(x))
        f2 = self.relu(self.conv2(f1))
        return f2

# ==========================================
# 2. 视频逐帧检测与可视化处理函数
# ==========================================
def process_video_frame_by_frame(video_path, output_video_name="video_features_visualized.mp4", frame_stride=1):
    """
    逐帧检测视频并输出可视化特征图视频
    
    :param video_path: 本地视频路径
    :param output_video_name: 输出的可视化视频文件名
    :param frame_stride: 采样间隔（默认为1表示逐帧检测；若视频较长可设为2或3提高速度）
    """
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取原视频的基本属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"开始处理视频: {video_path} | 总帧数: {total_frames} | 分辨率: {width}x{height} | FPS: {fps}")

    detector = SimpleFeatureDetector()
    detector.eval()

    # 初始化视频写入器（拼接后的画布尺寸宽度为原图的 3 倍）
    canvas_width = width * 3
    canvas_height = height
    output_video_path = os.path.join(OUTPUT_DIR, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps / frame_stride, (canvas_width, canvas_height))

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # 如果设置了采样间隔
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # 1. 预处理帧
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 2. 提取特征图
        with torch.no_grad():
            feature_maps = detector(img_tensor)[0].numpy()

        # 3. 计算热力图
        heatmap = np.mean(feature_maps, axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # 转换为 BGR 彩色热力图
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # 保持分辨率一致
        if heatmap_colored_bgr.shape[:2] != (height, width):
            heatmap_colored_bgr = cv2.resize(heatmap_colored_bgr, (width, height))

        # 4. 叠加热力图到原图帧
        overlay_bgr = cv2.addWeighted(frame_bgr, 0.6, heatmap_colored_bgr, 0.4, 0)

        # 5. 横向拼接三张图：原帧 | 特征热力图 | 叠加效果图
        canvas = np.hstack((frame_bgr, heatmap_colored_bgr, overlay_bgr))

        # 在图像上添加帧号标注
        cv2.putText(canvas, f"Frame: {frame_idx}/{total_frames}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 写入视频
        writer.write(canvas)

        # 保存单帧图片到文件夹（方便单独查看）
        frame_img_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_img_path, canvas)

        saved_count += 1
        frame_idx += 1

        if saved_count % 10 == 0 or frame_idx == total_frames:
            print(f"已完成帧进度: {saved_count}/{total_frames // frame_stride}")

    cap.release()
    writer.release()

    print(f"\n处理完成！")
    print(f"1. 逐帧可视化图像已存入目录: {FRAMES_DIR}")
    print(f"2. 拼接后的合成视频已保存至: {output_video_path}")

# ==========================================
# 3. 运行测试
# ==========================================
if __name__ == "__main__":
    # 替换为你本地的视频路径
    video_path = "test/video/10216928771641495851.mp4" 
    
    # 执行逐帧检测
    process_video_frame_by_frame(
        video_path=video_path, 
        output_video_name="processed_video_features.mp4",
        frame_stride=1  # 设为 1 表示逐帧检测，若视频较长可调整为 2 或 3
    )