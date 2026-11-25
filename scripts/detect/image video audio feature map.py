# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:25:41 2026

@author: joe
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn

# ==========================================
# 0. 创建统一保存目录
# ==========================================
OUTPUT_DIR = "output_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"所有特征图将统一保存至目录: {os.path.abspath(OUTPUT_DIR)}")

# ==========================================
# 1. 本地测试数据自动生成器
# ==========================================
def create_local_sample_files():
    if not os.path.exists("local_image.jpg"):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.circle(img, (112, 112), 60, (180, 200, 255), -1)
        cv2.putText(img, 'AI Test', (55, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        cv2.imwrite("local_image.jpg", img)

    if not os.path.exists("local_audio.wav"):
        sr = 22000
        t = np.linspace(0, 2, sr * 2)
        audio_signal = np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 8000 * t)
        audio_signal = (audio_signal / np.max(np.abs(audio_signal)) * 32767).astype(np.int16)
        wavfile.write("local_audio.wav", sr, audio_signal)

    if not os.path.exists("local_video.mp4"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("local_video.mp4", fourcc, 10, (224, 224))
        for i in range(15):
            f = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.circle(f, (30 + i * 10, 112), 40, (0, 255, 120), -1)
            out.write(f)
        out.release()

# ==========================================
# 2. 图像 / 视频帧特征图提取与保存
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

def save_image_feature_map(image_path, save_filename):
    detector = SimpleFeatureDetector()
    detector.eval()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"无法读取图片: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        feature_maps = detector(img_tensor)[0].numpy()

    heatmap = np.mean(feature_maps, axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image/Frame")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("CNN Feature Map")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Feature Heatmap Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已成功保存图片特征图 -> {save_path}")

# ==========================================
# 3. 音频 STFT 语谱图提取与保存
# ==========================================
import librosa
import librosa.display

def save_audio_spectrogram(audio_path, save_filename):
    if not os.path.exists(audio_path):
        print(f"音频文件不存在: {audio_path}")
        return

    try:
        # librosa.load 会自动处理格式转换、重采样与多通道合并（单声道）
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"音频读取失败 ({audio_path}): {e}")
        return

    # 计算短时傅里叶变换 (STFT)
    D = librosa.stft(audio_data)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # 绘制语谱图
    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")
    ax.set_title("Audio STFT Spectrogram Feature Map")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已成功保存音频特征图 -> {save_path}")

# ==========================================
# 4. 主程序运行
# ==========================================
if __name__ == "__main__":
    create_local_sample_files()

#    local_img_path = "2.jpg"
    local_video_path = "test/video/10216928771641495851.mp4"
#    local_audio_path = "MiniMax_2026-10-15_21_16_12_Upset_Girl.mp3"

    # 1. 保存图像特征图
#    save_image_feature_map(local_img_path, "image_feature_map.png")

    # 2. 抽离视频关键帧并保存特征图
    cap = cv2.VideoCapture(local_video_path)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("temp_video_frame.jpg", frame)
            save_image_feature_map("temp_video_frame.jpg", "video_frame_feature_map.png")
            if os.path.exists("temp_video_frame.jpg"):
                os.remove("temp_video_frame.jpg")
        cap.release()

    # 3. 保存音频语谱特征图
#    save_audio_spectrogram(local_audio_path, "audio_spectrogram_feature_map.png")

    print("\n所有模态的特征图处理完成，均存放在 output_features 文件夹中！")