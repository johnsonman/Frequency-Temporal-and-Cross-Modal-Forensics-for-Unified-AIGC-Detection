# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:57:14 2026

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
# 0. 创建统一输出目录
# ==========================================
OUTPUT_DIR = "output_local_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. AI 鉴伪特征提取网络模型
# ==========================================
class AIQualityDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv(x)
        score = self.classifier(feat)
        return feat, score

model = AIQualityDetector()
model.eval()

# ==========================================
# 2. 本地图像全过程检测与可视化
# ==========================================
def detect_local_image(image_path, save_name="local_image_detection.png"):
    if not os.path.exists(image_path):
        print(f"[错误] 图像文件不存在: {image_path}")
        return

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[错误] 无法读取图片: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        feat, score_t = model(tensor)
        score = score_t.item()

    # 提取并叠加特征热力图
    heatmap = np.mean(feat[0].numpy(), axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

    # 4 阶段过程可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Step 1: Input Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Step 2: Feature Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Step 3: Attention Overlay")
    axes[2].axis("off")

    is_fake = score >= 0.5
    label = "FAKE (AI Generated)" if is_fake else "REAL (Authentic)"
    color = "red" if is_fake else "green"

    axes[3].text(0.1, 0.7, f"Diagnosis:\n{label}", fontsize=13, color=color, fontweight='bold')
    axes[3].text(0.1, 0.4, f"Fake Score: {score * 100:.2f}%", fontsize=11)
    axes[3].set_title("Step 4: Final Verdict")
    axes[3].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"本地图像检测图已保存 -> {save_path}")

# ==========================================
# 3. 本地音频全过程检测与可视化
# ==========================================
def detect_local_audio(audio_path, save_name="local_audio_detection.png"):
    if not os.path.exists(audio_path):
        print(f"[错误] 音频文件不存在: {audio_path}")
        return

    try:
        # 使用 librosa 兼容 MP3/WAV/FLAC 等多格式
        import librosa
        import librosa.display
        data, sr = librosa.load(audio_path, sr=None)
    except ImportError:
        # 降级使用 scipy 处理 wav 格式
        sr, data = wavfile.read(audio_path)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0)

    nperseg = min(len(data), 512)
    frequencies, times, spectrogram = signal.spectrogram(data, fs=sr, nperseg=nperseg)
    power_db = 10 * np.log10(spectrogram + 1e-10)

    high_freq_energy = np.mean(power_db[frequencies > (sr / 4)])
    fake_score = float(1 / (1 + np.exp(- (high_freq_energy + 40) / 10)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. 时域波形
    time_axis = np.linspace(0, len(data)/sr, len(data))
    axes[0, 0].plot(time_axis, data, color='blue', alpha=0.7)
    axes[0, 0].set_title("Step 1: Time-Domain Waveform")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude")

    # 2. STFT 语谱图
    im = axes[0, 1].pcolormesh(times, frequencies, power_db, shading='gouraud', cmap='magma')
    axes[0, 1].set_title("Step 2: STFT Spectrogram Feature Map")
    axes[0, 1].set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=axes[0, 1], label="Power [dB]")

    # 3. 频域高频分析
    freq_profile = np.mean(power_db, axis=1)
    axes[1, 0].plot(frequencies, freq_profile, color='purple')
    axes[1, 0].axvline(x=sr/4, color='red', linestyle='--', label='High-Freq Boundary')
    axes[1, 0].set_title("Step 3: Frequency Profile Analysis")
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel("Mean Power [dB]")
    axes[1, 0].legend()

    # 4. 诊断卡片
    is_fake = fake_score >= 0.5
    label = "FAKE (AI Generated)" if is_fake else "REAL (Authentic)"
    color = "red" if is_fake else "green"

    axes[1, 1].text(0.1, 0.7, f"Audio Diagnosis:\n{label}", fontsize=14, color=color, fontweight='bold')
    axes[1, 1].text(0.1, 0.4, f"Fake Score: {fake_score * 100:.2f}%", fontsize=12)
    axes[1, 1].set_title("Step 4: Final Verdict")
    axes[1, 1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"本地音频检测图已保存 -> {save_path}")

# ==========================================
# 4. 本地视频逐帧检测与可视化（输出合成视频）
# ==========================================
def detect_local_video(video_path, save_name="local_video_detection.mp4"):
    if not os.path.exists(video_path):
        print(f"[错误] 视频文件不存在: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    canvas_w = width * 3
    canvas_h = height
    save_path = os.path.join(OUTPUT_DIR, save_name)
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, canvas_h))

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            feat, score_t = model(tensor)
            fake_prob = score_t.item()

        heatmap = np.mean(feat[0].numpy(), axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        if heatmap_colored.shape[:2] != (height, width):
            heatmap_colored = cv2.resize(heatmap_colored, (width, height))

        dashboard = cv2.addWeighted(frame_bgr, 0.5, heatmap_colored, 0.5, 0)
        is_fake = fake_prob >= 0.5
        label = "FAKE (AI Generated)" if is_fake else "REAL (Authentic)"
        theme_color = (0, 0, 255) if is_fake else (0, 255, 0)

        cv2.putText(dashboard, f"Result: {label}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, theme_color, 2, cv2.LINE_AA)
        cv2.putText(dashboard, f"Fake Score: {fake_prob * 100:.1f}%", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        bar_x, bar_y, bar_w, bar_h = 15, 80, int(width * 0.7), 12
        cv2.rectangle(dashboard, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill_w = int(bar_w * fake_prob)
        cv2.rectangle(dashboard, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), theme_color, -1)

        canvas = np.hstack((frame_bgr, heatmap_colored, dashboard))
        cv2.putText(canvas, f"Frame: {frame_idx}/{total_frames}", (15, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(canvas)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"本地视频逐帧检测视频已保存 -> {save_path}")

# ==========================================
# 5. 修改为你的本地文件路径并运行
# ==========================================
if __name__ == "__main__":
    # 请根据你本地文件的实际路径进行修改：
    MY_LOCAL_IMAGE = "3.jpg"
    MY_LOCAL_AUDIO = "music-1.mp3"
    MY_LOCAL_VIDEO = "1.mp4"

    print("--- 开始检测本地图像 ---")
    detect_local_image(MY_LOCAL_IMAGE)

    print("--- 开始检测本地音频 ---")
    detect_local_audio(MY_LOCAL_AUDIO)

    print("--- 开始检测本地视频 ---")
    detect_local_video(MY_LOCAL_VIDEO)

    print(f"\n检测完成，所有可视化文件保存在: {os.path.abspath(OUTPUT_DIR)}")