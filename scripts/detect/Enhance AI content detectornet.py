# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:05:31 2026

@author: joe
"""

# -*- coding: utf-8 -*-
"""
整合版 AI 伪造检测系统 (支持本地图像、音频、视频)
架构: SRM 高频噪声滤波 + ResNet18 空间主干 + Transformer 时序注意力
"""

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# 0. 全局配置与输出目录初始化
# ==========================================
OUTPUT_DIR = "output_local_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 运行设备: {device} ===")

# ==========================================
# 1. SRM 滤波层与双流鉴伪网络架构
# ==========================================
class SRMFilterLayer(nn.Module):
    """SRM (Spatial Rich Model) 5x5 经典高频噪声提取层"""
    def __init__(self):
        super().__init__()
        srm_kernel = np.array([
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        ], dtype=np.float32)

        srm_kernel[0] /= 4.0
        srm_kernel[1] /= 12.0
        srm_kernel[2] /= 2.0

        kernel_tensor = torch.zeros((3, 3, 5, 5), dtype=torch.float32)
        for i in range(3):
            kernel_tensor[i, 0] = torch.from_numpy(srm_kernel[i])
            kernel_tensor[i, 1] = torch.from_numpy(srm_kernel[i])
            kernel_tensor[i, 2] = torch.from_numpy(srm_kernel[i])

        self.srm_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.srm_conv.weight = nn.Parameter(kernel_tensor, requires_grad=False)

    def forward(self, x):
        noise = self.srm_conv(x)
        return torch.clamp(noise, -3.0, 3.0)

class PositionalEncoding(nn.Module):
    """Transformer 序列位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class UnifiedAIDetector(nn.Module):
    """全模态双流鉴伪网络 (RGB + SRM + Transformer 时序)"""
    def __init__(self, sequence_length=16):
        super().__init__()
        self.srm_layer = SRMFilterLayer()
        self.seq_len = sequence_length

        # 修改 ResNet18 接收 6 通道输入 (RGB + SRM 噪声)
        backbone = models.resnet18(weights=None)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data[:, :3] = old_conv.weight.data
        new_conv.weight.data[:, 3:] = old_conv.weight.data
        backbone.conv1 = new_conv

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 时序 Transformer 模块
        self.pos_encoder = PositionalEncoding(d_model=512, max_len=sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 单帧与序列双分类头
        self.single_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def extract_single_feat(self, x):
        """提取单张图像的特征与注意力图"""
        srm_noise = self.srm_layer(x)
        combined = torch.cat([x, srm_noise], dim=1) # [B, 6, H, W]
        feat = self.feature_extractor(combined)     # [B, 512, H_out, W_out]
        pooled = torch.flatten(self.avgpool(feat), 1) # [B, 512]
        logits = self.single_classifier(pooled)
        return feat, logits

    def forward_sequence(self, x_seq):
        """输入连续时序帧 [B, T, 3, H, W] 进行动态时序融合判定"""
        b, t, c, h, w = x_seq.shape
        x_reshaped = x_seq.view(b * t, c, h, w)

        feat, _ = self.extract_single_feat(x_reshaped)
        seq_feats = torch.flatten(self.avgpool(feat), 1).view(b, t, -1)

        seq_feats = self.pos_encoder(seq_feats)
        trans_out = self.temporal_transformer(seq_feats)
        temporal_feat = torch.mean(trans_out, dim=1)
        logits = self.single_classifier(temporal_feat)
        return logits

# 初始化与加载模型
model = UnifiedAIDetector().to(device)
model.eval()

# ==========================================
# 2. 本地图像检测与可视化
# ==========================================
def detect_local_image(image_path, save_name="image_result.png"):
    if not os.path.exists(image_path):
        print(f"[错误] 文件不存在: {image_path}")
        return

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 预处理为 256x256 标准输入
    resized_rgb = cv2.resize(img_rgb, (256, 256))
    tensor = torch.tensor(resized_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        feat, logits = model.extract_single_feat(tensor)
        score = torch.sigmoid(logits).item()

    # 特征热力图提取与对齐
    heatmap = np.mean(feat[0].cpu().numpy(), axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.resize(heatmap_colored, (orig_w, orig_h))
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

    # 4 阶段过程可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Step 1: Raw Image")
    axes[0].axis("off")

    axes[1].imshow(cv2.resize(heatmap, (orig_w, orig_h)), cmap="jet")
    axes[1].set_title("Step 2: SRM Feature Map")
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
    print(f"[图像检测完成] -> {save_path}")

# ==========================================
# 3. 本地音频检测 (Log-Mel 频谱特征)
# ==========================================
def detect_local_audio(audio_path, save_name="audio_result.png"):
    if not os.path.exists(audio_path):
        return

    try:
        import librosa
        data, sr = librosa.load(audio_path, sr=22050, duration=5.0)
        mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    except Exception as e:
        print(f"[错误] 无法提取音频 Log-Mel 语谱图: {e}")
        return

    # 将 Log-Mel 语谱转换为 3 通道图像后输入神经网络评估
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_img = cv2.applyColorMap(np.uint8(255 * mel_norm), cv2.COLORMAP_VIRIDIS)
    mel_rgb = cv2.cvtColor(mel_img, cv2.COLOR_BGR2RGB)
    
    resized_mel = cv2.resize(mel_rgb, (256, 256))
    tensor = torch.tensor(resized_mel).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        _, logits = model.extract_single_feat(tensor)
        score = torch.sigmoid(logits).item()

    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(mel_db, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("Log-Mel Spectrogram Analysis")
    axes[0].set_ylabel("Mel Frequency")
    axes[0].set_xlabel("Time Frames")

    is_fake = score >= 0.5
    label = "FAKE (AI Voice)" if is_fake else "REAL (Authentic)"
    color = "red" if is_fake else "green"

    axes[1].text(0.1, 0.6, f"Audio Verdict:\n{label}", fontsize=14, color=color, fontweight='bold')
    axes[1].text(0.1, 0.3, f"Fake Score: {score * 100:.2f}%", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[音频检测完成] -> {save_path}")

# ==========================================
# 4. 本地视频检测 (滑动窗口 + Transformer 时序融合)
# ==========================================
def detect_local_video(video_path, seq_len=16, save_name="video_result.mp4"):
    if not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    save_path = os.path.join(OUTPUT_DIR, save_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_buffer = []
    raw_frames = []

    print(f"[视频检测中] 正在分析时序特征: {video_path}")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(frame_rgb, (256, 256))
        tensor_frame = torch.tensor(resized_rgb).permute(2, 0, 1).float() / 255.0

        frame_buffer.append(tensor_frame)
        raw_frames.append(frame_bgr)

        # 当 Buffer 积累满 16 帧时，触发 Transformer 时序推理
        if len(frame_buffer) == seq_len:
            input_seq = torch.stack(frame_buffer, dim=0).unsqueeze(0).to(device) # [1, 16, 3, 256, 256]

            with torch.no_grad():
                logits = model.forward_sequence(input_seq)
                score = torch.sigmoid(logits).item()

            is_fake = score >= 0.5
            label = "FAKE (Deepfake)" if is_fake else "REAL (Authentic)"
            color = (0, 0, 255) if is_fake else (0, 255, 0)

            # 将结果写入画面
            out_frame = raw_frames[0].copy()
            cv2.putText(out_frame, f"Temporal Result: {label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
            cv2.putText(out_frame, f"Fake Score: {score * 100:.1f}%", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(out_frame)

            frame_buffer.pop(0)
            raw_frames.pop(0)

    # 补齐剩余尾帧处理
    for img in raw_frames:
        writer.write(img)

    cap.release()
    writer.release()
    print(f"[视频检测完成] -> {save_path}")

# ==========================================
# 5. 主调度逻辑
# ==========================================
def process_path(target_path):
    if not os.path.exists(target_path):
        print(f"[错误] 指定路径不存在: {target_path}")
        return

    # 若输入为单文件
    if os.path.isfile(target_path):
        files = [target_path]
        base_dir = os.path.dirname(target_path)
    else:
        # 若输入为文件夹，递归遍历
        base_dir = target_path
        files = [os.path.join(root, f) for root, _, filenames in os.walk(target_path) for f in filenames]

    print(f"\n=== 开始分析测试目标: {os.path.abspath(target_path)} ===")

    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        if ext in IMAGE_EXTS:
            print(f"\n[处理图像] {file_path}")
            detect_local_image(file_path, save_name=f"img_{base_name}_result.png")

        elif ext in AUDIO_EXTS:
            print(f"\n[处理音频] {file_path}")
            detect_local_audio(file_path, save_name=f"audio_{base_name}_result.png")

        elif ext in VIDEO_EXTS:
            print(f"\n[处理视频] {file_path}")
            detect_local_video(file_path, save_name=f"video_{base_name}_result.mp4")

    print(f"\n=== 检测完成！结果已存至: {os.path.abspath(OUTPUT_DIR)} ===")

if __name__ == "__main__":
    # 可以修改为本地测试文件或文件夹路径
    TEST_TARGET = "./testdata" 
    
    process_path(TEST_TARGET)