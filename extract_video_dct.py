# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:12:16 2026

@author: joe
"""
import cv2
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def process_frame_dct(frame, log_scale=True):
    """
    对单张图像帧提取 2D-DCT 频域特征图
    """
    # 1. 转换到 YCbCr 色彩空间，提取 Y (亮度) 通道
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr[:, :, 0].astype(np.float32)

    # 2. 执行 2D-DCT 变换 (沿着轴 0 和轴 1 连续执行 1D-DCT)
    dct_2d = dct(dct(y_channel.T, type=2, norm='ortho').T, type=2, norm='ortho')

    # 3. 对数缩放以便于可视化展示
    if log_scale:
        spectrum_map = np.log(np.abs(dct_2d) + 1e-5)
    else:
        spectrum_map = np.abs(dct_2d)

    # 4. 归一化到 [0, 255] 区间
    spectrum_normalized = cv2.normalize(spectrum_map, None, 0, 255, cv2.NORM_MINMAX)
    spectrum_uint8 = np.uint8(spectrum_normalized)

    return spectrum_uint8, dct_2d

def extract_video_dct(video_path, output_video_path=None, max_frames=100):
    """
    读取本地视频文件并提取 DCT 频谱特征
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法读取视频文件，请检查路径是否正确: {video_path}")
        return None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"打开视频成功! 分辨率: {width}x{height}, FPS: {fps:.2f}, 总帧数: {total_frames}")

    # 配置视频输出 Writer（如果提供了输出路径）
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    frame_count = 0
    dct_features = []

    print("开始提取本地视频 DCT 频谱...")
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        spectrum_map, dct_coeffs = process_frame_dct(frame)
        dct_features.append(dct_coeffs)

        if writer:
            writer.write(spectrum_map)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count}/{min(total_frames, max_frames)} 帧...")

    cap.release()
    if writer:
        writer.release()
        print(f"频谱特征视频已保存至: {output_video_path}")

    return np.array(dct_features)

# ================= 运行本地视频处理 =================
if __name__ == "__main__":
    # 替换为您本地视频的真实路径（支持相对路径或绝对路径）
    # 例如 Windows 路径: r"D:\Videos\my_test_video.mp4"
    # 例如 Linux/Mac 路径: "/Users/username/Movies/my_test_video.mp4"
    input_video = "1.mp4"
    output_video = "my_video_dct_spectrum.mp4"

    # 执行特征提取（可通过 max_frames 限制处理的最大帧数，设为 999999 处理全视频）
    features = extract_video_dct(input_video, output_video_path=output_video, max_frames=300)

    if features is not None:
        print(f"处理完成！输出特征数组 Shape: {features.shape} (帧数, 高度, 宽度)")

        # 读取本地视频的第一帧进行对比预览
        cap = cv2.VideoCapture(input_video)
        ret, first_frame = cap.read()
        cap.release()

        if ret:
            spectrum, _ = process_frame_dct(first_frame)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original Frame")
            plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("DCT Spectrum Map (Y Channel)")
            plt.imshow(spectrum, cmap="jet")
            plt.colorbar(shrink=0.8)
            plt.axis("off")

            plt.tight_layout()
            plt.show()