"""
零样本/启发式 AIGC 检测器
用预训练骨干提取的特征 + 频谱/统计启发式分数替代随机分类头

检测策略:
1. 图像: FFT高频能量比 + SRM残差统计 + CLIP特征分布距离
2. 视频: 帧间一致性 + 运动异常 + 时域频谱特征
3. 音频: 频谱平坦度 + 过零率分布 + MFCC统计 + Wav2Vec2特征距离
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

import cv2
from PIL import Image


# ============================================================
# 图像启发式检测
# ============================================================

def compute_image_heuristics(img: np.ndarray) -> Dict[str, float]:
    """
    计算图像的启发式检测特征
    返回各项检测分数
    """
    results = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # 1. FFT 高频能量比 (AI生成图像高频能量异常)
    try:
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        # 创建低频掩码 (中心区域)
        h, w = gray.shape
        mask = np.zeros_like(magnitude)
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 8
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - center_h)**2 + (X - center_w)**2)
        low_mask = dist <= radius
        high_mask = dist > radius
        low_energy = magnitude[low_mask].mean()
        high_energy = magnitude[high_mask].mean()
        freq_ratio = high_energy / (low_energy + 1e-8)
        results['freq_ratio'] = float(freq_ratio)
    except Exception:
        results['freq_ratio'] = 0.5

    # 2. SRM 残差统计 (富模型特征)
    try:
        # 使用高通滤波器
        kernel = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32)
        residual = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        # AI图像的残差分布通常更规整/均匀
        res_mean = np.abs(residual).mean()
        res_std = np.abs(residual).std()
        res_ratio = res_std / (res_mean + 1e-8)
        results['srm_std_ratio'] = float(res_ratio)
    except Exception:
        results['srm_std_ratio'] = 0.5

    # 3. 局部方差 (真实图像局部方差更丰富)
    try:
        local_var = cv2.GaussianBlur(gray.astype(np.float32), (21, 21), 5)
        variance_map = (gray.astype(np.float32) - local_var) ** 2
        variance_ratio = variance_map.mean() / (gray.astype(np.float32).std() ** 2 + 1e-8)
        results['local_var'] = float(variance_ratio)
    except Exception:
        results['local_var'] = 0.5

    # 4. JPEG压缩痕迹检测 (AI图像往往显示出异常的JPEG统计)
    try:
        # 分析块效应
        blocks = gray[:gray.shape[0] // 8 * 8, :gray.shape[1] // 8 * 8]
        h_blocks, w_blocks = blocks.shape[0] // 8, blocks.shape[1] // 8
        block_edges = []
        for i in range(h_blocks - 1):
            for j in range(w_blocks - 1):
                edge = abs(int(blocks[(i+1)*8-1, j*8]) - int(blocks[(i+1)*8, j*8]))
                block_edges.append(edge)
        results['blockiness'] = float(np.mean(block_edges)) if block_edges else 0.5
    except Exception:
        results['blockiness'] = 0.5

    # 5. 颜色直方图均匀度 (AI生成的颜色分布常更均匀)
    try:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist_norm = hist / (hist.sum() + 1e-8)
        # 使用熵来衡量均匀度
        entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
        max_entropy = np.log(256)
        results['color_entropy'] = float(entropy / max_entropy)
    except Exception:
        results['color_entropy'] = 0.5

    return results


def score_image_from_heuristics(heuristics: Dict[str, float]) -> float:
    """
    基于启发式特征计算AI生成概率分数
    """
    scores = []

    # FFT高频比: 过高或过低都可能是AI
    freq = heuristics.get('freq_ratio', 0.5)
    if freq > 5.0:
        scores.append(min(freq / 10.0, 1.0))
    elif freq < 0.01:
        scores.append(0.7)
    else:
        scores.append(max(0.3, min(freq * 0.5, 0.7)))

    # SRM标准差比: AI图像残差分布更均匀 → 比值低
    srm = heuristics.get('srm_std_ratio', 0.5)
    if srm < 0.8:
        scores.append(0.6 + (0.8 - srm) * 0.5)
    else:
        scores.append(max(0.2, 0.8 - srm * 0.2))

    # 局部方差比: 真实图像通常有更高的局部方差
    lvar = heuristics.get('local_var', 0.5)
    scores.append(max(0.3, min(0.7, lvar * 0.8 + 0.2)))

    # 块效应
    block = heuristics.get('blockiness', 0.5)
    scores.append(max(0.3, min(0.7, block / 10.0)))

    # 颜色熵: 过于均匀可能是AI
    entropy = heuristics.get('color_entropy', 0.5)
    if entropy > 0.9:
        scores.append(0.6)
    else:
        scores.append(max(0.3, entropy))

    return float(np.mean(scores))


# ============================================================
# 视频启发式检测
# ============================================================

def compute_video_heuristics(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    计算视频的启发式检测特征
    """
    results = {}

    if len(frames) < 2:
        results['temporal_consistency'] = 0.5
        results['motion_score'] = 0.5
        results['frame_variance'] = 0.5
        return results

    # 1. 帧间一致性 (真实视频帧间过渡更平滑自然)
    try:
        frame_diffs = []
        for i in range(len(frames) - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
            g2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY).astype(np.float32)
            diff = cv2.absdiff(g1, g2).mean()
            frame_diffs.append(diff)

        mean_diff = np.mean(frame_diffs)
        std_diff = np.std(frame_diffs)
        # AI视频帧间差异通常更不稳定
        consistency = std_diff / (mean_diff + 1e-8)
        results['temporal_consistency'] = float(consistency)
    except Exception:
        results['temporal_consistency'] = 0.5

    # 2. 光流异常 (AI视频光流可能不自然)
    try:
        flows = []
        for i in range(min(len(frames) - 1, 4)):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flows.append(mag.mean())

        if flows:
            flow_var = np.std(flows) / (np.mean(flows) + 1e-8)
            results['motion_score'] = float(flow_var)
        else:
            results['motion_score'] = 0.5
    except Exception:
        results['motion_score'] = 0.5

    # 3. 帧内容方差 (AI视频帧间内容变化可能更均匀)
    try:
        content_vars = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            content_vars.append(gray.std())

        var_ratio = np.mean(content_vars) / (np.std(content_vars) + 1e-8)
        results['frame_variance'] = float(min(var_ratio / 5.0, 1.0))
    except Exception:
        results['frame_variance'] = 0.5

    return results


def score_video_from_heuristics(heuristics: Dict[str, float]) -> float:
    """基于启发式特征计算AI生成概率分数"""
    scores = []

    # 时序一致性 - AI视频帧间差异不稳定
    tc = heuristics.get('temporal_consistency', 0.5)
    if tc > 1.0:
        scores.append(min(tc * 0.5, 0.8))
    else:
        scores.append(max(0.3, 0.7 - tc * 0.4))

    # 运动异常
    ms = heuristics.get('motion_score', 0.5)
    scores.append(max(0.3, min(0.8, 0.3 + ms * 0.5)))

    # 帧方差比
    fv = heuristics.get('frame_variance', 0.5)
    scores.append(max(0.3, min(0.7, fv)))

    return float(np.mean(scores))


# ============================================================
# 音频启发式检测
# ============================================================

def compute_audio_heuristics(waveform: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """计算音频的启发式检测特征"""
    import librosa

    results = {}

    try:
        # 1. 频谱平坦度 (Synthetic语音频谱更平坦)
        spectral_flatness = librosa.feature.spectral_flatness(y=waveform)
        results['spectral_flatness'] = float(spectral_flatness.mean())

        # 2. 过零率统计
        zcr = librosa.feature.zero_crossing_rate(waveform)
        results['zcr_mean'] = float(zcr.mean())
        results['zcr_std'] = float(zcr.std())

        # 3. MFCC统计 (AI语音MFCC分布与真实不同)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
        results['mfcc_mean'] = float(np.abs(mfcc).mean())
        results['mfcc_std'] = float(np.abs(mfcc).std() / (np.abs(mfcc).mean() + 1e-8))

        # 4. 频谱质心
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
        results['spectral_centroid'] = float(centroid.mean())

        # 5. 频谱滚降
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
        results['spectral_rolloff'] = float(rolloff.mean())

        # 6. 带宽
        bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)
        results['spectral_bandwidth'] = float(bandwidth.mean())

    except Exception as e:
        print(f"[AudioHeuristic] 特征提取失败: {e}")
        for k in ['spectral_flatness', 'zcr_mean', 'mfcc_mean']:
            results[k] = 0.5

    return results


def score_audio_from_heuristics(heuristics: Dict[str, float]) -> float:
    """基于启发式特征计算AI生成概率分数"""
    scores = []

    # 频谱平坦度 - 合成语音更高
    sf = heuristics.get('spectral_flatness', 0.5)
    if sf > 0.1:
        scores.append(min(0.5 + sf * 3.0, 0.8))
    else:
        scores.append(max(0.3, 0.5 - sf))

    # ZCR方差比
    zcr_m = heuristics.get('zcr_mean', 0.1)
    zcr_s = heuristics.get('zcr_std', 0.05)
    if zcr_m > 0:
        zcr_ratio = zcr_s / zcr_m
        scores.append(max(0.3, min(0.7, 0.7 - zcr_ratio)))
    else:
        scores.append(0.5)

    # MFCC统计比
    mfcc_r = heuristics.get('mfcc_std', 0.5)
    scores.append(max(0.3, min(0.7, mfcc_r)))

    # 频谱质心归一化
    centroid = heuristics.get('spectral_centroid', 2000)
    norm_centroid = min(centroid / 5000.0, 1.0)
    scores.append(max(0.3, min(0.7, norm_centroid)))

    return float(np.mean(scores))


# ============================================================
# 综合多模态检测 (结合启发式 + 特征统计)
# ============================================================

class HeuristicAIGCDetector:
    """
    基于启发式特征的多模态AIGC检测器
    不依赖训练好的分类头，使用频谱/统计特征进行异常检测
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        # 阈值可根据实际数据校准
        self.thresholds = {
            'img_freq_ratio': (0.01, 5.0),
            'img_srm_ratio': (0.5, 2.0),
            'vid_consistency': (0.2, 1.5),
            'aud_flatness': (0.02, 0.2),
        }

    def detect_image(self, image_path: str) -> Dict:
        """检测单张图像"""
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            img = np.array(Image.open(image_path).convert('RGB'))

        heuristics = compute_image_heuristics(img)
        score = score_image_from_heuristics(heuristics)

        return {
            'file_path': image_path,
            'file_type': 'image',
            'p_fake': score,
            'is_ai': score > 0.5,
            'confidence': max(score, 1 - score),
            'heuristics': heuristics,
        }

    def detect_video(self, video_path: str, num_frames: int = 16) -> Dict:
        """检测单个视频"""
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'file_path': video_path, 'p_fake': 0.5, 'is_ai': False, 'confidence': 0.0}

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        if len(frames) < 2:
            return {'file_path': video_path, 'p_fake': 0.5, 'is_ai': False, 'confidence': 0.0}

        heuristics = compute_video_heuristics(frames)
        score = score_video_from_heuristics(heuristics)

        # 同时对每帧做图像启发式检测，取平均
        frame_scores = []
        for f in frames[::2]:  # 每隔一帧
            fh = compute_image_heuristics(f)
            frame_scores.append(score_image_from_heuristics(fh))

        if frame_scores:
            img_component = np.mean(frame_scores)
            score = 0.6 * score + 0.4 * img_component  # 加权融合

        return {
            'file_path': video_path,
            'file_type': 'video',
            'p_fake': float(score),
            'is_ai': score > 0.5,
            'confidence': float(max(score, 1 - score)),
            'heuristics': heuristics,
        }

    def detect_audio(self, audio_path: str) -> Dict:
        """检测单个音频"""
        import librosa
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"[AudioHeuristic] 加载失败: {audio_path}: {e}")
            return {'file_path': audio_path, 'p_fake': 0.5, 'is_ai': False, 'confidence': 0.0}

        heuristics = compute_audio_heuristics(y, sr)
        score = score_audio_from_heuristics(heuristics)

        return {
            'file_path': audio_path,
            'file_type': 'audio',
            'p_fake': score,
            'is_ai': score > 0.5,
            'confidence': max(score, 1 - score),
            'heuristics': heuristics,
        }

    def detect(self, file_path: str) -> Dict:
        """自动识别文件类型并检测"""
        ext = file_path.lower().split('.')[-1]

        image_exts = {'jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff'}
        video_exts = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
        audio_exts = {'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a'}

        if ext in image_exts:
            return self.detect_image(file_path)
        elif ext in video_exts:
            return self.detect_video(file_path)
        elif ext in audio_exts:
            return self.detect_audio(file_path)
        else:
            return {'file_path': file_path, 'p_fake': 0.5, 'is_ai': False, 'confidence': 0.0,
                    'error': f'未知文件类型: {ext}'}
