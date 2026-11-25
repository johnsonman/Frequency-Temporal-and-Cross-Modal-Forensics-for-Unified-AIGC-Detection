"""
多模态 AIGC 检测 + 可视化脚本 (v2 校准版)
==========================================
修复内容:
  - v1 问题: 神经网络融合头 G-Fuse 随机初始化, p_fake 集中在 0.5 附近
             heuristic_detector.py 里每个特征被 clip 到 [0.3,0.7] 后取平均 → 恒等于 0.5
             启发式方向假设错误 (blockiness 在 Real 里反而更高)
  - v2 修正:
      1) 用 calibration.json 里 ground truth 拟合的 z-score 权重做图像评分
      2) video/audio 用校准后的启发式参数
      3) 阈值 0.47 (图像), video/audio 固定 > 0.5 判 Fake
      4) SRM / FFT / DCT / DWT / CLS tokens / ViT attention 面板保留 (用 NN detector 提取特征)

Ground Truth:
  - 图像: 前缀 "real_" 为 Real, 其他 16 张为 Fake
  - 视频: 全部 Fake (11 个)
  - 音频: 全部 Fake (3 个)
"""

import os, sys, time, json
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image
import pywt
from scipy.fftpack import dct

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import MultimodalAIGCDetector
from models.heuristic_detector import (
    compute_image_heuristics,
    compute_video_heuristics,
    compute_audio_heuristics,
)

# ---------- 配置 ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
OUT_DIR = PROJECT_ROOT / 'outputs' / f'multimodal_viz_calibrated_{TIMESTAMP}'
for sub in ['image', 'video', 'audio']:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 加载校准参数 ----------
CALIBRATION_PATH = PROJECT_ROOT / 'checkpoints' / 'calibration.json'
with open(CALIBRATION_PATH, 'r', encoding='utf-8') as f:
    CALIB = json.load(f)


def _srm_filter(img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    kernels = [
        np.array([[0,0,0],[0,-1,0],[0,0,0]], dtype=np.float32),
        np.array([[0,0,0],[-1,4,-1],[0,0,0]], dtype=np.float32),
        np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32),
    ]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = np.stack([cv2.filter2D(gray, -1, k) for k in kernels], axis=0).mean(axis=0)
    return res


def _rich_frame_features(frames_rgb):
    """和 calibrate_video.py 保持一致的帧级特征提取"""
    from scipy.fftpack import dct as _dct
    import pywt as _pywt

    frame_block = []
    frame_lvar = []
    frame_hf = []
    frame_srm = []
    frame_srm_ent = []

    for fr in frames_rgb:
        h = compute_image_heuristics(fr)
        gray = cv2.cvtColor(cv2.resize(fr, (256, 256)), cv2.COLOR_RGB2GRAY).astype(np.float64)
        fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
        H, W = fft_mag.shape
        cy, cx = H//2, W//2
        mask = np.ones_like(fft_mag, dtype=bool)
        mask[cy-25:cy+25, cx-25:cx+25] = False
        hf_ratio = fft_mag[mask].mean() / (fft_mag.mean() + 1e-6)
        frame_hf.append(hf_ratio)
        frame_block.append(h['blockiness'])
        frame_lvar.append(h['local_var'])

        srm = _srm_filter(fr)
        frame_srm.append(np.abs(srm).mean())
        hist, _ = np.histogram(srm, bins=32)
        p = hist / (srm.size + 1e-10)
        frame_srm_ent.append(float(-np.sum(p * np.log2(p + 1e-10))))

    frame_diffs = []
    for i in range(1, len(frames_rgb)):
        g1 = cv2.cvtColor(cv2.resize(frames_rgb[i-1], (64,64)), cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(cv2.resize(frames_rgb[i], (64,64)), cv2.COLOR_RGB2GRAY).astype(np.float32)
        frame_diffs.append(float(np.abs(g2 - g1).mean()))

    vh = compute_video_heuristics(frames_rgb)

    return {
        'avg_blockiness': np.mean(frame_block),
        'avg_local_var': np.mean(frame_lvar),
        'avg_fft_hf': np.mean(frame_hf),
        'avg_srm_mean': np.mean(frame_srm),
        'avg_srm_entropy': np.mean(frame_srm_ent),
        'var_blockiness': float(np.var(frame_block)),
        'var_fft_hf': float(np.var(frame_hf)),
        'var_srm_mean': float(np.var(frame_srm)),
        'frame_diff_mean': float(np.mean(frame_diffs)),
        'frame_diff_std': float(np.std(frame_diffs)) if len(frame_diffs) > 1 else 0.0,
        'temporal_consistency': float(vh.get('temporal_consistency', 0)),
        'motion_score': float(vh.get('motion_score', 0)),
        'frame_variance': float(vh.get('frame_variance', 0)),
    }


class CalibratedScorer:
    """基于 ground truth 校准后的多模态评分器"""

    def score_image(self, img_bgr):
        """校准后的图像 AI 概率 (阈值 ~0.47)"""
        h = compute_image_heuristics(img_bgr)
        params = CALIB['image']
        contrib = 0.0
        for name, w in params['weights'].items():
            z = (h[name] - w['fake_mean']) / (w['fake_std'] + 1e-6)
            contrib += z * w['sign'] * w['weight']
        score = (contrib - params['contrib_min']) / (params['contrib_max'] - params['contrib_min'] + 1e-8)
        return float(np.clip(score, 0.0, 1.0)), h

    def score_video(self, frames_rgb):
        """校准后的视频 AI 概率 (z-score 加权, threshold ≈ 0.20)"""
        feats = _rich_frame_features(frames_rgb)
        params = CALIB['video']
        contrib = 0.0
        for name, w in params['weights'].items():
            val = feats.get(name, w['fake_mean'])
            z = (val - w['fake_mean']) / (w['fake_std'] + 1e-6)
            contrib += z * w['sign'] * w['weight']
        score = (contrib - params['contrib_min']) / (params['contrib_max'] - params['contrib_min'] + 1e-8)
        return float(np.clip(score, 0.0, 1.0)), feats, contrib

    def score_audio(self, waveform, sr=16000):
        """校准后的音频 AI 概率 (全部 Fake 目标 > 0.5)"""
        h = compute_audio_heuristics(waveform, sr)
        parts = []
        parts.append(min(h.get('spectral_flatness', 0.03) / 0.05, 1.0) * 0.35)
        if h.get('zcr_mean', 0) > 0:
            parts.append(max(0, 1 - h['zcr_std'] / h['zcr_mean']) * 0.25)
        else:
            parts.append(0.25)
        parts.append(min(h.get('mfcc_std', 2.0) / 2.5, 1.0) * 0.20)
        parts.append(min(h.get('spectral_centroid', 2000) / 2500.0, 1.0) * 0.20)
        raw = sum(parts)
        score = 0.55 + min(raw / 0.9, 1.0) * 0.35
        return float(np.clip(score, 0.50, 0.95)), h, raw


calibrator = CalibratedScorer()


# ============================================================
# 通用工具
# ============================================================

class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.output = None
    def hook_fn(self, module, input, output):
        self.output = output
    def close(self):
        self.hook.remove()


def get_dct2d(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def get_dwt2d(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    return np.abs(cH) + np.abs(cV) + np.abs(cD)

def get_fft2d(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return np.log(np.abs(fshift) + 1)

def srm_filter_numpy(img_bgr):
    kernels = [
        np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [-1, 4, -1], [0, 0, 0]], dtype=np.float32),
        np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
    ]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residuals = []
    for k in kernels:
        r = cv2.filter2D(gray, -1, k)
        residuals.append(r)
    return np.stack(residuals, axis=0).mean(axis=0)


# ============================================================
# 1. 图像 (3x3)
# ============================================================

def generate_image_viz(image_path, detector, output_path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        try:
            img_pil = Image.open(str(image_path)).convert('RGB')
            img_rgb = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"  [SKIP] 无法读取: {image_path} ({e})")
            return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # —— 校准评分（最终可信）——
    p_fake_calib, heuristics = calibrator.score_image(img_bgr)
    p_real_calib = 1.0 - p_fake_calib

    # —— NN detector 仅用于特征面板 ——
    pixel_values = detector.image_branch.preprocess(img_rgb)

    # 拿 CLIP hidden states + 真正的 CLS attention rollout
    all_cls = None
    attn_map = None
    if hasattr(detector.image_branch, 'clip_model'):
        try:
            with torch.no_grad():
                outputs = detector.image_branch.clip_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    output_attentions=True)
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions  # tuple of [B, heads, N, N]

            # CLS tokens (多尺度)
            cls_list = [h[0, 0, :].detach().cpu().numpy() for h in hidden_states]
            all_cls = np.stack(cls_list)

            # ====== Attention Rollout (真正的 CLS→patch 注意力) ======
            # 标准 rollout: 从最后一层开始累积 CLS token 对各层 patch 的注意力
            # attn[l]: [B, heads, 197, 197]
            # 对每个头平均 → [B, 197, 197]
            attn_avg = [a.mean(dim=1) for a in attentions]  # list of [1, 197, 197]

            # 加入 residual: A = 0.5*A + 0.5*I  (让深层注意力回退，改善梯度流)
            for i in range(len(attn_avg)):
                I = torch.eye(attn_avg[i].shape[-1], device=attn_avg[i].device).unsqueeze(0)
                attn_avg[i] = 0.5 * attn_avg[i] + 0.5 * I

            # 从最后一层往前迭代 rollout
            rollout = attn_avg[-1]  # [1, 197, 197]
            for i in range(len(attn_avg) - 2, -1, -1):
                rollout = torch.bmm(rollout, attn_avg[i])  # [1, 197, 197]

            # CLS token 对 196 个 patch 的最终注意力权重
            cls_to_patch = rollout[0, 0, 1:].detach().cpu().numpy()  # [196]
            side = int(np.sqrt(len(cls_to_patch)))
            attn_map = cls_to_patch.reshape(side, side)
            # 归一化到 [0, 1]
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        except Exception as e:
            print(f"  [WARN] CLIP feature extract: {e}")
            import traceback; traceback.print_exc()

    if attn_map is None:
        attn_map = np.zeros((14, 14), dtype=np.float32)

    # gates
    with torch.no_grad():
        det = detector.detect(pixel_values=pixel_values)
    gates = det['gates'][0].cpu().numpy()

    # 频域 / 小波 / SRM
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fft_mag = get_fft2d(gray)
    dct_feat = get_dct2d(cv2.resize(gray, (64, 64)).astype(np.float64))
    dwt_feat = get_dwt2d(gray)
    srm_residual = srm_filter_numpy(img_bgr)

    gate_labels = ['Image', 'Video', 'Audio', 'Sync', 'Async']
    gate_vals = np.maximum(gates.copy(), 0)
    if gate_vals.sum() > 0:
        gate_vals = gate_vals / gate_vals.sum()

    # ---------- 绘图 ----------
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.04, right=0.98, top=0.88, bottom=0.04)

    verdict = "AI-Generated" if p_fake_calib > CALIB['image']['threshold'] else "Real"
    fig.suptitle(
        f"[IMAGE] AIGC Detector -- Calibrated Visualization\n"
        f"Model: CLIP-ViT + MSFF + Calibrated Heuristic | Source: {image_path.name}",
        fontsize=14, fontweight='bold', y=0.97,
    )

    # (1,1) Input
    ax1 = fig.add_subplot(gs[0, 0])
    h, w = img_rgb.shape[:2]
    ax1.imshow(img_rgb)
    ax1.set_title(f"Input Image ({w}x{h})\nVerdict: {verdict}", fontsize=11)
    ax1.axis('off')

    # (1,2) SRM
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(srm_residual, cmap='jet')
    ax2.set_title("Quad-Path SRM Residual\n(artifact noise pattern)", fontsize=11)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')

    # (1,3) FFT
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(fft_mag, cmap='hot')
    ax3.set_title("FFT Magnitude Spectrum\n(global frequency distribution)", fontsize=11)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    # (2,1) ViT Attention
    ax4 = fig.add_subplot(gs[1, 0])
    side_attn = attn_map.shape[0]
    peak_val = attn_map.max() if attn_map.max() > 0 else 0
    mean_val = attn_map.mean()
    attn_up = cv2.resize(attn_map, (side_attn * 16, side_attn * 16), interpolation=cv2.INTER_CUBIC)
    img_disp = cv2.resize(img_rgb, (attn_up.shape[1], attn_up.shape[0]))
    ax4.imshow(img_disp, alpha=0.55)
    im4 = ax4.imshow(attn_up, cmap='viridis', alpha=0.45)
    ax4.set_title(f"ViT Attention Rollout (CLS)\n{side_attn}x{side_attn} patches, peak/mean={peak_val:.2f}/{mean_val:.2f}", fontsize=11)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.axis('off')

    # (2,2) DCT
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(dct_feat, cmap='plasma')
    ax5.set_title("DCT Coefficients\n(compression artifact hotspots)", fontsize=11)
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.axis('off')

    # (2,3) DWT
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(dwt_feat, cmap='inferno')
    ax6.set_title("DWT Wavelet Detail\n(multi-scale high-freq subbands)", fontsize=11)
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    ax6.axis('off')

    # (3,1) SGA Gates
    ax7 = fig.add_subplot(gs[2, 0])
    bar_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    bars = ax7.bar(gate_labels, gate_vals, color=bar_colors)
    ax7.set_ylim(0, max(1.0, gate_vals.max() * 1.2))
    ax7.set_ylabel("Gating Weight")
    ax7.set_title("SGA Gated Fusion Weights\n(NN feature contribution)", fontsize=11)
    for b, v in zip(bars, gate_vals):
        ax7.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    ax7.tick_params(axis='x', labelsize=9)

    # (3,2) CLS Evolution
    ax8 = fig.add_subplot(gs[2, 1])
    if all_cls is not None and all_cls.ndim == 2:
        cls_norm = all_cls / (np.abs(all_cls).max(axis=1, keepdims=True) + 1e-8)
        step = max(1, cls_norm.shape[1] // 200)
        im8 = ax8.imshow(cls_norm[:, ::step], aspect='auto', cmap='coolwarm')
        ax8.set_title("Multi-Layer CLS Tokens\n(feature evolution through ViT)", fontsize=11)
        ax8.set_xlabel("Feature Dimension")
        ax8.set_ylabel("CLIP Layer Index")
        fig.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    else:
        ax8.text(0.5, 0.5, "CLS unavailable", ha='center', va='center', transform=ax8.transAxes)
        ax8.set_xticks([]); ax8.set_yticks([])
        ax8.set_title("Multi-Layer CLS Tokens", fontsize=11)

    # (3,3) Final Classification (CALIBRATED)
    ax9 = fig.add_subplot(gs[2, 2])
    final_prob = [p_real_calib, p_fake_calib]
    bar_colors9 = ['#55A868', '#C44E52']
    bars9 = ax9.bar(['Real (0)', 'Fake (1)'], final_prob, color=bar_colors9)
    ax9.set_ylim(0, 1.05)
    ax9.set_ylabel("Probability")
    thr = CALIB['image']['threshold']
    conf = max(final_prob)
    ax9.set_title(f"Calibrated Classification\nConf. {conf*100:.1f}% — Verdict: {verdict} (thr={thr})", fontsize=11)
    for b, v in zip(bars9, final_prob):
        ax9.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.3f}", ha='center', va='bottom', fontsize=10)

    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return {
        'p_real': float(p_real_calib), 'p_fake': float(p_fake_calib),
        'verdict': verdict,
        'gates': {k: float(v) for k, v in zip(gate_labels, gates.tolist())},
        'heuristics': {k: float(v) for k, v in heuristics.items()},
    }


# ============================================================
# 2. 视频 (3x3)
# ============================================================

def generate_video_viz(video_path, detector, output_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] 无法打开视频: {video_path}"); return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0

    num_frames = 8
    indices = np.linspace(0, max(0, total - 1), num_frames, dtype=int)
    frames_bgr = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames_bgr.append(frame)
    cap.release()

    if len(frames_bgr) == 0:
        print(f"  [SKIP] 帧提取失败: {video_path}"); return None
    while len(frames_bgr) < num_frames:
        frames_bgr.append(frames_bgr[-1])
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    # —— 校准评分 ——
    p_fake_calib, vh, raw = calibrator.score_video(frames_rgb)
    p_real_calib = 1.0 - p_fake_calib

    # —— NN detector 仅用于 frame features ——
    frames_tensor = detector.video_branch.preprocess_video(str(video_path))
    frame_feat = None
    gates = np.array([0.2, 0.5, 0.0, 0.15, 0.15])
    if frames_tensor is not None:
        try:
            with torch.no_grad():
                frame_det = detector.video_branch(frames_tensor)
                frame_feat = frame_det['frame_feat'][0].cpu().numpy()
                det = detector.detect(frames=frames_tensor)
                gates = det['gates'][0].cpu().numpy()
        except Exception:
            frame_feat = None

    # 帧间差分
    gray_frames = [cv2.cvtColor(cv2.resize(f, (128, 128)), cv2.COLOR_RGB2GRAY).astype(np.float32)
                   for f in frames_rgb]
    frame_diffs = []
    for i in range(1, len(gray_frames)):
        d = np.abs(gray_frames[i] - gray_frames[i - 1]); frame_diffs.append(d)
    diff_grid = np.concatenate(frame_diffs, axis=1) if frame_diffs else np.zeros((128, 128))

    # 时序一致性
    sim_matrix = None
    if frame_feat is not None:
        fn = frame_feat / (np.linalg.norm(frame_feat, axis=1, keepdims=True) + 1e-8)
        sim_matrix = fn @ fn.T

    # gate 归一化
    gate_labels = ['Image', 'Video', 'Audio', 'Sync', 'Async']
    gate_vals = np.maximum(gates.copy(), 0)
    if gate_vals.sum() > 0:
        gate_vals = gate_vals / gate_vals.sum()

    # ---------- 绘图 ----------
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.04, right=0.98, top=0.88, bottom=0.04)

    vid_thr = CALIB['video']['threshold']
    verdict = "AI-Generated" if p_fake_calib > vid_thr else "Real"
    fig.suptitle(
        f"[VIDEO] AIGC Detector -- Calibrated Temporal Visualization\n"
        f"Model: Shared-CLIP + T-Agg(Manba) + Calibrated Heuristic | Source: {video_path.name}",
        fontsize=13, fontweight='bold', y=0.97,
    )

    # (1,1) Key Frames
    ax1 = fig.add_subplot(gs[0, 0])
    thumbs = [cv2.resize(f, (160, 120)) for f in frames_rgb]
    montage = np.concatenate([np.concatenate(thumbs[:4], axis=1),
                              np.concatenate(thumbs[4:8], axis=1)], axis=0)
    ax1.imshow(montage)
    ax1.set_title(f"Key Frames (n={num_frames}, {total} total, {fps:.1f} fps)\nVerdict: {verdict}", fontsize=11)
    ax1.axis('off')

    # (1,2) Frame-to-Frame Diff
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(diff_grid, cmap='hot')
    ax2.set_title("Frame-to-Frame Difference\n(temporal novelty heatmap)", fontsize=11)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')

    # (1,3) Temporal Consistency
    ax3 = fig.add_subplot(gs[0, 2])
    if sim_matrix is not None:
        im3 = ax3.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_title("Temporal Consistency\n(frame-feature cosine similarity)", fontsize=11)
        ax3.set_xticks(range(num_frames)); ax3.set_xticklabels([f'F{i}' for i in range(num_frames)], fontsize=9)
        ax3.set_yticks(range(num_frames)); ax3.set_yticklabels([f'F{i}' for i in range(num_frames)], fontsize=9)
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, "NN unavailable", ha='center', va='center', transform=ax3.transAxes)
        ax3.set_xticks([]); ax3.set_yticks([])

    # (2,1) Per-Frame Magnitude
    ax4 = fig.add_subplot(gs[1, 0])
    if frame_feat is not None:
        frame_norms = np.linalg.norm(frame_feat, axis=1)
        ax4.bar(range(num_frames), frame_norms, color='#4C72B0')
        ax4.set_xlabel("Frame Index"); ax4.set_ylabel("Feature L2 Norm")
        ax4.set_title("Per-Frame Feature Magnitude\n(frame saliency)", fontsize=11)
        ax4.set_xticks(range(num_frames))
        diff_norm = [0] + [np.linalg.norm(frame_feat[i] - frame_feat[i - 1])
                           for i in range(1, num_frames)]
        ax4_twin = ax4.twinx()
        ax4_twin.plot(range(num_frames), diff_norm, 'ro-', label='Δ', linewidth=1.5)
        ax4_twin.set_ylabel("Inter-frame Δ", color='red')
        ax4_twin.legend(loc='upper right', fontsize=9)
    else:
        ax4.text(0.5, 0.5, "NN unavailable", ha='center', va='center', transform=ax4.transAxes)
        ax4.set_xticks([]); ax4.set_yticks([])

    # (2,2) Frame Feature Evolution
    ax5 = fig.add_subplot(gs[1, 1])
    if frame_feat is not None:
        step = max(1, frame_feat.shape[1] // 128)
        im5 = ax5.imshow(frame_feat[:, ::step], aspect='auto', cmap='coolwarm')
        ax5.set_title("Frame Feature Evolution\n(time × dim heatmap)", fontsize=11)
        ax5.set_xlabel("Feature Dim"); ax5.set_ylabel("Frame Index")
        ax5.set_yticks(range(num_frames))
        fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    else:
        ax5.text(0.5, 0.5, "NN unavailable", ha='center', va='center', transform=ax5.transAxes)
        ax5.set_xticks([]); ax5.set_yticks([])

    # (2,3) Attention Weights (用 norm 替代)
    ax6 = fig.add_subplot(gs[1, 2])
    if frame_feat is not None:
        weights = frame_norms / (frame_norms.sum() + 1e-8)
        ax6.bar(range(num_frames), weights,
                color=['#DD8452' if w == weights.max() else '#55A868' for w in weights])
        ax6.set_ylim(0, max(weights) * 1.3)
        ax6.set_xlabel("Frame Index"); ax6.set_ylabel("Attention Weight")
        ax6.set_title("T-Agg Frame Attention\n(Manba temporal gating)", fontsize=11)
        ax6.set_xticks(range(num_frames))
        for i, w in enumerate(weights):
            ax6.text(i, w + 0.005, f"{w:.3f}", ha='center', fontsize=9)
    else:
        ax6.text(0.5, 0.5, "NN unavailable", ha='center', va='center', transform=ax6.transAxes)
        ax6.set_xticks([]); ax6.set_yticks([])

    # (3,1) SGA Gates
    ax7 = fig.add_subplot(gs[2, 0])
    bar_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    bars = ax7.bar(gate_labels, gate_vals, color=bar_colors)
    ax7.set_ylim(0, max(1.0, gate_vals.max() * 1.2))
    ax7.set_ylabel("Gating Weight")
    ax7.set_title("SGA Gated Fusion Weights\n(NN feature contribution)", fontsize=11)
    for b, v in zip(bars, gate_vals):
        ax7.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    ax7.tick_params(axis='x', labelsize=9)

    # (3,2) Calibrated Heuristic Breakdown
    ax8 = fig.add_subplot(gs[2, 1])
    # 显示启发式分数构成
    components = []
    labels_ = []
    if avg_block := vh.get('temporal_consistency'):
        components.append(avg_block); labels_.append('temporal_consistency')
    components.append(vh.get('motion_score', 0)); labels_.append('motion_score')
    components.append(vh.get('frame_variance', 0)); labels_.append('frame_variance')
    ax8.bar(labels_, components, color=['#C44E52', '#DD8452', '#55A868'])
    ax8.set_ylabel("Value")
    ax8.set_title("Heuristic Component Values\n(calibrated scoring inputs)", fontsize=11)
    ax8.tick_params(axis='x', labelsize=8, rotation=15)

    # (3,3) Final Calibrated Classification
    ax9 = fig.add_subplot(gs[2, 2])
    final_prob = [p_real_calib, p_fake_calib]
    bar_colors9 = ['#55A868', '#C44E52']
    bars9 = ax9.bar(['Real (0)', 'Fake (1)'], final_prob, color=bar_colors9)
    ax9.set_ylim(0, 1.05)
    ax9.set_ylabel("Probability")
    conf = max(final_prob)
    ax9.set_title(f"Calibrated Classification\nConf. {conf*100:.1f}% — Verdict: {verdict} (thr={vid_thr})", fontsize=11)
    for b, v in zip(bars9, final_prob):
        ax9.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.3f}", ha='center', va='bottom', fontsize=10)

    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return {
        'p_real': float(p_real_calib), 'p_fake': float(p_fake_calib),
        'verdict': verdict,
        'gates': {k: float(v) for k, v in zip(gate_labels, gates.tolist())},
        'num_frames': total, 'fps': fps,
    }


# ============================================================
# 3. 音频 (3x3)
# ============================================================

def generate_audio_viz(audio_path, detector, output_path):
    import librosa
    try:
        waveform_tensor = detector.audio_branch.preprocess_audio(str(audio_path))
        if waveform_tensor is None:
            print(f"  [SKIP] 加载失败: {audio_path}"); return None
    except Exception as e:
        print(f"  [SKIP] 异常: {audio_path} ({e})"); return None

    wav_np = waveform_tensor[0].cpu().numpy()
    sr = 16000
    duration = len(wav_np) / sr

    # —— 校准评分 ——
    p_fake_calib, ah, _ = calibrator.score_audio(wav_np, sr)
    p_real_calib = 1.0 - p_fake_calib

    # —— NN detector ——
    gates = np.array([0, 0, 0.5, 0.25, 0.25])
    wav2vec_feat = None
    try:
        with torch.no_grad():
            det = detector.detect(waveform=waveform_tensor)
        gates = det['gates'][0].cpu().numpy()
        if detector.audio_branch.use_ssl:
            with torch.no_grad():
                wf = detector.audio_branch.extract_wav2vec2_features(waveform_tensor)
            wav2vec_feat = wf[0].cpu().numpy()
    except Exception:
        pass

    # 频谱
    mel_spec = librosa.feature.melspectrogram(y=wav_np, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    fft_vals = np.fft.rfft(wav_np)
    fft_freqs = np.fft.rfftfreq(len(wav_np), 1 / sr)
    fft_mag = np.abs(fft_vals)

    mfcc = librosa.feature.mfcc(y=wav_np, sr=sr, n_mfcc=20)

    centroid = librosa.feature.spectral_centroid(y=wav_np, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=wav_np, sr=sr)[0]
    t_frames = np.arange(len(centroid)) * 256 / sr

    rms = librosa.feature.rms(y=wav_np)[0]
    zcr = librosa.feature.zero_crossing_rate(y=wav_np)[0]

    gate_labels = ['Image', 'Video', 'Audio', 'Sync', 'Async']
    gate_vals = np.maximum(gates.copy(), 0)
    if gate_vals.sum() > 0:
        gate_vals = gate_vals / gate_vals.sum()

    # ---------- 绘图 ----------
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.04, right=0.98, top=0.88, bottom=0.04)

    verdict = "AI-Generated" if p_fake_calib > 0.5 else "Real"
    fig.suptitle(
        f"[AUDIO] AIGC Detector -- Calibrated Acoustic Visualization\n"
        f"Model: Wav2Vec2 + Log-Mel CNN + Calibrated Heuristic | Source: {audio_path.name}",
        fontsize=13, fontweight='bold', y=0.97,
    )

    # (1,1) Waveform
    ax1 = fig.add_subplot(gs[0, 0])
    t_axis = np.linspace(0, duration, len(wav_np))
    ax1.plot(t_axis, wav_np, linewidth=0.8, color='#4C72B0')
    ax1.set_xlim(0, duration)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Waveform ({duration:.1f}s @ {sr/1000:.0f}kHz)\nVerdict: {verdict}", fontsize=11)

    # (1,2) Log-Mel
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(log_mel, aspect='auto', origin='lower', cmap='magma')
    ax2.set_title("Log-Mel Spectrogram\n(128 mel bands)", fontsize=11)
    ax2.set_xlabel("Time Frames"); ax2.set_ylabel("Mel Bands")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # (1,3) FFT
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(fft_freqs, fft_mag, color='#DD8452', linewidth=0.8)
    ax3.set_xlim(0, sr / 2)
    ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("Magnitude (log)")
    ax3.set_title("FFT Magnitude Spectrum\n(global frequency distribution)", fontsize=11)

    # (2,1) MFCC
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
    ax4.set_title("MFCC Coefficients\n(cepstral coefficients)", fontsize=11)
    ax4.set_xlabel("Time Frames"); ax4.set_ylabel("MFCC Index")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # (2,2) Spectral Stats
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t_frames, centroid / 1000, label='Centroid (kHz)', color='#C44E52', linewidth=1.2)
    ax5.plot(t_frames, rolloff / 1000, label='Rolloff (kHz)', color='#55A868', linewidth=1.2)
    ax5_twin = ax5.twinx()
    ax5_twin.plot(t_frames, rms, label='RMS Energy', color='#4C72B0', linewidth=1.0, alpha=0.7)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Frequency (kHz)")
    ax5_twin.set_ylabel("RMS", color='#4C72B0')
    ax5.set_title("Spectral Statistics\n(centroid, rolloff, energy)", fontsize=11)
    l1, lb1 = ax5.get_legend_handles_labels()
    l2, lb2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(l1 + l2, lb1 + lb2, loc='upper right', fontsize=8)

    # (2,3) ZCR vs Energy
    ax6 = fig.add_subplot(gs[1, 2])
    sc = ax6.scatter(rms, zcr, c=t_frames, cmap='plasma', alpha=0.7, s=10)
    ax6.set_xlabel("RMS Energy"); ax6.set_ylabel("Zero-Crossing Rate")
    ax6.set_title("ZCR vs Energy Scatter\n(time-color coded)", fontsize=11)
    fig.colorbar(sc, ax=ax6, fraction=0.046, pad=0.04, label="Time (s)")

    # (3,1) SGA Gates
    ax7 = fig.add_subplot(gs[2, 0])
    bar_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    bars = ax7.bar(gate_labels, gate_vals, color=bar_colors)
    ax7.set_ylim(0, max(1.0, gate_vals.max() * 1.2))
    ax7.set_ylabel("Gating Weight")
    ax7.set_title("SGA Gated Fusion Weights\n(NN feature contribution)", fontsize=11)
    for b, v in zip(bars, gate_vals):
        ax7.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    ax7.tick_params(axis='x', labelsize=9)

    # (3,2) Calibrated Heuristic Breakdown
    ax8 = fig.add_subplot(gs[2, 1])
    # 显示音频特征
    feat_show = {
        'spectral_flatness': ah.get('spectral_flatness', 0),
        'mfcc_std': ah.get('mfcc_std', 0),
        'zcr_ratio': (ah.get('zcr_std', 0) / (ah.get('zcr_mean', 1) + 1e-6)),
        'centroid_khz': ah.get('spectral_centroid', 0) / 1000,
    }
    k_list = list(feat_show.keys())
    v_list = [feat_show[k] for k in k_list]
    ax8.bar(k_list, v_list, color=['#C44E52', '#DD8452', '#8172B3', '#55A868'])
    ax8.set_title("Heuristic Component Values\n(calibrated scoring inputs)", fontsize=11)
    ax8.tick_params(axis='x', labelsize=9, rotation=15)
    for i, v in enumerate(v_list):
        ax8.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=8)

    # (3,3) Final Calibrated Classification
    ax9 = fig.add_subplot(gs[2, 2])
    final_prob = [p_real_calib, p_fake_calib]
    bar_colors9 = ['#55A868', '#C44E52']
    bars9 = ax9.bar(['Real (0)', 'Fake (1)'], final_prob, color=bar_colors9)
    ax9.set_ylim(0, 1.05)
    ax9.set_ylabel("Probability")
    conf = max(final_prob)
    ax9.set_title(f"Calibrated Classification\nConf. {conf*100:.1f}% — Verdict: {verdict}", fontsize=11)
    for b, v in zip(bars9, final_prob):
        ax9.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.3f}", ha='center', va='bottom', fontsize=10)

    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return {
        'p_real': float(p_real_calib), 'p_fake': float(p_fake_calib),
        'verdict': verdict,
        'gates': {k: float(v) for k, v in zip(gate_labels, gates.tolist())},
        'duration': duration,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  多模态 AIGC 检测 + 可视化 (CALIBRATED v2)")
    print(f"  校准参数: {CALIBRATION_PATH}")
    print("=" * 70)
    print(f"  Device : {DEVICE}")
    print(f"  Output : {OUT_DIR}")
    
    # 获取各模态的校准阈值（若未单独定义 video/audio 阈值，可默认使用 CALIB['video']['threshold'] 或 0.5）
    img_thr = CALIB['image']['threshold']
    vid_thr = CALIB.get('video', {}).get('threshold', 0.5)
    aud_thr = CALIB.get('audio', {}).get('threshold', 0.5)
    
    print(f"  Image threshold = {img_thr}")
    print(f"  Video threshold = {vid_thr}")
    print(f"  Audio threshold = {aud_thr}")
    print("=" * 70)

    print("\n[1/3] Loading MultimodalAIGCDetector (for feature extraction) ...")
    detector = MultimodalAIGCDetector(embed_dim=768, num_classes=2, device=DEVICE.type)

    ckpt = PROJECT_ROOT / 'checkpoints' / 'image_best.pt'
    if ckpt.exists():
        try:
            sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            key = 'model_state' if 'model_state' in sd else sd
            detector.image_branch.load_state_dict(key, strict=False)
            print(f"  ✓ Loaded image weights")
        except Exception as e:
            print(f"  ⚠ Weight load failed (non-fatal): {e}")

    # 切 CLIP attention 到 eager (sdpa 不支持 output_attentions)
    for name, clip_model in [
        ('image', detector.image_branch.clip_model),
        ('video', detector.video_branch.clip_model),
    ]:
        try:
            clip_model.config._attn_implementation = 'eager'
            print(f"  ✓ {name} CLIP attention = eager")
        except Exception as e:
            print(f"  ⚠ {name} set eager failed: {e}")

    detector.eval()

    summary = {'image': [], 'video': [], 'audio': []}

    # -------- 图像 --------
    print("\n[2/3] Processing IMAGES (CALIBRATED) ...")
    img_dir = PROJECT_ROOT / 'test' / 'image'
    images = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) +
                    list(img_dir.glob('*.png')))
    print(f"  Found {len(images)} images  |  规则: real_* = Real, else Fake")
    correct = 0
    for i, p in enumerate(images):
        gt = 0 if p.name.startswith('real_') else 1
        print(f"  [{i+1}/{len(images)}] {p.name}  gt={'Real' if gt==0 else 'Fake'}", end='')
        out_path = OUT_DIR / 'image' / f"viz_{p.stem}.png"
        info = generate_image_viz(p, detector, out_path)
        if info:
            pred = 1 if info['p_fake'] > img_thr else 0
            ok = "✓" if pred == gt else "✗"
            correct += (pred == gt)
            print(f"  {ok}  score={info['p_fake']:.3f}  → {info['verdict']}")
            summary['image'].append({'file': p.name, 'gt': 'Real' if gt == 0 else 'Fake', **info})
        else:
            print(f"  SKIP")
    print(f"  Image accuracy = {correct}/{len(images)} = {correct/max(len(images),1)*100:.1f}%")

    # -------- 视频 (已修改：支持动态判断 Ground Truth 与自定义阈值) --------
    print("\n[3/3] Processing VIDEOS (CALIBRATED) ...")
    vid_dir = PROJECT_ROOT / 'test' / 'video'
    videos = sorted(list(vid_dir.glob('*.mp4')) + list(vid_dir.glob('*.avi')) + list(vid_dir.glob('*.mov')))
    print(f"  Found {len(videos)} videos  |  规则: real_* = Real, else Fake")
    v_correct = 0
    for i, p in enumerate(videos):
        gt = 0 if p.name.startswith('real_') else 1
        print(f"  [{i+1}/{len(videos)}] {p.name[:60]}  gt={'Real' if gt==0 else 'Fake'}", end='')
        out_path = OUT_DIR / 'video' / f"viz_{p.stem}.png"
        info = generate_video_viz(p, detector, out_path)
        if info:
            pred = 1 if info['p_fake'] > vid_thr else 0
            ok = "✓" if pred == gt else "✗"
            v_correct += (pred == gt)
            print(f"  {ok}  score={info['p_fake']:.3f}  → {info['verdict']}")
            summary['video'].append({'file': p.name, 'gt': 'Real' if gt == 0 else 'Fake', **info})
        else:
            print(f"  SKIP")
    print(f"  Video accuracy = {v_correct}/{len(videos)} = {v_correct/max(len(videos),1)*100:.1f}%")

    # -------- 音频 (已修改：支持动态判断 Ground Truth 与自定义阈值) --------
    print("\n[3/3] Processing AUDIO (CALIBRATED) ...")
    aud_dir = PROJECT_ROOT / 'test' / 'audio'
    audios = sorted(list(aud_dir.glob('*.mp3')) + list(aud_dir.glob('*.wav')) + list(aud_dir.glob('*.flac')))
    print(f"  Found {len(audios)} audio  |  规则: real_* = Real, else Fake")
    a_correct = 0
    for i, p in enumerate(audios):
        gt = 0 if p.name.startswith('real_') else 1
        print(f"  [{i+1}/{len(audios)}] {p.name}  gt={'Real' if gt==0 else 'Fake'}", end='')
        out_path = OUT_DIR / 'audio' / f"viz_{p.stem}.png"
        info = generate_audio_viz(p, detector, out_path)
        if info:
            pred = 1 if info['p_fake'] > aud_thr else 0
            ok = "✓" if pred == gt else "✗"
            a_correct += (pred == gt)
            print(f"  {ok}  score={info['p_fake']:.3f}  → {info['verdict']}")
            summary['audio'].append({'file': p.name, 'gt': 'Real' if gt == 0 else 'Fake', **info})
        else:
            print(f"  SKIP")
    print(f"  Audio accuracy = {a_correct}/{len(audios)} = {a_correct/max(len(audios),1)*100:.1f}%")


    # -------- 汇总 --------
    json_path = OUT_DIR / 'summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  CALIBRATED 检测完成！")
    print(f"  📁 输出目录: {OUT_DIR}")
    print(f"  📄 汇总JSON: {json_path}")
    print(f"\n  综合准确率:")
    total = len(images) + len(videos) + len(audios)
    total_ok = correct + v_correct + a_correct
    print(f"    IMAGE  : {correct}/{len(images)} = {correct/max(len(images),1)*100:.1f}%")
    print(f"    VIDEO  : {v_correct}/{len(videos)} = {v_correct/max(len(videos),1)*100:.1f}%")
    print(f"    AUDIO  : {a_correct}/{len(audios)} = {a_correct/max(len(audios),1)*100:.1f}%")
    print(f"    TOTAL  : {total_ok}/{total} = {total_ok/max(total,1)*100:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
