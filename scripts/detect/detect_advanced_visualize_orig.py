import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image
import pywt
from scipy.fftpack import dct

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import MultimodalAIGCDetector

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = PROJECT_ROOT / 'outputs' / 'advanced_viz_orig_size_1025_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def get_dct2d(img):
    """计算 2D DCT"""
    # 对于大图，为了可视化效果和计算效率，限制在 256x256 进行 DCT
    h, w = img.shape
    if h > 256 or w > 256:
        img_small = cv2.resize(img, (256, 256))
        return dct(dct(img_small.T, norm='ortho').T, norm='ortho')
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def get_dwt2d(img):
    """计算 2D DWT (Haar 小波)"""
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    # 结合高频分量
    return np.abs(cH) + np.abs(cV) + np.abs(cD)

def get_fft2d(img):
    """计算 2D FFT 振幅谱"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def compute_attention_map(detector, pixel_values):
    """
    Compute the ViT attention map via Attention Rollout (Abnar & Zuidema).

    Uses the real attention probability matrices produced by the CLIP vision
    encoder (output_attentions=True) instead of the self-attention *output*
    activations. Residual connections are accounted for by adding the identity
    matrix before row-normalizing each layer, then multiplying across layers.

    Returns:
        numpy.ndarray of shape [side, side] in [0, 1], or None if unavailable.
    """
    clip_model = detector.image_branch.clip_model

    # `sdpa` attention cannot return attention weights, so force the vision
    # encoder to the `eager` implementation before requesting them.
    if getattr(clip_model.config, '_attn_implementation', None) != 'eager':
        clip_model.config._attn_implementation = 'eager'

    with torch.no_grad():
        outputs = clip_model(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True,
        )

    attentions = getattr(outputs, 'attentions', None)
    if not attentions:
        return None

    num_tokens = attentions[0].shape[-1]
    device = attentions[0].device
    eye = torch.eye(num_tokens, device=device)

    rollout = eye
    for attn in attentions:
        # attn: [B, heads, N, N] -> average over heads
        a = attn[0].mean(dim=0)
        # account for residual connection
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout

    # CLS token attention towards all patch tokens
    cls_attn = rollout[0, 1:]
    side = int(round(cls_attn.numel() ** 0.5))
    if side * side != cls_attn.numel():
        return None

    grid = cls_attn.reshape(side, side).float().cpu().numpy()
    grid = grid - grid.min()
    if grid.max() > 1e-8:
        grid = grid / grid.max()
    return grid


def generate_viz(image_path, detector, output_path):
    """生成 3x3 高级可视化报告 (采用原图尺寸展示)"""
    # 1. 加载
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]
    
    # 2. 推理 (模型内部仍会 resize 为 224x224)
    pixel_values = detector.image_branch.preprocess(img_rgb)

    with torch.no_grad():
        det = detector.detect(pixel_values=pixel_values)
        
    # 提取结果
    p_fake = det['p_fake'][0].cpu().numpy()
    gates = det['gates'][0].cpu().numpy()
    
    # 提取 CLS Tokens
    with torch.no_grad():
        ms_features = detector.image_branch.extract_clip_features(pixel_values)
        cls_tokens = torch.stack(ms_features).squeeze(1).cpu().numpy() 
        if cls_tokens.shape[0] < 6:
            cls_tokens = np.tile(cls_tokens, (2, 1))[:6]

    # 3. 计算频域特征 (基于原图灰度图)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    fft_mag = get_fft2d(gray)
    dct_feat = get_dct2d(gray)
    dwt_feat = get_dwt2d(gray)
    
    # Quad-Path 频率特征 (使用 SRM 滤波)
    with torch.no_grad():
        srm_noise = detector.image_branch.forensics.srm(pixel_values)
        quad_feat = srm_noise[0].mean(dim=0).cpu().numpy()
        # Resize quad_feat to match original aspect ratio for display
        quad_feat_res = cv2.resize(quad_feat, (w_orig, h_orig))
    
    # Attention Map via Attention Rollout on the real CLIP attention weights
    attn_grid = compute_attention_map(detector, pixel_values)
    if attn_grid is None:
        attn_grid = np.zeros((14, 14), dtype=np.float32)

    attn_resized = cv2.resize(attn_grid, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    # Percentile stretch so the focus regions keep visible contrast
    lo, hi = np.percentile(attn_resized, [2, 99])
    attn_resized = np.clip((attn_resized - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    print(f"    Attention focus: grid={attn_grid.shape}, "
          f"max/mean={attn_grid.max() / (attn_grid.mean() + 1e-8):.2f}")

    # 4. 绘图 (3x3 Grid)
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 3, figure=fig)
    fig.suptitle(f"[IMAGE] AIGC Detector -- Original Resolution Visualization\nModel: CLIP-ViT + MSFF + Gated-Fusion | Source: {image_path.name}", 
                 fontsize=20, fontweight='bold')

    # (1,1) Input Image (Original Size)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title(f"Input Image ({w_orig}x{h_orig})\nVerdict: {'AI-Generated' if p_fake[1] > 0.5 else 'Real Image'}", fontsize=14)
    ax1.axis('off')

    # (1,2) Quad-Path Frequency Feature
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(quad_feat_res, cmap='jet')
    ax2.set_title("Quad-Path SRM Residual\n(artifact noise pattern)", fontsize=14)
    fig.colorbar(im2, ax=ax2)
    ax2.axis('off')

    # (1,3) FFT Magnitude
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(fft_mag, cmap='hot')
    ax3.set_title("FFT Magnitude Spectrum\n(global frequency distribution)", fontsize=14)
    fig.colorbar(im3, ax=ax3)
    ax3.axis('off')

    # (2,1) ViT Patch Attention (Attention Rollout)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(img_rgb, alpha=0.55)
    im4 = ax4.imshow(attn_resized, cmap='jet', alpha=0.55, vmin=0.0, vmax=1.0)
    ax4.set_title(f"ViT Attention Rollout (CLS)\n{attn_grid.shape[0]}x{attn_grid.shape[1]} patches, "
                  f"12 layers, peak/mean={attn_grid.max() / (attn_grid.mean() + 1e-8):.2f}", fontsize=13)
    fig.colorbar(im4, ax=ax4)
    ax4.axis('off')

    # (2,2) DCT High-Frequency
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(dct_feat, cmap='plasma')
    ax5.set_title("DCT Coefficients\n(compression artifact hotspots)", fontsize=14)
    fig.colorbar(im5, ax=ax5)
    ax5.axis('off')

    # (2,3) DWT Wavelet Detail
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(dwt_feat, cmap='inferno')
    ax6.set_title("DWT Wavelet Detail\n(multi-scale high-freq subbands)", fontsize=14)
    fig.colorbar(im6, ax=ax6)
    ax6.axis('off')

    # (3,1) SGA Gated Blend
    ax7 = fig.add_subplot(gs[2, 0])
    bar_labels = ['Image', 'Video', 'Audio', 'Sync', 'Async']
    bar_values = gates
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#937860']
    ax7.bar(bar_labels, bar_values, color=colors)
    ax7.set_ylim(0, 1.0)
    ax7.set_ylabel("Gating Weight")
    ax7.set_title(f"SGA Gated Fusion Weights\n(final feature contribution)", fontsize=14)
    for i, v in enumerate(bar_values):
        ax7.text(i, v + 0.02, f"{v:.2f}", ha='center')

    # (3,2) Layer-wise CLS Token
    ax8 = fig.add_subplot(gs[2, 1])
    im8 = ax8.imshow(cls_tokens, aspect='auto', cmap='coolwarm')
    ax8.set_title("Multi-Layer CLS Tokens\n(feature evolution through ViT)", fontsize=14)
    ax8.set_xlabel("Feature Dimension")
    ax8.set_ylabel("CLIP Layer Index")
    fig.colorbar(im8, ax=ax8)

    # (3,3) Classification Result
    ax9 = fig.add_subplot(gs[2, 2])
    prob_labels = ['Real (0)', 'Fake (1)']
    prob_values = [p_fake[0], p_fake[1]]
    ax9.bar(prob_labels, prob_values, color=['#55A868', '#C44E52'])
    ax9.set_ylim(0, 1.0)
    ax9.set_ylabel("Probability")
    ax9.set_title(f"Final Classification\nConf: {max(prob_values)*100:.1f}%", fontsize=14)
    for i, v in enumerate(prob_values):
        ax9.text(i, v + 0.02, f"{v:.3f}", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        plt.savefig(output_path, dpi=120)
        print(f"  [SUCCESS] Report saved: {output_path.name}")
    except OSError as e:
        # Target may be locked by an image viewer; fall back to a suffixed name
        alt_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        plt.savefig(alt_path, dpi=120)
        print(f"  [SUCCESS] {output_path.name} was locked ({e}); saved as {alt_path.name}")
    finally:
        plt.close()

def main():
    print("="*60)
    print("Advanced AIGC Detection Visualization (Original Resolution)")
    print("="*60)
    
    # 1. Initialize detector
    detector = MultimodalAIGCDetector(
        embed_dim=768,
        num_classes=2,
        device=DEVICE.type,
    )
    
    # Load weights
    ckpt_path = PROJECT_ROOT / 'checkpoints' / 'image_best.pt'
    if ckpt_path.exists():
        try:
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            if 'model_state' in state_dict:
                detector.image_branch.load_state_dict(state_dict['model_state'], strict=False)
            else:
                detector.image_branch.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {ckpt_path.name}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    detector.eval()

    # 2. Scan test images
    test_dir = PROJECT_ROOT / 'test' / 'image'
    images = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpeg')))
    
    if not images:
        print(f"No images found in {test_dir}")
        return

    print(f"Found {len(images)} images to process.")
    
    # 3. Processing loop
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] Processing {img_path.name}...")
        save_path = OUT_DIR / f"orig_viz_{img_path.stem}.png"
        generate_viz(img_path, detector, save_path)

    print("\n" + "="*60)
    print(f"Original size visualization complete!")
    print(f"Results saved to: {OUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
