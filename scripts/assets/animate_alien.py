"""
让图片里的外星人真正动起来 —— rembg 抠前景 + 分层程序化动画
每个外星人独立: 头部摆动 + 身体呼吸 + 微平移
背景独立: 慢速 Ken Burns 运动
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

INPUT  = Path(r"e:\johnsonman\Project\AIGC Detection Project\Code\A2\test\image\images-2-aliens.jpg")
OUTPUT = Path(r"e:\johnsonman\Project\AIGC Detection Project\Code\A2\outputs\aliens_alive.mp4")
DURATION = 8.0
FPS      = 30


def segment():
    print("🔍 rembg 前景分割 ...")
    from rembg import remove
    img = Image.open(INPUT).convert("RGB")
    result = remove(img, post_process_mask=True)
    result.save(Path(r"outputs\aliens_fg.png"))
    mask = np.array(result.split()[3])
    fg = np.array(result.convert("RGB"))
    bg = np.array(img)
    print(f"   前景 shape={fg.shape}, 前景像素={(mask>128).sum()}")
    return bg, fg, mask


def split_aliens(mask):
    """水平投影找中间缝 → 切成左右 → 各取最大连通域"""
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    H, W = binary.shape
    
    # 全图水平投影, 平滑后在合理范围找最小值
    h_proj = binary.sum(axis=0)
    smoothed = np.convolve(h_proj, np.ones(40)/40, mode='valid')
    search_start, search_end = 400, 1300
    cut_x = np.argmin(smoothed[search_start:search_end]) + search_start + 20
    print(f"   分割缝: x={cut_x}")

    def largest_component(m):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num <= 1: return m, 0, 0, 0, 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = np.argmax(areas) + 1
        ys, xs = np.where(labels == best)
        cx = xs.mean() if len(xs) else 0
        cy = ys.mean() if len(ys) else 0
        x0, x1 = xs.min(), xs.max() if len(xs) else (0, 0)
        return (labels == best).astype(np.uint8) * 255, cx, cy, x0, x1

    left_mask = binary.copy();  left_mask[:, cut_x:] = 0
    right_mask = binary.copy(); right_mask[:, :cut_x] = 0

    l_mask, lcx, lcy, lx0, lx1 = largest_component(left_mask)
    r_mask, rcx, rcy, rx0, rx1 = largest_component(right_mask)

    print(f"   左外星人: area={l_mask.sum()//255}, cx={lcx:.0f}")
    print(f"   右外星人: area={r_mask.sum()//255}, cx={rcx:.0f}")

    return [
        {"mask": l_mask, "cx": lcx, "cy": lcy, "x0": lx0, "x1": lx1, "side": "left"},
        {"mask": r_mask, "cx": rcx, "cy": rcy, "x0": rx0, "x1": rx1, "side": "right"},
    ]


def animate_frame(bg, fg, aliens, t):
    """
    每个外星人独立参数:
      - head_angle(t)  : ±1.8° 摇头
      - body_scale(t)  : ±0.4% 呼吸
      - body_tx/ty(t)  : ±2px 微晃
    """
    h, w = bg.shape[:2]

    # 背景 Ken Burns
    bg_scale = 1.003 + 0.003 * np.sin(2 * np.pi * t * 0.08)
    bg_tx = int(w * 0.002 * np.sin(2 * np.pi * t * 0.12))
    bg_ty = int(h * 0.001 * np.sin(2 * np.pi * t * 0.09))
    M_bg = cv2.getRotationMatrix2D((w/2, h/2), 0, bg_scale)
    M_bg[0, 2] += bg_tx; M_bg[1, 2] += bg_ty
    bg_frame = cv2.warpAffine(bg, M_bg, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 前景动画合成
    fg_out = np.zeros_like(bg)
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for i, alien in enumerate(aliens):
        # 左右外星人不同相位 → 独立动作
        head_phase  = 2 * np.pi * t * (0.55 if i == 0 else 0.47)
        body_phase  = 2 * np.pi * t * (0.33 if i == 0 else 0.28)
        sway_phase  = 2 * np.pi * t * (0.20 if i == 0 else 0.25)

        head_angle  = 1.8 * np.sin(head_phase) + 0.6 * np.sin(head_phase * 0.3)
        body_scale  = 1.0 + 0.004 * np.sin(body_phase + i * 0.7)
        body_ty     = 1.8 * np.sin(body_phase)
        body_tx     = 1.0 * np.sin(sway_phase)

        cx, cy = alien["cx"], alien["cy"]
        M = cv2.getRotationMatrix2D((cx, cy), head_angle, body_scale)
        M[0, 2] += body_tx; M[1, 2] += body_ty

        m_warp = cv2.warpAffine(alien["mask"], M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        # 从完整 fg 中取当前外星人区域 → 变换
        alien_rgb = cv2.bitwise_and(fg, fg, mask=alien["mask"])
        rgb_warp = cv2.warpAffine(alien_rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT)

        combined_mask = np.maximum(combined_mask, m_warp)
        fg_out[m_warp > 0] = rgb_warp[m_warp > 0]

    # 合成
    out = bg_frame.copy()
    out[combined_mask > 0] = fg_out[combined_mask > 0]
    return out


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # 若已缓存前景就跳过分割
    fg_cache = Path(r"outputs\aliens_fg.png")
    if fg_cache.exists():
        print("📦 使用缓存前景 ...")
        fg_pil = Image.open(fg_cache).convert("RGB")
        alpha  = Image.open(fg_cache).split()[3]
        mask   = np.array(alpha)
        fg     = np.array(fg_pil)
        bg     = np.array(Image.open(INPUT).convert("RGB"))
    else:
        bg, fg, mask = segment()

    aliens = split_aliens(mask)
    h, w = bg.shape[:2]
    total_frames = int(DURATION * FPS)

    print(f"\n🎬 生成 {DURATION}s × {FPS}fps = {total_frames} 帧, {w}×{h}")
    print(f"   左右独立摇头 + 呼吸 + 背景 Ken Burns\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (w, h))

    prev = None
    for i in range(total_frames):
        t = i / FPS
        frame = animate_frame(bg, fg, aliens, t)
        # 帧间运动平滑
        if prev is not None:
            frame = cv2.addWeighted(frame, 0.75, prev, 0.25, 0)
        prev = frame
        writer.write(frame)
        if (i + 1) % 60 == 0:
            print(f"   {i+1}/{total_frames}  ({t:.1f}s)")

    writer.release()
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"\n🎉 完成: {OUTPUT}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
