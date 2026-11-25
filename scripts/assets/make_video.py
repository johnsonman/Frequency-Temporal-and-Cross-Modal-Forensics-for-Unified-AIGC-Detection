"""
把单张图片做成带 Ken Burns 动画的视频 (5-10 秒)
用法: python make_video.py
"""
import cv2
import numpy as np
from pathlib import Path

INPUT  = Path(r"e:\johnsonman\Project\AIGC Detection Project\Code\A2\test\image\images-2-aliens.jpg")
OUTPUT = Path(r"e:\johnsonman\Project\AIGC Detection Project\Code\A2\outputs\aliens_video.mp4")

DURATION = 8.0    # 秒
FPS      = 30
W, H     = 1920, 1080   # 输出分辨率

def ken_burn_frame(img, t, total, W, H):
    """
    对原图做 Ken Burns: 从 1.0x 缩放 + 中心，缓慢推到 1.15x + 轻微平移
    t 当前时间, total 总时长
    """
    h, w = img.shape[:2]
    progress = t / total   # 0 → 1

    # 缩放: 从 1.0 → 1.12
    scale = 1.0 + 0.12 * progress
    # 平移: 中心略微向上右移动 (±3% 范围)
    tx = 0.03 * progress
    ty = -0.02 * progress

    # 裁剪区域 (基于原图比例, 缩放到 1/scale 后再 crop)
    # 目标 crop 尺寸 = 原图 / scale
    crop_w = int(w / scale)
    crop_h = int(h / scale)
    # crop 中心 = 原图中心 + 平移
    cx = w // 2 + int(tx * w)
    cy = h // 2 + int(ty * h)

    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)
    x1 = min(w, x0 + crop_w)
    y1 = min(h, y0 + crop_h)
    x0 = max(0, x1 - crop_w)
    y0 = max(0, y1 - crop_h)

    cropped = img[y0:y1, x0:x1]
    frame = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)
    return frame


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(INPUT))
    if img is None:
        print(f"❌ 无法读取图片: {INPUT}")
        return
    print(f"✅ 图片尺寸: {img.shape[1]}×{img.shape[0]}")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (W, H))
    if not writer.isOpened():
        # fallback
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (W, H))

    total_frames = int(DURATION * FPS)
    print(f"🎬 生成 {DURATION}s / {FPS}fps / {W}×{H} 视频, 共 {total_frames} 帧 ...")

    for i in range(total_frames):
        t = i / FPS
        frame = ken_burn_frame(img, t, DURATION, W, H)
        writer.write(frame)
        if (i + 1) % 60 == 0:
            print(f"   进度: {i+1}/{total_frames} 帧 ({t:.1f}s)")

    writer.release()
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"\n✅ 视频已保存: {OUTPUT}")
    print(f"   大小: {size_mb:.2f} MB  |  时长: {DURATION}s  |  {W}×{H}")


if __name__ == "__main__":
    main()
