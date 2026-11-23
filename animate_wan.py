"""
Wan2.1 图生视频 —— 让外星人动起来
模型已缓存: models--DeepBeepMeep--Wan2.1
GPU: RTX 3080 Ti (12GB)
"""
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
from pathlib import Path

INPUT  = Path(r"e:\johnsonman\Project\AIGC Detection Project\Code\A2\test\image\images-2-aliens.jpg")
OUTPUT = Path(r"e:\johnsonman\Project\AIGC Detection Project\Code\A2\outputs\aliens_wan.mp4")
MODEL  = "DeepBeepMeep/Wan2.1"

# —— 运动描述 ——
PROMPT = (
    "Two aliens with large bald heads sitting at an outdoor cafe table. "
    "The alien on the left slightly sways their head and moves their hand holding a transparent cup with a straw. "
    "The alien on the right subtly moves their V-sign hand and blinks. "
    "Realistic, natural motion, subtle head bob, smooth breathing-like movement. "
    "City street background with trees and parked cars."
)
NEGATIVE = "blurry, distorted face, extra fingers, jittery camera, watermark, text, low quality"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # 加载图片 —— Wan2.1 推荐 832×480 或 480×832 (landscape)
    img = Image.open(INPUT).convert("RGB")
    W_OUT, H_OUT = 832, 480
    img = img.resize((W_OUT, H_OUT), Image.LANCZOS)
    print(f"✅ 输入: {INPUT.name} → {W_OUT}×{H_OUT}")

    # 加载 pipeline
    print(f"🔄 加载 Wan2.1 (bfloat16, CUDA) ...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.enable_vae_slicing()
    print(f"✅ Pipeline 已加载, 显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # 推理 —— 49帧 ≈ 2.04s @ 24fps
    # 用 97 帧 ≈ 4.04s, 在 3080 Ti 上应该能跑
    NUM_FRAMES = 97
    STEPS      = 30
    print(f"🎬 推理: {NUM_FRAMES}帧 / {STEPS}步, 预计输出 {NUM_FRAMES/24:.1f}s ...")

    with torch.no_grad():
        result = pipe(
            image=img,
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            num_frames=NUM_FRAMES,
            height=H_OUT,
            width=W_OUT,
            guidance_scale=5.0,
            num_inference_steps=STEPS,
        )

    frames = result.frames[0]  # list of PIL.Image
    print(f"✅ 生成 {len(frames)} 帧")

    # 导出 MP4
    export_to_video(frames, str(OUTPUT), fps=24)
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"\n🎉 视频已保存: {OUTPUT}")
    print(f"   {W_OUT}×{H_OUT}, 24fps, {len(frames)/24:.1f}s, {size_mb:.1f} MB")
    print(f"\n提示: 想更长视频可改 NUM_FRAMES (上限取决于显存)")


if __name__ == "__main__":
    main()
