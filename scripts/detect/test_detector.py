"""
多模态 AIGC 检测器 - 测试脚本

对 test/ 目录下的图像、视频、音频文件进行真伪检测
"""

import os
import sys
import time
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import MultimodalAIGCDetector, DetectionResult
from utils.visualizer import (
    print_header, print_detection_result, print_summary,
    save_results_json, print_separator
)


def find_test_files(test_dir: str):
    """扫描测试目录，返回各类文件列表"""
    test_path = Path(test_dir)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    audio_exts = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}

    images = []
    videos = []
    audios = []

    for f in test_path.rglob('*'):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in image_exts:
                images.append(str(f))
            elif ext in video_exts:
                videos.append(str(f))
            elif ext in audio_exts:
                audios.append(str(f))

    return sorted(images), sorted(videos), sorted(audios)


def run_detection(detector, images, videos, audios, output_dir):
    """运行完整检测"""
    all_results = []

    # ============ 图像检测 ============
    if images:
        print_header(f"图像检测 ({len(images)} 个文件)")
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] 检测: {os.path.basename(img_path)}", end=' ', flush=True)
            t0 = time.time()
            result = detector.detect_image_file(img_path)
            elapsed = time.time() - t0
            all_results.append(result)
            label = "🤖" if result.is_ai else "📷"
            print(f"→ {label} p_fake={result.p_fake:.3f} ({elapsed:.1f}s)")

    # ============ 视频检测 ============
    if videos:
        print_header(f"视频检测 ({len(videos)} 个文件)")
        for i, vid_path in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}] 检测: {os.path.basename(vid_path)}", end=' ', flush=True)
            t0 = time.time()
            result = detector.detect_video_file(vid_path)
            elapsed = time.time() - t0
            all_results.append(result)
            label = "🤖" if result.is_ai else "📷"
            print(f"→ {label} p_fake={result.p_fake:.3f} ({elapsed:.1f}s)")

    # ============ 音频检测 ============
    if audios:
        print_header(f"音频检测 ({len(audios)} 个文件)")
        for i, aud_path in enumerate(audios, 1):
            print(f"[{i}/{len(audios)}] 检测: {os.path.basename(aud_path)}", end=' ', flush=True)
            t0 = time.time()
            result = detector.detect_audio_file(aud_path)
            elapsed = time.time() - t0
            all_results.append(result)
            label = "🤖" if result.is_ai else "📷"
            print(f"→ {label} p_fake={result.p_fake:.3f} ({elapsed:.1f}s)")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="多模态AIGC检测 - 测试脚本")
    parser.add_argument("--test-dir", type=str, default="test",
                        help="测试文件目录 (默认: test)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="输出目录 (默认: outputs)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备 (cuda/cpu, 默认: cuda)")
    parser.add_argument("--no-ssl", action="store_true",
                        help="不使用自监督预训练模型 (Wav2Vec2)")
    args = parser.parse_args()

    # 路径
    test_dir = str(PROJECT_ROOT / args.test_dir)
    output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print_header("多模态 AI 生成媒体检测器")
    print(f"测试目录: {test_dir}")
    print(f"输出目录: {output_dir}")
    print(f"计算设备: {args.device}")

    # 扫描测试文件
    images, videos, audios = find_test_files(test_dir)
    print_separator()
    print(f"发现测试文件:")
    print(f"  📷 图像: {len(images)} 个")
    print(f"  🎬 视频: {len(videos)} 个")
    print(f"  🎵 音频: {len(audios)} 个")

    if not (images or videos or audios):
        print("没有找到测试文件!")
        return

    # 创建检测器
    t0 = time.time()
    detector = MultimodalAIGCDetector(
        embed_dim=768,
        num_classes=2,
        device=args.device,
        num_frames=8,
    )
    model_load_time = time.time() - t0
    print(f"模型加载完成 ({model_load_time:.1f}s)")

    # 运行检测
    print()
    t_start = time.time()
    results = run_detection(detector, images, videos, audios, output_dir)
    total_time = time.time() - t_start

    # 打印详细结果
    print_header("详细检测结果")
    for i, r in enumerate(results, 1):
        print_detection_result(r, i)

    # 打印汇总
    print_summary(results)
    print(f"\n总耗时: {total_time:.1f}s")

    # 保存JSON
    import json
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"detection_results_{timestamp}.json")
    save_results_json(results, json_path)

    print_header("检测完成")


if __name__ == "__main__":
    main()
