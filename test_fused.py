"""
增强版多模态 AIGC 检测器测试脚本
融合: 预训练骨干特征 + 启发式频谱/统计检测

解决了随机分类头导致结果无区分度的问题
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

# 导入两部分检测器
from models.detector import MultimodalAIGCDetector
from models.heuristic_detector import HeuristicAIGCDetector
from utils.visualizer import (
    print_header, print_summary, save_results_json, print_separator,
)


def find_test_files(test_dir: str):
    """扫描测试目录"""
    test_path = Path(test_dir)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    audio_exts = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}

    images, videos, audios = [], [], []
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


class FusedResult:
    """融合检测结果"""
    def __init__(self, file_path: str, file_type: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.file_type = file_type
        self.p_fake = 0.0
        self.p_real = 0.0
        self.is_ai = False
        self.confidence = 0.0
        self.nn_score = None       # 神经网络分支分数
        self.heuristic_score = None  # 启发式分支分数
        self.fused_score = 0.0     # 融合分数
        self.heuristics = {}
        self.notes = ""


def run_fused_detection(
    nn_detector: MultimodalAIGCDetector,
    heuristic_detector: HeuristicAIGCDetector,
    images: list, videos: list, audios: list,
) -> list:
    """运行融合检测"""
    all_results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============ 图像 ============
    if images:
        print_header(f"图像检测 ({len(images)} 个文件)")
        for i, path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {os.path.basename(path):40s}", end=' ', flush=True)
            t0 = time.time()

            result = FusedResult(path, 'image')

            # 1. 神经网络分支
            try:
                det = nn_detector.detect_image_file(path)
                result.nn_score = det.p_fake
            except Exception as e:
                result.notes = f"NN错误: {e}"
                result.nn_score = None

            # 2. 启发式分支
            try:
                h_res = heuristic_detector.detect_image(path)
                result.heuristic_score = h_res['p_fake']
                result.heuristics = h_res.get('heuristics', {})
            except Exception as e:
                if not result.notes:
                    result.notes = f"Heuristic错误: {e}"
                result.heuristic_score = None

            # 3. 融合分数
            result.fused_score = fuse_scores(result.nn_score, result.heuristic_score)
            result.p_fake = result.fused_score
            result.p_real = 1.0 - result.fused_score
            result.is_ai = result.fused_score > 0.5
            result.confidence = max(result.p_fake, result.p_real)

            elapsed = time.time() - t0
            label = "🤖" if result.is_ai else "📷"
            print(f"→ {label} fused={result.fused_score:.3f} ({elapsed:.1f}s)")
            all_results.append(result)

    # ============ 视频 ============
    if videos:
        print_header(f"视频检测 ({len(videos)} 个文件)")
        for i, path in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}] {os.path.basename(path):50s}", end=' ', flush=True)
            t0 = time.time()

            result = FusedResult(path, 'video')

            # 神经网络
            try:
                det = nn_detector.detect_video_file(path)
                result.nn_score = det.p_fake
            except Exception as e:
                result.notes = f"NN错误: {e}"
                result.nn_score = None

            # 启发式
            try:
                h_res = heuristic_detector.detect_video(path)
                result.heuristic_score = h_res['p_fake']
                result.heuristics = h_res.get('heuristics', {})
            except Exception as e:
                if not result.notes:
                    result.notes = f"Heuristic错误: {e}"
                result.heuristic_score = None

            # 融合
            result.fused_score = fuse_scores(result.nn_score, result.heuristic_score)
            result.p_fake = result.fused_score
            result.p_real = 1.0 - result.fused_score
            result.is_ai = result.fused_score > 0.5
            result.confidence = max(result.p_fake, result.p_real)

            elapsed = time.time() - t0
            label = "🤖" if result.is_ai else "📷"
            print(f"→ {label} fused={result.fused_score:.3f} ({elapsed:.1f}s)")
            all_results.append(result)

    # ============ 音频 ============
    if audios:
        print_header(f"音频检测 ({len(audios)} 个文件)")
        for i, path in enumerate(audios, 1):
            print(f"[{i}/{len(audios)}] {os.path.basename(path):50s}", end=' ', flush=True)
            t0 = time.time()

            result = FusedResult(path, 'audio')

            # 神经网络
            try:
                det = nn_detector.detect_audio_file(path)
                result.nn_score = det.p_fake
            except Exception as e:
                result.notes = f"NN错误: {e}"
                result.nn_score = None

            # 启发式
            try:
                h_res = heuristic_detector.detect_audio(path)
                result.heuristic_score = h_res['p_fake']
                result.heuristics = h_res.get('heuristics', {})
            except Exception as e:
                if not result.notes:
                    result.notes = f"Heuristic错误: {e}"
                result.heuristic_score = None

            # 融合
            result.fused_score = fuse_scores(result.nn_score, result.heuristic_score)
            result.p_fake = result.fused_score
            result.p_real = 1.0 - result.fused_score
            result.is_ai = result.fused_score > 0.5
            result.confidence = max(result.p_fake, result.p_real)

            elapsed = time.time() - t0
            label = "🤖" if result.is_ai else "📷"
            print(f"→ {label} fused={result.fused_score:.3f} ({elapsed:.1f}s)")
            all_results.append(result)

    return all_results


def fuse_scores(nn_score: float, heuristic_score: float) -> float:
    """
    融合神经网络分数和启发式分数
    NN分数是预训练骨干的输出但分类头是随机的(0.5附近)
    启发式分数基于频谱/统计特征，有更多区分度

    策略: 启发式为主(0.7) + NN特征辅助(0.3)
    """
    if heuristic_score is not None and nn_score is not None:
        # NN分数进行校正 - 中心化后加权
        nn_centered = (nn_score - 0.5) * 0.6 + 0.5  # 向0.5收缩
        fused = 0.7 * heuristic_score + 0.3 * nn_centered
        return float(np.clip(fused, 0.0, 1.0))
    elif heuristic_score is not None:
        return float(heuristic_score)
    elif nn_score is not None:
        return float(nn_score)
    return 0.5


def print_fused_result(result: FusedResult, index: int):
    """打印融合结果"""
    print(f"\n[{index}] {result.file_name}")
    print(f"    类型: {result.file_type}")

    label = "🤖 AI生成" if result.is_ai else "📷 真实"
    print(f"    判定: {label} (置信度: {result.confidence:.2%})")
    print(f"    融合AI概率: {result.fused_score:.4f}")

    if result.nn_score is not None:
        print(f"    NN分支:     {result.nn_score:.4f} (预训练骨干+随机头)")
    else:
        print(f"    NN分支:     N/A")

    if result.heuristic_score is not None:
        print(f"    启发式分支: {result.heuristic_score:.4f} (频谱/统计特征)")
    else:
        print(f"    启发式分支: N/A")

    # 启发式详情
    if result.heuristics:
        top_keys = sorted(result.heuristics.items(), key=lambda x: -abs(x[1] - 0.5))[:4]
        feat_str = " | ".join(f"{k}={v:.4f}" for k, v in top_keys)
        print(f"    关键特征:   {feat_str}")

    if result.notes:
        print(f"    备注:       {result.notes}")


def print_fused_summary(results: list):
    """打印融合结果汇总"""
    total = len(results)
    success = [r for r in results if not r.notes]
    failed = [r for r in results if r.notes]

    print_header("融合检测汇总")
    print(f"总文件数: {total}")
    print(f"成功:     {len(success)}")
    print(f"失败:     {len(failed)}")

    if success:
        ai_count = sum(1 for r in success if r.is_ai)
        real_count = sum(1 for r in success if not r.is_ai)

        print(f"\n判定分布 (融合 p_fake > 0.5):")
        print(f"  🤖 AI生成: {ai_count} ({ai_count/max(len(success),1)*100:.1f}%)")
        print(f"  📷 真实:   {real_count} ({real_count/max(len(success),1)*100:.1f}%)")

        # 比较两个分支
        nn_scores = [r.nn_score for r in success if r.nn_score is not None]
        h_scores = [r.heuristic_score for r in success if r.heuristic_score is not None]
        fused_scores = [r.fused_score for r in success]

        print(f"\n分数统计:")
        if nn_scores:
            print(f"  NN分支    : mean={np.mean(nn_scores):.4f}, std={np.std(nn_scores):.4f}")
        if h_scores:
            print(f"  启发式    : mean={np.mean(h_scores):.4f}, std={np.std(h_scores):.4f}")
        print(f"  融合结果  : mean={np.mean(fused_scores):.4f}, std={np.std(fused_scores):.4f}")

        # 按类型
        by_type = {}
        for r in success:
            by_type.setdefault(r.file_type, []).append(r.fused_score)
        if len(by_type) > 1:
            print(f"\n按模态类型:")
            for t, scores in sorted(by_type.items()):
                print(f"  {t:8s}: n={len(scores):3d}, mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")


def save_fused_results(results: list, output_path: str):
    """保存融合结果JSON"""
    data = []
    for r in results:
        entry = {
            'file_path': r.file_path,
            'file_name': r.file_name,
            'file_type': r.file_type,
            'is_ai': r.is_ai,
            'fused_score': round(r.fused_score, 6),
            'nn_score': round(r.nn_score, 6) if r.nn_score is not None else None,
            'heuristic_score': round(r.heuristic_score, 6) if r.heuristic_score is not None else None,
            'confidence': round(r.confidence, 6),
            'heuristics': {k: round(v, 6) for k, v in r.heuristics.items()} if r.heuristics else None,
            'notes': r.notes,
        }
        data.append(entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="增强版多模态AIGC检测 - 融合NN+启发式")
    parser.add_argument("--test-dir", default="test")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    test_dir = str(PROJECT_ROOT / args.test_dir)
    output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print_header("增强版多模态 AI 生成媒体检测器")
    print(f"融合策略: 预训练骨干特征 + 频谱/统计启发式检测")
    print(f"测试目录: {test_dir}")
    print(f"计算设备: {args.device}")

    images, videos, audios = find_test_files(test_dir)
    print(f"\n发现测试文件:")
    print(f"  📷 图像: {len(images)} 个")
    print(f"  🎬 视频: {len(videos)} 个")
    print(f"  🎵 音频: {len(audios)} 个")

    # 加载检测器
    t0 = time.time()
    print("\n加载检测器...")
    nn_detector = MultimodalAIGCDetector(device=args.device)
    heuristic_detector = HeuristicAIGCDetector(device=args.device)
    print(f"检测器加载完成 ({time.time() - t0:.1f}s)")

    # 运行融合检测
    t_start = time.time()
    results = run_fused_detection(nn_detector, heuristic_detector, images, videos, audios)
    total_time = time.time() - t_start

    # 打印详细结果
    print_header("详细检测结果")
    for i, r in enumerate(results, 1):
        print_fused_result(r, i)

    # 汇总
    print_fused_summary(results)
    print(f"\n总耗时: {total_time:.1f}s")

    # 保存
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"fused_results_{timestamp}.json")
    save_fused_results(results, json_path)

    print_header("检测完成")


if __name__ == "__main__":
    main()
