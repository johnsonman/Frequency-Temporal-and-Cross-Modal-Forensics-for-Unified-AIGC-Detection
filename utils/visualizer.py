"""
检测结果可视化工具
"""

import json
import os
from datetime import datetime
from typing import List, Dict


def print_separator(char: str = '=', length: int = 70):
    print(char * length)


def print_header(title: str):
    print_separator()
    print(f"  {title}")
    print_separator()


def print_detection_result(result, index: int = 0):
    """打印单个检测结果"""
    from models.detector import DetectionResult

    print(f"\n[{index}] {os.path.basename(result.file_path)}")
    print(f"    类型: {result.file_type}")

    # 最终判定
    label = "🤖 AI生成" if result.is_ai else "📷 真实"
    print(f"    判定: {label} (置信度: {result.confidence:.2%})")
    print(f"    AI概率: {result.p_fake:.4f}, 真实概率: {result.p_real:.4f}")

    # 各分支概率
    branch_probs = []
    if result.p_img is not None:
        branch_probs.append(f"Img: {result.p_img:.3f}")
    if result.p_vid is not None:
        branch_probs.append(f"Vid: {result.p_vid:.3f}")
    if result.p_aud is not None:
        branch_probs.append(f"Aud: {result.p_aud:.3f}")
    if branch_probs:
        print(f"    分支概率: {' | '.join(branch_probs)}")

    # 门控权重
    if result.gates:
        active = [(k, v) for k, v in result.gates.items() if v > 0.01]
        active.sort(key=lambda x: -x[1])
        gate_str = " | ".join(f"{k}:{v:.3f}" for k, v in active)
        print(f"    门控权重: {gate_str}")

    # 可靠性
    if result.reliability:
        rel_str = " | ".join(f"{k}:{v:.3f}" for k, v in result.reliability.items() if v > 0.01)
        if rel_str:
            print(f"    可靠性:   {rel_str}")

    # 模态掩码
    if result.modal_mask:
        present = [k for k, v in result.modal_mask.items() if v]
        print(f"    模态:     {', '.join(present)}")

    if result.notes:
        print(f"    备注:     {result.notes}")


def print_summary(results: list):
    """打印汇总统计"""
    total = len(results)
    failed = [r for r in results if r.notes]
    success = [r for r in results if not r.notes]

    print_header("检测汇总")
    print(f"总文件数:   {total}")
    print(f"成功检测:   {len(success)}")
    print(f"失败:       {len(failed)}")

    if success:
        ai_count = sum(1 for r in success if r.is_ai)
        real_count = sum(1 for r in success if not r.is_ai)
        print(f"\n判定分布 (p_fake > 0.5):")
        print(f"  🤖 AI生成:  {ai_count} ({ai_count/max(len(success),1)*100:.1f}%)")
        print(f"  📷 真实:    {real_count} ({real_count/max(len(success),1)*100:.1f}%)")

        avg_fake = sum(r.p_fake for r in success) / len(success)
        avg_conf = sum(r.confidence for r in success) / len(success)
        print(f"\n  平均AI概率: {avg_fake:.4f}")
        print(f"  平均置信度: {avg_conf:.4f}")

        # 按类型分组
        by_type = {}
        for r in success:
            t = r.file_type
            by_type.setdefault(t, []).append(r)

        if len(by_type) > 1:
            print(f"\n按模态类型:")
            for t, rs in sorted(by_type.items()):
                avg = sum(r.p_fake for r in rs) / len(rs)
                print(f"  {t:8s}: n={len(rs):3d}, avg_p_fake={avg:.4f}")


def save_results_json(results: list, output_path: str):
    """保存结果为JSON"""
    data = []
    for r in results:
        entry = {
            'file_path': r.file_path,
            'file_name': os.path.basename(r.file_path),
            'file_type': r.file_type,
            'is_ai': r.is_ai,
            'p_fake': round(r.p_fake, 6),
            'p_real': round(r.p_real, 6),
            'confidence': round(r.confidence, 6),
            'p_img': round(r.p_img, 6) if r.p_img is not None else None,
            'p_vid': round(r.p_vid, 6) if r.p_vid is not None else None,
            'p_aud': round(r.p_aud, 6) if r.p_aud is not None else None,
            'gates': {k: round(v, 4) for k, v in r.gates.items()} if r.gates else None,
            'reliability': {k: round(v, 4) for k, v in r.reliability.items()} if r.reliability else None,
            'modal_mask': r.modal_mask,
            'notes': r.notes,
        }
        data.append(entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")
