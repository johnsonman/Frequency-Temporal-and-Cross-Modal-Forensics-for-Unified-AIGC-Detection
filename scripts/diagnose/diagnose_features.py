"""
诊断脚本: 分析启发式特征在 ground truth 下的区分度
用已知标注找最优阈值，作为最终修正检测方案的依据
"""
import json
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.heuristic_detector import (
    compute_image_heuristics, score_image_from_heuristics,
    compute_video_heuristics, score_video_from_heuristics,
    compute_audio_heuristics, score_audio_from_heuristics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_gt(label):
    """根据文件名前缀判断 ground truth"""
    name = label
    if name.startswith('real_'):
        return 0  # Real
    # 图像中非 real_ = Fake
    return 1  # Fake


def analyze_features(features_list, labels, modality):
    """对每个特征算 real vs fake 的均值/方差，并找最优阈值"""
    print(f"\n{'='*70}")
    print(f"  [{modality}] 特征区分度分析  (n={len(labels)})")
    print(f"{'='*70}")

    real_mask = np.array(labels) == 0
    fake_mask = np.array(labels) == 1

    feats = list(features_list[0].keys())
    for feat in feats:
        vals = np.array([f[feat] for f in features_list])
        r_vals = vals[real_mask] if real_mask.sum() > 0 else np.array([])
        f_vals = vals[fake_mask] if fake_mask.sum() > 0 else np.array([])

        # 如果是单侧（如 video/audio 全 fake），只报分布
        if len(r_vals) == 0:
            print(f"  {feat:20s}: Fake only | mean={f_vals.mean():.4f} ± {f_vals.std():.4f} | "
                  f"range=[{f_vals.min():.4f}, {f_vals.max():.4f}]")
        elif len(f_vals) == 0:
            print(f"  {feat:20s}: Real only | mean={r_vals.mean():.4f} ± {r_vals.std():.4f}")
        else:
            # Cohen's d 效应量
            pooled_std = np.sqrt((r_vals.std()**2 + f_vals.std()**2) / 2)
            cohens_d = (f_vals.mean() - r_vals.mean()) / (pooled_std + 1e-8)
            print(f"  {feat:20s}: Real {r_vals.mean():.4f}±{r_vals.std():.4f} | "
                  f"Fake {f_vals.mean():.4f}±{f_vals.std():.4f} | Cohen's d={cohens_d:.2f}")

    # 启发式综合分数
    if modality == 'image':
        scores = np.array([score_image_from_heuristics(f) for f in features_list])
    elif modality == 'video':
        scores = np.array([score_video_from_heuristics(f) for f in features_list])
    else:
        scores = np.array([score_audio_from_heuristics(f) for f in features_list])

    r_scores = scores[real_mask] if real_mask.sum() > 0 else np.array([])
    f_scores = scores[fake_mask] if fake_mask.sum() > 0 else np.array([])
    print(f"\n  {'综合分数':20s}: Real={r_scores.mean():.4f}±{r_scores.std():.4f} | "
          f"Fake={f_scores.mean():.4f}±{f_scores.std():.4f}")

    # 找最优阈值 (仅图像)
    if len(r_scores) > 0 and len(f_scores) > 0:
        print("\n  阈值扫描 (综合分数):")
        best_acc = 0
        best_thr = 0.5
        for thr in np.arange(0.30, 0.71, 0.01):
            preds = (scores > thr).astype(int)
            acc = (preds == np.array(labels)).mean()
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
        print(f"    最优阈值 = {best_thr:.2f}  →  准确率 = {best_acc*100:.1f}%")

        # 混淆矩阵
        preds = (scores > best_thr).astype(int)
        TP = ((preds == 1) & (np.array(labels) == 1)).sum()
        TN = ((preds == 0) & (np.array(labels) == 0)).sum()
        FP = ((preds == 1) & (np.array(labels) == 0)).sum()
        FN = ((preds == 0) & (np.array(labels) == 1)).sum()
        print(f"    Confusion: TP={TP} TN={TN} FP={FP} FN={FN}")

    return scores, best_thr if (len(r_scores) > 0 and len(f_scores) > 0) else 0.5


def main():
    import os
    import librosa

    # ================ 图像 ================
    print("分析图像特征...")
    img_dir = PROJECT_ROOT / 'test' / 'image'
    img_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
    ])

    img_feats, img_labels = [], []
    for f in img_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        feats = compute_image_heuristics(img)
        img_feats.append(feats)
        img_labels.append(get_gt(f.name))
        print(f"  {f.name:40s} gt={'Real' if get_gt(f.name)==0 else 'Fake':5s} freq={feats['freq_ratio']:.4f} srm={feats['srm_std_ratio']:.4f} block={feats['blockiness']:.2f}")

    img_scores, img_thr = analyze_features(img_feats, img_labels, 'IMAGE')

    # 图像中 blockiness 看起来很有区分度!
    # 让我单独看 blockiness 的阈值
    print("\n  --- blockiness 单独阈值 ---")
    block_scores = np.array([f['blockiness'] for f in img_feats])
    real_block = block_scores[np.array(img_labels) == 0]
    fake_block = block_scores[np.array(img_labels) == 1]
    print(f"    Real blockiness: {real_block.mean():.2f} ± {real_block.std():.2f}  (range [{real_block.min():.2f}, {real_block.max():.2f}])")
    print(f"    Fake blockiness:  {fake_block.mean():.2f} ± {fake_block.std():.2f}  (range [{fake_block.min():.2f}, {fake_block.max():.2f}])")

    # 搜索 blockiness 最优阈值 (blockiness 高 = AI)
    best_block_thr, best_block_acc = 0, 0
    for thr in np.arange(2.0, 15.0, 0.5):
        preds = (block_scores > thr).astype(int)
        acc = (preds == np.array(img_labels)).mean()
        if acc > best_block_acc:
            best_block_acc = acc
            best_block_thr = thr
    print(f"    blockiness 最优阈值 = {best_block_thr:.1f}  →  准确率 = {best_block_acc*100:.1f}%")

    # local_var
    print("\n  --- local_var 单独阈值 ---")
    lv_scores = np.array([f['local_var'] for f in img_feats])
    real_lv = lv_scores[np.array(img_labels) == 0]
    fake_lv = lv_scores[np.array(img_labels) == 1]
    print(f"    Real local_var: {real_lv.mean():.4f} ± {real_lv.std():.4f}")
    print(f"    Fake local_var:  {fake_lv.mean():.4f} ± {fake_lv.std():.4f}")
    best_lv_thr, best_lv_acc = 0, 0
    for thr in np.arange(0.02, 0.3, 0.01):
        preds = (lv_scores > thr).astype(int)
        acc = (preds == np.array(img_labels)).mean()
        if acc > best_lv_acc:
            best_lv_acc = acc
            best_lv_thr = thr
    print(f"    local_var 最优阈值 = {best_lv_thr:.2f}  →  准确率 = {best_lv_acc*100:.1f}%")

    # ================ 视频 ================
    print("\n\n分析视频特征...")
    vid_dir = PROJECT_ROOT / 'test' / 'video'
    vid_files = sorted([f for f in vid_dir.iterdir() if f.suffix.lower() == '.mp4'])

    vid_feats, vid_labels = [], []  # video 全 fake
    for f in vid_files:
        cap = cv2.VideoCapture(str(f))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, min(16, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) < 2:
            continue
        feats = compute_video_heuristics(frames)
        # 视频内每帧算一遍图像启发式
        frame_hs = [compute_image_heuristics(f) for f in frames[::2]]
        avg_freq = np.mean([h['freq_ratio'] for h in frame_hs])
        avg_block = np.mean([h['blockiness'] for h in frame_hs])
        feats['avg_frame_blockiness'] = avg_block
        feats['avg_frame_freq'] = avg_freq
        vid_feats.append(feats)
        vid_labels.append(1)  # all fake
        print(f"  {f.name[:50]:50s} tc={feats['temporal_consistency']:.4f} mv={feats['motion_score']:.4f} "
              f"fv={feats['frame_variance']:.4f} block={avg_block:.2f}")

    analyze_features(vid_feats, vid_labels, 'VIDEO (all fake)')

    # ================ 音频 ================
    print("\n\n分析音频特征...")
    aud_dir = PROJECT_ROOT / 'test' / 'audio'
    aud_files = sorted([f for f in aud_dir.iterdir() if f.suffix.lower() in {'.mp3', '.wav', '.flac'}])

    aud_feats, aud_labels = [], []  # all fake
    for f in aud_files:
        try:
            y, sr = librosa.load(str(f), sr=16000, mono=True)
            feats = compute_audio_heuristics(y, sr)
            aud_feats.append(feats)
            aud_labels.append(1)
            print(f"  {f.name[:50]:50s} flat={feats['spectral_flatness']:.4f} mfcc_std={feats['mfcc_std']:.4f} "
                  f"centroid={feats['spectral_centroid']:.0f}")
        except Exception as e:
            print(f"  ERROR {f.name}: {e}")

    analyze_features(aud_feats, aud_labels, 'AUDIO (all fake)')


if __name__ == '__main__':
    main()
