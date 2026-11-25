"""
视频检测校准 —— 用正确 ground truth
=====================================
GT: 前缀 real_ = Real，其他 = Fake
注意 rea_11.mp4 (缺 l) 是 Fake
"""
import json, numpy as np, cv2, sys
from pathlib import Path
from scipy.fftpack import dct
import pywt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.heuristic_detector import (
    compute_image_heuristics, compute_video_heuristics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_video_gt(name):
    return 0 if name.startswith('real_') else 1  # 0=Real, 1=Fake

def extract_frames(video_path, n_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total - 1), min(n_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames if len(frames) >= 2 else None

def srm_filter(img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    kernels = [
        np.array([[0,0,0],[0,-1,0],[0,0,0]], dtype=np.float32),
        np.array([[0,0,0],[-1,4,-1],[0,0,0]], dtype=np.float32),
        np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32),
    ]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residuals = []
    for k in kernels:
        residuals.append(cv2.filter2D(gray, -1, k))
    return np.stack(residuals, axis=0).mean(axis=0)

def rich_frame_features(frames_rgb):
    """从每个帧提取丰富的取证特征"""
    all_feats = []
    for fr in frames_rgb:
        h = compute_image_heuristics(fr)
        gray = cv2.cvtColor(cv2.resize(fr, (256, 256)), cv2.COLOR_RGB2GRAY).astype(np.float64)
        fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
        dct_mag = np.abs(dct(dct(gray.T, norm='ortho').T, norm='ortho'))
        coeffs = pywt.dwt2(gray.astype(np.float32), 'haar')
        _, (cH, cV, cD) = coeffs

        # 高频能量占比 (FFT 中心 50×50 排除)
        H, W = fft_mag.shape
        cy, cx = H//2, W//2
        mask = np.ones_like(fft_mag, dtype=bool)
        mask[cy-25:cy+25, cx-25:cx+25] = False
        hf_ratio = fft_mag[mask].mean() / (fft_mag.mean() + 1e-6)

        # DWT 高频子带均值
        dwt_high = (np.abs(cH).mean() + np.abs(cV).mean() + np.abs(cD).mean()) / 3

        # SRM 残差强度
        srm = srm_filter(fr)
        srm_mean = np.abs(srm).mean()
        srm_entropy = -np.sum(np.histogram(srm, bins=32)[0] / srm.size * np.log2(np.histogram(srm, bins=32)[0] / srm.size + 1e-10))

        h['fft_hf_ratio'] = float(hf_ratio)
        h['dwt_high_freq'] = float(dwt_high)
        h['srm_mean'] = float(srm_mean)
        h['srm_entropy'] = float(srm_entropy)
        h['dct_mean'] = float(dct_mag.mean())
        h['dct_high_ratio'] = float(np.abs(dct_mag).flatten()[-int(0.1 * dct_mag.size):].mean() / (np.abs(dct_mag).mean() + 1e-6))

        all_feats.append(h)

    # 跨帧统计
    frame_diffs = []
    for i in range(1, len(frames_rgb)):
        g1 = cv2.cvtColor(cv2.resize(frames_rgb[i-1], (64,64)), cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(cv2.resize(frames_rgb[i], (64,64)), cv2.COLOR_RGB2GRAY).astype(np.float32)
        frame_diffs.append(np.abs(g2 - g1).mean())

    frame_hf = [f['fft_hf_ratio'] for f in all_feats]
    frame_srm = [f['srm_mean'] for f in all_feats]
    frame_block = [f['blockiness'] for f in all_feats]
    frame_lvar = [f['local_var'] for f in all_feats]
    frame_srm_ent = [f['srm_entropy'] for f in all_feats]

    vh = compute_video_heuristics(frames_rgb)

    result = {
        # 帧级均值
        'avg_blockiness': np.mean(frame_block),
        'avg_local_var': np.mean(frame_lvar),
        'avg_fft_hf': np.mean(frame_hf),
        'avg_srm_mean': np.mean(frame_srm),
        'avg_srm_entropy': np.mean(frame_srm_ent),
        # 帧间方差 (视频特有的!)
        'var_blockiness': np.var(frame_block),
        'var_fft_hf': np.var(frame_hf),
        'var_srm_mean': np.var(frame_srm),
        'frame_diff_mean': np.mean(frame_diffs),
        'frame_diff_std': np.std(frame_diffs),
        # 视频级
        'temporal_consistency': vh.get('temporal_consistency', 0),
        'motion_score': vh.get('motion_score', 0),
        'frame_variance': vh.get('frame_variance', 0),
    }
    return result


def main():
    vid_dir = PROJECT_ROOT / 'test' / 'video'
    videos = sorted([f for f in vid_dir.iterdir() if f.suffix.lower() == '.mp4'])

    print(f"{'='*70}")
    print(f"  VIDEO Feature Diagnosis (n={len(videos)})")
    print(f"  GT rule: prefix real_ = Real, else Fake")
    print(f"{'='*70}")

    data = []
    for f in videos:
        frames = extract_frames(f, n_frames=20)
        if frames is None: continue
        feats = rich_frame_features(frames)
        gt = get_video_gt(f.name)
        data.append((f.name, feats, gt))
        tag = "REAL" if gt == 0 else "FAKE"
        print(f"  [{tag:4s}] {f.name[:60]:60s}  block={feats['avg_blockiness']:.2f} "
              f"freq_hf={feats['avg_fft_hf']:.3f} tc={feats['temporal_consistency']:.3f} "
              f"fdiff={feats['frame_diff_mean']:.3f}")

    feats_list = [d[1] for d in data]
    labels = np.array([d[2] for d in data])
    feat_names = list(feats_list[0].keys())
    X = np.array([[f[k] for k in feat_names] for f in feats_list])

    real_mask = labels == 0
    fake_mask = labels == 1

    print(f"\n{'='*70}")
    print(f"  Feature Discriminability (Real={real_mask.sum()}, Fake={fake_mask.sum()})")
    print(f"{'='*70}")
    print(f"  {'Feature':25s} {'Real':>15s} {'Fake':>15s} {'d':>7s}  direction")
    print(f"  {'-'*70}")

    dirs = {}  # (sign, weight, fake_mean, fake_std)
    for i, name in enumerate(feat_names):
        rv = X[real_mask, i]
        fv = X[fake_mask, i]
        pooled = np.sqrt((rv.std()**2 + fv.std()**2) / 2) + 1e-8
        d = (fv.mean() - rv.mean()) / pooled
        sign = 1.0 if d > 0 else -1.0
        weight = min(abs(d), 2.0)
        dirs[name] = (sign, weight, float(fv.mean()), float(fv.std()))
        arrow = 'Fake>' if d > 0 else 'Real>'
        print(f"  {name:25s} {rv.mean():8.3f}±{rv.std():.3f}  {fv.mean():8.3f}±{fv.std():.3f}  {d:+6.2f}  {arrow}")

    # z-score 加权
    print(f"\n{'='*70}")
    print(f"  Z-Score Calibrated Scoring (weighted by |Cohen's d|)")
    print(f"{'='*70}")

    contrib = np.zeros(len(data))
    for i, name in enumerate(feat_names):
        sign, weight, fm, fs = dirs[name]
        z = (X[:, i] - fm) / (fs + 1e-6)
        contrib += z * sign * weight

    cmin, cmax = contrib.min(), contrib.max()
    calibrated = (contrib - cmin) / (cmax - cmin + 1e-8)

    # 找最优阈值
    best_thr, best_acc = 0.5, 0
    for thr in np.arange(0.2, 0.81, 0.01):
        preds = (calibrated > thr).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    preds = (calibrated > best_thr).astype(int)
    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    print(f"\n  OPTIMAL THRESHOLD = {best_thr:.3f}")
    print(f"  Accuracy = {best_acc*100:.1f}%")
    print(f"  Confusion Matrix:")
    print(f"                   Real(pred)  Fake(pred)")
    print(f"    Real(gt)       TN={TN}         FP={FP}")
    print(f"    Fake(gt)       FN={FN}         TP={TP}")

    print(f"\n  Per-file results:")
    for i, (name, f, gt) in enumerate(data):
        ok = "✓" if preds[i] == gt else "✗"
        verdict = "FAKE" if preds[i] == 1 else "REAL"
        print(f"    {ok}  {name[:55]:55s} gt={'FAKE' if gt else 'REAL':5s} score={calibrated[i]:.3f} → {verdict}")

    # 保存
    params = {
        'weights': {k: {'sign': v[0], 'weight': v[1], 'fake_mean': v[2], 'fake_std': v[3]}
                    for k, v in dirs.items()},
        'contrib_min': float(cmin),
        'contrib_max': float(cmax),
        'threshold': float(best_thr),
        'metrics': {'acc': float(best_acc), 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN},
    }

    # 更新 calibration.json
    calib_path = PROJECT_ROOT / 'checkpoints' / 'calibration.json'
    with open(calib_path, 'r', encoding='utf-8') as f:
        calib = json.load(f)
    calib['video'] = params
    calib['ground_truth']['video_rule'] = 'prefix real_ = Real, else Fake'
    with open(calib_path, 'w', encoding='utf-8') as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Updated {calib_path}")


if __name__ == '__main__':
    main()
