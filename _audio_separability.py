"""真实 vs 伪造音频的特征可分性分析

读取 reports/audio_modules_en/*/meta.json 的模块统计量，
并在原始采样率下补充计算高频截止、动态范围、静音比等判别量。
"""
import os, json, glob
import numpy as np
import librosa

ROOT = "reports/audio_modules_en"
rows = []

for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    mj = os.path.join(p, "meta.json")
    if not os.path.isfile(mj):
        continue
    with open(mj, encoding="utf-8") as f:
        m = json.load(f)

    path = os.path.join("test/audio", m["file"])
    # 原生采样率加载，用于高频截止分析（模型侧只到 8kHz）
    y, sr = librosa.load(path, sr=None, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-8)

    # 平均功率谱
    S = np.abs(librosa.stft(y, n_fft=2048)) ** 2
    psd = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    psd_db = 10 * np.log10(psd + 1e-12)
    psd_db_n = psd_db - psd_db.max()

    # 高频截止: 从高往低找第一个高于 -50dB 的频点
    above = np.where(psd_db_n > -50)[0]
    cutoff = float(freqs[above[-1]]) if len(above) else 0.0

    # 动态范围：帧能量的 95 分位 - 5 分位 (dB)
    frame_e = 10 * np.log10(S.sum(axis=0) + 1e-12)
    dyn = float(np.percentile(frame_e, 95) - np.percentile(frame_e, 5))

    # 静音比：低于峰值 -40dB 的帧占比
    thr = frame_e.max() - 40
    silence = float((frame_e < thr).mean())

    # 帧间谱通量（变化快慢）
    Sn = S / (S.sum(axis=0, keepdims=True) + 1e-9)
    flux = float(np.sqrt(((np.diff(Sn, axis=1)) ** 2).sum(axis=0)).mean())

    rows.append({
        "name": m.get("display_name", os.path.splitext(m["file"])[0]),
        "file": m["file"],
        "gt": m["ground_truth"],
        "p_fake": m["p_fake_center"],
        "native_sr": sr,
        "cutoff_khz": cutoff / 1000,
        "centroid_hz": m["spectral"]["centroid_hz_mean"],
        "rolloff85_hz": m["spectral"]["rolloff85_hz_mean"],
        "flatness": m["spectral"]["flatness_mean"],
        "zcr": m["spectral"]["zcr_mean"],
        "band_0_2k": m["band_energy_ratio"]["0-2kHz"],
        "band_6_8k": m["band_energy_ratio"]["6-8kHz"],
        "dyn_range_db": dyn,
        "silence_ratio": silence,
        "spectral_flux": flux,
        "attn_mass": float(np.mean(m["attn_mass_on_patches"])),
    })

real = [r for r in rows if r["gt"] == "Real"]
fake = [r for r in rows if r["gt"] == "AI"]

cols = ["cutoff_khz", "centroid_hz", "rolloff85_hz", "flatness", "zcr",
        "band_0_2k", "band_6_8k", "dyn_range_db", "silence_ratio",
        "spectral_flux", "attn_mass", "p_fake"]

y = np.array([1 if r["gt"] == "AI" else 0 for r in rows])


def best_threshold_bacc(vals, yy):
    """扫描中点阈值，返回最大平衡准确率"""
    vals = np.asarray(vals, float)
    order = np.unique(vals)
    cand = np.concatenate([[order[0] - 1e-9], (order[:-1] + order[1:]) / 2,
                           [order[-1] + 1e-9]])
    best = 0.0
    for t in cand:
        pred = (vals > t).astype(int)
        rr = ((pred == 0) & (yy == 0)).sum() / max((yy == 0).sum(), 1)
        rf = ((pred == 1) & (yy == 1)).sum() / max((yy == 1).sum(), 1)
        best = max(best, (rr + rf) / 2)
    return best


print(f"real={len(real)}  fake={len(fake)}\n")
hdr = (f"{'metric':15s} {'Real min/mean/max':>24s} {'AI min/mean/max':>24s} "
       f"{'best BAcc':>10s}")
print(hdr); print("-" * len(hdr))
for c in cols:
    rv = np.array([r[c] for r in rows if r["gt"] == "Real"])
    fv = np.array([r[c] for r in rows if r["gt"] == "AI"])
    ba = best_threshold_bacc([r[c] for r in rows], y)
    print(f"{c:15s} {rv.min():7.4f}/{rv.mean():8.4f}/{rv.max():7.4f} "
          f"{fv.min():10.4f}/{fv.mean():8.4f}/{fv.max():7.4f} {ba*100:9.1f}%")

print("\n--- per file ---")
print(f"{'display_name':34s} {'gt':4s} {'p_fake':>7s} {'cut_kHz':>8s} {'centroid':>9s} "
      f"{'flat':>8s} {'dyn_db':>7s} {'silence':>8s} {'zcr':>7s}")
for r in rows:
    print(f"{r['name'][:34]:34s} {r['gt']:4s} {r['p_fake']:7.3f} "
          f"{r['cutoff_khz']:8.2f} {r['centroid_hz']:9.1f} {r['flatness']:8.5f} "
          f"{r['dyn_range_db']:7.1f} {r['silence_ratio']:8.3f} {r['zcr']:7.4f}")

with open("reports/audio_compare/separability_metrics.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
