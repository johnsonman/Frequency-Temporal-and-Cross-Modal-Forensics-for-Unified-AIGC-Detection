import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
"""Audio branch module-level testing for test/audio, each module saved separately.

Model: train.AudioPatchViT (Log-Mel -> Patch ViT -> Transformer x6 -> classifier)
Weights: checkpoints/audio_best.pt

Usage:
    python _audio_modules.py --lang zh --out reports/audio_modules
    python _audio_modules.py --lang en --out reports/audio_modules_en

Per-file output (16 PNG + meta.json):
    01_waveform.png              Center 5 s waveform
    02_logmel_input.png          Log-Mel fed to the model (z-score)
    03_logmel_db.png             Un-normalized dB spectrogram
    04_patch_tokens.png          Patch embedding output (channel L2 norm)
    05_attn_layer1..10_attn_layer6.png   Per-layer CLS->Patch attention
    11_attention_overview.png    All 6 layers, shared color scale
    12_cls_norm_by_layer.png     CLS token norm per layer
    13_probs.png                 Classification probabilities (center + 5 windows)
    14_spectral_features.png     Centroid / rolloff / flatness / ZCR
    15_band_energy.png           Band energy ratio
    16_embedding.png             Final CLS embedding
    meta.json
"""
import os, re, math, json, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000
DUR = 5.0
N_MELS = 128
N_FFT = 1024
HOP = 256

# ---------------- 文案 ----------------
ZH = {
    "wave_title": "波形（中心 {dur:.1f}s）",
    "time": "时间 (s)",
    "amp": "幅值",
    "logmel_in": "模型输入 Log-Mel 频谱（z-score 归一化）",
    "melbin": "Mel 频点 (0-128)",
    "logmel_db": "Log-Mel 频谱（dB，ref=max）",
    "db": "dB",
    "patch": "Patch Embedding 输出（192 通道模长，{hp}×{wp} patch）",
    "attn": "第 {i} 层 CLS→Patch 注意力（{h} 头均值）",
    "attn_all": "{n} 层 CLS→Patch 注意力总览（统一色标）",
    "cls_norm": "各层 CLS token 模长",
    "layer": "层",
    "prob_center": "中心窗分类概率",
    "prob_win": "多窗 p_fake（均值 {m:.3f} ± {s:.3f}）",
    "win_x": "窗口序号（全曲均分 5 窗）",
    "spec": "频谱统计特征",
    "spec_centroid": "频谱质心 (Hz)",
    "spec_rolloff": "85% 频谱滚降 (Hz)",
    "spec_flatness": "频谱平坦度",
    "spec_zcr": "过零率",
    "band": "分频段能量占比（Mel 四分位）",
    "energy": "能量占比",
    "emb": "最终 CLS 嵌入（{n} 维）",
    "gt_real": "真实",
    "gt_ai": "AI",
    "gt_unknown": "未知",
    "done": "完成",
    "skipped": "跳过",
}

EN = {
    "wave_title": "Waveform (center {dur:.1f} s)",
    "time": "Time (s)",
    "amp": "Amplitude",
    "logmel_in": "Model Input: Log-Mel Spectrogram (z-score normalized)",
    "melbin": "Mel bin (0-128)",
    "logmel_db": "Log-Mel Spectrogram (dB, ref=max)",
    "db": "dB",
    "patch": "Patch Embedding Output (L2 norm over 192 channels, {hp}x{wp} patches)",
    "attn": "Layer {i} CLS-to-Patch Attention ({h} heads, averaged)",
    "attn_all": "CLS-to-Patch Attention across {n} Layers (shared color scale)",
    "cls_norm": "CLS Token Norm per Layer",
    "layer": "Layer",
    "prob_center": "Classification Probability (center window)",
    "prob_win": "p_fake per Window (mean {m:.3f} +/- {s:.3f})",
    "win_x": "Window index (5 evenly spaced over the full track)",
    "spec": "Spectral Statistics",
    "spec_centroid": "Spectral Centroid (Hz)",
    "spec_rolloff": "85% Spectral Rolloff (Hz)",
    "spec_flatness": "Spectral Flatness",
    "spec_zcr": "Zero-Crossing Rate",
    "band": "Band Energy Ratio (Mel quartiles)",
    "energy": "Energy ratio",
    "emb": "Final CLS Embedding ({n}-dim)",
    "gt_real": "Real",
    "gt_ai": "AI",
    "gt_unknown": "Unknown",
    "done": "done",
    "skipped": "skipped",
}

BANDS = [(0, 32), (32, 64), (64, 96), (96, 128)]
BAND_LABELS = ["0-2kHz", "2-4kHz", "4-6kHz", "6-8kHz"]
AI_MARKS = ("minimax", "suno", "musiccreator", "seedance", "udio", "elevenlabs", "tts")

# ASCII display names, used by the English outputs (folders / tables / figures)
EN_NAMES = {
    "兰亭序-周杰伦#aeUM.mp3": "JayChou-LanTingXu",
    "庐州月-许嵩#b0LV.mp3": "XuSong-LuZhouYue",
    "光年之外-G.E.M.邓紫棋#eyW8s.mp3": "GEM-LightYearsAway",
    "musiccreatorAI_chinese_站稳别退.mp3": "musiccreatorAI-ZhanWenBieTui",
    "suno_music_chinese_雨停在你离开的那一天.mp3": "suno-YuTingZaiNiLiKaiDeNaYiTian",
}


def display_name(name):
    return EN_NAMES.get(name, os.path.splitext(name)[0])


def safe(name, lang="zh"):
    base = display_name(name) if lang == "en" else os.path.splitext(name)[0]
    return re.sub(r'[\\/:*?"<>|#]', "_", base)


def save(path, fig):
    """Write to a temp file then atomically replace (avoids Windows OSError)."""
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=110, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    ap.add_argument("--out", default="reports/audio_modules")
    ap.add_argument("--src", default="test/audio")
    args = ap.parse_args()
    T = EN if args.lang == "en" else ZH
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    from train import AudioPatchViT

    ck = torch.load("checkpoints/audio_best.pt", map_location="cpu", weights_only=False)
    model = AudioPatchViT(num_classes=2).to(DEV)
    model.load_state_dict(ck["model_state"])
    model.eval()
    N_LAYERS = len(model.encoder.layers)
    N_HEADS = model.encoder.layers[0].self_attn.num_heads
    print(f"device={DEV}  lang={args.lang}  ckpt epoch={ck.get('epoch')}  "
          f"val_acc={ck.get('val_acc')}  layers={N_LAYERS}  heads={N_HEADS}")

    # The fused encoder fast path bypasses MultiheadAttention, so disable it.
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception as e:
        print("fastpath toggle failed:", e)

    orig_mha = nn.MultiheadAttention.forward

    def forward_capture(x):
        caps = []

        def patched(self, *a, **kw):
            kw["need_weights"] = True
            kw["average_attn_weights"] = False
            out, w = orig_mha(self, *a, **kw)
            caps.append(w.detach().float().cpu())
            return out, w

        nn.MultiheadAttention.forward = patched
        try:
            patches = model.patch_embed(x)                  # [B,D,Hp,Wp]
            B, D, Hp, Wp = patches.shape
            seq = patches.flatten(2).transpose(1, 2)
            seq = torch.cat([model.cls_token.expand(B, -1, -1), seq], dim=1)
            N = seq.shape[1]
            if N <= model.pos_embed.shape[1]:
                seq = seq + model.pos_embed[:, :N]
            else:
                pos = F.interpolate(model.pos_embed.transpose(1, 2), size=N,
                                    mode="linear", align_corners=False).transpose(1, 2)
                seq = seq + pos
            cls_norms = []
            for layer in model.encoder.layers:
                seq = layer(seq)
                cls_norms.append(float(seq[0, 0].norm()))
            enc_norm = getattr(model.encoder, "norm", None)
            if enc_norm is not None:
                seq = enc_norm(seq)
            cls_out = model.norm(seq[:, 0])
            logits = model.classifier(cls_out)
        finally:
            nn.MultiheadAttention.forward = orig_mha

        return {
            "logits": logits,
            "attn": caps,
            "patch_grid": patches[0].detach().float().cpu().numpy(),
            "cls_out": cls_out[0].detach().float().cpu().numpy(),
            "cls_norms": cls_norms,
            "grid": (Hp, Wp),
        }

    def load_window(path, center=True, index=0, total=1):
        y, _ = librosa.load(path, sr=SR, mono=True)
        n = int(DUR * SR)
        if len(y) > n:
            s = (len(y) - n) // 2 if center else int(index * (len(y) - n) / max(total - 1, 1))
            y = y[s:s + n]
        elif len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        y = y / (np.max(np.abs(y)) + 1e-8)
        return y, y.shape[0] / SR

    def to_logmel(y):
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP)
        db = librosa.power_to_db(mel, ref=np.max)
        z = (db - db.mean()) / (db.std() + 1e-8)
        return mel, db, z

    def guess_gt(name):
        low = name.lower()
        if any(m in low for m in AI_MARKS):
            return 1, T["gt_ai"]
        # 真实歌曲命名形如 "歌名-歌手#短ID"
        if "#" in name or "周杰伦" in name or "邓紫棋" in name or "许嵩" in name:
            return 0, T["gt_real"]
        return -1, T["gt_unknown"]

    files = sorted(os.listdir(args.src))
    rows, skipped = [], []

    for fi, fname in enumerate(files, 1):
        path = os.path.join(args.src, fname)
        try:
            y5, dur5 = load_window(path)
        except Exception as e:
            print(f"[{T['skipped']}] {fname}: {type(e).__name__} {str(e)[:70]}")
            skipped.append({"file": fname, "reason": f"{type(e).__name__}: {str(e)[:120]}"})
            continue

        d = os.path.join(OUT, safe(fname, args.lang))
        os.makedirs(d, exist_ok=True)
        gt, gt_cn = guess_gt(fname)

        mel, db, z = to_logmel(y5)
        x = torch.tensor(z, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEV)

        with torch.no_grad():
            r = forward_capture(x)
            probs = F.softmax(r["logits"], -1)[0].float().cpu().numpy()
            wins = []
            for k in range(5):
                try:
                    yk, _ = load_window(path, center=False, index=k, total=5)
                    zk = to_logmel(yk)[2]
                    lk = model(torch.tensor(zk, dtype=torch.float32)
                               .unsqueeze(0).unsqueeze(0).to(DEV))
                    wins.append(float(F.softmax(lk, -1)[0, 1]))
                except Exception:
                    pass

        Hp, Wp = r["grid"]
        attn = r["attn"]
        Tlen = z.shape[1]
        pw_s = Wp * model.patch_size[1] * HOP / SR

        # 01 waveform
        fig, ax = plt.subplots(figsize=(9, 2.4), dpi=120)
        ax.plot(np.arange(len(y5)) / SR, y5, lw=0.4, color="#2c7fb8")
        ax.set_xlabel(T["time"]); ax.set_ylabel(T["amp"])
        ax.set_title(T["wave_title"].format(dur=dur5), fontsize=11)
        ax.grid(alpha=0.25)
        save(os.path.join(d, "01_waveform.png"), fig)

        # 02 log-mel input
        fig, ax = plt.subplots(figsize=(9, 3.2), dpi=120)
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="magma",
                       extent=[0, dur5, 0, N_MELS])
        ax.set_xlabel(T["time"]); ax.set_ylabel(T["melbin"])
        ax.set_title(T["logmel_in"], fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.03)
        save(os.path.join(d, "02_logmel_input.png"), fig)

        # 03 log-mel dB
        fig, ax = plt.subplots(figsize=(9, 3.2), dpi=120)
        im = ax.imshow(db, origin="lower", aspect="auto", cmap="viridis",
                       extent=[0, dur5, 0, N_MELS], vmin=-80, vmax=0)
        ax.set_xlabel(T["time"]); ax.set_ylabel(T["melbin"])
        ax.set_title(T["logmel_db"], fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.03, label=T["db"])
        save(os.path.join(d, "03_logmel_db.png"), fig)

        # 04 patch embedding
        pg = np.linalg.norm(r["patch_grid"], axis=0)
        fig, ax = plt.subplots(figsize=(7, 2.8), dpi=120)
        im = ax.imshow(pg, origin="lower", aspect="auto", cmap="cividis",
                       extent=[0, pw_s, 0, N_MELS])
        ax.set_xlabel(T["time"]); ax.set_ylabel(T["melbin"])
        ax.set_title(T["patch"].format(hp=Hp, wp=Wp), fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.03)
        save(os.path.join(d, "04_patch_tokens.png"), fig)

        # 05-10 per-layer attention
        maps = []
        for li, a in enumerate(attn):
            w = a[0, :, 0, 1:].mean(0).numpy().reshape(Hp, Wp)
            maps.append(w)
            fig, ax = plt.subplots(figsize=(7, 2.8), dpi=120)
            im = ax.imshow(w, origin="lower", aspect="auto", cmap="jet",
                           extent=[0, pw_s, 0, N_MELS])
            ax.set_xlabel(T["time"]); ax.set_ylabel(T["melbin"])
            ax.set_title(T["attn"].format(i=li + 1, h=N_HEADS), fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.03)
            save(os.path.join(d, f"{5 + li:02d}_attn_layer{li + 1}.png"), fig)

        # 11 attention overview
        fig, axes = plt.subplots(N_LAYERS, 1, figsize=(9, 1.5 * N_LAYERS), dpi=110,
                                 sharex=True)
        if N_LAYERS == 1:
            axes = [axes]
        vmax = max(float(m.max()) for m in maps) if maps else 1.0
        for li, (ax, w) in enumerate(zip(axes, maps)):
            ax.imshow(w, origin="lower", aspect="auto", cmap="jet", vmin=0, vmax=vmax,
                      extent=[0, pw_s, 0, N_MELS])
            ax.set_ylabel(f"L{li + 1}", fontsize=9)
            ax.set_yticks([])
            if li == 0:
                ax.set_title(T["attn_all"].format(n=N_LAYERS), fontsize=11)
        axes[-1].set_xlabel(T["time"])
        save(os.path.join(d, "11_attention_overview.png"), fig)

        # 12 CLS norm per layer
        fig, ax = plt.subplots(figsize=(5.5, 3), dpi=120)
        ax.plot(range(1, N_LAYERS + 1), r["cls_norms"], "o-", color="#d95f02")
        ax.set_xlabel(T["layer"]); ax.set_ylabel("||CLS||")
        ax.set_title(T["cls_norm"], fontsize=11)
        ax.grid(alpha=0.3)
        save(os.path.join(d, "12_cls_norm_by_layer.png"), fig)

        # 13 probabilities
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=120,
                                 gridspec_kw={"width_ratios": [1, 1.6]})
        ax = axes[0]
        b = ax.barh(["p_real", "p_fake"], [probs[0], probs[1]],
                    color=["#43a047", "#e53935"])
        ax.set_xlim(0, 1.05); ax.axvline(0.5, ls="--", c="k", lw=1)
        for bb, v in zip(b, [probs[0], probs[1]]):
            if v > 0.85:
                ax.text(v - 0.02, bb.get_y() + bb.get_height() / 2, f"{v:.4f}",
                        va="center", ha="right", color="w", fontweight="bold")
            else:
                ax.text(v + 0.02, bb.get_y() + bb.get_height() / 2, f"{v:.4f}", va="center")
        ax.set_title(T["prob_center"], fontsize=11); ax.grid(axis="x", alpha=0.3)

        ax = axes[1]
        if wins:
            ax.bar(range(1, len(wins) + 1), wins, color="#ff7043")
            ax.axhline(0.5, ls="--", c="k", lw=1)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel(T["win_x"]); ax.set_ylabel("p_fake")
            ax.set_title(T["prob_win"].format(m=np.mean(wins), s=np.std(wins)), fontsize=11)
            ax.grid(axis="y", alpha=0.3)
        save(os.path.join(d, "13_probs.png"), fig)

        # 14 spectral features
        cent = librosa.feature.spectral_centroid(y=y5, sr=SR, hop_length=HOP)[0]
        roll = librosa.feature.spectral_rolloff(y=y5, sr=SR, hop_length=HOP,
                                                roll_percent=0.85)[0]
        flat = librosa.feature.spectral_flatness(y=y5, hop_length=HOP)[0]
        zcr = librosa.feature.zero_crossing_rate(y5, hop_length=HOP)[0]
        tt = np.arange(len(cent)) * HOP / SR
        fig, axes = plt.subplots(4, 1, figsize=(9, 7), dpi=110, sharex=True)
        for ax, v, lab, c in zip(
                axes,
                [cent, roll, flat, zcr],
                [T["spec_centroid"], T["spec_rolloff"], T["spec_flatness"], T["spec_zcr"]],
                ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]):
            ax.plot(tt, v, lw=0.8, color=c)
            ax.set_ylabel(lab, fontsize=9)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel(T["time"])
        axes[0].set_title(T["spec"], fontsize=11)
        save(os.path.join(d, "14_spectral_features.png"), fig)

        # 15 band energy
        e = np.array([mel[a:b].sum() for a, b in BANDS])
        ratio = e / (e.sum() + 1e-9)
        fig, ax = plt.subplots(figsize=(5.5, 3), dpi=120)
        bb = ax.bar(BAND_LABELS, ratio, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
        for bx, v in zip(bb, ratio):
            ax.text(bx.get_x() + bx.get_width() / 2, v + 0.005, f"{v * 100:.1f}%",
                    ha="center", fontsize=9)
        ax.set_ylabel(T["energy"]); ax.set_ylim(0, max(0.6, ratio.max() * 1.25))
        ax.set_title(T["band"], fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        save(os.path.join(d, "15_band_energy.png"), fig)

        # 16 embedding
        emb = r["cls_out"]
        side = int(math.ceil(math.sqrt(emb.shape[0])))
        pad = np.full(side * side, np.nan, np.float32)
        pad[:emb.shape[0]] = emb
        fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
        im = ax.imshow(pad.reshape(side, side), cmap="RdBu_r",
                       vmin=-np.abs(emb).max(), vmax=np.abs(emb).max())
        ax.set_title(T["emb"].format(n=emb.shape[0]), fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
        save(os.path.join(d, "16_embedding.png"), fig)

        meta = {
            "file": fname,
            "display_name": display_name(fname),
            "lang": args.lang,
            "model": "AudioPatchViT",
            "ground_truth": gt_cn,
            "prediction": "AI" if probs[1] > 0.5 else "Real",
            "p_fake_center": float(probs[1]),
            "p_real_center": float(probs[0]),
            "p_fake_windows": wins,
            "p_fake_window_mean": float(np.mean(wins)) if wins else None,
            "p_fake_window_std": float(np.std(wins)) if wins else None,
            "n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP,
            "mel_frames": int(Tlen),
            "patches": int(Hp * Wp),
            "patch_grid": [int(Hp), int(Wp)],
            "attn_layers": N_LAYERS, "attn_heads": N_HEADS,
            "cls_norms": r["cls_norms"],
            "attn_mass_on_patches": [float(a[0, :, 0, 1:].sum(-1).mean()) for a in attn],
            "spectral": {
                "centroid_hz_mean": float(cent.mean()),
                "rolloff85_hz_mean": float(roll.mean()),
                "flatness_mean": float(flat.mean()),
                "zcr_mean": float(zcr.mean()),
            },
            "band_energy_ratio": {k: float(v) for k, v in zip(BAND_LABELS, ratio)},
        }
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

        rows.append({
            "display_name": display_name(fname), "dir": safe(fname, args.lang),
            "ground_truth": gt_cn,
            "p_fake_center": f"{probs[1]:.4f}",
            "p_fake_window_mean": f"{np.mean(wins):.4f}" if wins else "",
            "prediction": meta["prediction"],
            "mel_frames": Tlen, "patch_grid": f"{Hp}x{Wp}",
        })
        print(f"[{fi}/{len(files)}] {fname[:46]:46s} gt={gt_cn:8s} "
              f"p_fake={probs[1]:.4f}  win_mean={np.mean(wins):.4f}", flush=True)

    with open(os.path.join(OUT, "index.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["display_name", "dir", "ground_truth",
                                          "p_fake_center", "p_fake_window_mean",
                                          "prediction", "mel_frames", "patch_grid"])
        w.writeheader(); w.writerows(rows)
    if skipped:
        with open(os.path.join(OUT, "skipped.json"), "w", encoding="utf-8") as f:
            json.dump(skipped, f, ensure_ascii=False, indent=1)

    n_png = sum(len([x for x in fs if x.endswith(".png")]) for _, _, fs in os.walk(OUT))
    print(f"\n{T['done']}: {len(rows)} audio files, {n_png} png, "
          f"skipped={len(skipped)} -> {OUT}")


if __name__ == "__main__":
    main()
