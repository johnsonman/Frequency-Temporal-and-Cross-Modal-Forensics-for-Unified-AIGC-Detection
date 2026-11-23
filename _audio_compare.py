"""Real vs AI audio: separability comparison figures (English labels)

Output: reports/audio_compare/
    01_spectral_profile.png      Full-band average spectrum (0-22 kHz), cutoff marked
    02_logmel_side_by_side.png   Log-Mel side by side (shared color scale)
    03_attention_side_by_side.png CLS-to-Patch attention side by side
    04_metric_bars.png           Discriminative metrics grouped bars
    metrics.csv
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR, DUR, N_MELS, N_FFT, HOP = 16000, 5.0, 128, 1024, 256
OUT = "reports/audio_compare"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save(path, fig):
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, path)


# ---------------- model ----------------
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from train import AudioPatchViT

ck = torch.load("checkpoints/audio_best.pt", map_location="cpu", weights_only=False)
model = AudioPatchViT(num_classes=2).to(DEV)
model.load_state_dict(ck["model_state"])
model.eval()
try:
    torch.backends.mha.set_fastpath_enabled(False)
except Exception:
    pass

orig_mha = nn.MultiheadAttention.forward


def run(x):
    caps = []

    def patched(self, *a, **kw):
        kw["need_weights"] = True
        kw["average_attn_weights"] = False
        out, w = orig_mha(self, *a, **kw)
        caps.append(w.detach().float().cpu())
        return out, w

    nn.MultiheadAttention.forward = patched
    try:
        patches = model.patch_embed(x)
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
        for layer in model.encoder.layers:
            seq = layer(seq)
        enc = getattr(model.encoder, "norm", None)
        if enc is not None:
            seq = enc(seq)
        logits = model.classifier(model.norm(seq[:, 0]))
    finally:
        nn.MultiheadAttention.forward = orig_mha
    return logits, caps, (Hp, Wp)


def center5(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    n = int(DUR * SR)
    if len(y) > n:
        s = (len(y) - n) // 2
        y = y[s:s + n]
    elif len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y / (np.max(np.abs(y)) + 1e-8)


# ---------------- collect ----------------
items = []
for d in sorted(os.listdir("reports/audio_modules_en")):
    mj = os.path.join("reports/audio_modules_en", d, "meta.json")
    if not os.path.isfile(mj):
        continue
    with open(mj, encoding="utf-8") as f:
        m = json.load(f)
    path = os.path.join("test/audio", m["file"])

    y5 = center5(path)
    mel = librosa.feature.melspectrogram(y=y5, sr=SR, n_mels=N_MELS,
                                        n_fft=N_FFT, hop_length=HOP)
    z = librosa.power_to_db(mel, ref=np.max)
    z = (z - z.mean()) / (z.std() + 1e-8)

    with torch.no_grad():
        x = torch.tensor(z, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEV)
        logits, caps, (Hp, Wp) = run(x)
    p_fake = float(F.softmax(logits, -1)[0, 1])
    attn = np.mean([a[0, :, 0, 1:].mean(0).numpy().reshape(Hp, Wp) for a in caps], axis=0)

    # full-band spectrum
    y, sr = librosa.load(path, sr=None, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-8)
    S = np.abs(librosa.stft(y, n_fft=2048)) ** 2
    psd = S.mean(1)
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    db = 10 * np.log10(psd + 1e-12)
    db -= db.max()
    above = np.where(db > -50)[0]
    cutoff = float(fr[above[-1]]) / 1000 if len(above) else 0.0

    fe = 10 * np.log10(S.sum(0) + 1e-12)
    items.append({
        "name": m.get("display_name", os.path.splitext(m["file"])[0]),
        "file": m["file"], "gt": m["ground_truth"], "p_fake": p_fake,
        "z": z, "attn": attn, "freq": fr, "db": db, "cutoff": cutoff,
        "dyn": float(np.percentile(fe, 95) - np.percentile(fe, 5)),
        "flat": m["spectral"]["flatness_mean"],
    })
    print(f"{m['ground_truth']:5s} {items[-1]['name'][:40]:40s} "
          f"p_fake={p_fake:.3f} cutoff={cutoff:.2f}kHz dyn={items[-1]['dyn']:.1f}dB")

items.sort(key=lambda r: (r["gt"] != "Real", -r["p_fake"]))
real = [r for r in items if r["gt"] == "Real"]

# ---------------- 01 spectral profile ----------------
fig, ax = plt.subplots(figsize=(9.8, 4.8), dpi=130)
ai_labeled = False
for r in sorted(items, key=lambda x: x["gt"] != "Real"):
    is_real = r["gt"] == "Real"
    if is_real:
        ax.plot(r["freq"] / 1000, r["db"], lw=2.0, color="#d62728", alpha=0.95,
                label=f"Real: {r['name']}", zorder=5)
    else:
        ax.plot(r["freq"] / 1000, r["db"], lw=1.1, color="#1f77b4", alpha=0.45,
                label=None if ai_labeled else "AI-generated", zorder=2)
        ai_labeled = True
ax.axhline(-50, ls=":", c="k", lw=1)
ax.text(21.6, -47, "-50 dB", fontsize=8, ha="right")

ai_cut = [r["cutoff"] for r in items if r["gt"] != "Real"]
r_cut = [r["cutoff"] for r in items if r["gt"] == "Real"]
ax.axvspan(min(ai_cut), max(ai_cut), color="#1f77b4", alpha=0.10)
ax.axvline(max(ai_cut), ls="--", c="#1f5391", lw=1.1)
ax.axvline(min(r_cut), ls="--", c="#d62728", lw=1.1)
ax.annotate("", xy=(max(ai_cut), -88), xytext=(min(r_cut), -88),
            arrowprops=dict(arrowstyle="<->", color="k", lw=1))
ax.text((max(ai_cut) + min(r_cut)) / 2, -95,
        f"overlap {max(ai_cut):.1f}-{min(r_cut):.1f} kHz", ha="center", fontsize=8)
ax.text(np.mean(ai_cut), -70, "AI cutoff range", ha="center", fontsize=8.5,
        color="#1f5391")
ax.set_xlim(0, 22); ax.set_ylim(-100, 5)
ax.set_xlabel("Frequency (kHz)"); ax.set_ylabel("Relative power (dB)")
ax.set_title("Full-band average spectrum: real (red) vs AI-generated (blue)", fontsize=11)
ax.legend(fontsize=7.5, loc="lower left")
ax.grid(alpha=0.25)
save(os.path.join(OUT, "01_spectral_profile.png"), fig)

# ---------------- 02 log-mel grid ----------------
n = len(items)
cols = 4
rws = int(np.ceil(n / cols))
allz = np.concatenate([r["z"] for r in items], axis=1)
vmin, vmax = np.percentile(allz, 1), np.percentile(allz, 99)
fig, axes = plt.subplots(rws, cols, figsize=(4.0 * cols, 2.6 * rws), dpi=120)
axes = np.atleast_2d(axes)
for i, r in enumerate(items):
    ax = axes[i // cols, i % cols]
    im = ax.imshow(r["z"], origin="lower", aspect="auto", cmap="magma",
                   vmin=vmin, vmax=vmax, extent=[0, DUR, 0, N_MELS])
    tag = "REAL" if r["gt"] == "Real" else "AI"
    ax.set_title(f"{tag} | p_fake={r['p_fake']:.3f}\n{r['name'][:34]}", fontsize=8.5,
                 color="#c62828" if r["gt"] == "Real" else "#1f4e79")
    ax.set_xticks([]); ax.set_yticks([])
for j in range(n, rws * cols):
    axes[j // cols, j % cols].axis("off")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label="z-score")
fig.suptitle("Log-Mel spectrogram (center 5 s, shared color scale)", fontsize=12, y=1.0)
save(os.path.join(OUT, "02_logmel_side_by_side.png"), fig)

# ---------------- 03 attention grid ----------------
ava = np.concatenate([r["attn"].ravel() for r in items])
vmax_a = np.percentile(ava, 99)
fig, axes = plt.subplots(rws, cols, figsize=(4.0 * cols, 2.3 * rws), dpi=120)
axes = np.atleast_2d(axes)
for i, r in enumerate(items):
    ax = axes[i // cols, i % cols]
    im = ax.imshow(r["attn"], origin="lower", aspect="auto", cmap="jet",
                   vmin=0, vmax=vmax_a)
    tag = "REAL" if r["gt"] == "Real" else "AI"
    ax.set_title(f"{tag} | p_fake={r['p_fake']:.3f}", fontsize=8.5,
                 color="#c62828" if r["gt"] == "Real" else "#1f4e79")
    ax.set_xticks([]); ax.set_yticks([])
for j in range(n, rws * cols):
    axes[j // cols, j % cols].axis("off")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01,
             label="attention weight")
fig.suptitle("CLS-to-Patch attention, mean of 6 layers x 6 heads (shared scale)",
             fontsize=12, y=1.0)
save(os.path.join(OUT, "03_attention_side_by_side.png"), fig)

# ---------------- 04 metric bars ----------------
metrics = [("p_fake", "p_fake (model output)", 1.0),
           ("cutoff", "Spectral cutoff (kHz)", 22.0),
           ("dyn", "Dynamic range (dB)", None),
           ("flat", "Spectral flatness", None)]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), dpi=120)
xx = np.arange(len(items))
n_real = sum(1 for r in items if r["gt"] == "Real")
n_ai = len(items) - n_real
for ax, (k, lab, vmax_k) in zip(axes, metrics):
    vals = [r[k] for r in items]
    colors = ["#d62728" if r["gt"] == "Real" else "#1f77b4" for r in items]
    b = ax.bar(xx, vals, color=colors)
    for bx, v in zip(b, vals):
        ax.text(bx.get_x() + bx.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_title(lab, fontsize=10)
    ax.set_xticks(xx)
    ax.set_xticklabels([("R" if r["gt"] == "Real" else "A") + str(i + 1)
                        for i, r in enumerate(items)], fontsize=7.5, rotation=90)
    if vmax_k:
        ax.set_ylim(0, vmax_k)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.26)
fig.suptitle(f"Discriminative metrics: red = real (n={n_real}), blue = AI-generated (n={n_ai})\n"
             f"R1-R{n_real} and A1-A{n_ai} follow the ordering in the legend of panel 01",
             fontsize=10.5, y=1.06)
save(os.path.join(OUT, "04_metric_bars.png"), fig)

# csv
import csv
with open(os.path.join(OUT, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["display_name", "ground_truth", "p_fake", "spectral_cutoff_khz",
                "dynamic_range_db", "spectral_flatness"])
    for r in items:
        w.writerow([r["name"], r["gt"], f"{r['p_fake']:.4f}", f"{r['cutoff']:.2f}",
                    f"{r['dyn']:.1f}", f"{r['flat']:.5f}"])

print(f"\nsaved 4 figures + metrics.csv -> {OUT}")
