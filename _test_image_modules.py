"""test/image 全模块测试可视化

模型: models.image_branch.ImageBranch (CLIP ViT-B/16 + MSFF + SRM/FFT 局部取证 + G-Fuse)
权重: checkpoints/finetuned/finetuned.pt
输出: reports/image_modules/index.html + figs/*.png
"""
import os, glob, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "reports/image_modules"
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "checkpoints/finetuned/finetuned.pt"
THR = 0.5
LAYERS = (3, 6, 12)

# ---------------- 模型加载 ----------------
from models.image_branch import ImageBranch
from models.fusion import CrossModalConsistency, GatedFusion

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
br = ImageBranch(device=device)
br.clip_model.load_state_dict(ck["clip_vision"])
br.adapter.load_state_dict(ck["adapter"])
br.msff.load_state_dict(ck["msff"])
br.forensics.load_state_dict(ck["forensics"])
br.classifier.load_state_dict(ck["classifier"])
br.global_local_fusion.load_state_dict(ck["global_local_fusion"])
br.eval()

# SDPA 不返回注意力权重 → 切换到 eager 以获取 12 层 attention
br.clip_model.config._attn_implementation = "eager"
br.clip_model.attn_implementation = "eager"

# 融合模块（该 ckpt 未包含 → 随机初始化，仅作结构展示）
xcons = CrossModalConsistency(embed_dim=768).to(device).eval()
gfuse = GatedFusion(embed_dim=768, num_classes=2, num_modalities=3).to(device).eval()

CKPT_META = {"epoch": ck.get("epoch"), "val_acc": ck.get("val_acc"),
             "train_config": ck.get("train_config")}

print(f"device={device} | ckpt epoch={CKPT_META['epoch']} val_acc={CKPT_META['val_acc']}")


# ---------------- 工具函数 ----------------
def norm01(a, lo=2, hi=98):
    a = np.asarray(a, dtype=np.float32)
    l, h = np.percentile(a, lo), np.percentile(a, hi)
    if h - l < 1e-9:
        h = l + 1e-9
    return np.clip((a - l) / (h - l), 0, 1)


def montage(arrs, rows, cols, size=72, pad=3, signed=False):
    """把若干 2D 数组拼成一张图；signed=True 时零中心归一化到 [-1,1]"""
    canvas = np.zeros((rows * (size + pad) + pad, cols * (size + pad) + pad), np.float32)
    for i, a in enumerate(arrs[: rows * cols]):
        r, c = divmod(i, cols)
        a = np.asarray(a, dtype=np.float32)
        if signed:
            s = np.percentile(np.abs(a), 98) + 1e-9
            a = np.clip(a / s, -1, 1)
        else:
            a = norm01(a)
        im = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
        y0 = pad + r * (size + pad)
        x0 = pad + c * (size + pad)
        canvas[y0:y0 + size, x0:x0 + size] = im
    return canvas


def srm_arrays(srm_raw):
    """按 SRM 核类型分别归一化，保证 9 个响应可比且都能看清结构
    顺序: [k0c0,k1c0,k2c0, k0c1,k1c1,k2c1, k0c2,k1c2,k2c2]
    """
    out = [None] * 9
    for k in range(3):
        idx = [k, k + 3, k + 6]
        stack = np.stack([srm_raw[i] for i in idx])
        s = np.percentile(np.abs(stack), 98) + 1e-9
        for i in idx:
            out[i] = np.clip(srm_raw[i] / s, -1, 1)
    return out


def attention_rollout(attentions, discard_ratio=0.0):
    """CLIP ViT 注意力 rollout -> 14x14 图"""
    I = torch.eye(attentions[0].shape[-1], device=attentions[0].device)
    rollout = I.unsqueeze(0)
    for att in attentions:
        a = att.mean(dim=1)                      # [1, N, N]
        a = a + I
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(a, rollout)
    mask = rollout[0, 0, 1:]                     # patch tokens
    n = int(round(math.sqrt(mask.shape[0])))
    return mask.reshape(n, n).cpu().numpy()


def fft_spectra(x_gray):
    f = torch.fft.fftshift(torch.fft.fft2(x_gray))
    mag = torch.log(torch.abs(f) + 1.0)
    phase = torch.angle(f)
    return mag.cpu().numpy(), phase.cpu().numpy()


def hf_ratio(mag, r_frac=0.25):
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = math.sqrt(h * h + w * w) * r_frac
    m = (yy - cy) ** 2 + (xx - cx) ** 2 > r * r
    tot = mag.sum() + 1e-9
    return float(mag[m].sum() / tot)


# ---------------- 逐图推理 + 可视化 ----------------
paths = sorted(glob.glob("test/image/*"))
records = []

t_all = time.time()
for idx, p in enumerate(paths):
    name = os.path.basename(p)
    bgr = cv2.imread(p)
    if bgr is None:
        bgr = np.array(Image.open(p).convert("RGB"))[:, :, ::-1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gt = 0 if name.startswith("real_") else 1      # calibration.json 的 ground_truth rule

    x = br.preprocess(bgr)                          # [1,3,224,224]

    with torch.no_grad():
        clip_out = br.clip_model(pixel_values=x, output_hidden_states=True,
                                 output_attentions=True)
        hs = clip_out.hidden_states
        ms_feats = [br.adapter(hs[min(L, len(hs) - 1)][:, 0, :].float()) for L in LAYERS]
        f_global = br.msff(ms_feats)
        f_forensics = br.forensics(x)
        f_img = br.global_local_fusion(torch.cat([f_global, f_forensics], dim=-1))
        logits = br.classifier(f_img)
        p_img = F.softmax(logits, dim=-1)

        # 融合模块（未训练）
        f_sync_d = xcons(f_img=f_img)
        fused = gfuse(f_img=f_img, f_sync=f_sync_d["f_sync"],
                      f_async=f_sync_d["f_async"],
                      m_img=torch.ones(1, 1, device=device))
        p_fake = float(p_img[0, 1].cpu())
        p_real = float(p_img[0, 0].cpu())

        # MSFF 尺度权重
        proj = [pr(f) for pr, f in zip(br.msff.projections, ms_feats)]
        concat = torch.cat(proj, dim=-1)
        msff_w = br.msff.scale_attention(concat).float()[0].cpu().numpy()

        # 三层特征余弦相似度
        zs = [F.normalize(f.float(), dim=-1)[0] for f in ms_feats]
        cos3 = np.array([[float(torch.dot(a, b)) for b in zs] for a in zs])

        # SRM / FFT
        pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        srm_raw = F.conv2d(pad, br.forensics.srm.kernels, groups=3)[0].float().cpu().numpy()
        srm_proj = br.forensics.srm(x)[0].float().cpu().numpy()
        xg = x.mean(dim=1, keepdim=True)[0, 0]
        mag, phase = fft_spectra(xg.float())

        # 特征模长
        n_glob = float(f_global.norm(dim=-1).float()[0])
        n_fore = float(f_forensics.norm(dim=-1).float()[0])
        n_fuse = float(f_img.norm(dim=-1).float()[0])
        cos_gl = float(F.cosine_similarity(f_global.float(), f_forensics.float(), dim=-1)[0])

        # 注意力 rollout
        att_last = attention_rollout(clip_out.attentions)
        att_mid = attention_rollout(clip_out.attentions[: 6])
        att_early = attention_rollout(clip_out.attentions[: 3])

        gates = fused["gates"].float()[0].cpu().numpy()
        rel = {k: float(v[0, 0].cpu()) for k, v in fused["reliability"].items()}
        p_fused = float(F.softmax(fused["p_fake"].float(), dim=-1)[0, 1])
    # ---- 绘图 ----
    disp = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    xin = (x[0].float().cpu().permute(1, 2, 0).numpy() * np.array([0.26862954, 0.26130258, 0.27577711])
           + np.array([0.48145466, 0.4578275, 0.40821073]))
    xin = np.clip(xin, 0, 1)

    fig, axes = plt.subplots(3, 4, figsize=(19, 13))
    fig.suptitle(f"{name}   |   p_fake={p_fake:.4f}   |   "
                 f"判定={'AI生成' if p_fake > THR else '真实'}   "
                 f"(启发式GT={'Fake' if gt == 1 else 'Real'})",
                 fontsize=15, y=0.995)

    ax = axes[0, 0]
    ax.imshow(disp); ax.set_title("1.  输入原图", fontsize=11); ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(xin); ax.set_title("2.  CLIP 预处理 224×224（归一化反算）", fontsize=11); ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(disp)
    ax.imshow(cv2.resize(norm01(att_last), (224, 224), interpolation=cv2.INTER_CUBIC),
              cmap="jet", alpha=0.55)
    ax.set_title("3.  CLIP 注意力 Rollout（全12层）", fontsize=11); ax.axis("off")

    ax = axes[0, 3]
    ax.barh(["p_real", "p_fake"], [p_real, p_fake],
            color=["#4caf50", "#e53935"])
    ax.set_xlim(0, 1.05)
    ax.axvline(THR, ls="--", c="k", lw=1)
    for i, v in enumerate([p_real, p_fake]):
        if v > 0.85:
            ax.text(v - 0.02, i, f"{v:.4f}", va="center", ha="right", fontsize=10,
                    color="w", fontweight="bold")
        else:
            ax.text(v + 0.02, i, f"{v:.4f}", va="center", fontsize=10)
    ax.set_title("4.  图像分类头 Φ_img 输出", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 0]
    ax.imshow(montage(srm_arrays(srm_raw), 3, 3), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("5.  SRM 高频残差\n行=RGB通道, 列=3个SRM核 (红/蓝=正/负)", fontsize=9.5); ax.axis("off")

    ax = axes[1, 1]
    ax.imshow(montage(list(srm_proj), 4, 4), cmap="magma")
    ax.set_title("6.  SRM 1×1 投影 16 通道", fontsize=10); ax.axis("off")

    ax = axes[1, 2]
    ax.imshow(mag, cmap="inferno")
    ax.set_title(f"7.  FFT 对数幅度谱 (高频占比={hf_ratio(mag):.3f})", fontsize=11); ax.axis("off")

    ax = axes[1, 3]
    ax.imshow(phase, cmap="twilight")
    ax.set_title("8.  FFT 相位谱", fontsize=11); ax.axis("off")

    ax = axes[2, 0]
    bars = ax.bar([f"L{L}" for L in LAYERS], msff_w,
                  color=["#7986cb", "#4fc3f7", "#4db6ac"])
    for b, v in zip(bars, msff_w):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, max(0.6, msff_w.max() * 1.25))
    ax.set_title("9.  MSFF 多尺度注意力权重", fontsize=11)
    ax.set_ylabel("权重")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2, 1]
    im = ax.imshow(cos3, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([f"L{L}" for L in LAYERS]); ax.set_yticklabels([f"L{L}" for L in LAYERS])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cos3[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="w" if abs(cos3[i, j]) > 0.6 else "k")
    ax.set_title("10.  CLIP 三层 CLS 特征余弦相似度", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2, 2]
    names = ["全局\n(MSFF)", "取证\n(SRM+FFT)", "融合\nf_img"]
    vals = [n_glob, n_fore, n_fuse]
    bars = ax.bar(names, vals, color=["#5c6bc0", "#26a69a", "#8d6e63"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xlabel(f"cos(全局,取证) = {cos_gl:.3f}", fontsize=9)
    ax.set_title("11.  特征模长", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2, 3]
    lab = ["img", "vid", "aud", "sync", "async"]
    ax.bar(lab, gates, color="#ffa726", alpha=0.85)
    for i, v in enumerate(gates):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(0.6, gates.max() * 1.3))
    ax.set_title("12.  G-Fuse 门控（未训练）", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.975], h_pad=2.0)
    fig_path = os.path.join(FIG, f"{os.path.splitext(name)[0]}.png")
    fig.savefig(fig_path, dpi=80, bbox_inches="tight")
    plt.close(fig)

    records.append({
        "name": name, "path": p, "gt": gt, "p_fake": p_fake, "p_real": p_real,
        "pred": int(p_fake > THR), "fig": os.path.basename(fig_path),
        "msff_w": msff_w.tolist(), "cos3": cos3.tolist(),
        "n_glob": n_glob, "n_fore": n_fore, "n_fuse": n_fuse, "cos_gl": cos_gl,
        "hf_ratio": hf_ratio(mag),
        "srm_std": float(srm_raw.std()), "srm_absmean": float(np.abs(srm_raw).mean()),
        "phase_std": float(phase.std()),
        "gates": gates.tolist(), "rel": rel, "p_fused": p_fused,
        "att_mid_mean": float(norm01(att_mid).mean()),
        "att_last_mean": float(norm01(att_last).mean()),
    })
    print(f"[{idx+1}/{len(paths)}] {name:45s} p_fake={p_fake:.4f} "
          f"{'OK' if (p_fake > THR) == (gt == 1) else 'X'}", flush=True)

print(f"inference done in {time.time()-t_all:.1f}s")

# ---------------- 汇总图表 + HTML ----------------
from _test_image_report import build
metrics = build(records, CKPT_META)
print("metrics:", metrics)
