import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
"""为 test/image 每张图单独导出各模块特征图到独立文件夹

输出: reports/image_featuremaps/<图像名>/
    01_input.png                    原图 (RGB)
    02_clip_input.png               CLIP 预处理输入 224x224
    10_attn_rollout_all.png         CLIP 12层注意力 Rollout (14x14)
    11_attn_rollout_L3.png          前 3 层 Rollout
    12_attn_rollout_L6.png          前 6 层 Rollout
    13_attn_overlay.png             Rollout 叠加在原图上
    14_patch_token_norm.png         ViT patch token 特征模长 (14x14)
    15_patch_token_pca.png          ViT patch token PCA 前3主成分 (RGB)
    20_msff_scale_weights.png       MSFF 尺度注意力权重
    21_msff_feat_L3.png             MSFF L3 投影特征 (768-d)
    22_msff_feat_L6.png             MSFF L6 投影特征
    23_msff_feat_L12.png            MSFF L12 投影特征
    24_msff_global_fused.png        MSFF 融合输出
    30_srm_k0.png ... k8.png        SRM 9 个原始核响应
    31_srm_proj_c00.png ... c15.png SRM 1x1 投影 16 通道
    40_fft_magnitude.png            FFT 对数幅度谱
    41_fft_phase.png                FFT 相位谱
    50_forensics_feat.png           局部取证支路输出 (768-d)
    51_f_img_fused.png              全局+局部融合特征 f_img (768-d)
    meta.json                       该图的指标与文件名清单
"""
import os, glob, json, math
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "reports/image_featuremaps"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = (3, 6, 12)
FEAT_SHAPE = (24, 32)          # 768 = 24 x 32

os.makedirs(OUT, exist_ok=True)


# ---------------- 模型 ----------------
from models.image_branch import ImageBranch

ck = torch.load("checkpoints/finetuned/finetuned.pt", map_location="cpu", weights_only=False)
br = ImageBranch(device=DEV)
br.clip_model.load_state_dict(ck["clip_vision"])
br.adapter.load_state_dict(ck["adapter"])
br.msff.load_state_dict(ck["msff"])
br.forensics.load_state_dict(ck["forensics"])
br.classifier.load_state_dict(ck["classifier"])
br.global_local_fusion.load_state_dict(ck["global_local_fusion"])
br.eval()
br.clip_model.config._attn_implementation = "eager"
br.clip_model.attn_implementation = "eager"
print(f"device={DEV}  ckpt epoch={ck.get('epoch')}  val_acc={ck.get('val_acc'):.4f}")


# ---------------- 工具 ----------------
def norm01(a, lo=2, hi=98):
    a = np.asarray(a, dtype=np.float32)
    l, h = np.percentile(a, lo), np.percentile(a, hi)
    if h - l < 1e-9:
        h = l + 1e-9
    return np.clip((a - l) / (h - l), 0, 1)


def norm_signed(a, p=98):
    a = np.asarray(a, dtype=np.float32)
    s = np.percentile(np.abs(a), p) + 1e-9
    return np.clip(a / s, -1, 1)


def save_map(path, arr, cmap="inferno", size=None, signed=False):
    a = norm_signed(arr) if signed else norm01(arr)
    if size is not None:
        a = cv2.resize(a, size, interpolation=cv2.INTER_CUBIC)
    plt.imsave(path, a, cmap=cmap, vmin=-1 if signed else 0, vmax=1)


def save_vec(path, vec, cmap="viridis"):
    """768 维特征向量 -> 24x32 热力图"""
    a = np.asarray(vec, dtype=np.float32).reshape(FEAT_SHAPE)
    plt.imsave(path, norm01(a), cmap=cmap, vmin=0, vmax=1)


def attention_rollout(attentions):
    I = torch.eye(attentions[0].shape[-1], device=attentions[0].device)
    rollout = I.unsqueeze(0)
    for att in attentions:
        a = att.mean(dim=1) + I
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(a, rollout)
    n = int(round(math.sqrt(rollout.shape[-1] - 1)))
    return rollout[0, 0, 1:].reshape(n, n).cpu().numpy()


def patch_pca_rgb(tokens):
    """tokens: [N, D] -> [14,14,3] RGB (PCA 前3主成分)"""
    X = tokens - tokens.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = X @ Vt[:3].T                       # [N, 3]
    n = int(round(math.sqrt(tokens.shape[0])))
    out = np.zeros((n, n, 3), np.float32)
    for c in range(3):
        out[:, :, c] = norm01(pcs[:, c].reshape(n, n))
    return out


# ---------------- 主循环 ----------------
paths = sorted(glob.glob("test/image/*"))
summary = []

for i, p in enumerate(paths, 1):
    stem = os.path.splitext(os.path.basename(p))[0]
    d = os.path.join(OUT, stem)
    os.makedirs(d, exist_ok=True)

    bgr = cv2.imread(p)
    if bgr is None:
        bgr = np.array(Image.open(p).convert("RGB"))[:, :, ::-1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = br.preprocess(bgr)

    with torch.no_grad():
        o = br.clip_model(pixel_values=x, output_hidden_states=True, output_attentions=True)
        hs = o.hidden_states
        ms_feats = [br.adapter(hs[min(L, len(hs) - 1)][:, 0, :].float()) for L in LAYERS]
        f_global = br.msff(ms_feats)
        f_fore = br.forensics(x)
        f_img = br.global_local_fusion(torch.cat([f_global, f_fore], dim=-1))
        p_img = F.softmax(br.classifier(f_img), dim=-1)

        proj = [pr(f) for pr, f in zip(br.msff.projections, ms_feats)]
        msff_w = br.msff.scale_attention(torch.cat(proj, dim=-1))[0].float().cpu().numpy()

        pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        srm_raw = F.conv2d(pad, br.forensics.srm.kernels, groups=3)[0].float().cpu().numpy()
        srm_proj = br.forensics.srm(x)[0].float().cpu().numpy()

        gray = x.mean(dim=1)[0]
        fq = torch.fft.fftshift(torch.fft.fft2(gray))
        mag = torch.log(torch.abs(fq) + 1.0).cpu().numpy()
        phase = torch.angle(fq).cpu().numpy()

        att_all = attention_rollout(o.attentions)
        att_3 = attention_rollout(o.attentions[:3])
        att_6 = attention_rollout(o.attentions[:6])
        patch_tok = hs[-1][0, 1:, :].float().cpu().numpy()      # [196, 768]
        tok_norm = np.linalg.norm(patch_tok, axis=-1)

        p_fake = float(p_img[0, 1])
        p_real = float(p_img[0, 0])
        f_global_np = f_global[0].float().cpu().numpy()
        f_fore_np = f_fore[0].float().cpu().numpy()
        f_img_np = f_img[0].float().cpu().numpy()
        ms_np = [f[0].float().cpu().numpy() for f in ms_feats]

    # ---- 01 / 02 ----
    cv2.imwrite(os.path.join(d, "01_input.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    xin = x[0].float().cpu().permute(1, 2, 0).numpy()
    xin = np.clip(xin * np.array([0.26862954, 0.26130258, 0.27577711])
                  + np.array([0.48145466, 0.4578275, 0.40821073]), 0, 1)
    plt.imsave(os.path.join(d, "02_clip_input.png"), xin)

    # ---- 10~13 注意力 ----
    save_map(os.path.join(d, "10_attn_rollout_all.png"), att_all, "jet", size=(224, 224))
    save_map(os.path.join(d, "11_attn_rollout_L3.png"), att_3, "jet", size=(224, 224))
    save_map(os.path.join(d, "12_attn_rollout_L6.png"), att_6, "jet", size=(224, 224))
    hm = cv2.applyColorMap((norm01(att_all) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.resize(hm, (224, 224), interpolation=cv2.INTER_CUBIC)
    base = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(d, "13_attn_overlay.png"), cv2.addWeighted(base, 0.55, hm, 0.45, 0))

    # ---- 14 / 15 patch token ----
    save_map(os.path.join(d, "14_patch_token_norm.png"), tok_norm.reshape(14, 14), "viridis")
    plt.imsave(os.path.join(d, "15_patch_token_pca.png"), patch_pca_rgb(patch_tok))

    # ---- 20~24 MSFF ----
    fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=120)
    b = ax.bar([f"L{L}" for L in LAYERS], msff_w, color=["#7986cb", "#4fc3f7", "#4db6ac"])
    for bb, v in zip(b, msff_w):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(0.6, float(msff_w.max()) * 1.25))
    ax.set_ylabel("weight")
    ax.set_title("MSFF scale attention")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "20_msff_scale_weights.png"))
    plt.close(fig)

    for fn, v in zip(["21_msff_feat_L3.png", "22_msff_feat_L6.png", "23_msff_feat_L12.png"],
                     ms_np):
        save_vec(os.path.join(d, fn), v)
    save_vec(os.path.join(d, "24_msff_global_fused.png"), f_global_np)

    # ---- 30 SRM 原始 9 核（按核类型分别归一化）----
    for k in range(3):
        idx = [k, k + 3, k + 6]
        s = np.percentile(np.abs(np.stack([srm_raw[j] for j in idx])), 98) + 1e-9
        for j in idx:
            save_map(os.path.join(d, f"30_srm_k{j}.png"),
                     np.clip(srm_raw[j] / s, -1, 1), "coolwarm", signed=True)

    # ---- 31 SRM 投影 16 通道 ----
    for c in range(srm_proj.shape[0]):
        save_map(os.path.join(d, f"31_srm_proj_c{c:02d}.png"), srm_proj[c], "magma")

    # ---- 40 / 41 FFT ----
    save_map(os.path.join(d, "40_fft_magnitude.png"), mag, "inferno")
    save_map(os.path.join(d, "41_fft_phase.png"), phase, "twilight")

    # ---- 50 / 51 特征向量 ----
    save_vec(os.path.join(d, "50_forensics_feat.png"), f_fore_np, "magma")
    save_vec(os.path.join(d, "51_f_img_fused.png"), f_img_np, "cividis")

    files = sorted(os.listdir(d))
    meta = {
        "image": os.path.basename(p),
        "path": p,
        "p_fake": p_fake,
        "p_real": p_real,
        "pred": "AI" if p_fake > 0.5 else "Real",
        "msff_weights_L3_L6_L12": [float(v) for v in msff_w],
        "feat_norm": {"global": float(np.linalg.norm(f_global_np)),
                      "forensics": float(np.linalg.norm(f_fore_np)),
                      "fused": float(np.linalg.norm(f_img_np))},
        "n_files": len(files),
        "files": files,
    }
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    summary.append({"image": os.path.basename(p), "dir": stem, "p_fake": p_fake,
                    "pred": meta["pred"], "n_files": len(files)})
    print(f"[{i}/{len(paths)}] {stem:44s} p_fake={p_fake:.4f}  files={len(files)}", flush=True)

# 顶层索引
import csv
with open(os.path.join(OUT, "index.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["image", "dir", "p_fake", "pred", "n_files"])
    w.writeheader()
    w.writerows(summary)

tot = sum(s["n_files"] for s in summary)
sz = sum(os.path.getsize(os.path.join(dp, fn))
         for dp, _, fns in os.walk(OUT) for fn in fns) / 1024 ** 2
print(f"done: {len(summary)} images, {tot} png files, {sz:.1f} MB -> {OUT}")
