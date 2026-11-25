"""
Test-set image real/fake detection + detection-process visualization
------------------------------------------------
Model:  ImageDetectorViT (DeiT-Small backbone + MLP classifier, checkpoints/image_finetuned.pt)
Input:  RGB, 224x224, normalized to [-1, 1]  (same pipeline as train.py / finetune_gt.py)
Output: outputs/image_detection_results1025/   (new result folder)
    image_results_<ts>.json / .csv   per-image detection results
    01_p_fake_distribution.png       per-image AI probability
    02_all_images_grid.png           all images with verdict
    03_confusion_matrix.png          confusion matrix + verdict split
    04_errors_analysis.png           error cases (only when errors exist)
    process/<image>_process.png      detection-process figure per image
    image_detection_report.html      summary report
"""

import sys
import json
import time
import base64
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, str(PROJECT_ROOT))
from train import ImageDetectorViT

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

TEST_DIR = PROJECT_ROOT / 'test' / 'image'
CKPT = PROJECT_ROOT / 'checkpoints' / 'image_finetuned.pt'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'image_detection_results1025'
PROC_DIR = OUT_DIR / 'process'
THRESHOLD = 0.5          # image-branch decision threshold
FFT_BAND = 0.15          # low-frequency band half-width for the FFT heuristic
COLOR_AI, COLOR_REAL = '#EF4444', '#10B981'
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ============================================================
# Model
# ============================================================

def load_model():
    model = ImageDetectorViT(num_classes=2, pretrained=False)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def preprocess(img_bgr):
    """BGR -> RGB, 224x224, [-1,1]. Returns (tensor[1,3,224,224], rgb uint8)"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    t = torch.tensor(rgb).permute(2, 0, 1).float() / 255.0
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0), rgb


# ============================================================
# Heuristics / explainability
# ============================================================

def fft_high_ratio(gray, band=FFT_BAND):
    """Ratio of high-frequency energy (energy outside the central low-frequency region)"""
    fft_s = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(fft_s)
    h, w = mag.shape
    low = mag[h // 2 - int(h * band):h // 2 + int(h * band),
              w // 2 - int(w * band):w // 2 + int(w * band)].sum()
    return float(1 - low / (mag.sum() + 1e-10))


def saliency_map(model, t):
    """Gradient saliency of p_fake w.r.t. the input -> shows evidence regions"""
    t = t.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model(t)
        p = F.softmax(logits, dim=1)[0, 1]
        model.zero_grad(set_to_none=True)
        p.backward()
    s = t.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
    return (s - s.min()) / (s.max() - s.min() + 1e-8)


def srm_residual(gray):
    """High-pass residual via a 4-neighbour Laplacian kernel"""
    k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)
    return cv2.filter2D(gray, -1, k)


# ============================================================
# Visualization
# ============================================================

def save_fig(fig, path):
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def viz_process(rec, path):
    """Per-image process figure: Input | Saliency | FFT | SRM | Local variance | Probability"""
    rgb = rec['rgb']
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
    fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))) + 1)
    srm = srm_residual(g)
    blur = cv2.GaussianBlur(g, (31, 31), 0)
    lv = cv2.GaussianBlur((g - blur) ** 2, (31, 31), 0)

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))

    axes[0].imshow(rgb)
    axes[0].set_title('Input', fontsize=10)

    axes[1].imshow(rgb)
    axes[1].imshow(rec['saliency'], cmap='jet', alpha=0.45)
    axes[1].set_title('Saliency (gradient)', fontsize=10)

    axes[2].imshow(fft_mag, cmap='magma')
    axes[2].set_title(f'FFT spectrum\nHF ratio={rec["freq_ratio"]:.3f}', fontsize=10)

    axes[3].imshow(np.clip(srm, -30, 30), cmap='coolwarm')
    axes[3].set_title('SRM residual', fontsize=10)

    axes[4].imshow(lv, cmap='viridis')
    axes[4].set_title('Local variance', fontsize=10)

    axes[5].bar(['REAL', 'AI'], [1 - rec['p_fake'], rec['p_fake']],
                color=[COLOR_REAL, COLOR_AI], edgecolor='white')
    axes[5].axhline(THRESHOLD, color='gray', ls='--', linewidth=1.5)
    axes[5].set_ylim(0, 1)
    axes[5].set_ylabel('Probability')
    axes[5].set_title(f'p_fake={rec["p_fake"]:.4f}', fontsize=10)

    for ax in axes[:5]:
        ax.axis('off')

    verdict = 'AI-GENERATED' if rec['pred_fake'] else 'REAL'
    gt_str = 'REAL' if rec['gt_real'] else 'AI'
    mark = 'CORRECT' if rec['correct'] else 'WRONG'
    color = COLOR_AI if rec['pred_fake'] else COLOR_REAL
    fig.suptitle(f'[{verdict}]  {rec["file_name"]}   |   GT: {gt_str} ({mark})',
                 fontsize=12, fontweight='bold', color=color)
    plt.tight_layout()
    save_fig(fig, path)


def viz_summary(results, out_dir, accuracy):
    total = len(results)
    names = [r['file_name'] for r in results]
    p_fakes = [r['p_fake'] for r in results]
    colors = [COLOR_AI if r['pred_fake'] else COLOR_REAL for r in results]
    edge_c = ['#000000' if not r['correct'] else '#FFFFFF' for r in results]
    edge_w = [1.6 if not r['correct'] else 0 for r in results]

    # 1) p_fake distribution
    fig, ax = plt.subplots(figsize=(max(12, total * 0.6), 5))
    ax.bar(range(total), p_fakes, color=colors, edgecolor=edge_c, linewidth=edge_w)
    ax.axhline(THRESHOLD, color='gray', ls='--', linewidth=1.5, label=f'threshold {THRESHOLD}')
    ax.set_xticks(range(total))
    ax.set_xticklabels([n[:15] for n in names], rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('AI probability p_fake', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Image detection - DeiT-ViT (finetuned)  |  accuracy vs GT {accuracy:.1f}%',
                 fontsize=13, fontweight='bold')
    ax.legend([f'threshold {THRESHOLD}', 'red = judged AI', 'green = judged REAL',
               'black edge = wrong'], fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_dir / '01_p_fake_distribution.png')

    # 2) grid of all images
    ncols = 5
    nrows = int(np.ceil(total / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    flat = axes.flatten()
    for i, r in enumerate(results):
        ax = flat[i]
        ax.imshow(r['rgb'])
        color = COLOR_AI if r['pred_fake'] else COLOR_REAL
        gt_str = 'REAL' if r['gt_real'] else 'AI'
        mark = 'OK' if r['correct'] else 'MISS'
        ax.set_title(f"{r['file_name'][:18]}\np_fake={r['p_fake']:.4f}  [GT {gt_str} {mark}]",
                     fontsize=8, color=color, fontweight='bold')
        ax.axis('off')
    for j in range(total, len(flat)):
        flat[j].axis('off')
    fig.suptitle(f'All {total} test images | DeiT-ViT finetuned', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir / '02_all_images_grid.png')

    # 3) confusion matrix + verdict pie
    cm = np.zeros((2, 2), dtype=int)
    for r in results:
        cm[1 if r['gt_real'] else 0, 1 if r['pred_fake'] else 0] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax1.imshow(cm, cmap='Blues', aspect='equal')
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['REAL (pred)', 'AI (pred)'], fontsize=10)
    ax1.set_yticklabels(['REAL (GT)', 'AI (GT)'], fontsize=10)
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20,
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontweight='bold')
    ax1.set_title('Confusion matrix (vs filename GT)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax1, fraction=0.046)

    n_ai = sum(1 for r in results if r['pred_fake'])
    n_real = total - n_ai
    ax2.pie([n_ai, n_real], labels=[f'AI-generated ({n_ai})', f'Real ({n_real})'],
            colors=[COLOR_AI, COLOR_REAL], autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax2.set_title(f'Verdict split (n = {total})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, out_dir / '03_confusion_matrix.png')

    # 4) error analysis
    missed = [r for r in results if not r['correct']]
    if missed:
        fig, axes = plt.subplots(len(missed), 4, figsize=(16, 4 * len(missed)))
        axes = np.atleast_2d(axes)
        for row, r in enumerate(missed):
            g = cv2.cvtColor(r['rgb'], cv2.COLOR_RGB2GRAY).astype(float)
            axes[row, 0].imshow(r['rgb']); axes[row, 0].axis('off')
            axes[row, 0].set_title(f"{r['file_name'][:20]}\np_fake={r['p_fake']:.4f}",
                                   fontsize=8, color=COLOR_AI, fontweight='bold')
            axes[row, 1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))) + 1), cmap='magma')
            axes[row, 1].axis('off'); axes[row, 1].set_title('FFT spectrum', fontsize=8)
            axes[row, 2].imshow(np.clip(srm_residual(g), -30, 30), cmap='coolwarm')
            axes[row, 2].axis('off'); axes[row, 2].set_title('SRM residual', fontsize=8)
            axes[row, 3].bar(['REAL', 'AI'], [1 - r['p_fake'], r['p_fake']],
                             color=[COLOR_REAL, COLOR_AI])
            axes[row, 3].axhline(THRESHOLD, color='gray', ls='--')
            axes[row, 3].set_ylim(0, 1)
            axes[row, 3].set_title('Probability', fontsize=8)
        fig.suptitle(f'Misclassified images: {len(missed)}', fontsize=14,
                     fontweight='bold', color=COLOR_AI)
        plt.tight_layout()
        save_fig(fig, out_dir / '04_errors_analysis.png')
    return missed


def build_html(results, out_dir, ts, accuracy, missed):
    def b64img(p):
        return base64.b64encode(Path(p).read_bytes()).decode()

    n_ai = sum(1 for r in results if r['pred_fake'])
    n_real = len(results) - n_ai
    n_right = sum(1 for r in results if r['correct'])

    rows = ''
    for r in results:
        cls = 'ai' if r['pred_fake'] else 'real'
        pred = 'AI' if r['pred_fake'] else 'REAL'
        gt = 'REAL' if r['gt_real'] else 'AI'
        mark = '<span class="ok">OK</span>' if r['correct'] else '<span class="bad">MISS</span>'
        rows += (f'<tr><td>{r["file_name"]}</td>'
                 f'<td>{r["p_fake"]:.4f}</td><td class="{cls}">{pred}</td>'
                 f'<td>{gt}</td><td>{mark}</td><td>{r["freq_ratio"]:.4f}</td>'
                 f'<td>{r["inference_ms"]}ms</td>'
                 f'<td><a href="process/{r["proc_name"]}" target="_blank">view</a></td></tr>')

    blocks = ''
    for r in results:
        cls = 'ai' if r['pred_fake'] else 'real'
        pred = 'AI' if r['pred_fake'] else 'REAL'
        blocks += f"""<div class="card">
  <h3>{r['file_name']}</h3>
  <span class="badge {cls}">{pred} &middot; p_fake={r['p_fake']:.4f}</span>
  <a href="process/{r['proc_name']}" target="_blank"><img src="process/{r['proc_name']}"></a>
</div>"""

    err_block = ''
    if missed:
        err_block = ('<div class="card"><h3>Misclassified images</h3>'
                     '<a href="04_errors_analysis.png" target="_blank">'
                     '<img src="04_errors_analysis.png"></a></div>')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Image Real/Fake Detection Report</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f8fafc;color:#1E2761;margin:0}}
.top{{background:linear-gradient(135deg,#1E2761,#065A82);color:#fff;padding:28px 40px}}
.top h1{{font-size:28px;margin:0 0 6px}} .top .sub{{color:#CADCFC;font-size:13px}}
.ov{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;padding:20px 40px;background:#fff;border-bottom:1px solid #e2e8f0}}
.ov div{{background:#f1f5f9;border-radius:10px;padding:14px;border-top:3px solid #065A82}}
.ov .k{{font-size:12px;color:#64748b}} .ov .v{{font-size:26px;font-weight:bold}}
main{{padding:28px 40px;max-width:1500px;margin:auto}}
.card{{background:#fff;border-radius:12px;padding:20px;margin-bottom:22px;box-shadow:0 1px 3px rgba(0,0,0,.07)}}
.card h3{{font-size:15px;margin:0 0 8px;word-break:break-all}}
.card img{{width:100%;border-radius:8px;margin-top:10px}}
.badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:bold;color:#fff}}
.badge.ai{{background:#EF4444}} .badge.real{{background:#10B981}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff;border-radius:10px;overflow:hidden}}
th,td{{padding:9px 10px;text-align:left;border-bottom:1px solid #e2e8f0}}
th{{background:#f1f5f9;font-weight:600}}
.ai{{color:#EF4444;font-weight:bold}} .real{{color:#10B981;font-weight:bold}}
.ok{{color:#10B981;font-weight:bold}} .bad{{color:#EF4444;font-weight:bold}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.note{{background:#fff7ed;border-left:4px solid #F59E0B;padding:14px 18px;border-radius:8px;font-size:13px;margin-bottom:22px}}
</style></head><body>
<div class="top">
  <h1>Test-Set Image Real/Fake Detection Report</h1>
  <div class="sub">ImageDetectorViT (DeiT-Small) &middot; checkpoints/image_finetuned.pt &middot; detected at {ts}</div>
</div>
<div class="ov">
  <div><div class="k">Images scanned</div><div class="v">{len(results)}</div></div>
  <div><div class="k">Judged AI-generated</div><div class="v" style="color:#EF4444">{n_ai}</div></div>
  <div><div class="k">Judged Real</div><div class="v" style="color:#10B981">{n_real}</div></div>
  <div><div class="k">Accuracy vs GT</div><div class="v">{accuracy:.1f}% <span style="font-size:14px;color:#64748b">({n_right}/{len(results)})</span></div></div>
</div>
<main>
  <div class="note"><b>Caveat:</b> the GT labels are inferred from filenames (<code>real_*</code> = Real, otherwise AI),
    and <code>image_finetuned.pt</code> was fine-tuned on part of this same test set, so the reported accuracy is optimistic.</div>
  <div class="card"><h3>Per-image AI probability</h3>
    <img src="data:image/png;base64,{b64img(out_dir / '01_p_fake_distribution.png')}"></div>
  <div class="grid2">
    <div class="card"><h3>Confusion matrix</h3>
      <img src="data:image/png;base64,{b64img(out_dir / '03_confusion_matrix.png')}"></div>
    <div class="card"><h3>All images</h3>
      <a href="02_all_images_grid.png" target="_blank"><img src="02_all_images_grid.png"></a></div>
  </div>
  {err_block}
  <div class="card"><h3>Per-image results</h3>
    <table><tr><th>File</th><th>p_fake</th><th>Verdict</th><th>GT</th><th>Result</th>
      <th>HF ratio</th><th>Time</th><th>Process</th></tr>
      {rows}</table></div>
  <h2 style="color:#1E2761">Detection process per image</h2>
  {blocks}
</main></body></html>"""

    out = out_dir / 'image_detection_report.html'
    out.write_text(html, encoding='utf-8')
    return out


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUT_DIR}\n")

    print("Loading image model...")
    model = load_model()
    print(f"  checkpoints/{CKPT.name}\n")

    files = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"Images found: {len(files)}\n")

    results = []
    print(f"{'='*78}")
    print(f"{'file':<30} {'p_fake':>8} {'verdict':>8} {'GT':>6} {'ok':>5} {'HF ratio':>9}")
    print(f"{'='*78}")

    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  WARN: cannot read {p.name}, skipping")
            continue

        t, rgb = preprocess(img)
        t0 = time.time()
        with torch.no_grad():
            logits = model(t.to(DEVICE))
            p_fake = float(F.softmax(logits, dim=1)[0, 1].item())
        infer_ms = (time.time() - t0) * 1000

        pred_fake = p_fake > THRESHOLD
        gt_real = p.stem.lower().startswith('real_')
        correct = pred_fake == (not gt_real)
        freq = fft_high_ratio(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32))
        sal = saliency_map(model, t.to(DEVICE))

        proc_name = f"{i:02d}_{p.stem}_process.png"
        rec = {
            'file_name': p.name,
            'p_fake': round(p_fake, 4),
            'pred_fake': bool(pred_fake),
            'gt_real': bool(gt_real),
            'correct': bool(correct),
            'freq_ratio': round(float(freq), 4),
            'inference_ms': round(infer_ms, 1),
            'rgb': rgb,
            'saliency': sal,
            'proc_name': proc_name,
        }
        results.append(rec)
        viz_process(rec, PROC_DIR / proc_name)

        pred_str = 'AI' if pred_fake else 'REAL'
        gt_str = 'REAL' if gt_real else 'AI'
        print(f"  {p.name:<28} {p_fake:>8.4f} {pred_str:>8} {gt_str:>6} "
              f"{('OK' if correct else 'MISS'):>5} {freq:>9.4f}")

    if not results:
        print("No detection results available")
        return

    total = len(results)
    right = sum(1 for r in results if r['correct'])
    accuracy = right / total * 100
    print(f"\nAccuracy vs GT: {right}/{total} = {accuracy:.1f}%")

    print("\nGenerating summary visualizations...")
    missed = viz_summary(results, OUT_DIR, accuracy)

    ts_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    dump = [{k: v for k, v in r.items() if k not in ('rgb', 'saliency')} for r in results]
    (OUT_DIR / f'image_results_{ts_file}.json').write_text(
        json.dumps({'timestamp': ts, 'checkpoint': CKPT.name, 'threshold': THRESHOLD,
                    'accuracy_pct': round(accuracy, 2), 'results': dump},
                   indent=2, ensure_ascii=False), encoding='utf-8')

    with open(OUT_DIR / f'image_results_{ts_file}.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['file_name', 'p_fake', 'verdict', 'gt', 'correct', 'hf_ratio', 'inference_ms'])
        for r in results:
            w.writerow([r['file_name'], f"{r['p_fake']:.4f}",
                        'AI' if r['pred_fake'] else 'REAL',
                        'REAL' if r['gt_real'] else 'AI',
                        'OK' if r['correct'] else 'MISS',
                        f"{r['freq_ratio']:.4f}", r['inference_ms']])

    html = build_html(results, OUT_DIR, ts, accuracy, missed)

    n_ai = sum(1 for r in results if r['pred_fake'])
    print("\n" + "=" * 70)
    print("Detection summary")
    print("=" * 70)
    print(f"  Images total : {total}")
    print(f"  AI-generated : {n_ai}")
    print(f"  Real         : {total - n_ai}")
    print(f"  Wrong        : {len(missed)}")
    print(f"  Process figs : {len(list(PROC_DIR.glob('*.png')))}")
    print(f"\nResult folder: {OUT_DIR}")
    print(f"HTML report  : {html}")


if __name__ == '__main__':
    main()
