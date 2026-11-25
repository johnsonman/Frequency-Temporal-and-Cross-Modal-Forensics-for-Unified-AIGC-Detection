"""
Test-set video real/fake detection + detection-process visualization
------------------------------------------------
Model:  VideoDetectorViT (DeiT-Small per-frame features + temporal self-attention)
Fusion: neural-network output + heuristic temporal/frequency features
Output: outputs/video_detection_results1025/   (new result folder)
    video_results_<ts>.json / .csv   per-video detection results
    01_summary_bars.png              per-video AI probability comparison
    02_summary_pie.png               AI / REAL verdict split
    03_fusion_breakdown.png          NN vs heuristic vs fused
    process/<video>_process.png      detection-process figure per video
    video_detection_report.html      summary report
"""

import sys
import json
import time
import base64
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
from train import VideoDetectorViT

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = PROJECT_ROOT / 'outputs' / 'video_detection_results1025'
PROC_DIR = OUT_DIR / 'process'
NUM_FRAMES = 16          # temporal sequence length expected by the model
THRESHOLD = 0.45         # video-branch decision threshold (few training samples, kept from project policy)
COLOR_AI, COLOR_REAL = '#EF4444', '#10B981'
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


# ============================================================
# Model
# ============================================================

def load_video_model():
    ckpt_path = PROJECT_ROOT / 'checkpoints' / 'video_best.pt'
    model = VideoDetectorViT(num_classes=2, pretrained=False).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        meta = f"epoch={ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', '?')}"
    else:
        model.load_state_dict(ckpt)
        meta = 'raw state_dict'
    model.eval()

    # Capture last-layer temporal attention weights (for process visualization)
    attn_store = {}

    def hook(_module, _inp, out):
        if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
            attn_store['w'] = out[1].detach().float().cpu()

    model.temporal_attn[-1].attn.register_forward_hook(hook)
    return model, meta, attn_store


# ============================================================
# Video reading / sampling
# ============================================================

def sample_frames(path, n=NUM_FRAMES):
    """Uniformly sample n frames; return (list of RGB frames, video metadata)"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    picked = []
    if total > 0:
        for i in np.linspace(0, total - 1, n, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, fr = cap.read()
            if ret:
                picked.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    else:  # frame count unavailable -> sequential read
        all_frames = []
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            all_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        total = len(all_frames)
        if total:
            for i in np.linspace(0, total - 1, n, dtype=int):
                picked.append(all_frames[int(i)])
    cap.release()

    meta = {
        'total_frames': total,
        'fps': round(fps, 2),
        'resolution': f'{w}x{h}',
        'duration_s': round(total / fps, 2) if fps > 0 else None,
        'sampled_frames': len(picked),
    }
    return picked, meta


# ============================================================
# Heuristic features (temporal consistency + frequency domain)
# ============================================================

def fft_high_ratio(gray, band=0.10):
    """Ratio of high-frequency energy (energy outside the central low-frequency region)"""
    fft_s = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(fft_s)
    h, w = mag.shape
    low = mag[h // 2 - int(h * band):h // 2 + int(h * band),
              w // 2 - int(w * band):w // 2 + int(w * band)].sum()
    return float(1 - low / (mag.sum() + 1e-10))


def compute_metrics(frames_small):
    """Per-frame metrics + global temporal-consistency metrics"""
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames_small]
    frame_diff, edge_diff, fft_ratio = [], [], []
    for i, g in enumerate(grays):
        fft_ratio.append(fft_high_ratio(g))
        if i > 0:
            frame_diff.append(float(cv2.absdiff(grays[i - 1], g).mean() / 255.0))
            e0 = cv2.Canny(grays[i - 1].astype(np.uint8), 50, 150).astype(np.float32)
            e1 = cv2.Canny(g.astype(np.uint8), 50, 150).astype(np.float32)
            edge_diff.append(float(cv2.absdiff(e0, e1).mean() / 255.0))

    fd = np.array(frame_diff) if frame_diff else np.array([0.0])
    ed = np.array(edge_diff) if edge_diff else np.array([0.0])
    temporal_consistency = float(1 - fd.std() / (fd.mean() + 1e-6))
    edge_consistency = float(1 - ed.std() / (ed.mean() + 1e-6))
    mean_fft = float(np.mean(fft_ratio))

    # AI-video traits: overly consistent motion / too-stable edges / low high-frequency energy
    p_heur = 0.4 * (1 - temporal_consistency) + 0.3 * (1 - edge_consistency) + 0.3 * (1 - mean_fft)
    p_heur = float(np.clip(p_heur, 0, 1))

    return {
        'frame_diff': fd.tolist(),
        'edge_diff': ed.tolist(),
        'fft_ratio': fft_ratio,
        'temporal_consistency': temporal_consistency,
        'edge_consistency': edge_consistency,
        'mean_fft_ratio': mean_fft,
        'p_heur': p_heur,
    }


# ============================================================
# Inference
# ============================================================

def predict_video(model, attn_store, frames):
    """frames: list of RGB frames -> NN probability + heuristic probability + fused verdict"""
    small = [cv2.resize(f, (224, 224)) for f in frames]
    if len(small) < 2:
        return None

    arr = np.stack(small)
    if len(arr) < NUM_FRAMES:
        arr = np.concatenate([arr, np.repeat(arr[-1:], NUM_FRAMES - len(arr), axis=0)], axis=0)
    else:
        arr = arr[:NUM_FRAMES]

    t = torch.tensor(arr).permute(0, 3, 1, 2).float() / 255.0
    t = (t - 0.5) / 0.5

    attn_store.pop('w', None)
    t0 = time.time()
    with torch.no_grad():
        logits = model(t.unsqueeze(0).to(DEVICE))
        p_nn = float(F.softmax(logits, dim=1)[0, 1].item())
    infer_ms = (time.time() - t0) * 1000

    attn = attn_store.get('w')
    attn = attn[0].numpy() if attn is not None else None

    metrics = compute_metrics(small)
    p_heur = metrics['p_heur']

    # Fusion: prefer NN probability; fall back to heuristic when below threshold
    p_fused = p_nn if p_nn >= THRESHOLD else max(p_nn, 1 - p_heur)
    is_ai = p_fused > THRESHOLD

    return {
        'p_nn': round(p_nn, 4),
        'p_heur': round(p_heur, 4),
        'p_fused': round(float(p_fused), 4),
        'is_ai': bool(is_ai),
        'confidence': round(float(max(p_fused, 1 - p_fused)), 4),
        'inference_ms': round(infer_ms, 1),
        'attention': attn,
        'metrics': metrics,
        'frames': small,
    }


# ============================================================
# Visualization
# ============================================================

def save_fig(fig, path):
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def viz_process(rec, path):
    """Per-video process figure: sampled frames | frame diffs | temporal curves | FFT | attention | verdict"""
    frames = rec['frames']
    met = rec['metrics']
    n = len(frames)

    show_idx = np.linspace(0, n - 1, min(8, n), dtype=int)
    show = [frames[i] for i in show_idx]

    diffs = []
    for a, b in zip(show[:-1], show[1:]):
        g0 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        diffs.append(cv2.absdiff(g0, g1))

    fig = plt.figure(figsize=(26, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 0.95, 1.25], hspace=0.22)

    # --- Row 1: sampled frames ---
    g0 = gs[0].subgridspec(1, len(show), wspace=0.04)
    for i, fr in enumerate(show):
        ax = fig.add_subplot(g0[0, i])
        ax.imshow(fr)
        ax.set_title(f'frame #{show_idx[i]}', fontsize=9)
        ax.axis('off')

    # --- Row 2: frame-difference heatmaps ---
    g1 = gs[1].subgridspec(1, len(show), wspace=0.04)
    for i in range(len(show)):
        ax = fig.add_subplot(g1[0, i])
        if i < len(diffs):
            ax.imshow(diffs[i], cmap='hot')
            ax.set_title(f'|d(frame#{i}->#{i+1})|', fontsize=8)
        ax.axis('off')

    # --- Row 3: 4 analysis panels ---
    g2 = gs[2].subgridspec(1, 4, wspace=0.28)

    ax = fig.add_subplot(g2[0, 0])
    x = np.arange(1, len(met['frame_diff']) + 1)
    ax.plot(x, met['frame_diff'], 'o-', color='#065A82', label='Frame difference')
    ax.plot(x, met['edge_diff'], 's-', color='#F59E0B', label='Edge difference')
    ax.set_xlabel('Frame-pair index')
    ax.set_ylabel('Difference')
    ax.set_title(f'Temporal consistency = {met["temporal_consistency"]:.3f}', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(g2[0, 1])
    ax.plot(np.arange(len(met['fft_ratio'])), met['fft_ratio'], 'o-', color='#7C3AED')
    ax.axhline(met['mean_fft_ratio'], color='gray', ls='--',
               label=f'mean = {met["mean_fft_ratio"]:.3f}')
    ax.set_xlabel('Frame index')
    ax.set_ylabel('FFT high-frequency ratio')
    ax.set_title('High-frequency energy', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(g2[0, 2])
    if rec['attention'] is not None:
        im = ax.imshow(rec['attention'], cmap='viridis', aspect='auto')
        ax.set_xlabel('Attended frame (Key)')
        ax.set_ylabel('Query frame')
        ax.set_title('Temporal self-attention', fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.85)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.axis('off')

    ax = fig.add_subplot(g2[0, 3])
    names = ['NN', 'Heuristic', 'Fused']
    vals = [rec['p_nn'], rec['p_heur'], rec['p_fused']]
    cols = [COLOR_AI if v > THRESHOLD else COLOR_REAL for v in vals]
    bars = ax.bar(names, vals, color=cols, edgecolor='white')
    ax.axhline(THRESHOLD, color='gray', ls='--', label=f'threshold {THRESHOLD}')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.3f}',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('AI probability')
    ax.set_title('Probability breakdown', fontsize=11)
    ax.legend(fontsize=8)

    verdict = '[AI-GENERATED]' if rec['is_ai'] else '[REAL]'
    color = COLOR_AI if rec['is_ai'] else COLOR_REAL
    fig.suptitle(
        f"{verdict}   p_fake={rec['p_fused']:.4f}   confidence={rec['confidence']:.3f}   "
        f"[{rec['file_name']}]   {rec['meta']['resolution']}  {rec['meta']['total_frames']} frames / "
        f"{rec['meta']['duration_s']}s",
        fontsize=15, fontweight='bold', color=color)
    save_fig(fig, path)


def viz_summary(results, out_dir):
    """Summary figures: probability bars + verdict pie + fusion breakdown"""
    names = [r['file_name'] for r in results]
    p_fused = [r['p_fused'] for r in results]
    short = [n[:22] for n in names]

    # 1) Probability bars
    fig, ax = plt.subplots(figsize=(14, 6))
    cols = [COLOR_AI if p > THRESHOLD else COLOR_REAL for p in p_fused]
    ax.bar(range(len(p_fused)), p_fused, color=cols, edgecolor='white')
    ax.axhline(THRESHOLD, color='gray', ls='--', label=f'decision threshold {THRESHOLD}')
    for i, p in enumerate(p_fused):
        ax.text(i, p + 0.02, f'{p:.2f}', ha='center', fontsize=8)
    ax.set_xticks(range(len(p_fused)))
    ax.set_xticklabels(short, rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('AI probability p_fake')
    ax.set_ylim(0, 1.15)
    ax.set_title('Test-set video detection - AI probability distribution (red = AI / green = REAL)',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_dir / '01_summary_bars.png')

    # 2) Verdict pie
    n_ai = sum(1 for r in results if r['is_ai'])
    n_real = len(results) - n_ai
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.pie([n_ai, n_real], labels=[f'AI-generated ({n_ai})', f'Real ({n_real})'],
           colors=[COLOR_AI, COLOR_REAL], autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 12},
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title(f'Verdict split (n = {len(results)} videos)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, out_dir / '02_summary_pie.png')

    # 3) Fusion breakdown
    fig, ax = plt.subplots(figsize=(14, 6))
    idx = np.arange(len(results))
    w = 0.38
    ax.bar(idx - w / 2, [r['p_nn'] for r in results], w, label='NN', color='#065A82')
    ax.bar(idx + w / 2, [r['p_heur'] for r in results], w, label='Heuristic', color='#F59E0B')
    ax.plot(idx, p_fused, 'k^--', label='Fused', markersize=6)
    ax.axhline(THRESHOLD, color='gray', ls=':', label=f'threshold {THRESHOLD}')
    ax.set_xticks(idx)
    ax.set_xticklabels(short, rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('AI probability')
    ax.set_ylim(0, 1.15)
    ax.set_title('Neural network vs heuristic vs fused', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_dir / '03_fusion_breakdown.png')


def build_html(results, out_dir, ts, model_meta):
    def b64img(p):
        return base64.b64encode(Path(p).read_bytes()).decode()

    n_ai = sum(1 for r in results if r['is_ai'])
    n_real = len(results) - n_ai
    rows = ''
    for r in results:
        p = r['p_fused']
        cls = 'ai' if r['is_ai'] else 'real'
        label = 'AI' if r['is_ai'] else 'REAL'
        proc = PROC_DIR / f"{Path(r['file_name']).stem}_process.png"
        link = f'process/{proc.name}' if proc.exists() else ''
        rows += (f'<tr><td title="{r["file_name"]}">{r["file_name"][:40]}</td>'
                 f'<td>{r["meta"]["resolution"]}</td><td>{r["meta"]["total_frames"]}</td>'
                 f'<td>{r["meta"]["duration_s"]}s</td>'
                 f'<td>{r["p_nn"]:.4f}</td><td>{r["p_heur"]:.4f}</td>'
                 f'<td><b>{p:.4f}</b></td><td class="{cls}">{label}</td>'
                 f'<td>{r["confidence"]:.3f}</td>'
                 f'<td>{r["inference_ms"]}ms</td>'
                 f'<td><a href="{link}" target="_blank">view</a></td></tr>')

    blocks = ''
    for r in results:
        proc = PROC_DIR / f"{Path(r['file_name']).stem}_process.png"
        if not proc.exists():
            continue
        p = r['p_fused']
        cls = 'ai' if r['is_ai'] else 'real'
        label = 'AI' if r['is_ai'] else 'REAL'
        blocks += f"""<div class="card">
  <h3>{r['file_name']}</h3>
  <span class="badge {cls}">{label} &middot; p_fake={p:.4f} &middot; confidence {r['confidence']:.3f}</span>
  <a href="process/{proc.name}" target="_blank"><img src="process/{proc.name}"></a>
</div>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Video Real/Fake Detection Report</title>
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
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
</style></head><body>
<div class="top">
  <h1>Test-Set Video Real/Fake Detection Report</h1>
  <div class="sub">VideoDetectorViT (DeiT-Small + temporal self-attention) + heuristic frequency/temporal fusion &middot; detected at {ts} &middot; model {model_meta}</div>
</div>
<div class="ov">
  <div><div class="k">Videos scanned</div><div class="v">{len(results)}</div></div>
  <div><div class="k">Judged AI-generated</div><div class="v" style="color:#EF4444">{n_ai}</div></div>
  <div><div class="k">Judged Real</div><div class="v" style="color:#10B981">{n_real}</div></div>
  <div><div class="k">Decision threshold</div><div class="v">{THRESHOLD}</div></div>
</div>
<main>
  <div class="card"><h3>Per-video AI probability</h3>
    <img src="data:image/png;base64,{b64img(out_dir/'01_summary_bars.png')}"></div>
  <div class="grid2">
    <div class="card"><h3>Verdict split</h3>
      <img src="data:image/png;base64,{b64img(out_dir/'02_summary_pie.png')}"></div>
    <div class="card"><h3>Fusion breakdown</h3>
      <img src="data:image/png;base64,{b64img(out_dir/'03_fusion_breakdown.png')}"></div>
  </div>
  <div class="card"><h3>Per-video results</h3>
    <table><tr><th>File</th><th>Resolution</th><th>Frames</th><th>Duration</th>
      <th>NN</th><th>Heuristic</th><th>Fused p_fake</th><th>Verdict</th><th>Confidence</th><th>Time</th><th>Process</th></tr>
      {rows}</table></div>
  <h2 style="color:#1E2761">Detection process per video</h2>
  {blocks}
</main></body></html>"""

    out = out_dir / 'video_detection_report.html'
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

    print("Loading video model...")
    model, model_meta, attn_store = load_video_model()
    print(f"  checkpoints/video_best.pt  ({model_meta})\n")

    test_dir = PROJECT_ROOT / 'test'
    videos = sorted(p for p in test_dir.rglob('*') if p.suffix.lower() in VIDEO_EXTS)
    print(f"Videos found: {len(videos)}\n")

    results = []
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {vp.name}")
        try:
            frames, meta = sample_frames(vp)
            if len(frames) < 2:
                print("  WARN: not enough frames readable, skipping")
                continue
            rec = predict_video(model, attn_store, frames)
            if rec is None:
                print("  WARN: too few frames, skipping")
                continue

            rec.update({
                'file_name': vp.name,
                'relative_path': str(vp.relative_to(test_dir)),
                'meta': meta,
            })
            results.append(rec)

            badge = 'AI' if rec['is_ai'] else 'REAL'
            print(f"  {badge:4s}  p_fused={rec['p_fused']:.4f}  "
                  f"(nn={rec['p_nn']:.3f}, heur={rec['p_heur']:.3f})  "
                  f"{meta['resolution']} {meta['total_frames']} frames {meta['duration_s']}s")

            viz_process(rec, PROC_DIR / f"{vp.stem}_process.png")
        except Exception as e:
            print(f"  WARN: error: {e}")

    if not results:
        print("No detection results available")
        return

    # Summary figures
    print("\nGenerating summary visualizations...")
    viz_summary(results, OUT_DIR)

    # Save JSON / CSV
    ts_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    dump = []
    for r in results:
        dump.append({k: v for k, v in r.items() if k not in ('attention', 'frames')})
    (OUT_DIR / f'video_results_{ts_file}.json').write_text(
        json.dumps({'timestamp': ts, 'model': model_meta, 'threshold': THRESHOLD,
                    'results': dump}, indent=2, ensure_ascii=False), encoding='utf-8')

    import csv
    with open(OUT_DIR / f'video_results_{ts_file}.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['file_name', 'resolution', 'total_frames', 'fps', 'duration_s',
                    'p_nn', 'p_heur', 'p_fused', 'verdict', 'confidence', 'inference_ms'])
        for r in results:
            w.writerow([r['file_name'], r['meta']['resolution'], r['meta']['total_frames'],
                        r['meta']['fps'], r['meta']['duration_s'], r['p_nn'], r['p_heur'],
                        r['p_fused'], 'AI' if r['is_ai'] else 'REAL',
                        r['confidence'], r['inference_ms']])

    # HTML
    html = build_html(results, OUT_DIR, ts, model_meta)

    n_ai = sum(1 for r in results if r['is_ai'])
    print("\n" + "=" * 70)
    print("Detection summary")
    print("=" * 70)
    print(f"  Videos total : {len(results)}")
    print(f"  AI-generated : {n_ai}")
    print(f"  Real         : {len(results) - n_ai}")
    print(f"  Process figs : {len(list(PROC_DIR.glob('*.png')))}")
    print(f"\nResult folder: {OUT_DIR}")
    print(f"HTML report  : {html}")


if __name__ == '__main__':
    main()
