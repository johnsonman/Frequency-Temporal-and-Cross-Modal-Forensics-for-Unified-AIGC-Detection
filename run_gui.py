# -*- coding: utf-8 -*-
"""
多模态 AIGC 检测系统 —— 可视化操作界面
========================================
功能:
  1. 选择图片 / 视频 / 音频 / 文件夹 进行检测
  2. 后台线程推理 + UI 实时更新 (队列消息驱动)
  3. 显示融合概率、各模态独立分数、门控权重、启发式特征贡献
  4. 校准阈值判定 (checkpoints/calibration.json)
  5. 批量文件夹扫描 + 进度条 + 结果导出 JSON

依赖 (均为项目已有或标准库):
  - tkinter (标准库 GUI)
  - torch, transformers, cv2, numpy, PIL, soundfile
  - models/* (MultimodalAIGCDetector + HeuristicAIGCDetector)

运行: python run_gui.py
"""

import os
import sys
import json
import time
import queue
import threading
import numpy as np
import traceback
from pathlib import Path
from datetime import datetime

# ======================== 项目根路径 ========================
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ======================== 模型配置预集 ========================
# Detection Strategy: 检测策略 (融合方式)
STRATEGY_PRESETS = [
    ("Heuristic Only (Calibrated)",       "heuristic"),
    ("Hybrid (Calibrated Heur + NN)",     "hybrid"),
    ("Neural Only (MultimodalAIGC)",     "neural"),
]

# Video Frame Count: 视频帧采样数 (速度 vs 精度)
FRAME_OPTIONS = [
    ("4 frames (Fast)", 4),
    ("8 frames (Default)", 8),
    ("16 frames (Accurate)", 16),
]

# Backbone 选项 (保持 embed_dim=768 兼容 calibration.json)
IMG_BACKBONES = [
    ("openai/clip-vit-base-patch16",       "openai/clip-vit-base-patch16"),
]
AUD_BACKBONES = [
    ("facebook/wav2vec2-base-960h",        "facebook/wav2vec2-base-960h"),
]

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

# 支持的文件扩展名
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}


def classify_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return 'image'
    if ext in VIDEO_EXTS:
        return 'video'
    if ext in AUDIO_EXTS:
        return 'audio'
    return 'unknown'


def scan_folder(folder: str) -> list:
    files = []
    for p in Path(folder).rglob('*'):
        if p.is_file() and classify_file(str(p)) != 'unknown':
            files.append(str(p))
    return sorted(files)


# ============================================================
# Worker 线程 —— 加载模型 + 执行推理
# ============================================================
class DetectorWorker(threading.Thread):
    def __init__(self, msg_queue: queue.Queue, device: str = "cuda",
                 strategy: str = "hybrid", num_frames: int = 8,
                 img_model: str = "openai/clip-vit-base-patch16",
                 aud_model: str = "facebook/wav2vec2-base-960h"):
        super().__init__(daemon=True)
        self.msg_queue = msg_queue
        self.device = device
        self.strategy = strategy        # "hybrid" | "neural" | "heuristic"
        self.num_frames = num_frames
        self.img_model = img_model
        self.aud_model = aud_model
        self._stop_event = threading.Event()
        self.detector = None
        self.heuristic = None
        self.calibration = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self.msg_queue.put(('log', f"[Init] Loading model on {self.device} ..."))
            t0 = time.time()

            # 延迟导入避免 GUI 启动卡顿
            import torch
            from models.detector import MultimodalAIGCDetector
            from models.heuristic_detector import HeuristicAIGCDetector

            self.detector = MultimodalAIGCDetector(
                device=self.device,
                img_model=self.img_model,
                aud_model=self.aud_model,
                num_frames=self.num_frames,
            )
            self.heuristic = HeuristicAIGCDetector(device=self.device)

            self.msg_queue.put(('log',
                f"[Init] Strategy={self.strategy} | Frames={self.num_frames} | "
                f"img={self.img_model.split('/')[-1]} | aud={self.aud_model.split('/')[-1]}"))

            # 加载校准阈值
            cal_path = PROJECT_ROOT / 'checkpoints' / 'calibration.json'
            if cal_path.exists():
                with open(cal_path, 'r', encoding='utf-8') as f:
                    self.calibration = json.load(f)
                self.msg_queue.put(('log', f"[Init] Calibration loaded: image thr={self.calibration.get('image',{}).get('threshold','?')}, video thr={self.calibration.get('video',{}).get('threshold','?')}"))

            self.msg_queue.put(('log', f"[Init] Ready ({time.time() - t0:.1f}s)"))
            self.msg_queue.put(('ready', None))

        except Exception as e:
            err = traceback.format_exc()
            self.msg_queue.put(('log', f"[Init ERROR] {e}\n{err}"))
            self.msg_queue.put(('error', str(e)))

    # ---------- 推理方法 ----------
    def detect_one(self, file_path: str, strategy_override: str | None = None) -> dict:
        """检测单个文件, 返回完整结果 dict. strategy_override 可覆盖 Load Model 时的固定策略"""
        import numpy as np
        active_strat = strategy_override or self.strategy
        ftype = classify_file(file_path)
        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': ftype,
            'p_fake': None,
            'p_real': None,
            'is_ai': None,
            'confidence': None,
            'nn_score': None,
            'heuristic_score': None,
            'fused_score': 0.5,
            'p_img': None, 'p_vid': None, 'p_aud': None,
            'gates': None,
            'reliability': None,
            'modal_mask': {},
            'heuristics': {},
            'notes': '',
            'elapsed': 0,
        }

        t0 = time.time()
        try:
            # ---- NN 分支 (仅 hybrid / neural) ----
            det = None
            h_res = None
            if active_strat != "heuristic":
                if ftype == 'image':
                    det = self.detector.detect_image_file(file_path)
                elif ftype == 'video':
                    det = self.detector.detect_video_file(file_path)
                elif ftype == 'audio':
                    det = self.detector.detect_audio_file(file_path)
                else:
                    result['notes'] = "Unsupported file type"
                    return result

                if det and det.notes:
                    result['notes'] += det.notes
                if det:
                    result['nn_score'] = det.p_fake
                    result['p_img'] = det.p_img
                    result['p_vid'] = det.p_vid
                    result['p_aud'] = det.p_aud
                    result['gates'] = det.gates
                    result['reliability'] = det.reliability
                    result['modal_mask'] = det.modal_mask

            # ---- 启发式分支 (仅 hybrid / heuristic) ----
            if active_strat != "neural":
                try:
                    h_res = None
                    if ftype == 'image':
                        h_res = self.heuristic.detect_image(file_path)
                    elif ftype == 'video':
                        h_res = self.heuristic.detect_video(file_path)
                    elif ftype == 'audio':
                        h_res = self.heuristic.detect_audio(file_path)
                    if h_res:
                        result['heuristics'] = h_res.get('heuristics', {})
                        # 用 calibration.json 的 z-score×weight 公式得到校准后的分数
                        calibrated = self._calibrate_heuristic(
                            result['heuristics'], ftype
                        )
                        result['heuristic_score'] = calibrated or h_res.get('p_fake')
                        result['heuristic_score_raw'] = h_res.get('p_fake')
                except Exception as he:
                    result['notes'] += f"Heuristic error: {he}; "

            # ---- 融合 ----
            if active_strat == "neural":
                result['fused_score'] = result['nn_score'] or 0.5
            elif active_strat == "heuristic":
                result['fused_score'] = result['heuristic_score'] or 0.5
            else:  # hybrid - 主要用校准分数 + NN 轻微影响
                result['fused_score'] = self._fuse(
                    result['nn_score'], result['heuristic_score']
                )
            result['p_fake'] = result['fused_score']
            result['p_real'] = 1.0 - result['fused_score']

            # ---- 校准阈值判定 ----
            thr = self._get_threshold(ftype, strategy=active_strat)
            result['threshold'] = thr
            result['is_ai'] = result['fused_score'] > thr if thr is not None else (result['fused_score'] > 0.5)
            result['confidence'] = max(result['fused_score'], 1.0 - result['fused_score'])

        except Exception as e:
            result['notes'] = f"Detection error: {e}\n{traceback.format_exc()}"
        finally:
            result['elapsed'] = time.time() - t0

        # ---- 可视化生成 (独立于推理, 失败不影响主流程) ----
        try:
            orig_pil, viz_pil = self._generate_viz(ftype, file_path, result)
            result['orig_pil'] = orig_pil
            result['viz_pil'] = viz_pil
        except Exception as ve:
            result['orig_pil'] = None
            result['viz_pil'] = None
            self.msg_queue.put(('log', f"[Viz warn] {os.path.basename(file_path)}: {ve}"))

        return result

    def _fuse(self, nn_score, heuristic_score):
        """启发式为主(0.7) + NN辅助(0.3)"""
        if heuristic_score is not None and nn_score is not None:
            nn_centered = (nn_score - 0.5) * 0.6 + 0.5
            fused = 0.7 * heuristic_score + 0.3 * nn_centered
        elif heuristic_score is not None:
            fused = heuristic_score
        elif nn_score is not None:
            fused = nn_score
        else:
            fused = 0.5
        return float(np.clip(fused, 0.0, 1.0))

    def _get_threshold(self, ftype: str, strategy: str | None = None):
        if self.calibration is None:
            return None
        strat = strategy or self.strategy
        cal = self.calibration.get(ftype, {})
        # 策略特定阈值（优先）
        strat_thrs = cal.get('strategy_thresholds', {})
        if strat in strat_thrs:
            return strat_thrs[strat]
        # 通用阈值（启发式用的，给 heuristic 策略或 fallback）
        return cal.get('threshold', None)

    def _calibrate_heuristic(self, heuristics: dict, ftype: str) -> float | None:
        """用 calibration.json 的 z-score × weight 公式把启发式特征映射到校准后的 p_fake"""
        if self.calibration is None:
            return None
        cal = self.calibration.get(ftype, {})
        weights = cal.get('weights', {})
        if not weights or not heuristics:
            return None
        contrib = 0.0
        for name, wdef in weights.items():
            val = heuristics.get(name)
            if val is None:
                continue
            z = (val - wdef['fake_mean']) / (wdef['fake_std'] + 1e-6)
            contrib += z * wdef['sign'] * wdef['weight']
        cmin = cal.get('contrib_min', -5.0)
        cmax = cal.get('contrib_max', 5.0)
        calibrated = (contrib - cmin) / (cmax - cmin + 1e-8)
        return float(np.clip(calibrated, 0.0, 1.0))

    # ---------- 可视化生成 ----------
    def _generate_viz(self, ftype: str, file_path: str, result: dict):
        """后台线程生成可视化 — 仅 Attention Map + FFT Spectrum (深色底, 完整覆盖)"""
        import io
        import cv2
        from PIL import Image as PILImage
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Segoe UI', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        orig_pil = None
        viz_pil = None

        try:
            if ftype == 'image':
                bgr = cv2.imread(file_path)
                if bgr is None:
                    return None, None
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                orig_pil = PILImage.fromarray(rgb)
                h, w = rgb.shape[:2]

                fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#0A0F14')
                for ax in axes:
                    ax.set_facecolor('#0A0F14')

                # === Attention Map: Grad-CAM (resize to full image size + overlay RGB) ===
                try:
                    import torch
                    tensor = self.detector.image_branch.preprocess(rgb).clone().detach().requires_grad_(True)
                    with torch.enable_grad():
                        det_out = self.detector.image_branch(tensor)
                        p = det_out['p_img'][0, 1]
                        self.detector.zero_grad(set_to_none=True)
                        p.backward()
                    sal = tensor.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
                    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
                    # Resize saliency to FULL image size so it completely overlays
                    sal_full = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
                    extent = [0, w, h, 0]
                    axes[0].imshow(rgb, extent=extent, aspect='auto')
                    axes[0].imshow(sal_full, cmap='jet', alpha=0.6, extent=extent, aspect='auto')
                    axes[0].set_title('Attention Map (Grad-CAM)', color='#FFD700',
                                      fontsize=11, fontweight='bold')
                    axes[0].axis('off')
                except Exception:
                    axes[0].imshow(rgb, aspect='auto')
                    axes[0].set_title('Attention Map', color='#FFD700',
                                      fontsize=11, fontweight='bold')
                    axes[0].axis('off')

                # === FFT Frequency Spectrum ===
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
                fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
                axes[1].imshow(fft_mag, cmap='magma', aspect='auto')
                axes[1].set_title('FFT Frequency Spectrum', color='#00E5FF',
                                  fontsize=11, fontweight='bold')
                axes[1].axis('off')

            elif ftype == 'video':
                cap = cv2.VideoCapture(file_path)
                frames = []
                for _ in range(16):
                    ret, f = cap.read()
                    if not ret: break
                    frames.append(f)
                cap.release()
                if not frames:
                    return None, None
                first = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
                orig_pil = PILImage.fromarray(first)
                fh, fw = first.shape[:2]

                fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#0A0F14')
                for ax in axes:
                    ax.set_facecolor('#0A0F14')

                # === Temporal Attention: multi-frame diff heatmap ===
                try:
                    diffs = []
                    for i in range(min(4, len(frames) - 1)):
                        g0 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
                        g1 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY).astype(float)
                        diffs.append(np.abs(g0 - g1))
                    avg_diff = np.mean(diffs, axis=0) if diffs else np.zeros((64, 64))
                    # Resize to FULL frame size
                    avg_diff = cv2.resize(avg_diff, (fw, fh), interpolation=cv2.INTER_LINEAR)
                    axes[0].imshow(avg_diff, cmap='hot', aspect='auto')
                    axes[0].set_title('Temporal Attention (Frame Diff)',
                                      color='#FF6B6B', fontsize=11, fontweight='bold')
                    axes[0].axis('off')
                except Exception:
                    axes[0].imshow(first, aspect='auto')
                    axes[0].set_title('Temporal Attention',
                                      color='#FF6B6B', fontsize=11, fontweight='bold')
                    axes[0].axis('off')

                # === FFT Frequency Spectrum ===
                gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY).astype(float)
                fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
                axes[1].imshow(fft_mag, cmap='magma', aspect='auto')
                axes[1].set_title('FFT Frequency Spectrum', color='#00E5FF',
                                  fontsize=11, fontweight='bold')
                axes[1].axis('off')

            elif ftype == 'audio':
                try:
                    import soundfile as sf
                    waveform, sr = sf.read(file_path)
                    if waveform.ndim > 1:
                        waveform = waveform.mean(axis=1)
                except Exception:
                    return None, None
                orig_pil = None

                fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='#0A0F14')
                for ax in axes:
                    ax.set_facecolor('#0A0F14')

                # === Spectral Attention: Mel-Spectrogram ===
                try:
                    import librosa
                    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128)
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    axes[0].imshow(mel_db, aspect='auto', origin='lower', cmap='inferno')
                    axes[0].set_title('Spectral Attention (Mel)',
                                      color='#FF6B6B', fontsize=11, fontweight='bold')
                    axes[0].axis('off')
                except Exception:
                    axes[0].text(0.5, 0.5, 'librosa N/A', color='white',
                                 ha='center', va='center', fontsize=10)
                    axes[0].set_title('Spectral Attention',
                                      color='#FF6B6B', fontsize=11, fontweight='bold')
                    axes[0].axis('off')

                # === FFT Frequency Spectrum ===
                fft_vals = np.abs(np.fft.rfft(waveform[:sr*10]))
                freqs = np.fft.rfftfreq(len(waveform[:sr*10]), d=1.0/sr)
                fft_db = 20 * np.log10(fft_vals + 1e-10)
                half_n = len(freqs) // 2
                axes[1].plot(freqs[:half_n], fft_db[:half_n], color='#00E5FF', lw=0.8)
                axes[1].fill_between(freqs[:half_n], fft_db[:half_n], alpha=0.15, color='#00E5FF')
                axes[1].set_title('FFT Frequency Spectrum', color='#00E5FF',
                                  fontsize=11, fontweight='bold')
                axes[1].grid(alpha=0.2, color='#333')
                axes[1].set_xlabel('Hz', color='#888')
                axes[1].tick_params(colors='#888')

            else:
                return None, None

            plt.tight_layout(pad=1.0)
            verdict = 'AI-GENERATED' if result['is_ai'] else 'REAL'
            v_color = '#FF6B6B' if result['is_ai'] else '#55A868'
            f_score = result['fused_score']
            thr = result.get('threshold', 0.5) or 0.5
            fig.suptitle(
                f'[{verdict}]  p_fake={f_score:.4f} (thr={thr:.2f})  |  {result["file_name"]}',
                fontsize=12, fontweight='bold', color=v_color,
                backgroundcolor='#0A0F14'
            )

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0A0F14')
            plt.close(fig)
            buf.seek(0)
            viz_pil = PILImage.open(buf).convert('RGB')

        except Exception as e:
            print(f"[Viz error] {e}")
            traceback.print_exc()
            try:
                plt.close('all')
            except Exception:
                pass

        return orig_pil, viz_pil


# ============================================================
# 主窗口
# ============================================================
class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Multimodal AIGC Detector — GUI")

        # 自适应屏幕: 窗口 = 屏幕 85%, 居中
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        # 考虑 Windows 任务栏 (~48px)
        max_h = sh - 80
        win_w = int(min(sw * 0.85, 1600))
        win_h = int(min(max_h * 0.88, 900))
        x = max(0, (sw - win_w) // 2)
        y = max(0, (sh - win_h) // 2 - 20)
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        min_w = 780 if sw < 1400 else 860
        min_h = 520 if sh < 900 else 560
        self.root.minsize(min_w, min_h)
        self.root.resizable(True, True)
        self._init_w = win_w   # 给 _build_ui 的 PanedWindow 用
        self._init_h = win_h
        self._resize_job = None
        self._base_fonts = {}
        self._ui_scale = 1.0

        self.msg_queue = queue.Queue()
        self.worker: DetectorWorker = None
        self.is_ready = False

        # 当前任务状态
        self.task_files = []       # 待检测文件列表
        self.task_results = []     # 已完成结果列表（与 task_files 对应）
        self.is_running = False

        # PhotoImage 引用持有 (防止 GC)
        self._photos = {}

        self._build_ui()
        self.root.bind('<Configure>', self._schedule_responsive_layout, add='+')
        self._poll_queue()

    # ---------- UI 构建 ----------
    def _build_ui(self):
        # 颜色
        self.C = {
            'primary': '#1B3A5C',
            'secondary': '#2E86AB',
            'accent': '#A23B72',
            'bg': '#F4F7F9',
            'fake': '#C44E52',
            'real': '#55A868',
        }

        # 顶部工具栏分行，缩窄窗口时控件仍保持可见。
        toolbar = tk.Frame(self.root, bg=self.C['primary'])
        toolbar.pack(fill='x', side='top')
        toolbar_title = tk.Frame(toolbar, bg=self.C['primary'])
        toolbar_title.pack(fill='x')
        tk.Label(toolbar_title, text="Multimodal AIGC Detector",
                 bg=self.C['primary'], fg='white',
                 font=('Segoe UI', 14, 'bold')).pack(side='left', padx=16, pady=(8, 2))
        tk.Button(toolbar_title, text="Export JSON", command=self._on_export,
                  bg='#6B7B8B', fg='white', relief='flat',
                  padx=12, pady=5, font=('Segoe UI', 10)).pack(side='right', padx=16, pady=(4, 2))

        toolbar_options = tk.Frame(toolbar, bg=self.C['primary'])
        toolbar_options.pack(fill='x', padx=8, pady=2)
        # 设备选择
        tk.Label(toolbar_options, text="Device:", bg=self.C['primary'], fg='white').pack(side='left', padx=(4, 2))
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_var = tk.StringVar(value=dev)
        dev_combo = ttk.Combobox(toolbar_options, textvariable=self.device_var,
                                 values=["cuda", "cpu"], width=6, state='readonly')
        dev_combo.pack(side='left')

        # Strategy selector
        tk.Label(toolbar_options, text="Strategy:", bg=self.C['primary'], fg='white').pack(side='left', padx=(14, 2))
        self.strategy_var = tk.StringVar(value=STRATEGY_PRESETS[0][0])
        strat_combo = ttk.Combobox(toolbar_options, textvariable=self.strategy_var,
                                   values=[s[0] for s in STRATEGY_PRESETS],
                                   width=26, state='readonly')
        strat_combo.pack(side='left')

        # Video frames selector
        tk.Label(toolbar_options, text="Frames:", bg=self.C['primary'], fg='white').pack(side='left', padx=(10, 2))
        self.frames_var = tk.StringVar(value=FRAME_OPTIONS[1][0])
        frames_combo = ttk.Combobox(toolbar_options, textvariable=self.frames_var,
                                    values=[f[0] for f in FRAME_OPTIONS],
                                    width=18, state='readonly')
        frames_combo.pack(side='left')

        # Backbone info label (readonly, shows current config)
        self.model_info_lbl = tk.Label(toolbar_options, text="", bg=self.C['primary'], fg='#A0C4D8',
                                        font=('Segoe UI', 9))
        self.model_info_lbl.pack(side='left', padx=(14, 0), fill='x', expand=True)

        toolbar_actions = tk.Frame(toolbar, bg=self.C['primary'])
        toolbar_actions.pack(fill='x', padx=8, pady=(2, 7))
        self.btn_load = self._make_btn(toolbar_actions, "Load Model", self._on_load_model)
        self.btn_load.pack(side='left', padx=4)

        tk.Button(toolbar_actions, text="＋ File", command=self._on_add_file,
                  bg=self.C['secondary'], fg='white', relief='flat',
                  padx=12, pady=5, font=('Segoe UI', 10)).pack(side='left', padx=4)
        tk.Button(toolbar_actions, text="＋ Folder", command=self._on_add_folder,
                  bg=self.C['secondary'], fg='white', relief='flat',
                  padx=12, pady=5, font=('Segoe UI', 10)).pack(side='left', padx=4)
        tk.Button(toolbar_actions, text="Clear", command=self._on_clear,
                  bg='#6B7B8B', fg='white', relief='flat',
                  padx=12, pady=5, font=('Segoe UI', 10)).pack(side='left', padx=4)

        self.btn_start = tk.Button(toolbar_actions, text="▶  Start Detection", command=self._on_start,
                                   bg=self.C['accent'], fg='white', relief='flat',
                                   padx=16, pady=5, font=('Segoe UI', 10, 'bold'),
                                   state='disabled')
        self.btn_start.pack(side='left', padx=8)

        # 主体分栏
        body = tk.PanedWindow(self.root, orient='horizontal', sashwidth=6,
                              sashrelief='raised', bg=self.C['bg'])
        body.pack(fill='both', expand=True, side='top')

        # ---- 左栏: 文件列表 ----
        left = tk.Frame(body, bg=self.C['bg'])
        body.add(left, width=int(self._init_w * 0.32), minsize=220, stretch='always')

        tk.Label(left, text="Files to Detect", bg=self.C['bg'],
                 font=('Segoe UI', 11, 'bold'), fg=self.C['primary']).pack(anchor='w', padx=10, pady=(10, 4))

        # Treeview
        tv_frame = tk.Frame(left)
        tv_frame.pack(fill='both', expand=True, padx=8, pady=(0, 4))

        cols = ('status', 'type', 'file')
        self.tree = ttk.Treeview(tv_frame, columns=cols, show='headings', height=14)
        self.tree.heading('status', text='Status')
        self.tree.heading('type', text='Type')
        self.tree.heading('file', text='File')
        self.tree.column('status', width=60, anchor='center')
        self.tree.column('type', width=60, anchor='center')
        self.tree.column('file', width=320, anchor='w')
        self.tree.pack(side='left', fill='both', expand=True)
        self.tree.bind('<<TreeviewSelect>>', self._on_select)

        sb = ttk.Scrollbar(tv_frame, orient='vertical', command=self.tree.yview)
        sb.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=sb.set)

        # 进度条
        self.progress = ttk.Progressbar(left, mode='determinate')
        self.progress.pack(fill='x', padx=10, pady=6)

        # ---- 右栏: 详情面板 ----
        right = tk.Frame(body, bg='white')
        body.add(right, width=int(self._init_w * 0.68), minsize=440, stretch='always')

        tk.Label(right, text="Detection Details", bg='white',
                 font=('Segoe UI', 11, 'bold'), fg=self.C['primary']).pack(anchor='w', padx=12, pady=(10, 4))

        # 可拖动上下分栏，用户可自由调整可视化和文字详情的空间。
        right_split = tk.PanedWindow(right, orient='vertical', sashwidth=6,
                                     sashrelief='raised', bg='white')
        right_split.pack(fill='both', expand=True, padx=12, pady=(4, 4))

        viz_host = tk.Frame(right_split, bg='white')
        self._build_viz_panel(viz_host)
        right_split.add(viz_host, height=int(self._init_h * 0.38), minsize=180,
                        stretch='always')

        detail_container = tk.Frame(right_split, bg='white')
        right_split.add(detail_container, height=int(self._init_h * 0.42), minsize=220,
                        stretch='always')

        # 摘要卡片
        self.summary_frame = tk.Frame(detail_container, bg='white')
        self.summary_frame.pack(fill='x', pady=4)
        self._build_summary_card(self.summary_frame)

        # 概率 Meter
        self.meter_frame = tk.Frame(detail_container, bg='white')
        self.meter_frame.pack(fill='x', pady=4)
        self._build_meter(self.meter_frame)

        # 下分两栏: 门控 + 启发式
        detail_split = tk.Frame(detail_container, bg='white')
        detail_split.pack(fill='both', expand=True, pady=4)

        gate_frame = tk.LabelFrame(detail_split, text="Gated Fusion Weights", bg='white',
                                    font=('Segoe UI', 10, 'bold'), fg=self.C['primary'])
        gate_frame.pack(side='left', fill='both', expand=True, padx=(0, 4))
        self._build_gate_panel(gate_frame)

        heur_frame = tk.LabelFrame(detail_split, text="Heuristic Features", bg='white',
                                    font=('Segoe UI', 10, 'bold'), fg=self.C['primary'])
        heur_frame.pack(side='left', fill='both', expand=True, padx=(4, 0))
        self._build_heur_panel(heur_frame)

        # 模态掩码 + 各模态独立分数
        self.modal_frame = tk.LabelFrame(detail_container, text="Per-Modality Results", bg='white',
                                          font=('Segoe UI', 10, 'bold'), fg=self.C['primary'])
        self.modal_frame.pack(fill='x', pady=(4, 4))
        self._build_modal_panel(self.modal_frame)

        # 底部日志
        log_frame = tk.LabelFrame(self.root, text="Log", font=('Segoe UI', 9, 'bold'), fg=self.C['primary'])
        log_frame.pack(fill='x', side='bottom', padx=8, pady=4)
        self.log_text = tk.Text(log_frame, height=6, font=('Consolas', 9), wrap='word',
                                bg='#1E1E1E', fg='#D4D4D4', relief='flat')
        self.log_text.pack(fill='x', padx=4, pady=4, side='left')
        log_sb = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        log_sb.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=log_sb.set, state='disabled')

    def _build_viz_panel(self, parent):
        """可视化面板: 左原图 + 右检测结果 side-by-side"""
        import tkinter as tk

        outer = tk.LabelFrame(parent, text="Visualization", bg='white',
                               font=('Segoe UI', 10, 'bold'), fg=self.C['primary'])
        outer.pack(fill='both', expand=True, pady=(0, 4))

        row = tk.Frame(outer, bg='white')
        row.pack(fill='both', expand=True, padx=6, pady=6)

        # 左: 原图
        orig_cell = tk.Frame(row, bg='#2E3A46', highlightbackground='#C5CDD4', highlightthickness=1)
        orig_cell.pack(side='left', fill='both', expand=True, padx=(0, 4))
        tk.Label(orig_cell, text="Original", bg='#2E3A46', fg='white',
                 font=('Segoe UI', 9, 'bold')).pack(anchor='w', padx=6, pady=(3, 0))
        self.orig_canvas = tk.Canvas(orig_cell, bg='#1E272E', highlightthickness=0)
        self.orig_canvas.pack(fill='both', expand=True, padx=4, pady=4)
        self.orig_canvas.bind('<Configure>', lambda e: self._redraw_cached('orig'))
        self._orig_pil_cache = None

        # 右: 检测结果
        viz_cell = tk.Frame(row, bg='#2E3A46', highlightbackground='#C5CDD4', highlightthickness=1)
        viz_cell.pack(side='left', fill='both', expand=True, padx=(4, 0))
        tk.Label(viz_cell, text="Detection", bg='#2E3A46', fg='white',
                 font=('Segoe UI', 9, 'bold')).pack(anchor='w', padx=6, pady=(3, 0))
        self.viz_canvas = tk.Canvas(viz_cell, bg='#1E272E', highlightthickness=0)
        self.viz_canvas.pack(fill='both', expand=True, padx=4, pady=4)
        self.viz_canvas.bind('<Configure>', lambda e: self._redraw_cached('viz'))
        self._viz_pil_cache = None

        # 占位文字 (初始)
        self._show_viz_placeholder()

    def _schedule_responsive_layout(self, event):
        """Debounce root resizes before scaling fonts and fixed-width controls."""
        if event.widget is not self.root:
            return
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except tk.TclError:
                pass
        self._resize_job = self.root.after(120, self._apply_responsive_layout)

    def _apply_responsive_layout(self, force=False):
        self._resize_job = None
        width = max(self.root.winfo_width(), 1)
        height = max(self.root.winfo_height(), 1)
        width_ratio = width / max(self._init_w, 1)
        height_ratio = height / max(self._init_h, 1)
        scale = max(0.72, min(1.45, (width_ratio * height_ratio) ** 0.5))
        if not force and abs(scale - self._ui_scale) < 0.015:
            return
        self._ui_scale = scale

        for widget in self._walk_widgets(self.root):
            try:
                font_value = widget.cget('font')
                if font_value:
                    key = str(widget)
                    if key not in self._base_fonts:
                        actual = tkfont.Font(root=self.root, font=font_value).actual()
                        self._base_fonts[key] = actual
                    actual = self._base_fonts[key]
                    base_size = int(actual['size'])
                    scaled_size = max(7, round(abs(base_size) * scale))
                    if base_size < 0:
                        scaled_size = -scaled_size
                    widget.configure(font=(
                        actual['family'], scaled_size, actual['weight'], actual['slant'],
                        actual['underline'], actual['overstrike']
                    ))
            except (tk.TclError, TypeError, ValueError):
                pass

            if isinstance(widget, ttk.Progressbar):
                try:
                    key = str(widget)
                    if not hasattr(self, '_base_progress_lengths'):
                        self._base_progress_lengths = {}
                    if key not in self._base_progress_lengths:
                        self._base_progress_lengths[key] = int(widget.cget('length'))
                    widget.configure(length=max(70, round(self._base_progress_lengths[key] * scale)))
                except (tk.TclError, TypeError, ValueError):
                    pass

        if hasattr(self, 'tree'):
            if not hasattr(self, '_base_tree_widths'):
                self._base_tree_widths = {
                    col: int(self.tree.column(col, 'width'))
                    for col in ('status', 'type', 'file')
                }
            for col, base_width in self._base_tree_widths.items():
                self.tree.column(col, width=max(48, round(base_width * scale)))

        if hasattr(self, 'meter_canvas'):
            self.meter_canvas.configure(height=max(20, round(28 * scale)))

        # Keep the feature-list canvas window as wide as its viewport.
        if hasattr(self, 'heur_canvas') and hasattr(self, 'heur_canvas_window'):
            try:
                self.heur_canvas.itemconfigure(
                    self.heur_canvas_window, width=max(1, self.heur_canvas.winfo_width())
                )
            except tk.TclError:
                pass

        # Canvas <Configure> handlers redraw cached images at the new dimensions.

    @staticmethod
    def _walk_widgets(parent):
        for child in parent.winfo_children():
            yield child
            yield from MainWindow._walk_widgets(child)

    def _show_viz_placeholder(self):
        self._draw_canvas_placeholder(self.orig_canvas, "—  select a file to view  —")
        self._draw_canvas_placeholder(self.viz_canvas, "—  select a file to view  —")

    @staticmethod
    def _draw_canvas_placeholder(canvas, text):
        canvas.delete('all')
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 120)
        canvas.create_text(width // 2, height // 2, text=text,
                           fill='#6B7B8B', font=('Segoe UI', 10))

    def _redraw_cached(self, which):
        """Canvas resize 时重新绘制缓存的 PIL 图"""
        if which == 'orig':
            if self._orig_pil_cache is not None:
                self._render_pil_to_canvas(self.orig_canvas, self._orig_pil_cache)
            else:
                self._draw_canvas_placeholder(self.orig_canvas, "—  select a file to view  —")
        elif which == 'viz':
            if self._viz_pil_cache is not None:
                self._render_pil_to_canvas(self.viz_canvas, self._viz_pil_cache)
            else:
                self._draw_canvas_placeholder(self.viz_canvas, "—  select a file to view  —")

    def _render_pil_to_canvas(self, canvas, pil_img):
        """将 PIL.Image 缩放后绘制到 Canvas, PhotoImage 加入 self._photos 防 GC"""
        from PIL import Image, ImageTk
        canvas.delete('all')
        cw = max(canvas.winfo_width(), 200)
        ch = max(canvas.winfo_height(), 120)
        img = pil_img.copy()
        max_w, max_h = max(cw - 12, 16), max(ch - 12, 12)
        scale = min(max_w / max(img.width, 1), max_h / max(img.height, 1))
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        if img.size != size:
            img = img.resize(size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._photos[canvas] = photo
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor='center')

    def _display_images(self, result: dict):
        """根据 result 更新可视化面板"""
        orig_pil = result.get('orig_pil')
        viz_pil = result.get('viz_pil')
        self._orig_pil_cache = orig_pil
        self._viz_pil_cache = viz_pil

        # 左: 原图 (音频无原图, 清空白)
        if orig_pil is not None:
            self._orig_pil_cache = orig_pil
            self._render_pil_to_canvas(self.orig_canvas, orig_pil)
        else:
            self._draw_canvas_placeholder(self.orig_canvas, "(no original preview)")

        # 右: 检测结果
        if viz_pil is not None:
            self._viz_pil_cache = viz_pil
            self._render_pil_to_canvas(self.viz_canvas, viz_pil)
        else:
            self._draw_canvas_placeholder(self.viz_canvas, "(visualization unavailable)")

    def _make_btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         bg='#E1E5E8', fg=self.C['primary'],
                         relief='flat', padx=10, pady=6,
                         font=('Segoe UI', 10))

    # ---------- 子面板构建 ----------
    def _build_summary_card(self, parent):
        self.summary = {}
        row = tk.Frame(parent, bg='white')
        row.pack(fill='x')
        for i, (label, key) in enumerate([
            ("File", "file"), ("Type", "type"), ("Threshold", "threshold")
        ]):
            cell = tk.Frame(row, bg=self.C['bg'], highlightbackground='#C5CDD4', highlightthickness=1)
            cell.grid(row=0, column=i, sticky='nsew', padx=3, pady=2)
            row.grid_columnconfigure(i, weight=1)
            tk.Label(cell, text=label, bg=self.C['bg'], fg='#6B7B8B',
                     font=('Segoe UI', 8)).pack(anchor='w', padx=8, pady=(4, 0))
            lbl = tk.Label(cell, text="—", bg=self.C['bg'], fg=self.C['primary'],
                           font=('Segoe UI', 10, 'bold'))
            lbl.pack(anchor='w', padx=8, pady=(0, 4))
            self.summary[key] = lbl

        row2 = tk.Frame(parent, bg='white')
        row2.pack(fill='x', pady=(4, 0))
        self.verdict_lbl = tk.Label(row2, text="—  waiting for detection  —", bg='white',
                                     font=('Segoe UI', 16, 'bold'), fg=self.C['primary'])
        self.verdict_lbl.pack(side='left')

        self.conf_lbl = tk.Label(row2, text="", bg='white', font=('Segoe UI', 11), fg='#6B7B8B')
        self.conf_lbl.pack(side='right')

    def _build_meter(self, parent):
        tk.Label(parent, text="Fused  AI Probability", bg='white',
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        self.meter_canvas = tk.Canvas(parent, height=28, bg='#E1E5E8', highlightthickness=0)
        self.meter_canvas.pack(fill='x', pady=4)
        self.meter_canvas.bind('<Configure>', lambda _e: self._draw_meter(self._last_p_fake))
        self._last_p_fake = None
        self.meter_text = tk.Label(parent, text="—", bg='white', font=('Segoe UI', 12, 'bold'),
                                    fg=self.C['primary'])
        self.meter_text.pack(anchor='center')

    def _build_gate_panel(self, parent):
        self.gate_widgets = {}
        for key, label in [('img', 'Image'), ('vid', 'Video'), ('aud', 'Audio'),
                            ('sync', 'Sync'), ('async', 'Async')]:
            row = tk.Frame(parent, bg='white')
            row.pack(fill='x', padx=8, pady=2)
            tk.Label(row, text=label, bg='white', width=7, anchor='w',
                     font=('Segoe UI', 9)).pack(side='left')
            bar = ttk.Progressbar(row, mode='determinate', maximum=1.0, length=180)
            bar.pack(side='left', padx=6)
            val_lbl = tk.Label(row, text="—", bg='white', width=6, anchor='w',
                                font=('Segoe UI', 9))
            val_lbl.pack(side='left')
            self.gate_widgets[key] = (bar, val_lbl)

    def _build_heur_panel(self, parent):
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        self.heur_inner = tk.Frame(canvas, bg='white')
        self.heur_inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        self.heur_canvas = canvas
        self.heur_canvas_window = canvas.create_window((0, 0), window=self.heur_inner, anchor='nw')
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(
            self.heur_canvas_window, width=max(1, e.width)))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

    def _build_modal_panel(self, parent):
        row = tk.Frame(parent, bg='white')
        row.pack(fill='x', padx=8, pady=6)
        self.modal_labels = {}
        for i, (label, key) in enumerate([
            ("Image p_fake", 'p_img'), ("Video p_fake", 'p_vid'),
            ("Audio p_fake", 'p_aud'), ("Modal Mask", 'mask')
        ]):
            cell = tk.Frame(row, bg=self.C['bg'], highlightbackground='#C5CDD4', highlightthickness=1)
            cell.grid(row=0, column=i, sticky='nsew', padx=4)
            row.grid_columnconfigure(i, weight=1)
            tk.Label(cell, text=label, bg=self.C['bg'], fg='#6B7B8B',
                     font=('Segoe UI', 8)).pack(anchor='w', padx=8, pady=(4, 0))
            lbl = tk.Label(cell, text="—", bg=self.C['bg'], fg=self.C['primary'],
                           font=('Consolas', 11, 'bold'))
            lbl.pack(anchor='w', padx=8, pady=(0, 6))
            self.modal_labels[key] = lbl

    # ---------- UI 更新 ----------
    def _set_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_text.configure(state='normal')
        self.log_text.insert('end', line)
        self.log_text.see('end')
        # 限制行数
        lines = self.log_text.get('1.0', 'end-1c').split('\n')
        if len(lines) > 500:
            self.log_text.delete('1.0', f'{len(lines)-500}.0')
        self.log_text.configure(state='disabled')

    def _reset_detail(self):
        for k, lbl in self.summary.items():
            lbl.config(text="—")
        self.verdict_lbl.config(text="—  waiting for detection  —", fg=self.C['primary'])
        self.conf_lbl.config(text="")
        self._draw_meter(None)
        self._show_viz_placeholder()
        self._orig_pil_cache = None
        self._viz_pil_cache = None
        for key, (bar, lbl) in self.gate_widgets.items():
            bar.config(value=0)
            lbl.config(text="—")
        for w in self.heur_inner.winfo_children():
            w.destroy()
        for k, lbl in self.modal_labels.items():
            lbl.config(text="—")

    def _draw_meter(self, p_fake):
        self._last_p_fake = p_fake
        c = self.meter_canvas
        c.delete('all')
        w = max(c.winfo_width(), 200)
        h = 28
        # 刻度线 0 / 0.5 / 1.0
        for x in (w * 0.0, w * 0.5, w * 1.0):
            c.create_line(x, 0, x, h, fill='#B0B8C0', width=1)
        if p_fake is None:
            self.meter_text.config(text="—", fg='#6B7B8B')
            return
        # 渐变填充 (绿→黄→红)
        thr_x = w * (self.current_threshold if hasattr(self, 'current_threshold') and self.current_threshold else 0.5)
        c.create_rectangle(0, 0, p_fake * w, h, fill=self.C['fake'] if p_fake > (self.current_threshold if hasattr(self,'current_threshold') and self.current_threshold else 0.5) else self.C['real'], outline='')
        c.create_line(thr_x, 0, thr_x, h, fill=self.C['accent'], width=2, dash=(4,2))
        self.meter_text.config(text=f"AI Probability = {p_fake:.4f}",
                                fg=self.C['fake'] if p_fake > (self.current_threshold if hasattr(self,'current_threshold') and self.current_threshold else 0.5) else self.C['real'])

    def _update_detail(self, result: dict):
        self.current_threshold = result.get('threshold', 0.5)

        self.summary['file'].config(text=result['file_name'])
        self.summary['type'].config(text=result['file_type'].upper())
        thr = result.get('threshold')
        self.summary['threshold'].config(text=f"{thr:.2f}" if thr is not None else "0.50")

        if result['is_ai'] is None:
            self.verdict_lbl.config(text="—", fg=self.C['primary'])
            self.conf_lbl.config(text="")
        else:
            verdict = "🤖 AI GENERATED" if result['is_ai'] else "📷 REAL"
            fg = self.C['fake'] if result['is_ai'] else self.C['real']
            self.verdict_lbl.config(text=verdict, fg=fg)
            self.conf_lbl.config(text=f"Confidence {result['confidence']:.2%}  |  {result['elapsed']:.1f}s")

        # Meter
        self._draw_meter(result.get('fused_score'))

        # Gates
        gates = result.get('gates') or {}
        for key, (bar, lbl) in self.gate_widgets.items():
            val = gates.get(key)
            if val is not None:
                bar.config(value=val)
                lbl.config(text=f"{val:.3f}")
            else:
                bar.config(value=0)
                lbl.config(text="—")

        # Heuristics
        for w in self.heur_inner.winfo_children():
            w.destroy()
        heur = result.get('heuristics') or {}
        if heur:
            top_items = sorted(heur.items(), key=lambda kv: -abs(kv[1] - 0.5))[:8]
            for name, val in top_items:
                row = tk.Frame(self.heur_inner, bg='white')
                row.pack(fill='x', padx=6, pady=1)
                tk.Label(row, text=name[:22], bg='white', anchor='w',
                         font=('Consolas', 9)).pack(side='left')
                bar = ttk.Progressbar(row, mode='determinate', maximum=1.0, length=140)
                bar.config(value=float(val) if 0 <= val <= 1 else max(0.0, min(1.0, val)))
                bar.pack(side='left', padx=6)
                tk.Label(row, text=f"{val:.4f}", bg='white', anchor='w',
                         font=('Consolas', 9)).pack(side='left')
        else:
            tk.Label(self.heur_inner, text="No heuristic features (audio / error)",
                     bg='white', fg='#6B7B8B', font=('Segoe UI', 9)).pack(padx=8, pady=12)

        # Modal panel
        self.modal_labels['p_img'].config(
            text=f"{result['p_img']:.3f}" if result['p_img'] is not None else "—")
        self.modal_labels['p_vid'].config(
            text=f"{result['p_vid']:.3f}" if result['p_vid'] is not None else "—")
        self.modal_labels['p_aud'].config(
            text=f"{result['p_aud']:.3f}" if result['p_aud'] is not None else "—")
        mask = result.get('modal_mask') or {}
        if mask:
            parts = []
            for k in ['img', 'vid', 'aud']:
                parts.append(f"{k}={'✓' if mask.get(k) else '·'}")
            self.modal_labels['mask'].config(text=' '.join(parts))
        else:
            self.modal_labels['mask'].config(text="—")

        # 动态生成的启发式控件也应用当前窗口缩放比例。
        self._apply_responsive_layout(force=True)

        # 更新可视化面板
        self._display_images(result)

    # ---------- 事件处理 ----------
    def _get_model_config(self):
        """从下拉框读取当前配置"""
        strat_label = self.strategy_var.get()
        frames_label = self.frames_var.get()
        # 反查 preset value
        strategy = next((v for label, v in STRATEGY_PRESETS if label == strat_label), "hybrid")
        num_frames = next((v for label, v in FRAME_OPTIONS if label == frames_label), 8)
        return strategy, num_frames

    def _on_load_model(self):
        if self.is_running:
            return
        dev = self.device_var.get()
        strategy, num_frames = self._get_model_config()

        self._set_log(f"Loading model on {dev} | strategy={strategy} | frames={num_frames} ...")
        self.btn_load.config(state='disabled', text="Loading...")
        self.btn_start.config(state='disabled')
        self.is_ready = False

        self.model_info_lbl.config(
            text=f"{strategy} · frames={num_frames} · clip-vit-b/16 · wav2vec2-base-960h"
        )

        self.worker = DetectorWorker(
            self.msg_queue,
            device=dev,
            strategy=strategy,
            num_frames=num_frames,
        )
        self.worker.start()

    def _on_add_file(self):
        paths = filedialog.askopenfilenames(
            title="Select files",
            filetypes=[("Media", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.mp3 *.wav *.flac *.aac *.ogg"),
                       ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("Audio", "*.mp3 *.wav *.flac *.aac *.ogg"),
                       ("All", "*.*")]
        )
        for p in paths:
            if classify_file(p) != 'unknown':
                self._add_item(p)

    def _on_add_folder(self):
        folder = filedialog.askdirectory(title="Select folder to scan")
        if not folder:
            return
        self._set_log(f"Scanning folder: {folder}")
        files = scan_folder(folder)
        self._set_log(f"Found {len(files)} files")
        for p in files:
            self._add_item(p)

    def _add_item(self, path: str):
        if path in self.task_files:
            return
        ftype = classify_file(path)
        self.task_files.append(path)
        self.task_results.append(None)
        self.tree.insert('', 'end', iid=str(len(self.task_files) - 1),
                         values=("⏳", ftype, path))

    def _on_clear(self):
        self.task_files = []
        self.task_results = []
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.progress.config(value=0)
        self._reset_detail()

    def _on_start(self):
        if self.is_running or not self.is_ready:
            return
        pending = [(i, f) for i, f in enumerate(self.task_files) if self.task_results[i] is None]
        if not pending:
            messagebox.showinfo("Info", "No pending files")
            return
        self.is_running = True
        self.btn_start.config(text="Running...", state='disabled')
        self.progress.config(value=0, maximum=len(pending))
        threading.Thread(target=self._run_batch, args=(pending,), daemon=True).start()

    def _run_batch(self, pending):
        completed = 0
        # 读 GUI 当前下拉框的 strategy（每次 batch 开始时取一次）
        strat_label = self.strategy_var.get()
        active_strategy = next(
            (v for label, v in STRATEGY_PRESETS if label == strat_label),
            self.worker.strategy
        )
        for idx, path in pending:
            if self.worker._stop_event.is_set():
                break
            self.msg_queue.put(('log', f"[Detect] {os.path.basename(path)}"))
            result = self.worker.detect_one(path, strategy_override=active_strategy)
            self.task_results[idx] = result
            self.msg_queue.put(('result', (idx, result)))
            completed += 1
            self.msg_queue.put(('progress', completed))
        self.msg_queue.put(('batch_done', len(pending)))

    def _on_select(self, _evt):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        r = self.task_results[idx]
        if r:
            self._update_detail(r)

    def _on_export(self):
        if not any(self.task_results):
            messagebox.showinfo("Info", "No results to export")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(PROJECT_ROOT / 'outputs' / f"gui_results_{ts}.json")
        out = filedialog.asksaveasfilename(
            title="Export results", defaultextension=".json",
            initialfile=default, filetypes=[("JSON", "*.json")]
        )
        if not out:
            return
        data = []
        for r in self.task_results:
            if r is None:
                continue
            # 转存 numpy float
            entry = {}
            for k, v in r.items():
                if isinstance(v, dict):
                    entry[k] = {kk: float(vv) if hasattr(vv, '__float__') else vv for kk, vv in v.items()}
                else:
                    try:
                        entry[k] = float(v) if hasattr(v, '__float__') and not isinstance(v, str) else v
                    except Exception:
                        entry[k] = v
            data.append(entry)
        try:
            os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._set_log(f"Exported {len(data)} results to {out}")
            messagebox.showinfo("OK", f"Exported {len(data)} results\n→ {out}")
        except Exception as e:
            self._set_log(f"Export failed: {e}")
            messagebox.showerror("Error", str(e))

    # ---------- 队列消息轮询 ----------
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                tag, payload = msg
                if tag == 'log':
                    self._set_log(payload)
                elif tag == 'ready':
                    self.is_ready = True
                    self.btn_load.config(text="✓ Model Ready", bg=self.C['real'], fg='white',
                                          state='normal')
                    self.btn_start.config(state='normal')
                    self._set_log("Model is ready. Add files and click Start Detection.")
                elif tag == 'error':
                    self.btn_load.config(state='normal', text="Load Model")
                    self.is_ready = False
                    messagebox.showerror("Error", f"Model load failed:\n{payload}")
                elif tag == 'result':
                    idx, result = payload
                    # 更新 Treeview
                    status = "🤖" if result['is_ai'] else "📷" if result['is_ai'] is False else "?"
                    extra = f" {result['fused_score']:.3f}" if result['fused_score'] is not None else ""
                    self.tree.item(str(idx), values=(f"{status}{extra}", result['file_type'],
                                                     result['file_path']))
                    # 自动选中最新结果
                    self.tree.selection_set(str(idx))
                    self._update_detail(result)
                elif tag == 'progress':
                    self.progress.config(value=payload)
                elif tag == 'batch_done':
                    self.is_running = False
                    self.btn_start.config(text="▶ Start Detection", state='normal')
                    # 汇总
                    done = [r for r in self.task_results if r is not None]
                    ai_cnt = sum(1 for r in done if r['is_ai'])
                    real_cnt = sum(1 for r in done if r['is_ai'] is False)
                    self._set_log(f"[Done] {len(done)} files — AI: {ai_cnt}, Real: {real_cnt}")
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)


# ============================================================
# 入口
# ============================================================
def main():
    root = tk.Tk()
    try:
        # Windows 下高 DPI 支持
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # ttk 主题
    style = ttk.Style()
    available = style.theme_names()
    for t in ('vista', 'clam', 'default'):
        if t in available:
            style.theme_use(t)
            break

    MainWindow(root)
    root.mainloop()


if __name__ == '__main__':
    main()
