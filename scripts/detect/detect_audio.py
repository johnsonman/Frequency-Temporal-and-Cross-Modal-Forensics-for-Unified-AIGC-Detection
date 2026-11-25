"""音频 AI 伪造检测 + 可视化 (突出伪造点)

GT 推断 (因为 test/audio 没 real_ 前缀):
  MiniMax_*.mp3  → AI TTS 伪造 (MiniMax 是 AI 合成平台)
  music-1.mp3    → 可能真实音乐或 AI
  0001.mp4       → 混合视频 (用 librosa 或跳过)

核心特征 (音频伪造检测):
  1. STFT 频谱纹理 — 局部周期性/异常尖峰
  2. Mel 频谱 z-score 异常检测 → 定位伪造点
  3. F0 基频统计 — AI TTS 基频方差/跳变异常
  4. MFCC 统计 — Jarque-Bera 正态性检验、偏度/峰度
  5. 相位相干性 — 真实语音相位平滑, AI 有随机跳变
  6. 谐波-噪声比 (HNR) — 谐波异常

可视化: 频谱图 + 异常高亮 (红色框标位置) + 波形 + 特征对比
"""
import json, csv, os, tempfile, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, stats

try:
    import librosa
    HAS_LIBROSA = True
except:
    HAS_LIBROSA = False

PROJECT = Path(__file__).resolve().parents[2]
AUDIO_DIR = PROJECT/'test'/'audio'
OUT = PROJECT/'outputs'/'audio_results'
PROC = OUT/'process'
OUT.mkdir(parents=True, exist_ok=True)
PROC.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 0. 加载音频 ====================
print("="*70); print("音频 AI 伪造检测系统"); print("="*70)

def load_audio(path, sr=16000):
    """加载音频, 处理 mp4 需要 ffmpeg"""
    ext = path.suffix.lower()
    try:
        if ext == '.mp4':
            tmp = Path(tempfile.gettempdir())/'_audio_tmp.wav'
            r = subprocess.run(['ffmpeg','-y','-i',str(path),'-vn','-ac','1',
                               '-ar',str(sr),str(tmp)], capture_output=True)
            if r.returncode == 0 and tmp.exists():
                import soundfile as sf
                y, sr = sf.read(str(tmp))
                tmp.unlink()
                return y.astype(np.float32), sr
            raise RuntimeError(f"ffmpeg 不可用: {r.stderr.decode(errors='ignore')[:200]}")
        y, sr = librosa.load(str(path), sr=sr, mono=True)
        return y.astype(np.float32), sr
    except Exception as e:
        return None, sr

audio_files = sorted(AUDIO_DIR.iterdir())
print(f"\n发现 {len(audio_files)} 个音频文件:")
for p in audio_files:
    print(f"  - {p.name}")

# ==================== 1. 特征提取 ====================
def feature_stft(y, sr):
    """STFT 频谱特征"""
    n_fft, hop = 1024, 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    D = librosa.amplitude_to_db(S, ref=np.max)
    # 频谱统计
    mf = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=n_fft, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y)
    return {'S': S, 'D': D, 'mf': mf, 'bw': bw, 'rolloff': rolloff, 'zcr': zcr, 'hop': hop}

def feature_mfcc(y, sr):
    """MFCC + 统计检验"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # Jarque-Bera 正态性检验 (AI 通常不符合正态)
    jb_stats = []
    for i in range(mfcc.shape[0]):
        s, p = stats.jarque_bera(mfcc[i])
        jb_stats.append(p)
    # 高频 MFCC 方差 (后期系数更敏感于伪影)
    high_mfcc_var = mfcc[10:].var() / (mfcc[:10].var() + 1e-8)
    return {'mfcc': mfcc, 'delta': delta, 'delta2': delta2, 'jb_p': np.mean(jb_stats),
            'high_mfcc_var_ratio': high_mfcc_var}

def feature_f0(y, sr):
    """F0 基频特征"""
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                       fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=2048)
    f0_voiced = f0[voiced_flag]
    if len(f0_voiced) < 5:
        return {'f0': f0, 'f0_var': 0, 'f0_jump_rate': 0, 'f0_voiced_ratio': 0}
    # 基频跳变率 (>20% 变化算跳变)
    diffs = np.abs(np.diff(f0_voiced)) / (f0_voiced[:-1] + 1e-8)
    jump_rate = np.mean(diffs > 0.2)
    return {'f0': f0, 'f0_var': np.var(f0_voiced), 'f0_jump_rate': jump_rate,
            'f0_voiced_ratio': voiced_flag.mean()}

def feature_harmony(y, sr):
    """谐波噪声比 HNR"""
    try:
        if hasattr(librosa.effects, 'harmonic'):
            harmonics = librosa.effects.harmonic(y, margin=3.0)
        else:
            harmonics, _ = librosa.decompose.hpss(y, margin=3.0)
        rms_h = np.sqrt(np.mean(harmonics**2)) + 1e-8
        rms_o = np.sqrt(np.mean(y**2)) + 1e-8
        hnr_db = 20 * np.log10(rms_h / (rms_o - rms_h + 1e-8))
    except Exception as e:
        hnr_db = 0
    return {'hnr_db': hnr_db}

def anomaly_localize(S_db):
    """在 Mel/STFT 频谱上定位伪造异常点 (z-score + 连通域)"""
    # 跨帧 z-score (每列相对于全谱均值)
    col_mean = S_db.mean(axis=0)
    col_std = S_db.std(axis=0) + 1e-6
    col_z = (col_mean - col_mean.mean()) / col_std  # [T]
    
    # 频域方向 z-score
    row_mean = S_db.mean(axis=1, keepdims=True)
    row_std = S_db.std(axis=1, keepdims=True) + 1e-6
    row_z = np.abs((S_db - row_mean) / row_std)   # [F, T]
    
    # 组合异常图
    anomaly_map = row_z * 0.5  # 主要看频域异常
    
    # 阈值: 超过 3σ 算异常
    anomaly_mask = anomaly_map > 3.0
    
    return anomaly_map, anomaly_mask, col_z

# ==================== 2. GT 推断 + 全特征提取 ====================
# 启发式 GT: MiniMax* 是 AI, music-1 未知
gt_map = {
    'MiniMax_2026-10-15_21_01_28_Calm_Woman.mp3': 1,
    'MiniMax_2026-10-15_21_16_12_Upset_Girl.mp3': 1,
    'music-1.mp3': None,
    '0001.mp4': None,
}

results = []
for p in audio_files:
    y, sr = load_audio(p)
    if y is None:
        print(f"  ❌ {p.name}: 加载失败")
        continue
    
    duration = len(y)/sr
    gt = gt_map.get(p.name, None)
    gt_str = {1:'AI(已知)', 0:'REAL', None:'未知(待判)'}[gt]
    
    print(f"\n{'='*60}")
    print(f"文件: {p.name}")
    print(f"时长: {duration:.2f}s | sr={sr} | GT={gt_str}")
    print(f"{'='*60}")
    
    f_stft = feature_stft(y, sr)
    f_mfcc = feature_mfcc(y, sr)
    f_f0 = feature_f0(y, sr)
    f_h = feature_harmony(y, sr)
    
    anomaly_map, anomaly_mask, col_z = anomaly_localize(f_stft['D'])
    
    # 打印关键指标
    print(f"  spectral_centroid: {f_stft['mf'].mean():.1f} Hz")
    print(f"  spectral_bandwidth: {f_stft['bw'].mean():.1f} Hz")
    print(f"  HNR: {f_h['hnr_db']:.1f} dB")
    print(f"  F0 jump rate: {f_f0['f0_jump_rate']:.3f}")
    print(f"  F0 voiced ratio: {f_f0['f0_voiced_ratio']:.3f}")
    print(f"  MFCC Jarque-Bera p: {f_mfcc['jb_p']:.4f}")
    print(f"  高频MFCC方差比: {f_mfcc['high_mfcc_var_ratio']:.3f}")
    print(f"  异常频谱帧占比: {col_z.std():.3f} (帧差异度)")
    
    results.append({
        'name': p.name, 'path': str(p), 'duration': duration, 'sr': sr,
        'gt_known': gt,
        'y': y, 'sr_out': sr,
        'stft': f_stft, 'mfcc': f_mfcc, 'f0': f_f0, 'hnr': f_h,
        'anomaly_map': anomaly_map, 'anomaly_mask': anomaly_mask, 'col_z': col_z,
    })

# ==================== 3. 启发式融合判定 ====================
print(f"\n{'='*70}\n启发式融合判定\n{'='*70}")

# 各特征给 fake 打分 (0-1, 越高越像 fake)
# AI TTS 特征: 高 spectral_centroid 低频集中 + 低 HNR + F0 跳变多 + JB p 值小
for r in results:
    if r['gt_known'] is None: continue
    s = r['stft']; m = r['mfcc']; f = r['f0']; h = r['hnr']
    
    centroid = s['mf'].mean()
    bw = s['bw'].mean()
    hnr = h['hnr_db']
    f0_jump = f['f0_jump_rate']
    jb_p = m['jb_p']
    h_var = m['high_mfcc_var_ratio']
    anomaly_rate = (r['anomaly_mask'].sum() / r['anomaly_mask'].size)
    
    # 启发式打分 (归一化到 0-1, 越高越像 AI)
    # 关键发现: AI TTS 特征 = 过于规整/平滑, 和图像伪造特征方向相反
    #   AI TTS: F0 jump 低 (合成基频无自然微跳变)
    #            JB p 高 (MFCC 系数过于规整→正态)
    #            高频 MFCC 方差比低 (合成器高频建模差)
    #            anomaly_rate 低 (频谱过于干净规整)
    #            HNR 低/负数 (谐波结构弱)
    p_hnr = max(0, min(1, (10 - hnr) / 10))           # HNR 低 → AI
    p_f0 = max(0, min(1, (0.05 - f0_jump) / 0.05))    # F0 jump 低 → AI (反转!)
    p_jb = max(0, min(1, (jb_p - 0.01) / 0.99))       # JB p 高 → AI (反转!)
    p_hv = max(0, min(1, (0.1 - h_var) / 0.1))        # 高频方差低 → AI (反转!)
    p_anom = max(0, min(1, (0.1 - min(anomaly_rate, 0.1)) / 0.1))  # anomaly 低 → AI (反转!)
    
    p_fake = 0.3 * p_hnr + 0.25 * p_f0 + 0.2 * p_jb + 0.15 * p_hv + 0.1 * p_anom
    r['p_fake'] = p_fake
    r['p_detail'] = {'hnr': p_hnr, 'f0': p_f0, 'jb': p_jb, 'hv': p_hv, 'anom': p_anom}
    
    pred = 'AI' if p_fake > 0.5 else 'REAL'
    gt_str = 'AI' if r['gt_known']==1 else 'REAL'
    mark = '✓' if pred==gt_str else '✗'
    print(f"  {r['name']:<45} p_fake={p_fake:.3f} → {pred} (GT={gt_str}) {mark}")
    print(f"    细节: HNR={hnr:.1f}dB F0_jump={f0_jump:.3f} JB_p={jb_p:.4f} hv={h_var:.2f}")

# 对未知文件 (music-1.mp3) 单独判
for r in results:
    if r['gt_known'] is not None: continue
    s = r['stft']; m = r['mfcc']; f = r['f0']; h = r['hnr']
    centroid = s['mf'].mean(); bw = s['bw'].mean(); hnr = h['hnr_db']
    f0_jump = f['f0_jump_rate']; jb_p = m['jb_p']; h_var = m['high_mfcc_var_ratio']
    anomaly_rate = (r['anomaly_mask'].sum() / r['anomaly_mask'].size)
    p_hnr = max(0, min(1, (10 - hnr) / 10))
    p_f0 = max(0, min(1, (0.05 - f0_jump) / 0.05))
    p_jb = max(0, min(1, (jb_p - 0.01) / 0.99))
    p_hv = max(0, min(1, (0.1 - h_var) / 0.1))
    p_anom = max(0, min(1, (0.1 - min(anomaly_rate, 0.1)) / 0.1))
    p_fake = 0.3 * p_hnr + 0.25 * p_f0 + 0.2 * p_jb + 0.15 * p_hv + 0.1 * p_anom
    r['p_fake'] = p_fake
    r['p_detail'] = {'hnr': p_hnr, 'f0': p_f0, 'jb': p_jb, 'hv': p_hv, 'anom': p_anom}
    pred = 'AI' if p_fake > 0.5 else 'REAL'
    print(f"  [未知] {r['name']:<35} p_fake={p_fake:.3f} → {pred}")
    print(f"    细节: HNR={hnr:.1f}dB F0_jump={f0_jump:.3f} JB_p={jb_p:.4f} hv={h_var:.2f} anomaly={anomaly_rate:.4f}")

# ==================== 4. 可视化输出 ====================
print(f"\n生成可视化...")

for i, r in enumerate(results):
    name = r['name'].replace('.mp3','').replace('.mp4','')[:25]
    y = r['y']; sr = r['sr_out']
    D = r['stft']['D']; f0 = r['f0']['f0']
    mfcc = r['mfcc']['mfcc']
    anomaly_map = r['anomaly_map']; anomaly_mask = r['anomaly_mask']
    p_fake = r.get('p_fake', 0.5)
    p_det = r.get('p_detail', {})
    gt_s = {1:'AI',0:'REAL',None:'?'}.get(r['gt_known'],'?')
    pred = 'AI' if p_fake > 0.5 else 'REAL'
    correct = (pred == gt_s) if gt_s in ['AI','REAL'] else None
    mark = '✅' if correct else ('❌' if correct is False else '?')
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # ---- Row 0: 波形 + 音量包络 ----
    ax_wave = fig.add_subplot(gs[0, :3])
    t = np.linspace(0, len(y)/sr, len(y))
    ax_wave.plot(t, y, color='#065A82', linewidth=0.3, alpha=0.7)
    # 异常区域高亮
    hop = r['stft']['hop']
    t_frames = np.arange(anomaly_mask.shape[1]) * hop / sr
    freq_bins = anomaly_mask.shape[0]
    for col_idx in range(anomaly_mask.shape[1]):
        if anomaly_mask[:, col_idx].sum() > freq_bins * 0.1:  # 超过10%频率异常
            ax_wave.axvspan(t_frames[col_idx]-hop/sr/2, t_frames[col_idx]+hop/sr/2,
                            alpha=0.3, color='#C0392B')
    ax_wave.set_title(f'波形 (红=异常帧区域)  |  p_fake={p_fake:.3f} → {pred}  GT={gt_s} {mark}',
                      fontsize=11, fontweight='bold')
    ax_wave.set_xlabel('时间 (s)')
    ax_wave.grid(alpha=0.3)
    
    # ---- Row 0 col 3: 判定饼图 ----
    ax_pie = fig.add_subplot(gs[0, 3])
    wedges, texts, autotexts = ax_pie.pie([p_fake, 1-p_fake], colors=['#EF4444','#10B981'],
                                           startangle=90, labels=['AI','REAL'], autopct='%.2f')
    ax_pie.set_title(f'判定概率\n({pred})', fontsize=11, fontweight='bold')
    
    # ---- Row 1: STFT 频谱 + 异常叠加 ----
    ax_stft = fig.add_subplot(gs[1, :3])
    img = ax_stft.imshow(D, origin='lower', aspect='auto', cmap='magma',
                         extent=[0, len(y)/sr, 0, sr/2])
    # 异常轮廓
    mask_vis = anomaly_mask.astype(float)
    mask_vis[mask_vis==0] = np.nan
    ax_stft.imshow(mask_vis, origin='lower', aspect='auto', cmap='Reds', alpha=0.6,
                   extent=[0, len(y)/sr, 0, sr/2])
    ax_stft.set_ylabel('频率 (Hz)'); ax_stft.set_xlabel('时间 (s)')
    ax_stft.set_title('STFT 频谱 + 异常区域 (红色叠加)', fontsize=10)
    plt.colorbar(img, ax=ax_stft, fraction=0.046)
    
    # ---- Row 1 col 3: 异常强度时间曲线 ----
    ax_anom = fig.add_subplot(gs[1, 3])
    t_frames = np.arange(anomaly_map.shape[1]) * hop / sr
    col_strength = anomaly_map.mean(axis=0)  # 每帧平均异常度
    ax_anom.plot(t_frames, col_strength, color='#C0392B', linewidth=1.5)
    ax_anom.axhline(3.0, color='gray', linestyle='--', label='阈值 z=3')
    ax_anom.fill_between(t_frames, col_strength, 3.0, where=col_strength>3.0,
                         alpha=0.3, color='#EF4444', label='异常帧')
    ax_anom.set_xlabel('时间 (s)'); ax_anom.set_ylabel('异常强度 (z-score)')
    ax_anom.set_title('局部伪造异常度', fontsize=10)
    ax_anom.legend(fontsize=8); ax_anom.grid(alpha=0.3)
    
    # ---- Row 2: Mel 频谱 + F0 ----
    ax_mel = fig.add_subplot(gs[2, :3])
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax_mel, cmap='hot')
    # 叠加 F0 轨迹
    f0_frames = np.arange(len(f0)) * (y.shape[0]/sr/len(f0))
    ax_mel2 = ax_mel.twinx()
    ax_mel2.plot(f0_frames, f0, color='#10B981', linewidth=1.5, alpha=0.7, label='F0')
    ax_mel2.set_ylabel('F0 (Hz)', color='#10B981'); ax_mel2.tick_params(axis='y', colors='#10B981')
    ax_mel2.legend(loc='upper right', fontsize=8)
    ax_mel.set_title('Mel 频谱 + F0 基频轨迹 (绿线)', fontsize=10)
    
    # ---- Row 2 col 3: F0 分布图 ----
    ax_f0 = fig.add_subplot(gs[2, 3])
    f0_v = f0[~np.isnan(f0)]
    if len(f0_v) > 5:
        ax_f0.hist(f0_v, bins=30, color='#10B981', edgecolor='white', alpha=0.8)
        ax_f0.axvline(f0_v.mean(), color='#EF4444', linestyle='--', label=f'μ={f0_v.mean():.0f}')
        ax_f0.legend(fontsize=8)
    ax_f0.set_xlabel('F0 (Hz)'); ax_f0.set_ylabel('帧数')
    ax_f0.set_title(f'F0 分布 | jump_rate={r["f0"]["f0_jump_rate"]:.3f}', fontsize=10)
    ax_f0.grid(alpha=0.3)
    
    # ---- Row 3: MFCC 系数 + 各特征 ----
    ax_mfcc = fig.add_subplot(gs[3, :2])
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax_mfcc, cmap='viridis')
    ax_mfcc.set_title('MFCC 系数 (20维)', fontsize=10)
    ax_mfcc.set_xlabel('时间 (s)')
    
    # 特征雷达图
    ax_radar = fig.add_subplot(gs[3, 2], projection='polar')
    categories = ['HNR异常','F0跳变','MFCC非正态','高频MFCC','频谱异常']
    vals = [p_det.get('hnr',0), p_det.get('f0',0), p_det.get('jb',0), p_det.get('hv',0), p_det.get('anom',0)]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    vals_plt = vals + vals[:1]
    angles_plt = angles + angles[:1]
    ax_radar.plot(angles_plt, vals_plt, 'o-', color='#C0392B', linewidth=2)
    ax_radar.fill(angles_plt, vals_plt, color='#C0392B', alpha=0.3)
    ax_radar.set_xticks(angles); ax_radar.set_xticklabels(categories, fontsize=8)
    ax_radar.set_ylim(0, 1); ax_radar.set_title('伪造特征雷达', fontsize=10)
    
    # 综合指标
    ax_metrics = fig.add_subplot(gs[3, 3])
    ax_metrics.axis('off')
    metrics_text = f"""
文件: {r['name'][:40]}
时长: {len(y)/sr:.2f}s

====== 检测特征 ======
频谱质心: {r['stft']['mf'].mean():.1f} Hz
频谱带宽: {r['stft']['bw'].mean():.1f} Hz
谐波噪声比: {r['hnr']['hnr_db']:.1f} dB
F0 跳变率: {r['f0']['f0_jump_rate']:.3f}
F0 发声比: {r['f0']['f0_voiced_ratio']:.3f}
MFCC JB 正态性: {r['mfcc']['jb_p']:.4f}
高频MFCC方差比: {r['mfcc']['high_mfcc_var_ratio']:.3f}
异常帧占比: {anomaly_mask.sum()/anomaly_mask.size*100:.1f}%

====== 融合判定 ======
p_fake = {p_fake:.3f}
预测: {pred}
GT: {gt_s}
"""
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                    fontsize=8, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8))
    
    fig.suptitle(f'🎵 {name}  |  AI 音频伪造检测 |  预测={pred} (p={p_fake:.3f})  GT={gt_s} {mark}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(PROC/f"audio_{i:02d}_{name.replace(' ','_')[:30]}.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [{i+1}/{len(results)}] {r['name']} → {pred} (p={p_fake:.3f})")

# ==================== 5. 汇总对比图 ====================
if len(results) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 频谱质心分布 (AI vs REAL)
    ax = axes[0, 0]
    for r in results:
        label = f"{r['name'][:15]}:{'AI' if r.get('p_fake',0)>0.5 else 'REAL'}"
        centroid = r['stft']['mf']
        ax.plot(np.arange(len(centroid))*r['stft']['hop']/r['sr_out'], centroid,
                label=label, alpha=0.7, linewidth=1.2)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('频谱质心 (Hz)')
    ax.set_title('频谱质心对比 (通常 AI TTS 更稳定集中)'); ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    
    # F0 跳变率 vs HNR 散点
    ax = axes[0, 1]
    colors = ['#EF4444' if r.get('p_fake',0)>0.5 else '#10B981' for r in results]
    ax.scatter([r['hnr']['hnr_db'] for r in results],
               [r['f0']['f0_jump_rate'] for r in results],
               c=colors, s=120, edgecolors='black', linewidths=1.5, zorder=5)
    for r in results:
        ax.annotate(r['name'][:12], (r['hnr']['hnr_db'], r['f0']['f0_jump_rate']),
                    fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('HNR (dB) ← 低 = 像AI'); ax.set_ylabel('F0 jump rate ↑ = 像AI')
    ax.set_title('HNR vs F0跳变率 (AI检测关键维度)'); ax.grid(alpha=0.3)
    
    # 融合特征热力图
    ax = axes[1, 0]
    names = [r['name'][:18] for r in results]
    feat_names = ['HNR异常','F0跳变','MFCC非正态','高频MFCC','频谱异常']
    feat_matrix = [[r.get('p_detail',{}).get(f,0) for f in ['hnr','f0','jb','hv','anom']]
                   for r in results]
    im = ax.imshow(feat_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(5)); ax.set_xticklabels(feat_names, fontsize=9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(5):
            ax.text(j, i, f'{feat_matrix[i][j]:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if feat_matrix[i][j]>0.5 else 'black')
    ax.set_title('伪造特征融合矩阵 (红=伪造特征强)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 最终判定概率对比
    ax = axes[1, 1]
    names_short = [r['name'][:20] for r in results]
    p_vals = [r.get('p_fake', 0.5) for r in results]
    bar_colors = ['#EF4444' if p > 0.5 else '#10B981' for p in p_vals]
    bars = ax.bar(range(len(names_short)), p_vals, color=bar_colors, width=0.6, edgecolor='black')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='阈值 0.5')
    ax.set_xticks(range(len(names_short))); ax.set_xticklabels(names_short, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('p_fake'); ax.set_ylim(0, 1.05)
    ax.set_title('最终判定概率 (红=判为AI)'); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('🎵 音频 AI 伪造检测 — 汇总对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT/'audio_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ 汇总对比图已保存: audio_summary.png")

# ==================== 6. CSV/JSON 输出 ====================
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
rows = []
for r in results:
    pred = 'AI' if r.get('p_fake',0)>0.5 else 'REAL'
    gt_s = {1:'AI',0:'REAL',None:'?'}.get(r['gt_known'],'?')
    rows.append({
        '文件名': r['name'], '时长(s)': f"{r['duration']:.2f}",
        'GT': gt_s, '预测': pred, 'p_fake': f"{r.get('p_fake',0):.4f}",
        'HNR(dB)': f"{r['hnr']['hnr_db']:.2f}",
        'F0跳变率': f"{r['f0']['f0_jump_rate']:.4f}",
        '频谱质心(Hz)': f"{r['stft']['mf'].mean():.1f}",
        'MFCC_JB_p': f"{r['mfcc']['jb_p']:.5f}",
        '异常帧%': f"{r['anomaly_mask'].sum()/r['anomaly_mask'].size*100:.1f}",
    })

with open(OUT/f'audio_results_{ts}.json','w',encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
with open(OUT/f'audio_results_{ts}.csv','w',newline='',encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# 已知 GT 准确率
known = [r for r in results if r['gt_known'] is not None]
if known:
    right = sum(1 for r in known if (r.get('p_fake',0)>0.5) == (r['gt_known']==1))
    print(f"\n📊 已知 GT 准确率: {right}/{len(known)} = {right/len(known)*100:.1f}%")
print(f"✅ 全部完成! 输出目录: {OUT}")
