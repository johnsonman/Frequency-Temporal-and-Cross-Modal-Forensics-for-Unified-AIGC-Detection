"""
用训练好的 Transformer 模型在 test/ 文件夹进行检测
支持图像 / 视频 / 音频 三种模态
"""

import os
import json
import time
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
TEST_DIR = PROJECT_ROOT / 'test'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 从 train.py 导入模型类
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from train import ImageDetectorViT, VideoDetectorViT, AudioPatchViT, MelAudioDataset


def load_checkpoint(model, ckpt_path):
    """加载 checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"  加载: {ckpt_path}")
    print(f"    Epoch: {ckpt.get('epoch', '?')}, Val Acc: {ckpt.get('val_acc', '?')}")
    return model


def predict_image(model, img_path):
    """单张图像推理"""
    model.eval()
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.array(Image.open(img_path).convert('RGB'))
        img = img[:, :, ::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item()
        ai_prob = probs[0, 1].item()

    return {
        'file_name': Path(img_path).name,
        'file_type': 'image',
        'is_ai': pred == 1,
        'ai_probability': round(ai_prob, 4),
        'confidence': round(confidence, 4),
        'predicted_label': 'ai' if pred == 1 else 'real',
    }


def predict_video(model, frames_path):
    """视频帧序列推理"""
    model.eval()
    frames = torch.load(str(frames_path), weights_only=False)
    frames = frames.float()
    if frames.max() > 1.0:
        frames = frames / 255.0
    frames = (frames - 0.5) / 0.5
    frames = frames.unsqueeze(0).to(DEVICE)  # [1, T, 3, 224, 224]

    with torch.no_grad():
        logits = model(frames)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item()
        ai_prob = probs[0, 1].item()

    return {
        'file_name': Path(frames_path).name,
        'file_type': 'video',
        'is_ai': pred == 1,
        'ai_probability': round(ai_prob, 4),
        'confidence': round(confidence, 4),
        'predicted_label': 'ai' if pred == 1 else 'real',
    }


def predict_video_mp4(model, mp4_path, num_frames=16):
    """直接从 mp4 视频文件推理"""
    model.eval()
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()

    if len(frames) < 2:
        return {'file_name': Path(mp4_path).name, 'file_type': 'video',
                'error': '无法读取视频帧'}

    # 转 tensor [T, 3, 224, 224]
    arr = np.stack(frames)
    tensor = torch.tensor(arr).permute(0, 3, 1, 2).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item()
        ai_prob = probs[0, 1].item()

    return {
        'file_name': Path(mp4_path).name,
        'file_type': 'video',
        'is_ai': pred == 1,
        'ai_probability': round(ai_prob, 4),
        'confidence': round(confidence, 4),
        'predicted_label': 'ai' if pred == 1 else 'real',
    }


def predict_audio(model, audio_path, sr=16000, duration=5.0):
    """音频推理 - 转为 Log-Mel 频谱输入"""
    model.eval()
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception as e:
        return {'file_name': Path(audio_path).name, 'file_type': 'audio',
                'error': str(e)}

    n_samples = int(duration * sr)
    if len(y) > n_samples:
        start = random.randint(0, len(y) - n_samples)
        y = y[start:start + n_samples]
    elif len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item()
        ai_prob = probs[0, 1].item()

    return {
        'file_name': Path(audio_path).name,
        'file_type': 'audio',
        'is_ai': pred == 1,
        'ai_probability': round(ai_prob, 4),
        'confidence': round(confidence, 4),
        'predicted_label': 'ai' if pred == 1 else 'real',
    }


def main():
    print(f"设备: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Test 目录: {TEST_DIR}")

    # ---- 加载模型 ----
    print("\n" + "=" * 60)
    print("加载训练好的模型...")
    print("=" * 60)

    results = []

    # 图像模型
    img_ckpt = CHECKPOINT_DIR / 'image_best.pt'
    if img_ckpt.exists():
        img_model = ImageDetectorViT(num_classes=2, pretrained=False).to(DEVICE)
        load_checkpoint(img_model, img_ckpt)

        # 图像检测
        print("\n[图像检测]")
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for f in sorted(TEST_DIR.rglob('*')):
            if f.suffix.lower() in img_exts:
                t0 = time.time()
                r = predict_image(img_model, f)
                r['inference_time_ms'] = round((time.time() - t0) * 1000, 1)
                results.append(r)
                badge = "🤖 AI" if r['is_ai'] else "📷 REAL"
                print(f"  {badge} {f.name}  p_ai={r['ai_probability']:.4f}  "
                      f"conf={r['confidence']:.4f}  ({r['inference_time_ms']}ms)")

    # 视频模型
    vid_ckpt = CHECKPOINT_DIR / 'video_best.pt'
    if vid_ckpt.exists():
        vid_model = VideoDetectorViT(num_classes=2, pretrained=False).to(DEVICE)
        load_checkpoint(vid_model, vid_ckpt)

        print("\n[视频检测]")
        vid_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        for f in sorted(TEST_DIR.rglob('*')):
            if f.suffix.lower() in vid_exts:
                t0 = time.time()
                r = predict_video_mp4(vid_model, f)
                r['inference_time_ms'] = round((time.time() - t0) * 1000, 1)
                results.append(r)
                badge = "🤖 AI" if r.get('is_ai', False) else "📷 REAL"
                if 'error' in r:
                    print(f"  ⚠ {f.name}  错误: {r['error']}")
                else:
                    print(f"  {badge} {f.name}  p_ai={r['ai_probability']:.4f}  "
                          f"conf={r['confidence']:.4f}  ({r['inference_time_ms']}ms)")

        # 也检测 .pt 视频帧
        for f in sorted(TEST_DIR.rglob('*.pt')):
            t0 = time.time()
            r = predict_video(vid_model, f)
            r['inference_time_ms'] = round((time.time() - t0) * 1000, 1)
            results.append(r)
            badge = "🤖 AI" if r['is_ai'] else "📷 REAL"
            print(f"  {badge} {f.name}  p_ai={r['ai_probability']:.4f}  "
                  f"conf={r['confidence']:.4f}  ({r['inference_time_ms']}ms)")

    # 音频模型
    aud_ckpt = CHECKPOINT_DIR / 'audio_best.pt'
    if aud_ckpt.exists():
        aud_model = AudioPatchViT(num_classes=2).to(DEVICE)
        load_checkpoint(aud_model, aud_ckpt)

        print("\n[音频检测]")
        aud_exts = {'.mp3', '.wav', '.flac', '.ogg', '.aac', '.m4a'}
        for f in sorted(TEST_DIR.rglob('*')):
            if f.suffix.lower() in aud_exts:
                t0 = time.time()
                r = predict_audio(aud_model, f)
                r['inference_time_ms'] = round((time.time() - t0) * 1000, 1)
                results.append(r)
                badge = "🤖 AI" if r.get('is_ai', False) else "📷 REAL"
                if 'error' in r:
                    print(f"  ⚠ {f.name}  错误: {r['error']}")
                else:
                    print(f"  {badge} {f.name}  p_ai={r['ai_probability']:.4f}  "
                          f"conf={r['confidence']:.4f}  ({r['inference_time_ms']}ms)")

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("检测汇总")
    print("=" * 60)

    modal_stats = {}
    for r in results:
        ft = r.get('file_type', 'unknown')
        if ft not in modal_stats:
            modal_stats[ft] = {'total': 0, 'ai': 0, 'real': 0}
        modal_stats[ft]['total'] += 1
        if r.get('is_ai', False):
            modal_stats[ft]['ai'] += 1
        else:
            modal_stats[ft]['real'] += 1

    for ft, stats in modal_stats.items():
        label_cn = {'image': '图像', 'video': '视频', 'audio': '音频'}.get(ft, ft)
        ai_ratio = stats['ai'] / stats['total'] * 100 if stats['total'] else 0
        print(f"  {label_cn}: 共 {stats['total']} 个, AI={stats['ai']}, Real={stats['real']} (AI占比 {ai_ratio:.1f}%)")

    # 保存结果
    output_dir = TEST_DIR.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = output_dir / f'transformer_results_{ts}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
