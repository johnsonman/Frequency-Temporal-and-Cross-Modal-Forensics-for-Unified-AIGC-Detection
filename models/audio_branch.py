"""
音频分支 (Audio Branch Φ_aud)
架构: 音频预处理 → WavLM/XLS-R编码器 + Log-Mel 2D CNN → 注意力池化 → 音频分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    HAS_WAV2VEC2 = True
except ImportError:
    HAS_WAV2VEC2 = False


class LogMelCNN(nn.Module):
    """Log-Mel 频谱 + 2D CNN 特征提取"""

    def __init__(self, n_mels: int = 128, embed_dim: int = 768):
        super().__init__()
        self.n_mels = n_mels

        self.cnn = nn.Sequential(
            # [B, 1, n_mels, T] -> [B, 32, n_mels, T]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 32)),
            nn.Dropout(0.2),
        )

        # 全局特征池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: [B, n_mels, T] Log-Mel频谱
        Returns:
            [B, embed_dim] CNN特征
        """
        # 增加channel维度
        x = mel_spec.unsqueeze(1)  # [B, 1, n_mels, T]
        x = self.cnn(x)
        x = self.gap(x).flatten(1)
        return self.proj(x)


class AttentionPooling(nn.Module):
    """注意力池化 - 对时序特征进行加权聚合"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] 时序特征
        Returns:
            [B, D] 池化特征
        """
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)
        attn_out, weights = self.attn(query, x, x)
        return self.norm(attn_out.squeeze(1))


class MeanPooling(nn.Module):
    """简单的均值池化作为备选"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, D]"""
        return x.mean(dim=1)


class AudioBranch(nn.Module):
    """
    音频分支 Φ_aud
    WavLM/XLS-R编码器 (自监督预训练) + Log-Mel CNN 双支路 → 注意力池化 → 音频分类头
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        embed_dim: int = 768,
        num_classes: int = 2,
        sample_rate: int = 16000,
        device: str = "cuda",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sample_rate = sample_rate
        self.device = device

        # 自监督编码器
        if HAS_WAV2VEC2:
            try:
                print(f"[AudioBranch] 加载 Wav2Vec2/XLS-R模型: {model_name}")
                self.wav_processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.wav_model = Wav2Vec2Model.from_pretrained(model_name)
                hidden_size = self.wav_model.config.hidden_size
                self.wav_adapter = nn.Linear(hidden_size, embed_dim)
                self.use_ssl = True
            except Exception as e:
                print(f"[AudioBranch] SSL模型加载失败: {e}")
                self.use_ssl = False
        else:
            self.use_ssl = False

        if not self.use_ssl:
            print("[AudioBranch] 使用原始波形特征作为Fallback")

        # Log-Mel + 2D CNN 支路
        self.mel_cnn = LogMelCNN(n_mels=128, embed_dim=embed_dim)

        # 声学语义分支注意力池化
        self.attn_pool = AttentionPooling(embed_dim)
        self.mean_pool = MeanPooling()

        # 双支路融合
        if self.use_ssl:
            self.branch_fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.branch_fusion = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )

        # 音频分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self.to(device)

    def extract_wav2vec2_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """提取SSL特征 [B, T, D]"""
        with torch.no_grad():
            outputs = self.wav_model(waveform)
        feat = outputs.last_hidden_state  # [B, T, hidden_size]
        return self.wav_adapter(feat)

    def compute_logmel(self, waveform: torch.Tensor) -> torch.Tensor:
        """计算Log-Mel频谱 [B, n_mels, T]"""
        import librosa

        B = waveform.shape[0]
        mel_specs = []

        for i in range(B):
            wav = waveform[i].cpu().numpy()
            # 计算Mel频谱
            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=self.sample_rate,
                n_mels=128,
                n_fft=1024,
                hop_length=256,
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            # 归一化
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
            mel_specs.append(log_mel)

        # 统一时间长度
        max_len = max(m.shape[1] for m in mel_specs)
        padded = []
        for m in mel_specs:
            if m.shape[1] < max_len:
                pad = np.zeros((128, max_len - m.shape[1]))
                m = np.concatenate([m, pad], axis=1)
            padded.append(m)

        return torch.tensor(np.stack(padded), dtype=torch.float32, device=self.device)

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            waveform: [B, T] 音频波形 (采样率16kHz)

        Returns:
            dict:
                - f_aud: 音频表征特征 [B, D]
                - p_aud: 分类概率 [B, num_classes]
                - mel_feat: Log-Mel CNN特征 [B, D]
        """
        B = waveform.shape[0]

        # Log-Mel + CNN 支路
        mel_spec = self.compute_logmel(waveform)  # [B, 128, T]
        mel_feat = self.mel_cnn(mel_spec)  # [B, D]

        if self.use_ssl:
            # SSL 编码器支路
            ssl_feat_seq = self.extract_wav2vec2_features(waveform)  # [B, T, D]
            ssl_feat = self.attn_pool(ssl_feat_seq)  # [B, D]

            # 双支路融合
            f_aud = self.branch_fusion(torch.cat([ssl_feat, mel_feat], dim=-1))
        else:
            f_aud = self.branch_fusion(mel_feat)

        # 分类
        p_aud = self.classifier(f_aud)

        return {
            "f_aud": f_aud,
            "p_aud": F.softmax(p_aud, dim=-1),
            "mel_feat": mel_feat,
        }

    def preprocess_audio(self, audio_path: str, target_len: float = 10.0) -> Optional[torch.Tensor]:
        """
        预处理音频文件
        Args:
            audio_path: 音频文件路径
            target_len: 目标时长(秒)

        Returns:
            [1, T] 波形张量 或 None
        """
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            if y is None or len(y) < 1000:
                print(f"[AudioBranch] 音频太短: {audio_path}")
                return None

            # 截断或填充到目标长度
            target_samples = int(target_len * self.sample_rate)
            if len(y) > target_samples:
                # 取中间部分
                start = (len(y) - target_samples) // 2
                y = y[start:start + target_samples]
            elif len(y) < target_samples:
                # 尾部填充静音
                y = np.pad(y, (0, target_samples - len(y)), mode='constant')

            # 归一化
            y = y / (np.max(np.abs(y)) + 1e-8)

            return torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self.device)

        except Exception as e:
            print(f"[AudioBranch] 音频加载失败 {audio_path}: {e}")
            return None
