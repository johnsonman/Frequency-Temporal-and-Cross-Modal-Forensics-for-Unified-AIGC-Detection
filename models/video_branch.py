"""
视频分支 (Video Branch Φ_vid)
架构: 视频帧采样 → 共享CLIP/VIT编码器 → T-Agg时序聚合(Manba/Transformer) → 视频分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List

try:
    from transformers import CLIPVisionModel, CLIPImageProcessor
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


class ManbaAggregation(nn.Module):
    """
    Manba时序聚合 - 基于门控注意力的时序建模
    简化版Manba: Temporal Multi-head Attention with Gating
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 时序自注意力
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 门控机制
        self.gate_query = nn.Linear(embed_dim, embed_dim)
        self.gate_key = nn.Linear(embed_dim, embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 帧级权重
        self.frame_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] 帧级特征序列
        Returns:
            [B, D] 时序聚合特征
        """
        # 自注意力
        attn_out, _ = self.temporal_attn(
            self.norm1(x), self.norm1(x), self.norm1(x)
        )
        x = x + attn_out

        # FFN
        x = x + self.ffn(x)

        # 帧级加权
        weights = self.frame_weights(x)  # [B, T, 1]
        weighted = (x * weights).sum(dim=1)  # [B, D]

        return weighted


class CrossFrameAttention(nn.Module):
    """跨帧注意力 - 捕获帧间依赖"""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        q = self.norm(x)
        k = v = self.norm(x)
        attn_out, _ = self.cross_attn(q, k, v)
        return x + attn_out


class TAggregation(nn.Module):
    """
    T-Agg 时序聚合模块
    Manba/Transformer时序建模 + 帧级特征融合
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            ManbaAggregation(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.cross_frame = CrossFrameAttention(embed_dim, num_heads)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] 帧级特征序列
        Returns:
            [B, D] 聚合后的视频表征
        """
        # 应用Manba聚合层
        frame_features = x
        for layer in self.layers:
            frame_features = self.cross_frame(frame_features)

        # 最终聚合
        aggregated = self.layers[-1](frame_features)
        return self.final_norm(aggregated)


class VideoBranch(nn.Module):
    """
    视频分支 Φ_vid
    共享CLIP/VIT编码器 → T-Agg时序聚合 → 视频分类头
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        embed_dim: int = 768,
        num_classes: int = 2,
        num_frames: int = 8,
        device: str = "cuda",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.device = device

        if HAS_CLIP:
            print(f"[VideoBranch] 加载共享CLIP模型: {model_name}")
            self.clip_processor = CLIPImageProcessor.from_pretrained(model_name)
            self.clip_model = CLIPVisionModel.from_pretrained(model_name)
            hidden_size = self.clip_model.config.hidden_size
            self.adapter = nn.Linear(hidden_size, embed_dim)
            self.use_clip = True
        else:
            self.use_clip = False
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(128 * 49, embed_dim),
            )

        # T-Agg时序聚合
        self.t_agg = TAggregation(embed_dim=embed_dim, num_heads=8, num_layers=2)

        # 视频分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self.to(device)

    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        提取每帧的特征
        Args:
            frames: [B, T, 3, H, W]
        Returns:
            [B, T, D]
        """
        B, T = frames.shape[:2]
        # 展平批次和时间维度
        flat_frames = frames.view(B * T, *frames.shape[2:])  # [B*T, 3, H, W]

        if self.use_clip:
            with torch.no_grad():
                outputs = self.clip_model(flat_frames)
            # 取CLS token
            frame_feat = outputs.last_hidden_state[:, 0, :]
            frame_feat = self.adapter(frame_feat)
        else:
            frame_feat = self.backbone(flat_frames)

        # 恢复时序维度
        frame_feat = frame_feat.view(B, T, -1)  # [B, T, D]
        return frame_feat

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: [B, T, 3, 224, 224] 归一化后的视频帧张量

        Returns:
            dict:
                - f_vid: 视频表征特征 [B, D]
                - p_vid: 分类logits [B, num_classes]
                - frame_feat: 帧级特征 [B, T, D]
        """
        # 帧级特征提取
        frame_feat = self.extract_frame_features(frames)  # [B, T, D]

        # T-Agg时序聚合
        f_vid = self.t_agg(frame_feat)  # [B, D]

        # 分类
        p_vid = self.classifier(f_vid)

        return {
            "f_vid": f_vid,
            "p_vid": F.softmax(p_vid, dim=-1),
            "frame_feat": frame_feat,
        }

    def preprocess_video(self, video_path: str) -> Optional[torch.Tensor]:
        """
        预处理视频文件 - 均匀采样帧
        Returns:
            [1, T, 3, 224, 224] 或 None
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[VideoBranch] 无法打开视频: {video_path}")
            return None

        # 获取视频属性
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        # 均匀采样帧
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return None

        # 如果帧数不足，重复最后一帧
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        # 处理每帧
        if self.use_clip and hasattr(self, 'clip_processor'):
            processed = []
            from PIL import Image
            for f in frames[:self.num_frames]:
                pil_img = Image.fromarray(f)
                inputs = self.clip_processor(pil_img, return_tensors="pt")
                processed.append(inputs.pixel_values.squeeze(0))
            frames_tensor = torch.stack(processed)  # [T, 3, 224, 224]
        else:
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            frames_tensor = torch.stack([transform(f) for f in frames[:self.num_frames]])

        return frames_tensor.unsqueeze(0).to(self.device)  # [1, T, 3, 224, 224]
