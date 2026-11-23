"""
图像分支 (Image Branch Φ_img)
架构: CLIP/VIT编码器 → MSFF多尺度特征融合 + SRM/FFT局部取证支路 → 图像分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

try:
    from transformers import CLIPVisionModel, CLIPImageProcessor
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


class SRMFilter(nn.Module):
    """SRM (Spatial Rich Model) 滤镜 - 用于提取高频取证特征"""

    def __init__(self):
        super().__init__()
        # 3个SRM滤镜核
        srm_kernels = [
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]],        # 内核
            [[0, 0, 0], [-1, 4, -1], [0, 0, 0]],       # 边缘核
            [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],    # 点核
        ]
        kernels = torch.tensor(srm_kernels, dtype=torch.float32)
        kernels = kernels.unsqueeze(1).repeat(3, 1, 1, 1)  # [9, 1, 3, 3] for 3 channels
        self.register_buffer('kernels', kernels)
        # 输出映射
        self.proj = nn.Conv2d(9, 16, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> [B, 16, H, W]"""
        padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        filtered = F.conv2d(padded, self.kernels, groups=3)
        return self.proj(filtered)


class FFTFeatureExtractor(nn.Module):
    """FFT频域特征提取"""

    def __init__(self, out_channels: int = 16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> [B, out_channels, 16, 16]"""
        # 转换为灰度计算FFT
        gray = x.mean(dim=1, keepdim=True)
        # 对每个样本计算2D FFT
        B = gray.shape[0]
        fft_mag = []
        fft_phase = []
        for i in range(B):
            img = gray[i, 0]
            fft = torch.fft.fft2(img)
            fft_shift = torch.fft.fftshift(fft)
            mag = torch.log(torch.abs(fft_shift) + 1.0)
            phase = torch.angle(fft_shift)
            fft_mag.append(mag)
            fft_phase.append(phase)
        fft_mag = torch.stack(fft_mag).unsqueeze(1)
        fft_phase = torch.stack(fft_phase).unsqueeze(1)
        fft_feat = torch.cat([fft_mag, fft_phase], dim=1)
        return self.proj(fft_feat)


class MultiScaleFeatureFusion(nn.Module):
    """
    MSFF - 多尺度特征融合模块
    特征投影 + 层级注意力加权 + 残差融合
    """

    def __init__(self, embed_dim: int = 768, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales

        # 每层的投影
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
            ) for _ in range(num_scales)
        ])

        # 层级注意力加权
        self.scale_attention = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, num_scales),
            nn.Softmax(dim=-1),
        )

        # 融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim * num_scales),
            nn.Linear(embed_dim * num_scales, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # 残差连接投影
        self.residual_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, features: list) -> torch.Tensor:
        """
        features: list of [B, D] (各层CLIP特征)
        返回: [B, D] 融合特征
        """
        # 投影每层特征
        projected = [proj(f) for proj, f in zip(self.projections, features)]

        # 拼接用于注意力
        concat = torch.cat(projected, dim=-1)  # [B, D*num_scales]

        # 计算每层权重
        weights = self.scale_attention(concat)  # [B, num_scales]

        # 加权融合
        weighted = torch.stack(projected, dim=-1) * weights.unsqueeze(1)
        weighted_sum = weighted.sum(dim=-1)  # [B, D]

        # MLP融合
        fused = self.fusion_mlp(concat)

        # 残差融合
        output = fused + self.residual_proj(weighted_sum)
        return output


class LocalForensicsBranch(nn.Module):
    """
    局部取证支路 - SRM高频残差 + FFT频域特征
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.srm = SRMFilter()
        self.fft = FFTFeatureExtractor(out_channels=16)

        # SRM特征处理
        self.srm_processor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, embed_dim),
        )

        # FFT特征处理
        self.fft_processor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, embed_dim),
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> [B, D]"""
        srm_feat = self.srm_processor(self.srm(x))
        fft_feat = self.fft_processor(self.fft(x))
        return self.fusion(torch.cat([srm_feat, fft_feat], dim=-1))


class ImageBranch(nn.Module):
    """
    图像分支 Φ_img
    CLIP ViT 编码器 + MSFF + 局部取证支路
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        embed_dim: int = 768,
        num_classes: int = 2,
        device: str = "cuda",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device

        if HAS_CLIP:
            print(f"[ImageBranch] 加载 CLIP 模型: {model_name}")
            self.clip_processor = CLIPImageProcessor.from_pretrained(model_name)
            self.clip_model = CLIPVisionModel.from_pretrained(model_name)
            # 获取中间层输出
            self.num_hidden_layers = self.clip_model.config.num_hidden_layers
            hidden_size = self.clip_model.config.hidden_size

            # 适配CLIP的hidden_size到目标embed_dim
            self.adapter = nn.Linear(hidden_size, embed_dim)
            self.use_clip = True
        else:
            print("[ImageBranch] transformers未安装，使用EfficientNet作为backbone")
            self.use_clip = False
            # Fallback: 简单CNN backbone
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(256 * 49, embed_dim),
            )

        # MSFF多尺度融合
        self.msff = MultiScaleFeatureFusion(embed_dim=embed_dim, num_scales=3)

        # 局部取证支路
        self.forensics = LocalForensicsBranch(embed_dim=embed_dim)

        # 全局+局部特征融合
        self.global_local_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 图像分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self.to(device)

    def extract_clip_features(self, pixel_values: torch.Tensor) -> list:
        """提取CLIP三层特征 (Layer 3, 6, 12)"""
        outputs = self.clip_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # tuple of [B, N, D]

        # 选择三层: 浅层(3), 中层(6), 深层(最后一层)
        layer_indices = [
            min(3, len(hidden_states) - 1),
            min(6, len(hidden_states) - 1),
            len(hidden_states) - 1,
        ]

        features = []
        for idx in layer_indices:
            feat = hidden_states[idx]  # [B, N, D]
            # 取CLS token
            cls_feat = feat[:, 0, :]  # [B, D]
            features.append(self.adapter(cls_feat))

        return features

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: [B, 3, 224, 224] 归一化后的图像张量

        Returns:
            dict:
                - f_img: 图像表征特征 [B, D]
                - p_img: 分类logits [B, num_classes]
                - forensics_feat: 取证特征 [B, D]
        """
        B = pixel_values.shape[0]

        if self.use_clip:
            # 多尺度CLIP特征提取
            ms_features = self.extract_clip_features(pixel_values)
            # MSFF融合
            fused_global = self.msff(ms_features)
        else:
            fused_global = self.backbone(pixel_values)

        # 局部取证特征
        forensics_feat = self.forensics(pixel_values)

        # 全局+局部融合
        f_img = self.global_local_fusion(torch.cat([fused_global, forensics_feat], dim=-1))

        # 分类
        p_img_logits = self.classifier(f_img)
        p_img = F.softmax(p_img_logits, dim=-1)

        return {
            "f_img": f_img,
            "p_img": p_img,
            "p_img_logits": p_img_logits,
            "forensics_feat": forensics_feat,
        }

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理单张图像"""
        if self.use_clip and hasattr(self, 'clip_processor'):
            # CLIP预处理
            from PIL import Image
            if isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image[:, :, ::-1] if image.shape[2] == 3 else image)
            else:
                pil_img = image
            inputs = self.clip_processor(pil_img, return_tensors="pt")
            return inputs.pixel_values.to(self.device)
        else:
            # 简单预处理
            from PIL import Image
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            return transform(image).unsqueeze(0).to(self.device)
