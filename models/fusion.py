"""
跨模态一致性 (X-Cons) + 门控融合 (G-Fuse)
X-Cons: 同步对比学习 + 音画对齐建模
G-Fuse: 特征对齐 → 可靠性评估 → 门控加权融合 → 全连接分类器 → 最终真伪判定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CrossModalConsistency(nn.Module):
    """
    X-Cons 跨模态一致性模块
    同步对比学习 (f_sync) + 音画对齐建模 (f_async)
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim

        # 同步特征投影 - 将不同模态特征映射到共享空间
        self.img_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.vid_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.aud_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 音画对齐相似度计算
        self.alignment_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        # 同步一致性度量
        self.sync_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 异步特征 (音画对齐差异)
        self.async_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def compute_cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """计算余弦相似度矩阵"""
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        return torch.matmul(a_norm, b_norm.T) / self.temperature

    def forward(
        self,
        f_img: Optional[torch.Tensor] = None,
        f_vid: Optional[torch.Tensor] = None,
        f_aud: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算跨模态一致性特征

        Args:
            f_img: [B, D] 图像特征 (可选)
            f_vid: [B, D] 视频特征 (可选)
            f_aud: [B, D] 音频特征 (可选)

        Returns:
            dict:
                - f_sync: 同步一致性特征 [B, D]
                - f_async: 异步一致性特征 [B, D]
                - align_score: 音画对齐分数 [B, 1]
        """
        device = f_img.device if f_img is not None else (
            f_vid.device if f_vid is not None else f_aud.device
        )
        B = 1  # 单样本默认
        dummy = torch.zeros(1, self.embed_dim, device=device)

        # 投影到共享空间
        img_p = self.img_proj(f_img) if f_img is not None else dummy
        vid_p = self.vid_proj(f_vid) if f_vid is not None else dummy
        aud_p = self.aud_proj(f_aud) if f_aud is not None else dummy

        # 同步特征 (三模态融合)
        sync_input = torch.cat([img_p, vid_p, aud_p], dim=-1)
        f_sync = self.sync_mlp(sync_input)

        # 异步特征 - 音画对齐差异
        if f_vid is not None and f_aud is not None:
            # 音画对齐分数
            align_input = torch.cat([vid_p, aud_p], dim=-1)
            align_score = torch.sigmoid(self.alignment_head(align_input))

            # 异步特征 (差异)
            f_async = self.async_mlp(torch.cat([vid_p - aud_p, vid_p * aud_p], dim=-1))
        else:
            f_async = self.async_mlp(torch.cat([img_p - vid_p, img_p * vid_p], dim=-1))
            align_score = torch.ones(img_p.shape[0], 1, device=device) * 0.5

        return {
            "f_sync": f_sync,
            "f_async": f_async,
            "align_score": align_score,
        }


class ReliabilityAssessor(nn.Module):
    """
    可靠性评估器 - 评估每个模态的可信度
    基于特征范数、不确定性和模态特定属性
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.reliability_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Dict[str, Optional[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        评估各模态可靠性

        Args:
            features: {'img': f_img, 'vid': f_vid, 'aud': f_aud}

        Returns:
            {'img': w_img, 'vid': w_vid, 'aud': w_aud} - 各模态权重 [B, 1]
        """
        weights = {}
        # 找到第一个非None的特征来获取device和batch_size
        ref_feat = next((f for f in features.values() if f is not None), None)
        if ref_feat is not None:
            device = ref_feat.device
            batch_size = ref_feat.shape[0]
        else:
            device = 'cpu'
            batch_size = 1

        for key, feat in features.items():
            if feat is not None:
                # 基于特征本身评估可靠性
                w = self.reliability_head(feat)
                # 额外加入特征范数作为可信度因子
                norm_factor = torch.sigmoid(feat.norm(dim=-1, keepdim=True) / (feat.shape[-1] ** 0.5))
                weights[key] = w * norm_factor
            else:
                # 缺失模态权重为0
                weights[key] = torch.zeros(batch_size, 1, device=device)

        return weights


class GatedFusion(nn.Module):
    """
    G-Fuse 门控融合模块
    特征对齐 → 可靠性评估 → 门控加权融合 → 全连接分类器 → 最终真伪判定

    支持模态掩码 m_k (处理缺失模态)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_modalities: int = 3,  # img, vid, aud
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # 特征对齐 - 将不同模态特征映射到统一空间
        self.align_projs = nn.ModuleDict({
            'img': nn.Linear(embed_dim, embed_dim),
            'vid': nn.Linear(embed_dim, embed_dim),
            'aud': nn.Linear(embed_dim, embed_dim),
            'sync': nn.Linear(embed_dim, embed_dim),
            'async': nn.Linear(embed_dim, embed_dim),
        })

        # 可靠性评估
        self.reliability = ReliabilityAssessor(embed_dim)

        # 门控网络 - 学习各模态的融合权重
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * (num_modalities + 2), embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, num_modalities + 2),
            nn.Softmax(dim=-1),
        )

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 5, embed_dim // 2),  # +5 for reliability + consistency features
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, num_classes),
        )

    def forward(
        self,
        f_img: Optional[torch.Tensor] = None,
        f_vid: Optional[torch.Tensor] = None,
        f_aud: Optional[torch.Tensor] = None,
        f_sync: Optional[torch.Tensor] = None,
        f_async: Optional[torch.Tensor] = None,
        m_img: Optional[torch.Tensor] = None,
        m_vid: Optional[torch.Tensor] = None,
        m_aud: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        门控融合推理

        Args:
            f_img, f_vid, f_aud: 各模态特征
            f_sync, f_async: 跨模态一致性特征
            m_img, m_vid, m_aud: 模态掩码 (1=存在, 0=缺失)

        Returns:
            dict:
                - p_fake: 最终AI生成概率 [B, num_classes]
                - gates: 门控权重 [B, 5]
                - reliability: 各模态可靠性分数 dict
        """
        # 获取batch size
        device = f_img.device if f_img is not None else (
            f_vid.device if f_vid is not None else f_aud.device
        )
        B = 1
        dummy = torch.zeros(B, self.embed_dim, device=device)

        # 特征对齐
        aligned = {}
        aligned['img'] = self.align_projs['img'](f_img) if f_img is not None else dummy
        aligned['vid'] = self.align_projs['vid'](f_vid) if f_vid is not None else dummy
        aligned['aud'] = self.align_projs['aud'](f_aud) if f_aud is not None else dummy
        aligned['sync'] = self.align_projs['sync'](f_sync) if f_sync is not None else dummy
        aligned['async'] = self.align_projs['async'](f_async) if f_async is not None else dummy

        # 可靠性评估
        reliability = self.reliability({
            'img': f_img, 'vid': f_vid, 'aud': f_aud
        })

        # 构建门控输入 (拼接所有对齐特征)
        gate_input = torch.cat([
            aligned['img'], aligned['vid'], aligned['aud'],
            aligned['sync'], aligned['async'],
        ], dim=-1)

        # 计算门控权重
        gates = self.gate_network(gate_input)  # [B, 5]

        # 模态掩码 - 缺失模态权重置0
        masks = []
        for m in [m_img, m_vid, m_aud]:
            masks.append(m if m is not None else torch.ones(B, 1, device=device))
        mask_tensor = torch.cat(masks + [
            torch.ones(B, 1, device=device),  # sync始终存在
            torch.ones(B, 1, device=device),  # async始终存在
        ], dim=1)  # [B, 5]

        gated_weights = gates * mask_tensor
        gated_weights = gated_weights / (gated_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 加权融合
        stacked_features = torch.stack([
            aligned['img'], aligned['vid'], aligned['aud'],
            aligned['sync'], aligned['async'],
        ], dim=1)  # [B, 5, D]

        fused = torch.sum(stacked_features * gated_weights.unsqueeze(-1), dim=1)  # [B, D]

        # 构建辅助特征 (可靠性分数 + 一致性指标)
        rel_scores = torch.cat([
            reliability['img'], reliability['vid'], reliability['aud']
        ], dim=-1)  # [B, 3]
        consistency_feat = torch.cat([
            F.cosine_similarity(aligned['img'], aligned['vid'], dim=-1).unsqueeze(-1),
            F.cosine_similarity(aligned['vid'], aligned['aud'], dim=-1).unsqueeze(-1),
        ], dim=-1)  # [B, 2]
        aux_feat = torch.cat([rel_scores, consistency_feat], dim=-1)  # [B, 5]

        # 最终分类
        classifier_input = torch.cat([fused, aux_feat], dim=-1)
        p_fake = self.classifier(classifier_input)

        return {
            "p_fake": F.softmax(p_fake, dim=-1),
            "gates": gated_weights,
            "reliability": reliability,
            "fused_feat": fused,
        }
