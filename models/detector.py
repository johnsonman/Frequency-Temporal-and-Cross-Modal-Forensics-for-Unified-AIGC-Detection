"""
多模态 AI 生成媒体检测器 (Multimodal AIGC Detector)
整合: 图像分支 + 视频分支 + 音频分支 + X-Cons跨模态一致性 + G-Fuse门控融合

支持单独/组合模态输入，最终输出真伪检测结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Union, Any
from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    """检测结果数据类"""
    file_path: str = ""
    file_type: str = ""  # image, video, audio
    p_fake: float = 0.0  # AI生成概率
    p_real: float = 0.0  # 真实概率
    is_ai: bool = False
    confidence: float = 0.0
    p_img: Optional[float] = None
    p_vid: Optional[float] = None
    p_aud: Optional[float] = None
    gates: Optional[Dict[str, float]] = None
    reliability: Optional[Dict[str, float]] = None
    modal_mask: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""


class MultimodalAIGCDetector(nn.Module):
    """
    完整的多模态AI生成媒体检测器
    基于架构图中的多模态融合框架
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        device: str = "cuda",
        img_model: str = "openai/clip-vit-base-patch16",
        aud_model: str = "facebook/wav2vec2-base-960h",
        num_frames: int = 8,
    ):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        print("=" * 60)
        print("多模态 AI 生成媒体检测器 (Multimodal AIGC Detector)")
        print("=" * 60)

        # 延迟导入以避免循环依赖
        from .image_branch import ImageBranch
        from .video_branch import VideoBranch
        from .audio_branch import AudioBranch
        from .fusion import CrossModalConsistency, GatedFusion

        # 图像分支
        self.image_branch = ImageBranch(
            model_name=img_model,
            embed_dim=embed_dim,
            num_classes=num_classes,
            device=device,
        )
        print("  ✓ 图像分支 Φ_img 已加载")

        # 视频分支
        self.video_branch = VideoBranch(
            model_name=img_model,  # 共享CLIP
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_frames=num_frames,
            device=device,
        )
        print("  ✓ 视频分支 Φ_vid 已加载 (共享CLIP)")

        # 音频分支
        self.audio_branch = AudioBranch(
            model_name=aud_model,
            embed_dim=embed_dim,
            num_classes=num_classes,
            device=device,
        )
        print("  ✓ 音频分支 Φ_aud 已加载")

        # 跨模态一致性 X-Cons
        self.cross_modal = CrossModalConsistency(embed_dim=embed_dim)
        print("  ✓ 跨模态一致性 X-Cons 已加载")

        # 门控融合 G-Fuse
        self.gated_fusion = GatedFusion(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_modalities=3,
        ).to(device)
        print("  ✓ 门控融合 G-Fuse 已加载")

        print("=" * 60)
        print(f"模型参数量: {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 60)

        self.to(device)
        self.eval()

    def extract_image_features(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取图像特征"""
        with torch.no_grad():
            return self.image_branch(pixel_values)

    def extract_video_features(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取视频特征"""
        with torch.no_grad():
            return self.video_branch(frames)

    def extract_audio_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取音频特征"""
        with torch.no_grad():
            return self.audio_branch(waveform)

    def detect(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        frames: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        执行多模态检测

        Args:
            pixel_values: [B, 3, 224, 224] 图像张量 (图像模态)
            frames: [B, T, 3, 224, 224] 视频帧张量 (视频模态)
            waveform: [B, L] 音频波形张量 (音频模态)

        Returns:
            完整的检测结果字典
        """
        B = 1  # 默认批大小
        device = self.device

        # 模态掩码
        m_img = torch.ones(B, 1, device=device) if pixel_values is not None else torch.zeros(B, 1, device=device)
        m_vid = torch.ones(B, 1, device=device) if frames is not None else torch.zeros(B, 1, device=device)
        m_aud = torch.ones(B, 1, device=device) if waveform is not None else torch.zeros(B, 1, device=device)

        # 各分支特征提取
        f_img = p_img = None
        f_vid = p_vid = None
        f_aud = p_aud = None

        if pixel_values is not None:
            img_result = self.extract_image_features(pixel_values)
            f_img = img_result['f_img']
            p_img = img_result['p_img']

        if frames is not None:
            vid_result = self.extract_video_features(frames)
            f_vid = vid_result['f_vid']
            p_vid = vid_result['p_vid']

        if waveform is not None:
            aud_result = self.extract_audio_features(waveform)
            f_aud = aud_result['f_aud']
            p_aud = aud_result['p_aud']

        # 跨模态一致性
        xcons_result = self.cross_modal(
            f_img=f_img, f_vid=f_vid, f_aud=f_aud
        )
        f_sync = xcons_result['f_sync']
        f_async = xcons_result['f_async']

        # 门控融合
        fusion_result = self.gated_fusion(
            f_img=f_img, f_vid=f_vid, f_aud=f_aud,
            f_sync=f_sync, f_async=f_async,
            m_img=m_img, m_vid=m_vid, m_aud=m_aud,
        )

        return {
            # 最终结果
            'p_fake': fusion_result['p_fake'],  # [B, 2]
            'gates': fusion_result['gates'],
            'reliability': fusion_result['reliability'],
            'fused_feat': fusion_result['fused_feat'],

            # 各模态独立结果
            'p_img': p_img,
            'p_vid': p_vid,
            'p_aud': p_aud,

            # 模态特征
            'f_img': f_img,
            'f_vid': f_vid,
            'f_aud': f_aud,
            'f_sync': f_sync,
            'f_async': f_async,

            # 模态掩码
            'modal_mask': {'img': m_img, 'vid': m_vid, 'aud': m_aud},

            # 跨模态指标
            'align_score': xcons_result['align_score'],
        }

    @torch.no_grad()
    def detect_image_file(self, image_path: str) -> DetectionResult:
        """检测单张图像"""
        from PIL import Image
        import cv2

        result = DetectionResult(file_path=image_path, file_type='image')

        try:
            # 加载图像
            img = cv2.imread(image_path)
            if img is None:
                img = np.array(Image.open(image_path).convert('RGB'))
                img = img[:, :, ::-1]  # RGB -> BGR for cv2

            # 预处理
            pixel_values = self.image_branch.preprocess(img)

            # 检测
            det = self.detect(pixel_values=pixel_values)
            self._fill_result(result, det)

        except Exception as e:
            result.notes = f"处理失败: {e}"
            print(f"[Detector] 图像检测失败 {image_path}: {e}")

        return result

    @torch.no_grad()
    def detect_video_file(self, video_path: str) -> DetectionResult:
        """检测单个视频"""
        result = DetectionResult(file_path=video_path, file_type='video')

        try:
            frames = self.video_branch.preprocess_video(video_path)
            if frames is None:
                result.notes = "视频帧提取失败"
                return result

            det = self.detect(frames=frames)
            self._fill_result(result, det)

        except Exception as e:
            result.notes = f"处理失败: {e}"
            print(f"[Detector] 视频检测失败 {video_path}: {e}")

        return result

    @torch.no_grad()
    def detect_audio_file(self, audio_path: str) -> DetectionResult:
        """检测单个音频"""
        result = DetectionResult(file_path=audio_path, file_type='audio')

        try:
            waveform = self.audio_branch.preprocess_audio(audio_path)
            if waveform is None:
                result.notes = "音频加载失败"
                return result

            det = self.detect(waveform=waveform)
            self._fill_result(result, det)

        except Exception as e:
            result.notes = f"处理失败: {e}"
            print(f"[Detector] 音频检测失败 {audio_path}: {e}")

        return result

    @torch.no_grad()
    def detect_multimodal(
        self,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> DetectionResult:
        """多模态联合检测 (视频的话会自动用video分支)"""
        result = DetectionResult(file_path=video_path or image_path or audio_path, file_type='multimodal')

        pixel_values = None
        frames = None
        waveform = None

        try:
            if image_path:
                from PIL import Image
                import cv2
                img = cv2.imread(image_path)
                if img is None:
                    img = np.array(Image.open(image_path).convert('RGB'))
                    img = img[:, :, ::-1]
                pixel_values = self.image_branch.preprocess(img)

            if video_path:
                frames = self.video_branch.preprocess_video(video_path)

            if audio_path:
                waveform = self.audio_branch.preprocess_audio(audio_path)

            det = self.detect(
                pixel_values=pixel_values,
                frames=frames,
                waveform=waveform,
            )
            self._fill_result(result, det)

        except Exception as e:
            result.notes = f"多模态检测失败: {e}"
            print(f"[Detector] 多模态检测失败: {e}")

        return result

    def _fill_result(self, result: DetectionResult, det: Dict[str, Any]):
        """将检测结果填充到DetectionResult"""
        p_fake = det['p_fake']  # [B, 2] - index 0=真实, index 1=AI生成
        result.p_real = float(p_fake[0, 0].cpu())
        result.p_fake = float(p_fake[0, 1].cpu())
        result.is_ai = result.p_fake > 0.5
        result.confidence = max(result.p_fake, result.p_real)

        if det['p_img'] is not None:
            result.p_img = float(det['p_img'][0, 1].cpu())
        if det['p_vid'] is not None:
            result.p_vid = float(det['p_vid'][0, 1].cpu())
        if det['p_aud'] is not None:
            result.p_aud = float(det['p_aud'][0, 1].cpu())

        # 门控权重
        gate_values = det['gates'][0].cpu().numpy()
        result.gates = {
            'img': float(gate_values[0]),
            'vid': float(gate_values[1]),
            'aud': float(gate_values[2]),
            'sync': float(gate_values[3]),
            'async': float(gate_values[4]),
        }

        # 模态可靠性
        result.reliability = {
            k: float(v[0, 0].cpu()) for k, v in det['reliability'].items()
        }

        # 模态掩码
        result.modal_mask = {
            k: bool(v[0, 0].cpu() > 0.5) for k, v in det['modal_mask'].items()
        }

    def forward(self, **kwargs):
        return self.detect(**kwargs)
