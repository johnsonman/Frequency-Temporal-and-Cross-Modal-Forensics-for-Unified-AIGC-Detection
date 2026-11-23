import torch
from PIL import Image
import numpy as np
from transformers import CLIPImageProcessor

# 训练时的预处理 (train_clip.py)
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# ImageBranch 推理时的预处理 (image_branch.py preprocess)
# 让我们看它到底做了什么

# 加载测试图像
img_path = "test/image/0001.jpg"
img = Image.open(img_path).convert("RGB")

# === 方式 A: 训练用的 CLIPImageProcessor ===
inp_a = proc(img, return_tensors="pt")
pv_a = inp_a.pixel_values.squeeze(0)  # [3, 224, 224]
print(f"A (train CLIPImageProcessor): mean={pv_a.mean().item():.4f} std={pv_a.std().item():.4f} min={pv_a.min():.4f} max={pv_a.max():.4f}")

# === 方式 B: ImageBranch.preprocess() ===
# 看源码它调的是 CLIPImageProcessor — 应该一样
# 让我直接 import 看
import sys; sys.path.insert(0, '.')
from models.image_branch import ImageBranch
ib = ImageBranch(embed_dim=768, num_classes=2)

img_arr = np.array(img)
pv_b = ib.preprocess(img_arr)
print(f"B (ImageBranch.preprocess): mean={pv_b.mean().item():.4f} std={pv_b.std().item():.4f} min={pv_b.min():.4f} max={pv_b.max():.4f}")

# === 方式 C: run_gui DetectorWorker 里的处理 ===
# run_gui 里 detect_image_file 调的是 MultimodalAIGCDetector.detect_image_file
# 让我看 detector.py detect_image_file

# === 关键: 直接把 A 和 B 的 tensor 比较 ===
diff = (pv_a.cpu() - pv_b.cpu()).abs().max().item()
print(f"\nMax diff between A and B: {diff:.6f}")
print(f"Are they equal? {torch.equal(pv_a.cpu(), pv_b.cpu())}")

# 另外看: CLIPImageProcessor 输出是 float32, 范围大约 [-1.8, 1.8]
# ImageBranch.preprocess 会不会用了不同的 normalize？
