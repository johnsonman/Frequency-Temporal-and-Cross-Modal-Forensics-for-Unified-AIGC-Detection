import sys, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from PIL import Image
import numpy as np
from transformers import CLIPImageProcessor

# 1. 创建 2 个 processor
proc_a = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

from models.image_branch import ImageBranch
ib = ImageBranch(embed_dim=768, num_classes=2)
proc_b = ib.clip_processor

print(f"proc_a normalize: mean={proc_a.image_mean}, std={proc_a.image_std}")
print(f"proc_b normalize: mean={proc_b.image_mean}, std={proc_b.image_std}")
print(f"proc_a size: {proc_a.size}, proc_b size: {proc_b.size}")

# 2. 同一张图
img = Image.open("test/image/0001.jpg").convert("RGB")

inp_a = proc_a(img, return_tensors="pt").pixel_values.squeeze(0)
inp_b = proc_b(img, return_tensors="pt").pixel_values.squeeze(0)

print(f"\ninp_a: mean={inp_a.mean():.4f} std={inp_a.std():.4f} shape={inp_a.shape}")
print(f"inp_b: mean={inp_b.mean():.4f} std={inp_b.std():.4f} shape={inp_b.shape}")
print(f"max diff: {(inp_a - inp_b).abs().max().item():.8f}")

# 3. 也试试直接 numpy array
arr = np.array(img)
inp_c = proc_b(arr, return_tensors="pt").pixel_values.squeeze(0)
print(f"\ninp_c (numpy): mean={inp_c.mean():.4f}")
print(f"max diff b vs c: {(inp_b - inp_c).abs().max().item():.8f}")

# 4. 看看训练时的图像路径是不是 test/image 里的同一张
# 训练集是 data/image_v3, 测试集是 test/image — 完全不同！
print("\n=== 训练集 vs 测试集图像格式 ===")
import glob, os
train_img = glob.glob("data/image_v3/fake/*")[0]
test_img = glob.glob("test/image/*")[0]
print(f"train: {train_img}  size={Image.open(train_img).size}")
print(f"test:  {test_img}  size={Image.open(test_img).size}")
