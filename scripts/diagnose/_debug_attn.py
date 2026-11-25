"""调试 CLIP attention 提取"""
import torch, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
import cv2, numpy as np

from models.detector import MultimodalAIGCDetector
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detector = MultimodalAIGCDetector(embed_dim=768, num_classes=2, device=DEVICE.type)
detector.eval()

clip = detector.image_branch.clip_model

# 搞清楚结构
print("=== CLIPVisionModel children ===")
for n, _ in clip.named_children():
    print(f"  {n}")

print(f"\n=== has attr ===")
for n in ['vision_model', 'visual', 'text_model', 'text_projection']:
    print(f"  has {n}: {hasattr(clip, n)}")

# 尝试切换到 eager
print(f"\n=== Set attention eager ===")
try:
    clip.config._attn_implementation = 'eager'
    print(f"  config._attn_implementation = {clip.config._attn_implementation}")
except Exception as e:
    print(f"  fail: {e}")

# 找所有 CLIPVisionModel 里的 VisionModel
def find_vision_model(mod):
    for n, c in mod.named_children():
        cls_name = type(c).__name__
        if 'Vision' in cls_name or 'Encoder' in cls_name:
            print(f"  Found: {n} -> {cls_name}")
        find_vision_model(c)

print("\n=== Vision-like modules ===")
find_vision_model(clip)

# 尝试前向 + output_attentions
img = cv2.imread('test/image/0001.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixel_values = detector.image_branch.preprocess(img_rgb)

print("\n=== Forward with output_attentions ===")
try:
    with torch.no_grad():
        out = clip(pixel_values=pixel_values, output_hidden_states=True, output_attentions=True)
    print(f"  out keys: {[k for k in dir(out) if not k.startswith('_')]}")
    if hasattr(out, 'attentions') and out.attentions is not None:
        attn = out.attentions
        print(f"  len(attn) = {len(attn)}")
        print(f"  attn[-1].shape = {attn[-1].shape}")  # [B, heads, N, N]
        # CLS 的注意力 = attn[0, :, 0, 1:] → 对 patch 的权重
        last_cls_attn = attn[-1][0, :, 0, 1:]  # [heads, 196]
        print(f"  last_cls_attn shape = {last_cls_attn.shape}")
        # 所有 head 平均
        mean_cls_attn = last_cls_attn.mean(dim=0).cpu().numpy()
        print(f"  mean cls attn = {mean_cls_attn[:10]}...  (sum={mean_cls_attn.sum():.3f})")
        side = int(np.sqrt(len(mean_cls_attn)))
        print(f"  side = {side}")
        attn_map = mean_cls_attn.reshape(side, side)
        print(f"  attn_map range = [{attn_map.min():.3f}, {attn_map.max():.3f}]")
    else:
        print(f"  no attentions available: type={type(out)}")
except Exception as e:
    print(f"  forward failed: {e}")
    import traceback; traceback.print_exc()
