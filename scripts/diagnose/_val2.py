import sys, torch, json, os, glob
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

# 1. 先检查 history.json 或者手动解析输出
# 2. 直接训练脚本里的 model 加载方式来加载权重

CKPT = "checkpoints/finetuned/finetuned.pt"
print("Loading:", CKPT)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print("Keys:", list(ckpt.keys()))
print("val_acc:", ckpt.get("val_acc"), "epoch:", ckpt.get("epoch"))

# 3. 用 DetectorWorker 官方加载方式 + 手动覆盖 finetuned 权重
from run_gui import DetectorWorker
import queue, time

msg_q = queue.Queue()
w = DetectorWorker(msg_q, device='cuda', strategy='heuristic', num_frames=8)
w.start()
t0 = time.time()
while time.time() - t0 < 120:
    try:
        tag, payload = msg_q.get(timeout=1)
        if tag == 'ready': break
    except queue.Empty: pass

# 手动覆盖 ImageBranch 权重
ib = w.detector.image_branch
ib.clip_model.load_state_dict(ckpt['clip_vision'], strict=True)
ib.adapter.load_state_dict(ckpt['image_branch_heads']['adapter'])
ib.msff.load_state_dict(ckpt['image_branch_heads']['msff'])
ib.forensics.load_state_dict(ckpt['image_branch_heads']['forensics'])
ib.classifier.load_state_dict(ckpt['image_branch_heads']['classifier'])
ib.global_local_fusion.load_state_dict(ckpt['image_branch_heads']['global_local_fusion'])
print("✓ Finetuned weights injected into worker")

# 4. 跑测试
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor

proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

pairs = [(p,'image') for p in glob.glob('test/image/*') if os.path.isfile(p)]
print(f"\n=== NeuraL Only (finetuned) on test images ===")

ok = tot = 0
mistakes = []
scores_real = []; scores_fake = []

for path, ftype in pairs:
    name = os.path.basename(path)
    gt_real = name.lower().startswith('real_')
    r = w.detect_one(path, strategy_override='neural')
    # 用 ImageBranch.classifier 的 p_fake 概率
    p_fake = r.get('nn_score', 0.5)
    # 也直接 forward 拿更准的
    try:
        img = Image.open(path).convert('RGB')
        inp = proc(img, return_tensors="pt")
        pv = inp.pixel_values.cuda()
        with torch.no_grad():
            out = ib(pv)
        p_fake = torch.nn.functional.softmax(out['p_img_logits'], dim=-1)[0,1].item()
    except Exception as e:
        pass

    pred_ai = p_fake > 0.5  # Neural Only 直接用 0.5 阈值
    correct = pred_ai == (not gt_real)
    ok += int(correct); tot += 1
    if gt_real: scores_real.append(p_fake)
    else: scores_fake.append(p_fake)
    if not correct:
        gt_s = "Real" if gt_real else "Fake"
        pr_s = "AI" if pred_ai else "Real"
        mistakes.append((name, p_fake, gt_s, pr_s))

print(f"Acc: {ok}/{tot} = {ok/tot:.1%}")
print(f"real p_fake: mean={np.mean(scores_real):.3f} range=[{min(scores_real):.3f},{max(scores_real):.3f}]")
print(f"fake p_fake: mean={np.mean(scores_fake):.3f} range=[{min(scores_fake):.3f},{max(scores_fake):.3f}]")
print(f"mistakes ({len(mistakes)}):")
for m in mistakes[:10]:
    print(f"  XX p_fake={m[1]:.3f} gt={m[2]:4s} pred={m[3]:4s}  {m[0]}")
