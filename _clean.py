"""直接用 run_in_background 跑完整训练"""
import subprocess, sys, os
# 删除旧 checkpoint 让我们清楚知道训练是否在推进
if os.path.exists("checkpoints/finetuned/finetuned.pt"):
    os.remove("checkpoints/finetuned/finetuned.pt")
print("Cleaned old checkpoint, starting fresh...")
