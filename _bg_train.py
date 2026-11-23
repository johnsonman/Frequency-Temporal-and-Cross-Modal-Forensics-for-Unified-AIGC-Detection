"""后台训练 — 简单 wrapper"""
import subprocess, sys, time

proc = subprocess.Popen(
    [sys.executable, "-u", "train_clip.py"],
    stdout=open("train_out.log", "w"),
    stderr=subprocess.STDOUT,
    bufsize=1,
)
print(f"PID: {proc.pid}")

# 监控进程，每 5min 打印状态
while True:
    ret = proc.poll()
    if ret is not None:
        print(f"DONE rc={ret}")
        break
    time.sleep(60)
    print(f"  [{time.strftime('%H:%M:%S')}] alive, log size={__import__('os').path.getsize('train_out.log')}")
