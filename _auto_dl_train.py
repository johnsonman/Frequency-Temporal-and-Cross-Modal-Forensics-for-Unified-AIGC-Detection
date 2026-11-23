"""后台下载 + 训练 — 下载完自动开始训练"""
import os, sys, time, random, threading, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO
import requests

out_dir = "data/image_real"
os.makedirs(out_dir, exist_ok=True)

TARGET = 2500
WORKERS = 30

done = [0]
lock = threading.Lock()

def dl(_):
    try:
        r = requests.get(f"https://picsum.photos/400/400", timeout=15, 
                        headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 5000:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img = img.resize((224, 224), Image.LANCZOS)
            with lock:
                idx = done[0]; done[0] += 1
            img.save(os.path.join(out_dir, f"real_{idx:05d}.jpg"), "JPEG", quality=85)
            return True
    except:
        pass
    return False

print(f"[DL] Starting download: {TARGET} real photos with {WORKERS} workers...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = [pool.submit(dl, i) for i in range(TARGET + 800)]
    for fut in as_completed(futures):
        if done[0] % 500 == 0 and done[0] > 0:
            print(f"[DL] {done[0]}/{TARGET}  ({time.time()-t0:.0f}s)")
        if done[0] >= TARGET:
            break

elapsed = time.time() - t0
print(f"[DL] Done: {done[0]} photos in {elapsed:.0f}s ({done[0]/elapsed:.1f} img/s)")

# === 自动开始训练 ===
print("\n" + "="*60)
print("[TRAIN] Launching train_v2.py...")
print("="*60)

import subprocess
result = subprocess.run(
    [sys.executable, "train_v2.py"],
    cwd=os.path.dirname(os.path.abspath(__file__))
)
print(f"\n[TRAIN] Exit code: {result.returncode}")
