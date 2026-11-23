"""Picsum 并行下载 — 快速获取真实照片"""
import os, sys, time, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO
import requests

out_dir = "data/image_real"
os.makedirs(out_dir, exist_ok=True)

print(f"Downloading real photos → {out_dir}")
t0 = time.time()
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

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = [pool.submit(dl, i) for i in range(TARGET + 500)]
    for fut in as_completed(futures):
        if done[0] % 200 == 0 and done[0] > 0:
            print(f"  {done[0]}/{TARGET}  ({time.time()-t0:.0f}s)")
        if done[0] >= TARGET:
            break

print(f"\n✓ {done[0]} real photos in {time.time()-t0:.0f}s")
