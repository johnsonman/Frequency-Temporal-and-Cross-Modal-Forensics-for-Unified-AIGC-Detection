"""下载更多样化的真实照片 — 多源"""
import os, sys, time, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageEnhance
from io import BytesIO
import requests

out_dir = "data/image_real2"
os.makedirs(out_dir, exist_ok=True)

TARGET = 2000
WORKERS = 30
done = [0]
lock = threading.Lock()

SOURCES = [
    # Picsum (随机真实照片)
    lambda: f"https://picsum.photos/400/400",
    # ThisPersonDoesNotExist 的人物（但可能是 AI？算了）
]

def dl(_):
    # 随机选择 Picsum 或使用 unsplash 关键词
    queries = ["nature", "city", "portrait", "animal", "food", "architecture", "sports", "travel"]
    query = random.choice(queries)
    
    # 尝试 Picsum
    try:
        r = requests.get(f"https://picsum.photos/400/400?random={random.randint(0,999999)}", 
                       timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.content) > 5000:
            img = Image.open(BytesIO(r.content)).convert("RGB")
        else:
            return False
    except:
        return False
    
    # 数据增强：随机 JPEG 压缩质量 (让不同文件大小)
    try:
        w, h = img.size
        if min(w, h) < 150: return False
        side = min(w, h)
        left = (w - side) // 2; top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((224, 224), Image.LANCZOS)
        
        # 随机颜色抖动
        if random.random() < 0.5:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.5 + random.random())
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.7 + random.random()*0.6)
        
        # 随机 JPEG 质量保存
        quality = 60 + int(random.random() * 35)  # 60-95
        
        with lock:
            idx = done[0]; done[0] += 1
        out_path = os.path.join(out_dir, f"real2_{idx:05d}.jpg")
        img.save(out_path, "JPEG", quality=quality)
        return True
    except:
        return False

print(f"Downloading {TARGET} augmented real photos → {out_dir}")
t0 = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = [pool.submit(dl, i) for i in range(TARGET + 800)]
    for fut in as_completed(futures):
        if done[0] % 200 == 0 and done[0] > 0:
            print(f"  {done[0]}/{TARGET}  ({time.time()-t0:.0f}s)")
        if done[0] >= TARGET:
            break

print(f"\n✓ {done[0]} augmented real photos in {time.time()-t0:.0f}s")
