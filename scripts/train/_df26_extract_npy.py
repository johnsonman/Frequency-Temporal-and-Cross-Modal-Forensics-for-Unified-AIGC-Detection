"""DF26 抽帧 v2: 无损保存为 npy [T,224,224,3] uint8 (避免JPEG压缩损失取证信号)"""
import os, json, time, glob
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor

T = 8
SIZE = 224
SRC = "data/DF26"
DST = "data/DF26_npy"
MANIFEST = os.path.join(DST, "manifest.json")

t0 = time.time()
items = []
for label in ("real", "fake"):
    for p in glob.glob(os.path.join(SRC, label, "**", "*.mp4"), recursive=True):
        items.append((p, label))
print(f"found {len(items)} videos")


def extract(job):
    src, label = job
    rel = os.path.relpath(src, SRC)
    out = os.path.join(DST, os.path.splitext(rel)[0] + ".npy")
    stem = os.path.splitext(os.path.basename(src))[0]
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None
    idx = sorted({int(i * (total - 1) / (T - 1)) for i in range(T)})
    while len(idx) < T:
        idx.append(idx[-1]); idx = sorted(idx)
    buf, cur, k = [], 0, 0
    while k < len(idx):
        if not cap.grab():
            break
        if cur == idx[k]:
            ok, fr = cap.retrieve()
            if ok:
                h, w = fr.shape[:2]
                s = min(h, w)
                y0, x0 = (h - s) // 2, (w - s) // 2
                sq = fr[y0:y0 + s, x0:x0 + s]
                sq = cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                buf.append(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
            k += 1
        cur += 1
    cap.release()
    if len(buf) < 3:
        return None
    while len(buf) < T:
        buf.append(buf[-1])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, np.stack(buf).astype(np.uint8))
    return {
        "npy": out, "label": 0 if label == "real" else 1, "group": stem,
        "category": rel.split(os.sep)[1] if len(rel.split(os.sep)) > 2 else "",
    }


records = []
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, rec in enumerate(ex.map(extract, items), 1):
        if rec:
            records.append(rec)
        if i % 300 == 0:
            print(f"  {i}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

with open(MANIFEST, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=1)

n_r = sum(1 for r in records if r["label"] == 0)
n_f = sum(1 for r in records if r["label"] == 1)
sz = sum(os.path.getsize(r["npy"]) for r in records) / 1024 ** 3
print(f"done: {len(records)} videos real={n_r} fake={n_f} "
      f"groups={len({r['group'] for r in records})} {sz:.2f}GB {time.time()-t0:.0f}s")
