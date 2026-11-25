"""预处理：所有图像统一重编码为标准 JPEG，彻底抹平源文件特征"""
import os, glob, random, io
from PIL import Image

def preprocess_all():
    # 统一配置
    TARGET_SIZE = 224
    JPEG_QUALITY = 85
    
    dirs = {
        "real": "data/image_real",       # Picsum 真实照片
        "fake": "data/image_v3/fake",    # AI 图像
    }
    
    out_dirs = {
        "real": "data/_prep/real",
        "fake": "data/_prep/fake",
    }
    
    for od in out_dirs.values():
        os.makedirs(od, exist_ok=True)
    
    for label, in_dir in dirs.items():
        out_dir = out_dirs[label]
        files = sorted([f for f in glob.glob(os.path.join(in_dir, "*")) 
                        if os.path.isfile(f)])
        print(f"\nProcessing {label}: {len(files)} files")
        
        saved = 0
        for f in files:
            try:
                img = Image.open(f).convert("RGB")
                
                # 统一变换：center crop + resize
                w, h = img.size
                side = min(w, h)
                left = (w - side) // 2; top = (h - side) // 2
                img = img.crop((left, top, left + side, top + side))
                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                
                # 统一重编码为 JPEG (quality=85) — 彻底抹去源文件格式特征
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=JPEG_QUALITY)
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
                
                out_name = f"{label}_{saved:06d}.jpg"
                img.save(os.path.join(out_dir, out_name), "JPEG", quality=JPEG_QUALITY)
                saved += 1
                if saved % 1000 == 0:
                    print(f"  {saved}/{len(files)}")
            except Exception as e:
                continue
        
        print(f"  ✓ {label}: {saved}/{len(files)} → {out_dir}")
    
    print(f"\nDone! All images preprocessed to {TARGET_SIZE}x{TARGET_SIZE} JPEG q={JPEG_QUALITY}")
    print("This should eliminate source-specific artifacts (Picsum JPEG, PNG chunks, etc.)")

if __name__ == "__main__":
    preprocess_all()
