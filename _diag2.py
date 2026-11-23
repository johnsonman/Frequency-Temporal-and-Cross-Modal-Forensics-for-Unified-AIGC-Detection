import sys, time
print("0. Start", flush=True); t=time.time()

print("1. import torch...", flush=True)
import torch; print(f"   done ({time.time()-t:.1f}s)", flush=True)

print("2. import numpy...", flush=True)
import numpy; print(f"   done ({time.time()-t:.1f}s)", flush=True)

print("3. import transformers CLIPImageProcessor...", flush=True)
from transformers import CLIPImageProcessor; print(f"   done ({time.time()-t:.1f}s)", flush=True)

print("4. import transformers CLIPVisionModel...", flush=True)
from transformers import CLIPVisionModel; print(f"   done ({time.time()-t:.1f}s)", flush=True)

print("5. import PIL...", flush=True)
from PIL import Image; print(f"   done ({time.time()-t:.1f}s)", flush=True)

sys.path.insert(0, '.')
print("6. from models.image_branch import ImageBranch...", flush=True)
from models.image_branch import ImageBranch; print(f"   done ({time.time()-t:.1f}s)", flush=True)

print("ALL OK", flush=True)
