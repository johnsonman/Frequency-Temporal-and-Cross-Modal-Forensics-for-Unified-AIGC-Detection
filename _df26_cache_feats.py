"""缓存 DF26 帧的 CLIP CLS 特征 (含水平翻转版本), 用于快速训练 T-Agg

输出: data/DF26_feats/<rel>.npy  形状 [2,T,768] float16  (0=原图, 1=水平翻转)
"""
import os, json, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
T = 8


class NpyDS(Dataset):
    def __init__(self, recs):
        self.recs = recs
        self.norm = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        arr = np.load(self.recs[i]["npy"])  # [T,224,224,3] uint8
        x = torch.from_numpy(arr).float().div_(255.).permute(0, 3, 1, 2)
        return self.norm(x), i


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open("data/DF26_npy/manifest.json", encoding="utf-8") as f:
        recs = json.load(f)
    print(f"videos {len(recs)}")

    from transformers import CLIPVisionModel
    clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()

    dl = DataLoader(NpyDS(recs), batch_size=16, num_workers=4, shuffle=False,
                    pin_memory=True, persistent_workers=True)
    t0 = time.time()
    with torch.no_grad():
        for bi, (x, ids) in enumerate(dl):
            B = x.shape[0]
            flat = x.view(B * T, 3, 224, 224).to(device, non_blocking=True)
            flat_f = torch.flip(flat, dims=[3]).contiguous()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                o1 = clip(flat).last_hidden_state[:, 0, :]
                o2 = clip(flat_f).last_hidden_state[:, 0, :]
            f = torch.stack([o1.view(B, T, -1), o2.view(B, T, -1)]).float().cpu().numpy().astype(np.float16)
            for j, idx in enumerate(ids.tolist()):
                out = os.path.join(
                    "data/DF26_feats",
                    os.path.splitext(os.path.relpath(recs[idx]["npy"], "data/DF26_npy"))[0] + ".npy")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                np.save(out, f[:, j])
                recs[idx]["feat"] = out
            if (bi + 1) % 30 == 0:
                print(f"  {bi+1}/{len(dl)} batches  {time.time()-t0:.0f}s", flush=True)

    with open("data/DF26_feats/manifest.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)
    print(f"done {len(recs)} videos, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
