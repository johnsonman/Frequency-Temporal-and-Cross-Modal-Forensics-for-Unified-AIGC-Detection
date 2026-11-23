"""Smoke test train_clip.py — 2 batches only"""
import sys, torch
sys.path.insert(0, '.')
torch.manual_seed(42)

# Patch CFG for smoke test
import train_clip as tc
tc.CFG.epochs = 1
tc.CFG.batch_size = 4
tc.CFG.num_workers = 0

# Limit dataset to 40 images
import glob, os
orig_init = tc.ImageV3Dataset.__init__
def patched_init(self, root, img_processor):
    self.img_processor = img_processor
    self.samples = []
    for label, sub in [(1, "fake"), (0, "real")]:
        files = glob.glob(os.path.join(root, sub, "*"))[:20]  # 20 each
        for f in files:
            if os.path.isfile(f):
                self.samples.append((f, label))
    print(f"  [SMOKE] Dataset: {len(self.samples)} images")
tc.ImageV3Dataset.__init__ = patched_init

# Monkey-patch train to stop after 2 batches
orig_train = tc.train
def smoke_train():
    from transformers import CLIPImageProcessor
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    ds = tc.ImageV3Dataset(tc.CFG.data_dir, processor)
    dl = torch.utils.data.DataLoader(ds, batch_size=tc.CFG.batch_size, shuffle=False,
                                      num_workers=0, drop_last=False)
    model = tc.build_model_with_lora()
    model.train()
    groups = tc.param_groups(model)
    optim = torch.optim.AdamW(groups)
    bce = torch.nn.BCEWithLogitsLoss()

    print("\n[SMOKE] Forward + backward test (2 batches)")
    for i, (pv, labels) in enumerate(dl):
        if i >= 2: break
        pv = pv.to(tc.CFG.device)
        labels = labels.to(tc.CFG.device)
        det = model.detect(pixel_values=pv)
        ms = model.image_branch.extract_clip_features(pv)
        fg = model.image_branch.msff(ms)
        fo = model.image_branch.forensics(pv)
        fi = model.image_branch.global_local_fusion(torch.cat([fg, fo], dim=-1))
        logits = model.image_branch.classifier(fi)
        lab = torch.nn.functional.one_hot(labels, 2).float()
        loss = bce(logits, lab) + 0.3 * bce(torch.log(det['p_fake'] + 1e-7), lab)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"  batch {i}: loss={loss.item():.4f} p_fake={det['p_fake'].mean(dim=0).cpu().numpy().round(3)}")
    print("[SMOKE] PASSED ✓")
smoke_train()
