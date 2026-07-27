#!/usr/bin/env python3
"""Extract USFM (ViT-B/16) FULL patch-token grid at block 6 (index 5) for all
cohort images — NO mean pooling. Each image -> (196, 768) grid (14x14 patches,
CLS dropped). Enables spatial/attention pooling and region masking that the
patch-mean discards.

Storage: ~0.57 MB/image float32 -> ~35 GB for 61,804 images. Sharded .npz
(compressed) for resume; float16 to roughly halve on-disk size.

Block 6 chosen because it is the most FGR-relevant layer (SGA AUC 0.653 vs
0.613 final layer; see multilayer_usfm_results.json).
"""
import sys, os, time, numpy as np, pandas as pd, torch
from functools import partial
import torch.nn as nn
from torchvision import transforms
from PIL import Image

WS = "/Users/tiago/.claude-science/orgs/2e50fc88-f384-4a2e-9328-c60d613fd12a/workspaces/50e875d8-edc0-4647-b8d5-b6df7858c9cb"
sys.path.insert(0, f"{WS}/USFM-master")
from usdsgen.modules.backbone.vision_transformer import VisionTransformer

WORK = "/Users/tiago/dev/fgr-geometry/results/img_align/_ml_worklist.csv"
SHARD_DIR = "/Users/tiago/dev/fgr-geometry/results/img_align/_patchgrid_shards"
WEIGHTS = "/Users/tiago/Downloads/USFM_latest.pth"
BLOCK = 5              # block 6 (1-indexed)
BATCH = 32; NTHREADS = 10; SHARD_EVERY = 2048   # smaller shards: grids are big
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
META = ["new_filename", "fetus_id", "in_cohort", "ga_weeks_recovered"]
torch.set_num_threads(NTHREADS)
os.makedirs(SHARD_DIR, exist_ok=True)

df = pd.read_csv(WORK)
done = set()
for fn in sorted(os.listdir(SHARD_DIR)):
    if fn.endswith(".npz"):
        z = np.load(os.path.join(SHARD_DIR, fn), allow_pickle=True)
        done.update(z["new_filename"].tolist())
todo = df[~df.new_filename.isin(done)].reset_index(drop=True)
print(f"to embed: {len(todo)} / {len(df)} ({len(done)} done)", flush=True)

model = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
    depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.0, init_values=0.1,
    use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=True,
    use_mean_pooling=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
ck = torch.load(WEIGHTS, map_location="cpu")
model.load_state_dict(ck.get("model", ck.get("state_dict", ck)), strict=False)
model.eval()

grid = {}
def hook(mod, inp, out):
    grid["g"] = out[:, 1:, :].detach()   # (B, 196, 768) — drop CLS, keep all patches
model.blocks[BLOCK].register_forward_hook(hook)

pre = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                          transforms.Normalize(MEAN, STD)])
t0 = time.time(); buf_img = []; buf_meta = []
gbuf = []; metas = []; shard_i = len(os.listdir(SHARD_DIR)); n = 0

def dump():
    global gbuf, metas, shard_i
    if not metas: return
    G = np.concatenate(gbuf).astype(np.float16)   # (n,196,768) fp16
    M = {k: np.array([m[k] for m in metas], dtype=object) for k in META}
    np.savez_compressed(os.path.join(SHARD_DIR, f"grid_{shard_i:04d}.npz"), grid=G, **M)
    shard_i += 1; gbuf = []; metas = []

for idx, r in todo.iterrows():
    try: img = pre(Image.open(r.png).convert("RGB"))
    except Exception: continue
    buf_img.append(img); buf_meta.append(r)
    if len(buf_img) == BATCH or idx == len(todo) - 1:
        with torch.no_grad():
            _ = model.forward_features(torch.stack(buf_img))
        gbuf.append(grid["g"].float().cpu().numpy()); metas.extend(buf_meta)
        n += len(buf_img); buf_img = []; buf_meta = []
        if len(metas) >= SHARD_EVERY: dump()
        if n % 4096 < BATCH:
            print(f"{n}/{len(todo)} {n/(time.time()-t0):.1f} img/s", flush=True)
dump()
print(f"DONE {n} images -> block6 patch grids (fp16) in {SHARD_DIR} in {(time.time()-t0)/60:.1f} min", flush=True)
