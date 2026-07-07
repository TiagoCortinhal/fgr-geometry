#!/usr/bin/env python3
"""Extract USFM (ViT-B/16) patch-mean embeddings at MULTIPLE transformer
blocks (1/6/12 -> indices 0/5/11) in a single forward pass via forward hooks.

Each layer: mean over patch tokens (drop CLS) -> 768-d. Output .npz holds one
(N,768) matrix per layer (emb_l0, emb_l5, emb_l11) + metadata. Cohort images
only (61,804). Shard-checkpointed for resume. ~44 min CPU, 10 threads.

Motivation: the single final-layer mean pooling used everywhere else may be
too semantic/coarse. Early blocks (l0) carry edge/texture/echogenicity; mid
(l5) intermediate; late (l11) anatomy/semantics. Testing whether a richer
multi-layer image representation changes any image-vs-growth result.
"""
import sys, os, time, numpy as np, pandas as pd, torch
from functools import partial
import torch.nn as nn
from torchvision import transforms
from PIL import Image

WS = "/Users/tiago/.claude-science/orgs/2e50fc88-f384-4a2e-9328-c60d613fd12a/workspaces/50e875d8-edc0-4647-b8d5-b6df7858c9cb"
sys.path.insert(0, f"{WS}/USFM-master")
from usdsgen.modules.backbone.vision_transformer import VisionTransformer

WORK = "/Users/tiago/PythonProject/fgr-geometry/results/img_align/_ml_worklist.csv"
OUT = "/Users/tiago/PythonProject/fgr-geometry/results/img_align/emb_usfm_multilayer.npz"
SHARD_DIR = "/Users/tiago/PythonProject/fgr-geometry/results/img_align/_ml_shards"
WEIGHTS = "/Users/tiago/Downloads/USFM_latest.pth"
LAYERS = [0, 5, 11]
BATCH = 64; NTHREADS = 10; SHARD_EVERY = 5120
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
META = ["new_filename", "dataset_type", "fetus_id", "in_cohort", "study_date",
        "ga_weeks_recovered", "same_pregnancy"]
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

feats = {}
def mk(i):
    def hook(mod, inp, out): feats[i] = out[:, 1:, :].mean(1).detach()
    return hook
for i in LAYERS:
    model.blocks[i].register_forward_hook(mk(i))

pre = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                          transforms.Normalize(MEAN, STD)])

t0 = time.time(); buf_img = []; buf_meta = []
lbuf = {i: [] for i in LAYERS}; metas = []; shard_i = len(os.listdir(SHARD_DIR)); n = 0

def dump():
    global lbuf, metas, shard_i
    if not metas: return
    M = {k: np.array([m[k] for m in metas], dtype=object) for k in META}
    kw = {f"emb_l{i}": np.concatenate(lbuf[i]).astype(np.float32) for i in LAYERS}
    np.savez(os.path.join(SHARD_DIR, f"shard_{shard_i:04d}.npz"), **kw, **M)
    shard_i += 1; lbuf = {i: [] for i in LAYERS}; metas = []

for idx, r in todo.iterrows():
    try: img = pre(Image.open(r.png).convert("RGB"))
    except Exception: continue
    buf_img.append(img); buf_meta.append(r)
    if len(buf_img) == BATCH or idx == len(todo) - 1:
        with torch.no_grad():
            _ = model.forward_features(torch.stack(buf_img))
        for i in LAYERS: lbuf[i].append(feats[i].float().cpu().numpy())
        metas.extend(buf_meta); n += len(buf_img); buf_img = []; buf_meta = []
        if len(metas) >= SHARD_EVERY: dump()
        if n % 5120 < BATCH:
            print(f"{n}/{len(todo)} {n/(time.time()-t0):.1f} img/s", flush=True)
dump()

# merge shards
shards = sorted(f for f in os.listdir(SHARD_DIR) if f.endswith(".npz"))
acc = {f"emb_l{i}": [] for i in LAYERS}; macc = {k: [] for k in META}
for fn in shards:
    z = np.load(os.path.join(SHARD_DIR, fn), allow_pickle=True)
    for i in LAYERS: acc[f"emb_l{i}"].append(z[f"emb_l{i}"])
    for k in META: macc[k].append(z[k])
final = {k: np.concatenate(v) for k, v in acc.items()}
final.update({k: np.concatenate(v) for k, v in macc.items()})
final["layers"] = np.array(LAYERS)
np.savez(OUT, **final)
print(f"DONE {len(final['new_filename'])} images x {len(LAYERS)} layers -> {OUT} in {(time.time()-t0)/60:.1f} min", flush=True)
