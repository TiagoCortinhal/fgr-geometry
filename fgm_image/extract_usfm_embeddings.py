#!/usr/bin/env python3
"""Extract frozen USFM (ViT-B/16, 768-d mean-pooled) embeddings for all
preprocessed IMPACT+clinical images -> single .npz (float32 matrix + metadata).
Faithful to fgm/embeddings.py USFMBackbone. Shard-checkpoints for resume."""
import sys, os, time, numpy as np, pandas as pd, torch
from functools import partial
import torch.nn as nn
from torchvision import transforms
from PIL import Image
WS="/Users/tiago/.claude-science/orgs/2e50fc88-f384-4a2e-9328-c60d613fd12a/workspaces/50e875d8-edc0-4647-b8d5-b6df7858c9cb"
sys.path.insert(0, f"{WS}/USFM-master")
from usdsgen.modules.backbone.vision_transformer import VisionTransformer

WORK="/Users/tiago/dev/fgr-geometry/results/img_align/_embed_worklist.csv"
OUT="/Users/tiago/dev/fgr-geometry/results/img_align/emb_usfm_all.npz"
SHARD_DIR="/Users/tiago/dev/fgr-geometry/results/img_align/_usfm_shards"
WEIGHTS="/Users/tiago/Downloads/USFM_latest.pth"
BATCH=64; NTHREADS=10; SHARD_EVERY=5120
IMAGENET_MEAN=(0.485,0.456,0.406); IMAGENET_STD=(0.229,0.224,0.225)
torch.set_num_threads(NTHREADS)
os.makedirs(SHARD_DIR, exist_ok=True)

df=pd.read_csv(WORK)
# resume from any completed shards
done=set()
for fn in sorted(os.listdir(SHARD_DIR)):
    if fn.endswith(".npz"):
        z=np.load(os.path.join(SHARD_DIR,fn), allow_pickle=True)
        done.update(z["new_filename"].tolist())
if done: print(f"resuming: {len(done)} already done in shards", flush=True)
todo=df[~df.new_filename.isin(done)].reset_index(drop=True)
print(f"to embed: {len(todo)} / {len(df)}", flush=True)

model=VisionTransformer(img_size=224,patch_size=16,in_chans=3,num_classes=0,embed_dim=768,depth=12,num_heads=12,
    mlp_ratio=4,qkv_bias=True,drop_path_rate=0.0,init_values=0.1,use_abs_pos_emb=False,use_rel_pos_bias=False,
    use_shared_rel_pos_bias=True,use_mean_pooling=True,norm_layer=partial(nn.LayerNorm,eps=1e-6))
ck=torch.load(WEIGHTS,map_location="cpu")
model.load_state_dict(ck.get("model",ck.get("state_dict",ck)),strict=False); model.eval()
pre=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
META=["new_filename","dataset_type","fetus_id","in_cohort","study_date","ga_weeks_recovered","same_pregnancy"]

t0=time.time(); embs=[]; metas=[]; buf_img=[]; buf_meta=[]; n_done=0; shard_i=len(os.listdir(SHARD_DIR))
def dump_shard():
    global embs, metas, shard_i
    if not embs: return
    E=np.concatenate(embs).astype(np.float32)
    M={k:np.array([m[k] for m in metas], dtype=object) for k in META}
    np.savez(os.path.join(SHARD_DIR, f"shard_{shard_i:04d}.npz"), emb=E, **M)
    shard_i+=1; embs=[]; metas=[]

for i,r in todo.iterrows():
    try: img=pre(Image.open(r.png).convert("RGB"))
    except Exception: continue
    buf_img.append(img); buf_meta.append(r)
    if len(buf_img)==BATCH or i==len(todo)-1:
        with torch.no_grad():
            f=model.forward_features(torch.stack(buf_img))
            if f.ndim==3: f=f.mean(1)
        embs.append(f.float().cpu().numpy()); metas.extend(buf_meta)
        n_done+=len(buf_img); buf_img=[]; buf_meta=[]
        if sum(len(e) for e in embs)>=SHARD_EVERY: dump_shard()
        if n_done % (BATCH*20)==0:
            el=time.time()-t0; rate=n_done/el
            print(f"  {n_done}/{len(todo)}  {rate:.1f} img/s  ETA {(len(todo)-n_done)/rate/60:.0f} min", flush=True)
dump_shard()

# consolidate all shards -> single npz
allE=[]; allM={k:[] for k in META}
for fn in sorted(os.listdir(SHARD_DIR)):
    if not fn.endswith(".npz"): continue
    z=np.load(os.path.join(SHARD_DIR,fn), allow_pickle=True)
    allE.append(z["emb"])
    for k in META: allM[k].extend(z[k].tolist())
E=np.concatenate(allE).astype(np.float32)
np.savez_compressed(OUT, emb=E, **{k:np.array(v,dtype=object) for k,v in allM.items()},
                    emb_cols=np.array([f"emb_{i}" for i in range(768)]))
print(f"DONE {E.shape[0]} embeddings, dim {E.shape[1]} -> {OUT}  in {(time.time()-t0)/60:.1f} min", flush=True)
