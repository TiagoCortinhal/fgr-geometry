#!/usr/bin/env python3
"""HPC (GPU) extraction of the full USF-MAE token stack + layer/pooling probes.

Runs end-to-end on the CiTIUS HPC (V100S, torch+timm+CUDA):
  1) EXTRACT: all 12 transformer blocks x 197 tokens (CLS+196 patches) x 768-d, per frame,
     GPU-batched, written SHARDED to OUT_DIR (resumable). ~148 GB for 20,413 frames.
     Also writes a COMPACT summary file (LS = per-layer [CLS, mean-patch] (N,12,1536)
     + PT = last-layer patches (N,196,768)) that the probes read instantly.
  2) PROBE #1 (layer-weighting): per-single-layer GA-clock + learned softmax layer-mix,
     fetus-grouped 5-fold CV -> which depth carries GA best (we've only used layer 12).
  3) PROBE #2 (attention-pool): learned query over 196 last-layer patches vs flat .mean().

Config is the block below — paths already set for the HPC. Run in the Apptainer terminal:
    cd /mnt/beegfs/groups/collage/<...>/fgr-geometry
    python fgm_image/pregnancy_representation/experiments_2026-07-22/hpc_extract_and_probe.py

Stages are independent + resumable: --extract, --probe, or default (both).
"""
import os, sys, time, glob, json, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn, timm
from functools import partial
from PIL import Image
import torchvision.transforms as T

# ----------------------------- CONFIG (edit if paths move) -----------------------------
IMG_DIR = "/mnt/beegfs/groups/collage/data/IMPACT_FULL/processed/IMPACT_FULL/preprocessed"
WEIGHTS = "USF-MAE_full_pretrain_43dataset_100epochs.pt"        # in repo root
INDEX   = "fgm_image/pregnancy_representation/experiments_2026-07-22/ga_cnn/ga_cnn_index.csv"
# /mnt/beegfs/groups/collage is READ-ONLY. Outputs go INSIDE THE REPO (writable, under
# your home), NOT the group path. Default: <repo>/fgm_image/.../experiments_2026-07-22/out_usfmae/.
# Override with env GA_OUT_DIR if you want a different writable path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../experiments_2026-07-22
OUT_DIR = os.environ.get("GA_OUT_DIR", os.path.join(_SCRIPT_DIR, "out_usfmae"))
SUMMARY = os.path.join(OUT_DIR, "summaries.npz")                # compact file for probes (~5 GB)
SAVE_SHARDS = os.environ.get("GA_SAVE_SHARDS","0")=="1"         # full 148GB token store OFF by default
SHARD   = 512                                                   # frames/shard (~3.7 GB, only if SAVE_SHARDS)
BATCH   = 64                                                    # GPU forward batch
DEV     = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------------------

tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class MAEEncoder(nn.Module):
    """Standard MAE ViT-B/16 encoder; returns all 12 block outputs (CLS+196 patches)."""
    def __init__(s,img=224,patch=16,ed=768,depth=12,heads=12):
        super().__init__()
        s.patch_embed=timm.models.vision_transformer.PatchEmbed(img,patch,3,ed)
        n=s.patch_embed.num_patches
        s.cls_token=nn.Parameter(torch.zeros(1,1,ed)); s.pos_embed=nn.Parameter(torch.zeros(1,n+1,ed))
        s.blocks=nn.ModuleList([timm.models.vision_transformer.Block(ed,heads,4.,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6)) for _ in range(depth)])
        s.norm=nn.LayerNorm(ed,eps=1e-6)
    @torch.no_grad()
    def all_layers(s,x):
        h=s.patch_embed(x)+s.pos_embed[:,1:,:]
        cls=s.cls_token+s.pos_embed[:,:1,:]; h=torch.cat([cls.expand(x.shape[0],-1,-1),h],1)
        outs=[]
        for b in s.blocks: h=b(h); outs.append(h)
        return torch.stack(outs,1)          # (B,12,197,768)

def load_encoder():
    ck=torch.load(WEIGHTS,map_location="cpu",weights_only=False)
    sd=ck.get("model",ck) if isinstance(ck,dict) else ck
    enc=MAEEncoder(); miss,unexp=enc.load_state_dict({k:v for k,v in sd.items() if not k.startswith(("decoder","mask_token"))},strict=False)
    print(f"encoder loaded (missing {len(miss)} unexpected {len(unexp)})",flush=True)
    return enc.to(DEV).eval()

def build_frame_table():
    df=pd.read_csv(INDEX).copy()
    df["img"]=df["new_filename"].astype(str).apply(
        lambda n: os.path.join(IMG_DIR, n if n.endswith(".png") else n+".png"))
    ok=df["img"].apply(os.path.exists)
    print(f"index {len(df)} | resolved on HPC {ok.sum()}",flush=True)
    return df[ok].reset_index(drop=True)

# ----------------------------- STAGE 1: EXTRACT -----------------------------
def extract():
    os.makedirs(OUT_DIR,exist_ok=True)
    df=build_frame_table(); enc=load_encoder()
    n=len(df); nsh=(n+SHARD-1)//SHARD
    print(f"EXTRACT {n} frames -> {nsh} shards x {SHARD} | dev={DEV} batch={BATCH} "
          f"| SAVE_SHARDS={SAVE_SHARDS} | out={OUT_DIR}",flush=True)
    # Compact summaries written PER-SHARD (peak RAM ~1 shard, not the whole set), then
    # concatenated from the small per-shard files at the end. The giant (12,197,768) token
    # tensor is reduced to LS+PT on the GPU and freed immediately.
    sumdir=os.path.join(OUT_DIR,"_summ_parts"); os.makedirs(sumdir,exist_ok=True)
    t0=time.time()
    for si in range(nsh):
        part=os.path.join(sumdir,f"part_{si:04d}.npz")
        if os.path.exists(part) and not SAVE_SHARDS: continue      # resume: skip done parts
        sl=df.iloc[si*SHARD:(si+1)*SHARD]
        ls=np.zeros((len(sl),12,1536),np.float32); pt=np.zeros((len(sl),196,768),np.float32)
        toks_full = np.zeros((len(sl),12,197,768),np.float32) if SAVE_SHARDS else None
        for b0 in range(0,len(sl),BATCH):
            bs=sl.iloc[b0:b0+BATCH]
            x=torch.stack([tf(Image.open(p).convert("RGB")) for p in bs["img"]]).to(DEV)
            tk=enc.all_layers(x)                                   # (B,12,197,768) on GPU
            ls[b0:b0+len(bs)]=torch.cat([tk[:,:,0,:],tk[:,:,1:,:].mean(2)],-1).cpu().numpy()
            pt[b0:b0+len(bs)]=tk[:,-1,1:,:].cpu().numpy()
            if SAVE_SHARDS: toks_full[b0:b0+len(bs)]=tk.cpu().numpy()
        if SAVE_SHARDS:
            np.savez(os.path.join(OUT_DIR,f"shard_{si:04d}.npz"), tokens=toks_full,
                     names=sl["new_filename"].values, ga=sl["ga_weeks_recovered"].values,
                     plane=sl["plane_prop"].values, nid=sl["nid"].astype(str).values); del toks_full
        np.savez(part, LS=ls, PT=pt, ga=sl["ga_weeks_recovered"].values,
                 nid=sl["nid"].astype(str).values, plane=sl["plane_prop"].values,
                 names=sl["new_filename"].values); del ls, pt
        print(f"  shard {si+1}/{nsh} ({len(sl)} fr) {time.time()-t0:.0f}s "
              f"{(si+1)*SHARD/(time.time()-t0):.0f} fr/s",flush=True)
    # concat small parts -> single summary
    parts=sorted(glob.glob(os.path.join(sumdir,"part_*.npz")))
    acc={k:[] for k in ["LS","PT","ga","nid","plane","names"]}
    for p in parts:
        z=np.load(p,allow_pickle=True)
        for k in acc: acc[k].append(z[k])
    np.savez(SUMMARY, **{k:np.concatenate(v) for k,v in acc.items()})
    print(f"EXTRACT done {time.time()-t0:.0f}s | summary -> {SUMMARY} "
          f"({os.path.getsize(SUMMARY)/1e9:.1f} GB)",flush=True)

# ----------------------------- STAGE 2: PROBES -----------------------------
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def cv_ridge(X,y,grp):
    pred=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,groups=grp):
        sc=StandardScaler().fit(X[tr])
        pred[te]=Ridge(alpha=10).fit(sc.transform(X[tr]),y[tr]).predict(sc.transform(X[te]))
    return float(pearsonr(pred,y)[0]), float(np.abs(pred-y).mean())

def probe_layers(LS,ga,nid):
    res={"per_layer":{}}
    for L in range(12):
        r,mae=cv_ridge(LS[:,L,:],ga,nid); res["per_layer"][f"L{L+1}"]={"r":r,"mae":mae}
    rs=[]
    for tr,te in GroupKFold(5).split(LS,groups=nid):
        sc=StandardScaler().fit(LS[tr].reshape(len(tr),-1))
        Xtr=torch.tensor(sc.transform(LS[tr].reshape(len(tr),-1)).reshape(len(tr),12,1536),dtype=torch.float32,device=DEV)
        Xte=torch.tensor(sc.transform(LS[te].reshape(len(te),-1)).reshape(len(te),12,1536),dtype=torch.float32,device=DEV)
        ytr=torch.tensor(ga[tr],dtype=torch.float32,device=DEV)
        lw=nn.Parameter(torch.zeros(12,device=DEV)); lin=nn.Linear(1536,1).to(DEV)
        opt=torch.optim.Adam([lw]+list(lin.parameters()),1e-3)
        for _ in range(300):
            opt.zero_grad(); w=torch.softmax(lw,0)
            p=lin((Xtr*w[None,:,None]).sum(1)).squeeze(1)
            (((p-ytr)**2).mean()+1e-3*(lin.weight**2).mean()).backward(); opt.step()
        with torch.no_grad():
            w=torch.softmax(lw,0); pte=lin((Xte*w[None,:,None]).sum(1)).squeeze(1).cpu().numpy()
        rs.append((pearsonr(pte,ga[te])[0],np.abs(pte-ga[te]).mean(),w.detach().cpu().numpy()))
    res["learned_mix"]={"r":float(np.mean([x[0] for x in rs])),"mae":float(np.mean([x[1] for x in rs])),
                        "layer_weights":np.mean([x[2] for x in rs],0).round(3).tolist()}
    return res

def probe_pool(PT,ga,nid):
    r_mean,mae_mean=cv_ridge(PT.mean(1),ga,nid)
    rs=[]
    for tr,te in GroupKFold(5).split(PT,groups=nid):
        Xtr=torch.tensor(PT[tr],device=DEV); Xte=torch.tensor(PT[te],device=DEV)
        ytr=torch.tensor(ga[tr],dtype=torch.float32,device=DEV)
        q=nn.Parameter(torch.randn(768,device=DEV)*0.02); head=nn.Linear(768,1).to(DEV)
        opt=torch.optim.Adam([q]+list(head.parameters()),1e-3)
        for _ in range(300):
            opt.zero_grad()
            att=torch.softmax((Xtr@q)/768**0.5,1)
            p=head((att[:,:,None]*Xtr).sum(1)).squeeze(1)
            ((p-ytr)**2).mean().backward(); opt.step()
        with torch.no_grad():
            att=torch.softmax((Xte@q)/768**0.5,1)
            pte=head((att[:,:,None]*Xte).sum(1)).squeeze(1).cpu().numpy()
        rs.append((pearsonr(pte,ga[te])[0],np.abs(pte-ga[te]).mean()))
    return {"flat_mean":{"r":float(r_mean),"mae":float(mae_mean)},
            "attention_pool":{"r":float(np.mean([x[0] for x in rs])),"mae":float(np.mean([x[1] for x in rs]))}}

def probe():
    z=np.load(SUMMARY,allow_pickle=True)
    LS,PT,ga,nid,pl=z["LS"],z["PT"],z["ga"].astype(np.float32),z["nid"].astype(str),z["plane"]
    outdir="fgm_image/pregnancy_representation/experiments_2026-07-22/out_probe"; os.makedirs(outdir,exist_ok=True)
    for plane in [None,"cerebral"]:
        m=np.isfinite(ga)&(ga>=6)&(ga<=42)
        if plane: m&=(pl==plane)
        tag=plane or "all_planes"
        print(f"[{tag}] n={m.sum()} fetuses={len(set(nid[m]))}",flush=True)
        out={"n":int(m.sum()),"probe1_layers":probe_layers(LS[m],ga[m],nid[m]),
             "probe2_pooling":probe_pool(PT[m],ga[m],nid[m])}
        json.dump(out,open(os.path.join(outdir,f"probe_{tag}.json"),"w"),indent=2)
        p1=out["probe1_layers"]; best=max(p1["per_layer"].items(),key=lambda kv:kv[1]["r"])
        print("  per-layer r: "+" ".join(f"{k[1:]}:{v['r']:.2f}" for k,v in p1["per_layer"].items()),flush=True)
        print(f"  BEST single {best[0]} r={best[1]['r']:.3f} | learned-mix r={p1['learned_mix']['r']:.3f} | L12 r={p1['per_layer']['L12']['r']:.3f}",flush=True)
        print(f"  weights {p1['learned_mix']['layer_weights']}",flush=True)
        print(f"  pooling: flat-mean r={out['probe2_pooling']['flat_mean']['r']:.3f} -> attn r={out['probe2_pooling']['attention_pool']['r']:.3f}",flush=True)
    print(f"PROBE done -> {outdir}",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--extract",action="store_true"); ap.add_argument("--probe",action="store_true")
    a=ap.parse_args()
    if a.extract or not (a.extract or a.probe): extract()
    if a.probe   or not (a.extract or a.probe): probe()
