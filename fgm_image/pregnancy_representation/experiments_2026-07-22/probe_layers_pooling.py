#!/usr/bin/env python3
"""Probe the full USF-MAE token stack for a BETTER GA representation than layer-12 mean-pool.

Two lightweight heads on FROZEN tokens (no encoder training), fetus-grouped 5-fold CV:
  #1 LAYER-WEIGHTING: learn a softmax weight per layer (12) over the CLS+mean-patch of each
     layer -> which depth carries GA best; compare each single layer + the learned mix.
  #2 ATTENTION-POOL: a learned query attends over the 196 patch tokens (replaces the crude
     .mean()) at the best layer -> does spatial attention beat flat averaging?

Baseline to beat: layer-12 concat[CLS, mean-patch] linear clock, GA-r ~0.94 (cerebral).

Reads shards from /Users/tiago/usb/usfmae_all_layers/shard_*.npz (skips in-progress/bad ones).
Run: /Users/tiago/.claude-science/conda/envs/fgrgeom/bin/python probe_layers_pooling.py
Outputs: ./out_probe/{layer_pool_results.json}
"""
import os, glob, json, zipfile, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

HERE=os.path.dirname(os.path.abspath(__file__))
SHARD_DIR="/Users/tiago/usb/usfmae_all_layers"
OUT=os.path.join(HERE,"out_probe"); os.makedirs(OUT,exist_ok=True)
DEV="mps" if torch.backends.mps.is_available() else "cpu"

def load_shards(plane=None, max_shards=None):
    """Return per-frame: layer_summ (N,12,1536)=concat[CLS,mean-patch] per layer,
       patch_toks (N,196,768) at LAST layer, ga (N), nid (N). Skips bad/in-progress shards.
    MEMORY-SAFE: each shard's giant (N,12,197,768) tokens array is summarized to the small
    LS/PT arrays and freed BEFORE loading the next shard (never holds >1 shard of raw tokens)."""
    fs=sorted(glob.glob(f"{SHARD_DIR}/shard_*.npz"))
    if max_shards: fs=fs[:max_shards]
    LS=[]; PT=[]; GA=[]; NID=[]; PL=[]
    for f in fs:
        try: z=np.load(f,allow_pickle=True)
        except zipfile.BadZipFile: print(f"  skip in-progress {os.path.basename(f)}",flush=True); continue
        tok=z["tokens"]                       # (N,12,197,768) — the big transient
        LS.append(np.concatenate([tok[:,:,0,:], tok[:,:,1:,:].mean(2)],-1).astype(np.float32))  # (N,12,1536)
        PT.append(tok[:,-1,1:,:].astype(np.float32))    # (N,196,768) last-layer patches
        GA.append(z["ga"]); NID.append(z["nid"].astype(str)); PL.append(z["plane"])
        del tok, z                            # free the ~3.7GB shard before next iter
    LS=np.concatenate(LS); PT=np.concatenate(PT); GA=np.concatenate(GA)
    NID=np.concatenate(NID); PL=np.concatenate(PL)
    ok=np.isfinite(GA)&(GA>=6)&(GA<=42)
    if plane: ok&=(PL==plane)
    return LS[ok], PT[ok], GA[ok].astype(np.float32), NID[ok]

def cv_ridge(X,y,grp):
    pred=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,groups=grp):
        sc=StandardScaler().fit(X[tr])
        pred[te]=Ridge(alpha=10).fit(sc.transform(X[tr]),y[tr]).predict(sc.transform(X[te]))
    return pearsonr(pred,y)[0], np.abs(pred-y).mean()

def probe1_layers(LS,ga,nid):
    """Per-single-layer clock + a learned softmax layer-mix (small torch head, grouped CV)."""
    res={"per_layer":{}}
    for L in range(12):
        r,mae=cv_ridge(LS[:,L,:],ga,nid); res["per_layer"][f"L{L+1}"]={"r":float(r),"mae":float(mae)}
    # learned layer-mix: softmax weights over 12 layers, then linear to GA (grouped CV)
    rs=[]
    for tr,te in GroupKFold(5).split(LS,groups=nid):
        sc=StandardScaler().fit(LS[tr].reshape(len(tr),-1))
        Xtr=torch.tensor(sc.transform(LS[tr].reshape(len(tr),-1)).reshape(len(tr),12,1536),dtype=torch.float32,device=DEV)
        Xte=torch.tensor(sc.transform(LS[te].reshape(len(te),-1)).reshape(len(te),12,1536),dtype=torch.float32,device=DEV)
        ytr=torch.tensor(ga[tr],dtype=torch.float32,device=DEV)
        lw=nn.Parameter(torch.zeros(12,device=DEV)); lin=nn.Linear(1536,1).to(DEV)
        opt=torch.optim.Adam([lw]+list(lin.parameters()),1e-3)
        for ep in range(300):
            opt.zero_grad(); w=torch.softmax(lw,0)
            mix=(Xtr*w[None,:,None]).sum(1); p=lin(mix).squeeze(1)
            l=((p-ytr)**2).mean()+1e-3*(lin.weight**2).mean(); l.backward(); opt.step()
        with torch.no_grad():
            w=torch.softmax(lw,0); pte=lin((Xte*w[None,:,None]).sum(1)).squeeze(1).cpu().numpy()
        rs.append((pearsonr(pte,ga[te])[0],np.abs(pte-ga[te]).mean(),w.detach().cpu().numpy()))
    res["learned_mix"]={"r":float(np.mean([x[0] for x in rs])),"mae":float(np.mean([x[1] for x in rs])),
                        "layer_weights":np.mean([x[2] for x in rs],0).round(3).tolist()}
    return res

def probe2_attnpool(PT,ga,nid):
    """Attention-pool over 196 last-layer patch tokens vs flat mean, grouped CV."""
    # baseline flat mean
    r_mean,mae_mean=cv_ridge(PT.mean(1),ga,nid)
    rs=[]
    for tr,te in GroupKFold(5).split(PT,groups=nid):
        Xtr=torch.tensor(PT[tr],device=DEV); Xte=torch.tensor(PT[te],device=DEV)
        ytr=torch.tensor(ga[tr],dtype=torch.float32,device=DEV)
        q=nn.Parameter(torch.randn(768,device=DEV)*0.02); head=nn.Linear(768,1).to(DEV)
        opt=torch.optim.Adam([q]+list(head.parameters()),1e-3)
        for ep in range(300):
            opt.zero_grad()
            att=torch.softmax((Xtr@q)/768**0.5,1)          # (B,196)
            pooled=(att[:,:,None]*Xtr).sum(1); p=head(pooled).squeeze(1)
            l=((p-ytr)**2).mean(); l.backward(); opt.step()
        with torch.no_grad():
            att=torch.softmax((Xte@q)/768**0.5,1); pooled=(att[:,:,None]*Xte).sum(1)
            pte=head(pooled).squeeze(1).cpu().numpy()
        rs.append((pearsonr(pte,ga[te])[0],np.abs(pte-ga[te]).mean()))
    return {"flat_mean":{"r":float(r_mean),"mae":float(mae_mean)},
            "attention_pool":{"r":float(np.mean([x[0] for x in rs])),"mae":float(np.mean([x[1] for x in rs]))}}

def main():
    for plane in [None,"cerebral"]:
        LS,PT,ga,nid=load_shards(plane=plane)
        tag=plane or "all_planes"
        print(f"[{tag}] frames={len(ga)} fetuses={len(set(nid))} GA {ga.min():.1f}-{ga.max():.1f}",flush=True)
        out={"n":len(ga),"probe1_layers":probe1_layers(LS,ga,nid),"probe2_pooling":probe2_attnpool(PT,ga,nid)}
        json.dump(out,open(os.path.join(OUT,f"probe_{tag}.json"),"w"),indent=2)
        p1=out["probe1_layers"]; best=max(p1["per_layer"].items(),key=lambda kv:kv[1]["r"])
        print(f"  best single layer: {best[0]} r={best[1]['r']:.3f} | learned-mix r={p1['learned_mix']['r']:.3f} | L12 r={p1['per_layer']['L12']['r']:.3f}",flush=True)
        print(f"  pooling: flat-mean r={out['probe2_pooling']['flat_mean']['r']:.3f} -> attn-pool r={out['probe2_pooling']['attention_pool']['r']:.3f}",flush=True)

if __name__=="__main__": main()
