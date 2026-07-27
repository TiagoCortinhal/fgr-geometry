#!/usr/bin/env python3
"""Clinical WIDE-GA FetalCLIP clock — the run that recovers the ~0.8-0.9 GA regime and a
meaningful MAE (comparable to the npj paper), because clinical GA spans 6-42wk (SD ~6.9wk)
vs IMPACT's narrow 26-41wk (SD 1.3wk).

FetalCLIP (open_clip ViT-L/14) per-layer summaries on the 30,257 clinical fetal frames,
then per-layer GA clock (fetus-grouped OOF) with r AND MAE, plus a per-GA-window MAE table
(like the paper). Best layer reported.

Needs the clinical images on the HPC. SET CLINICAL_IMG_DIR below (probe it first):
  find /mnt/beegfs/groups/collage -path '*CLINICAL*preprocessed*' -name '*.png' | head
Index: clinical_index.csv (nid,new_filename,ga_weeks_recovered,plane_prop,split) — in repo.
Weights: FetalCLIP_weights.pt in repo root.

USAGE:
  python hpc_clinical_fetalclip_clock.py --check          # resolve images + load model, exit
  python hpc_clinical_fetalclip_clock.py --extract        # build clinical FetalCLIP summary
  python hpc_clinical_fetalclip_clock.py --clock          # per-layer wide-GA clock (r + MAE)
  python hpc_clinical_fetalclip_clock.py                  # extract then clock
Outputs: out_usfmae/summaries_FetalCLIP_clinical.npz + out_probe/clinical_ga_clock.json/.png
"""
import os, sys, glob, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
# ---- SET THIS to the clinical preprocessed image dir on the HPC (see --check) ----
CLINICAL_IMG_DIR=os.environ.get("CLINICAL_IMG_DIR",
    "/mnt/beegfs/groups/collage/data/IMPACT_CLINICAL/processed/IMPACT_CLINICAL/preprocessed")
INDEX=os.path.join(HERE,"clinical_index.csv")
WEIGHTS=os.path.join(ROOT,"FetalCLIP_weights.pt")
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae")); os.makedirs(OUT,exist_ok=True)
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
SUMM=os.path.join(OUT,"summaries_FetalCLIP_clinical.npz")
DEV="cuda" if torch.cuda.is_available() else "cpu"
BATCH=48; SHARD=512
CLIP_MEAN=((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))
tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize(*CLIP_MEAN)])

def build_fetalclip():
    import open_clip
    model=open_clip.create_model("ViT-L-14",pretrained=None)
    ck=torch.load(WEIGHTS,map_location="cpu",weights_only=False)
    sd=ck.get("state_dict",ck.get("model",ck)); sd={k.replace("module.",""):v for k,v in sd.items()}
    vis={k[len("visual."):]:v for k,v in sd.items() if k.startswith("visual.")}
    model.visual.load_state_dict(vis,strict=False); vt=model.visual.to(DEV).eval()
    @torch.no_grad()
    def layers(x):
        v=vt; h=v.conv1(x); h=h.reshape(h.shape[0],h.shape[1],-1).permute(0,2,1)
        cls=v.class_embedding.to(h.dtype)+torch.zeros(h.shape[0],1,h.shape[-1],device=h.device,dtype=h.dtype)
        h=torch.cat([cls,h],1)+v.positional_embedding.to(h.dtype); h=v.ln_pre(h); h=h.permute(1,0,2)
        out=[]
        for blk in v.transformer.resblocks:
            h=blk(h); t=h.permute(1,0,2); out.append(torch.cat([t[:,0],t[:,1:].mean(1)],-1))
        return torch.stack(out,1)   # (B,24,2048)
    return layers

def frame_table():
    df=pd.read_csv(INDEX).copy()
    df["img"]=df["new_filename"].astype(str).apply(lambda n: os.path.join(CLINICAL_IMG_DIR, n if n.endswith(".png") else n+".png"))
    ok=df["img"].apply(os.path.exists)
    print(f"clinical index {len(df)} | resolved on HPC {ok.sum()} at {CLINICAL_IMG_DIR}",flush=True)
    return df[ok].reset_index(drop=True)

def check():
    df=pd.read_csv(INDEX)
    print(f"index rows {len(df)} | GA {df.ga_weeks_recovered.min():.1f}-{df.ga_weeks_recovered.max():.1f} SD {df.ga_weeks_recovered.std():.2f}",flush=True)
    hits=sum(os.path.exists(os.path.join(CLINICAL_IMG_DIR,str(n)+(""if str(n).endswith('.png')else'.png'))) for n in df.new_filename.head(200))
    print(f"path probe: {hits}/200 of first frames resolve at {CLINICAL_IMG_DIR}",flush=True)
    if hits==0:
        print("  -> WRONG DIR. Find it: find /mnt/beegfs/groups/collage -path '*CLINICAL*' -name '*.png' | head",flush=True); return
    lf=build_fetalclip()
    x=torch.stack([tf(Image.open(os.path.join(CLINICAL_IMG_DIR,str(df.new_filename.iloc[0])+'.png')).convert('RGB'))]).to(DEV)
    print("FetalCLIP OK, out",tuple(lf(x).shape),flush=True)

def extract():
    if os.path.exists(SUMM): print("summary exists, skip extract",flush=True); return
    df=frame_table(); lf=build_fetalclip()
    import time; t0=time.time(); LS=[]
    for b0 in range(0,len(df),BATCH):
        bs=df.iloc[b0:b0+BATCH]
        x=torch.stack([tf(Image.open(p).convert("RGB")) for p in bs["img"]]).to(DEV)
        LS.append(lf(x).float().cpu().numpy())
        if (b0//BATCH)%20==0: print(f"  {b0}/{len(df)} {time.time()-t0:.0f}s {(b0+len(bs))/max(time.time()-t0,1):.0f} fr/s",flush=True)
    LS=np.concatenate(LS)
    np.savez(SUMM, LS=LS, ga=df.ga_weeks_recovered.values, nid=df.nid.astype(str).values,
             plane=df.plane_prop.values, names=df.new_filename.values)
    print(f"EXTRACT done {LS.shape} -> {SUMM} ({time.time()-t0:.0f}s)",flush=True)

def clock():
    z=np.load(SUMM,allow_pickle=True); LS=z["LS"]; ga=z["ga"].astype(np.float32); nid=z["nid"].astype(str)
    m=np.isfinite(ga)&(ga>=6)&(ga<=42); LS=LS[m]; ga=ga[m]; nid=nid[m]
    nl=LS.shape[1]; res={"n":int(m.sum()),"GA_span":[float(ga.min()),float(ga.max())],"per_layer":{}}
    print(f"clinical clock: n={m.sum()} fetuses={len(set(nid))} GA {ga.min():.1f}-{ga.max():.1f} SD {ga.std():.2f}",flush=True)
    best=(-1,None,None)
    for L in range(nl):
        X=LS[:,L,:]; pred=np.zeros(len(ga))
        for tr,te in GroupKFold(5).split(X,groups=nid):
            sc=StandardScaler().fit(X[tr])
            Xtr=PCA(64,random_state=0).fit(sc.transform(X[tr]))
            pred[te]=Ridge(alpha=100).fit(Xtr.transform(sc.transform(X[tr])),ga[tr]).predict(Xtr.transform(sc.transform(X[te])))
        r=float(pearsonr(pred,ga)[0]); mae=float(np.abs(pred-ga).mean())
        res["per_layer"][L+1]={"r":r,"mae_wk":mae}
        if r>best[0]: best=(r,L+1,(pred.copy(),mae))
        print(f"  L{L+1}/{nl}: r={r:.3f} MAE={mae:.2f}wk",flush=True)
    r,bl,(pred,mae)=best
    res["best_layer"]=bl; res["best_r"]=r; res["best_mae_wk"]=mae
    # per-GA-window MAE (like the paper)
    wins=[(6,14),(14,18),(18,24),(24,32),(32,42)]; res["window_mae"]={}
    for lo,hi in wins:
        w=(ga>=lo)&(ga<hi)
        if w.sum()>20: res["window_mae"][f"{lo}-{hi}wk"]={"mae_wk":float(np.abs(pred[w]-ga[w]).mean()),"mae_days":float(np.abs(pred[w]-ga[w]).mean()*7),"n":int(w.sum())}
    json.dump(res,open(os.path.join(OUTP,"clinical_ga_clock.json"),"w"),indent=2)
    print(f"\nBEST layer L{bl}: r={r:.3f} MAE={mae:.2f}wk ({mae*7:.1f} days)",flush=True)
    print("per-GA-window MAE:",flush=True)
    for k,v in res["window_mae"].items(): print(f"  {k}: {v['mae_days']:.1f} days (n={v['n']})",flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(12,4.3))
        Ls=sorted(res["per_layer"]); ax[0].plot(Ls,[res["per_layer"][l]["r"] for l in Ls],"-o")
        ax[0].set_title(f"clinical FetalCLIP GA clock r by layer (best L{bl} r={r:.2f})"); ax[0].set_xlabel("layer"); ax[0].set_ylabel("GA-r"); ax[0].grid(alpha=.3)
        ax[1].scatter(ga,pred,s=3,alpha=.2); ax[1].plot([6,42],[6,42],"r--")
        ax[1].set_title(f"pred vs true GA (MAE {mae*7:.1f}d, wide 6-42wk)"); ax[1].set_xlabel("true GA (wk)"); ax[1].set_ylabel("pred GA (wk)")
        fig.tight_layout(); fig.savefig(os.path.join(OUTP,"clinical_ga_clock.png"),dpi=150,bbox_inches="tight")
        print("figure -> out_probe/clinical_ga_clock.png",flush=True)
    except Exception as ex: print("fig skipped",ex,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--check",action="store_true"); ap.add_argument("--extract",action="store_true"); ap.add_argument("--clock",action="store_true")
    a=ap.parse_args()
    if a.check: check()
    elif a.extract: extract()
    elif a.clock: clock()
    else: extract(); clock()
