#!/usr/bin/env python3
"""Four-encoder GA / lag / placental comparison, built FROM the full-token shards already
extracted (out_usfmae/fulltok_<enc>/). No re-extraction.

Step A: for each encoder, stream its full-token shards -> per-layer summary
        LS (N, n_layers, 2*dim) = concat[CLS, mean(patch tokens)] per block. Cache to
        out_usfmae/summaries_<enc>.npz (skip if present).
Step B: per encoder, per layer: GA clock r, lag-axis, placental image<->Doppler CCA
        (GA-residualized), fetus-grouped CV. Best layer per axis per encoder.
Output: out_probe/encoder_comparison.json + encoder_comparison.png (per-layer curves, 4 enc).

Run: python hpc_compare_encoders.py            # does A then B for all found encoders
     python hpc_compare_encoders.py --summarize-only
"""
import os, sys, glob, json, argparse, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
ECHO=os.path.join(ROOT,"data_local","IMPACT_ecocardio_zscores_corrected.xlsx")
DOPPLER=["Zscore_UTA","Zscore_AU","Zscore_ACM","Zscore_CPR","Zscore_DV","Zscore_Aortic_Ithsmus"]
ENCODERS=["USF-MAE","USFM","FetalCLIP","DINOv2"]

# ---------- Step A: summaries from full-token shards ----------
def summarize(enc):
    out=os.path.join(OUT,f"summaries_{enc}.npz")
    if os.path.exists(out): print(f"  {enc}: summary exists",flush=True); return out
    d=os.path.join(OUT,f"fulltok_{enc}")
    shards=sorted(glob.glob(os.path.join(d,"shard_*.npz")))
    if not shards: print(f"  {enc}: NO shards at {d}",flush=True); return None
    LS=[]; ga=[]; nid=[]; plane=[]; names=[]
    for i,f in enumerate(shards):
        z=np.load(f,allow_pickle=True); tok=z["tokens"]        # (n, L, Ntok, dim) fp16
        cls=tok[:,:,0,:].astype(np.float32); mp=tok[:,:,1:,:].mean(2).astype(np.float32)
        LS.append(np.concatenate([cls,mp],-1))
        ga.append(z["ga"]); nid.append(z["nid"].astype(str)); plane.append(z["plane"]); names.append(z["names"])
        del tok,z
        if i%10==0: print(f"    {enc} shard {i+1}/{len(shards)}",flush=True)
    np.savez(out, LS=np.concatenate(LS), ga=np.concatenate(ga), nid=np.concatenate(nid),
             plane=np.concatenate(plane), names=np.concatenate(names))
    print(f"  {enc}: summary -> {out} {np.concatenate(LS).shape}",flush=True); return out

# ---------- Step B: probes ----------
def cv_ridge(X,y,grp):
    pred=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,groups=grp):
        sc=StandardScaler().fit(X[tr]); pred[te]=Ridge(alpha=10).fit(sc.transform(X[tr]),y[tr]).predict(sc.transform(X[te]))
    return float(pearsonr(pred,y)[0]), pred

def cv_cca(X,C,grp,npc=12):
    pa,pb=[],[]
    for tr,te in GroupKFold(5).split(X,groups=grp):
        sc=StandardScaler().fit(X[tr]); pca=PCA(npc,random_state=0).fit(sc.transform(X[tr]))
        Xtr=pca.transform(sc.transform(X[tr])); Xte=pca.transform(sc.transform(X[te]))
        cs=StandardScaler().fit(C[tr]); m=CCA(1,max_iter=800).fit(Xtr,cs.transform(C[tr]))
        a,b=m.transform(Xte,cs.transform(C[te])); pa.append(a[:,0]); pb.append(b[:,0])
    return float(pearsonr(np.concatenate(pa),np.concatenate(pb))[0])

def resid(A,g): G=np.column_stack([np.ones_like(g),g,g**2]); return A-G@np.linalg.lstsq(G,A,rcond=None)[0]

def probe(enc, summ, dop):
    z=np.load(summ,allow_pickle=True,mmap_mode="r")
    LS=np.asarray(z["LS"]); ga=np.asarray(z["ga"]).astype(np.float32); nid=np.asarray(z["nid"]).astype(str)
    m=np.isfinite(ga)&(ga>=6)&(ga<=42); LS=LS[m]; ga=ga[m]; nid=nid[m]
    nlayer=LS.shape[1]
    D=dop.reindex(nid).values; okD=~np.isnan(D).any(1)
    r={"n":int(m.sum()),"n_layers":nlayer,"GA_per_layer":{},"PLAC_per_layer":{}}
    # GA clock per layer + lag from best-GA layer
    for L in range(nlayer):
        rr,_=cv_ridge(LS[:,L,:],ga,nid); r["GA_per_layer"][L+1]=rr
    bestGA=max(r["GA_per_layer"],key=r["GA_per_layer"].get)
    _,gapred=cv_ridge(LS[:,bestGA-1,:],ga,nid); lag=gapred-ga
    r["GA_best_layer"]=bestGA; r["GA_best_r"]=r["GA_per_layer"][bestGA]
    # placental per layer (GA-resid CCA)
    if okD.sum()>=100:
        gD=ga[okD]; CD=resid(D[okD],gD)
        for L in range(nlayer):
            r["PLAC_per_layer"][L+1]=cv_cca(resid(LS[okD][:,L,:],gD),CD,nid[okD])
        bp=max(r["PLAC_per_layer"],key=r["PLAC_per_layer"].get)
        r["PLAC_best_layer"]=bp; r["PLAC_best_cc"]=r["PLAC_per_layer"][bp]; r["PLAC_n"]=int(okD.sum())
    print(f"[{enc}] n={r['n']} layers={nlayer} | GA best L{bestGA} r={r['GA_best_r']:.3f}"
          + (f" | PLAC best L{r['PLAC_best_layer']} cc={r['PLAC_best_cc']:.3f}" if okD.sum()>=100 else ""),flush=True)
    return r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--summarize-only",action="store_true"); a=ap.parse_args()
    e=pd.read_excel(ECHO); e["nid"]=e["Cod"].map(lambda x:str(int(float(x))) if pd.notna(x) else None)
    ed=e.dropna(subset=["nid"]).copy()
    for c in DOPPLER: ed[c]=pd.to_numeric(ed[c],errors="coerce")
    dop=ed.groupby("nid")[DOPPLER].median()
    res={}
    for enc in ENCODERS:
        print(f"=== {enc} ===",flush=True)
        summ=summarize(enc)
        if summ is None or a.summarize_only: continue
        res[enc]=probe(enc,summ,dop)
    if a.summarize_only: print("summaries done"); return
    json.dump(res,open(os.path.join(OUTP,"encoder_comparison.json"),"w"),indent=2)
    # summary table
    print("\n==== ENCODER COMPARISON ====",flush=True)
    print(f"{'encoder':10} {'GA best (layer)':22} {'placental best (layer)':24}",flush=True)
    for enc,r in res.items():
        ga=f"r={r['GA_best_r']:.3f} (L{r['GA_best_layer']}/{r['n_layers']})"
        pl=f"cc={r.get('PLAC_best_cc',float('nan')):.3f} (L{r.get('PLAC_best_layer','-')})" if "PLAC_best_cc" in r else "n/a"
        print(f"{enc:10} {ga:22} {pl:24}",flush=True)
    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(13,4.5))
        for enc,r in res.items():
            xs=[k/r["n_layers"] for k in sorted(r["GA_per_layer"])]  # relative depth
            ax[0].plot(xs,[r["GA_per_layer"][k] for k in sorted(r["GA_per_layer"])],"-o",ms=3,label=enc)
            if r["PLAC_per_layer"]:
                xp=[k/r["n_layers"] for k in sorted(r["PLAC_per_layer"])]
                ax[1].plot(xp,[r["PLAC_per_layer"][k] for k in sorted(r["PLAC_per_layer"])],"-o",ms=3,label=enc)
        ax[0].set_title("GA clock r by relative depth"); ax[1].set_title("placental image<->Doppler cc by depth")
        for a_ in ax: a_.set_xlabel("relative depth (layer/total)"); a_.legend(fontsize=8); a_.grid(alpha=0.3)
        ax[0].set_ylabel("GA-r"); ax[1].set_ylabel("placental cc")
        fig.suptitle("Four-encoder comparison (IMPACT): where each axis is read best")
        fig.tight_layout(); fig.savefig(os.path.join(OUTP,"encoder_comparison.png"),dpi=150,bbox_inches="tight")
        print("figure -> out_probe/encoder_comparison.png",flush=True)
    except Exception as ex: print("figure skipped:",ex,flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
