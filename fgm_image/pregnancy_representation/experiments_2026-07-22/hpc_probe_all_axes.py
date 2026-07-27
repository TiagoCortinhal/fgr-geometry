#!/usr/bin/env python3
"""Richer end-to-end probe of the full USF-MAE token stack — best LAYER and best POOLING
for EVERY axis (not just GA): gestational age, placental (image<->Doppler CCA), and
appearance-lag (GA-clock residual). Per-plane. Summary table + figure.

Self-contained: builds the Doppler panel from the repo's echo xlsx and the lag from the
GA-clock, so it needs NO workspace CSVs — only what's in the repo + the extracted summary.

Reads: out_usfmae/summaries.npz  (LS (N,12,1536), PT (N,196,768), ga, nid, plane, names)
       data_local/IMPACT_ecocardio_zscores_corrected.xlsx   (Doppler z + Cod)
       ga_cnn/ga_cnn_index.csv   (new_filename -> nid/GA/plane, for the Cod join)
Writes: out_probe/all_axes_results.json  +  out_probe/all_axes_summary.png
Run:  python fgm_image/pregnancy_representation/experiments_2026-07-22/hpc_probe_all_axes.py
"""
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr

HERE=os.path.dirname(os.path.abspath(__file__))
SUMMARY=os.path.join(os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae")),"summaries.npz")
ECHO=os.path.join(HERE,"..","..","..","data_local","IMPACT_ecocardio_zscores_corrected.xlsx")
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")
OUTDIR=os.path.join(HERE,"out_probe"); os.makedirs(OUTDIR,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
DOPPLER=["Zscore_UTA","Zscore_AU","Zscore_ACM","Zscore_CPR","Zscore_DV","Zscore_Aortic_Ithsmus"]

# ------------------------------- load + build panels -------------------------------
def load():
    z=np.load(SUMMARY,allow_pickle=True)
    LS,PT=z["LS"],z["PT"]; ga=z["ga"].astype(np.float32)
    nid=z["nid"].astype(str); plane=z["plane"].astype(str); names=z["names"].astype(str)
    # Doppler panel by fetus (Cod == nid)
    e=pd.read_excel(ECHO); e["nid"]=e["Cod"].map(lambda x:str(int(float(x))) if pd.notna(x) else None)
    dop=e.dropna(subset=["nid"]).copy()
    for c in DOPPLER: dop[c]=pd.to_numeric(dop[c],errors="coerce")
    # echo file can have duplicate Cod rows -> collapse to one row/fetus (median) so reindex is unique
    dop=dop.groupby("nid")[DOPPLER].median()
    return LS,PT,ga,nid,plane,names,dop

def clock_lag(LS,ga,nid):
    """Layer-12 GA clock (Ridge), OOF; lag = pred - true. LS[:,-1,:] is layer-12 [CLS,meanpatch]."""
    X=LS[:,-1,:]; pred=np.zeros(len(ga))
    for tr,te in GroupKFold(5).split(X,groups=nid):
        sc=StandardScaler().fit(X[tr]); pred[te]=Ridge(alpha=10).fit(sc.transform(X[tr]),ga[tr]).predict(sc.transform(X[te]))
    return pred-ga

# ------------------------------- metrics -------------------------------
def cv_ridge(X,y,grp):
    pred=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,groups=grp):
        sc=StandardScaler().fit(X[tr]); pred[te]=Ridge(alpha=10).fit(sc.transform(X[tr]),y[tr]).predict(sc.transform(X[te]))
    return float(pearsonr(pred,y)[0])

def cv_cca(X,C,grp,npc=12):
    """Held-out canonical corr between image feats X and tabular block C (GA already out if desired)."""
    pa,pb=[],[]
    for tr,te in GroupKFold(5).split(X,groups=grp):
        sc=StandardScaler().fit(X[tr]); Xtr=PCA(npc,random_state=0).fit_transform(sc.transform(X[tr]))
        pca=PCA(npc,random_state=0).fit(sc.transform(X[tr])); Xte=pca.transform(sc.transform(X[te]))
        cs=StandardScaler().fit(C[tr]); Ctr=cs.transform(C[tr]); Cte=cs.transform(C[te])
        m=CCA(1,max_iter=1000).fit(Xtr,Ctr); a,b=m.transform(Xte,Cte); pa.append(a[:,0]); pb.append(b[:,0])
    return float(pearsonr(np.concatenate(pa),np.concatenate(pb))[0])

def attn_pool_r(PT,y,grp):
    rs=[]
    for tr,te in GroupKFold(5).split(PT,groups=grp):
        Xtr=torch.tensor(PT[tr],device=DEV); Xte=torch.tensor(PT[te],device=DEV)
        ytr=torch.tensor(y[tr],dtype=torch.float32,device=DEV)
        q=nn.Parameter(torch.randn(768,device=DEV)*0.02); head=nn.Linear(768,1).to(DEV)
        opt=torch.optim.Adam([q]+list(head.parameters()),1e-3)
        for _ in range(300):
            opt.zero_grad(); att=torch.softmax((Xtr@q)/768**0.5,1)
            p=head((att[:,:,None]*Xtr).sum(1)).squeeze(1); ((p-ytr)**2).mean().backward(); opt.step()
        with torch.no_grad():
            att=torch.softmax((Xte@q)/768**0.5,1); pte=head((att[:,:,None]*Xte).sum(1)).squeeze(1).cpu().numpy()
        rs.append(pearsonr(pte,y[te])[0])
    return float(np.mean(rs))

# ------------------------------- main -------------------------------
def main():
    LS,PT,ga,nid,plane,names,dop=load()
    lag=clock_lag(LS,ga,nid)
    res={}
    for pl in ["all_planes","cerebral","abdominal","femur"]:
        m=np.isfinite(ga)&(ga>=6)&(ga<=42)
        if pl!="all_planes": m&=(plane==pl)
        if m.sum()<100: continue
        sub_nid=nid[m]; sub_ga=ga[m]; sub_lag=lag[m]
        # doppler per-frame (broadcast fetus value); GA-residualized for placental axis
        D=dop.reindex(sub_nid).values
        okD=~np.isnan(D).any(1)
        entry={"n":int(m.sum())}
        # AXIS: GA — per layer + attn-pool
        entry["GA_per_layer"]={f"L{L+1}":cv_ridge(LS[m][:,L,:],sub_ga,sub_nid) for L in range(12)}
        entry["GA_L12"]=entry["GA_per_layer"]["L12"]
        entry["GA_attn_pool"]=attn_pool_r(PT[m],sub_ga,sub_nid)
        entry["GA_flat_mean"]=cv_ridge(PT[m].mean(1),sub_ga,sub_nid)
        # AXIS: LAG — per layer (does a layer read maturation-lag better?) + attn
        entry["LAG_per_layer"]={f"L{L+1}":cv_ridge(LS[m][:,L,:],sub_lag,sub_nid) for L in range(12)}
        entry["LAG_attn_pool"]=attn_pool_r(PT[m],sub_lag,sub_nid)
        # AXIS: PLACENTAL — image<->Doppler CCA, GA-residualized, per layer
        if okD.sum()>=100:
            def resid(A,g): G=np.column_stack([np.ones_like(g),g,g**2]); return A-G@np.linalg.lstsq(G,A,rcond=None)[0]
            gD=sub_ga[okD]; CD=resid(D[okD],gD)
            entry["PLAC_per_layer"]={f"L{L+1}":cv_cca(resid(LS[m][okD][:,L,:],gD),CD,sub_nid[okD]) for L in range(12)}
            entry["PLAC_n"]=int(okD.sum())
        res[pl]=entry
        # print
        gl=entry["GA_per_layer"]; bestGA=max(gl,key=gl.get)
        print(f"[{pl}] n={entry['n']}",flush=True)
        print(f"  GA  per-layer best {bestGA} r={gl[bestGA]:.3f} | L12 r={entry['GA_L12']:.3f} | attn-pool r={entry['GA_attn_pool']:.3f} (flat {entry['GA_flat_mean']:.3f})",flush=True)
        ll=entry["LAG_per_layer"]; bestL=max(ll,key=ll.get)
        print(f"  LAG per-layer best {bestL} r={ll[bestL]:.3f} | L12 r={ll['L12']:.3f} | attn-pool r={entry['LAG_attn_pool']:.3f}",flush=True)
        if "PLAC_per_layer" in entry:
            pp=entry["PLAC_per_layer"]; bestP=max(pp,key=pp.get)
            print(f"  PLAC per-layer best {bestP} cc={pp[bestP]:.3f} | L12 cc={pp['L12']:.3f} (n={entry['PLAC_n']})",flush=True)
    json.dump(res,open(os.path.join(OUTDIR,"all_axes_results.json"),"w"),indent=2)
    # figure: per-layer curves for GA / LAG / PLAC (all_planes + cerebral)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4.2))
        for ax,axis,key,ylab in [(axes[0],"GA","GA_per_layer","GA-clock r"),
                                  (axes[1],"LAG","LAG_per_layer","lag r"),
                                  (axes[2],"PLAC","PLAC_per_layer","placental cc")]:
            for pl,style in [("all_planes","-o"),("cerebral","-s")]:
                if pl in res and key in res[pl]:
                    v=res[pl][key]; ax.plot(range(1,13),[v[f"L{i}"] for i in range(1,13)],style,label=pl)
            ax.set_title(axis); ax.set_xlabel("layer"); ax.set_ylabel(ylab); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle("USF-MAE: which LAYER carries each axis best (per-layer probe)")
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"all_axes_summary.png"),dpi=150,bbox_inches="tight")
        print("figure -> out_probe/all_axes_summary.png",flush=True)
    except Exception as ex: print("figure skipped:",ex,flush=True)
    print("DONE -> out_probe/all_axes_results.json",flush=True)

if __name__=="__main__": main()
