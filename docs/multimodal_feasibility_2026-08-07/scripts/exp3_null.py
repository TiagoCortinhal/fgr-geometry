"""Prespec requires: permutation null per endpoint, BH across all tests, and the
PRIMARY comparison is fused vs BEST SINGLE BLOCK -- not vs chance."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/Users/tiago/dev/fgr-geometry/tools/impact_fetal_panel")
from fgm_tools import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
fgm_setup(); P=fgm_panel(); Z,cols,bl,fids=P["Z"],P["cols"],np.array(P["blocks"]),P["fids"]
NAMES=["growth","maternal","Doppler","cardiac"]
G={b:[i for i in range(len(cols)) if bl[i]==b] for b in NAMES}
o=pd.read_csv("data/impact_outcomes.csv"); o["fid"]=pd.to_numeric(o.Cod,errors="coerce")
def flag(c,src=None):
    s=(src if src is not None else o).set_index("fid")[c]
    def m(v):
        if isinstance(v,str): return 1.0 if v.strip().lower() in ("yes","si","sí","1") else 0.0
        return float(v) if np.isfinite(v) else np.nan
    d={int(k):m(v) for k,v in s.items() if np.isfinite(k)}
    return np.array([d.get(int(f),np.nan) for f in fids])
mer=pd.read_excel("data/IMPACT_merged_by_Cod.xlsx"); mer["fid"]=pd.to_numeric(mer["Cod"],errors="coerce")
SGA=flag("SGA_birth (<p10)"); SEV=flag("severeSGA (<p3)"); LGA=flag("LGA_birth (>p90)")
NIC=flag("NICU",mer)
AGA=np.where(np.isfinite(SGA)&np.isfinite(LGA),((SGA==0)&(LGA==0)).astype(float),np.nan)
END={"SGA":SGA,"severeSGA":SEV,"AGA":AGA,"LGA":LGA,"NICU":NIC}
def run(y,seed=0):
    m=np.isfinite(y); yy=y[m].astype(int)
    S={}
    for b,ix in G.items():
        X=np.where(np.isfinite(Z[m][:,ix]),Z[m][:,ix],0.0); p=np.zeros(len(yy))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,yy):
            p[te]=LogisticRegression(max_iter=2000).fit(X[tr],yy[tr]).predict_proba(X[te])[:,1]
        S[b]=p
    eps=1e-6
    lo=sum(np.log(np.clip(S[b],eps,1-eps))-np.log(np.clip(1-S[b],eps,1-eps)) for b in G)
    return yy,S,lo
print("PRIMARY TEST: does FUSED beat its own BEST SINGLE BLOCK? (prespec stop rule)\n")
print(f"{'endpoint':11s} {'best block':11s} {'best AUC':>9s} {'fused':>8s} {'delta':>8s} {'boot CI':>18s} {'p(delta>0)':>11s}")
out={}
rng=np.random.default_rng(0)
for name,y in END.items():
    yy,S,lo=run(y)
    a={b:fgm_auc(S[b],yy) for b in NAMES}
    bb=max(a,key=a.get); af=fgm_auc(lo,yy); dl=af-a[bb]
    ds=[]
    for _ in range(400):
        i=rng.integers(0,len(yy),len(yy))
        if yy[i].sum()<5 or (1-yy[i]).sum()<5: continue
        ds.append(fgm_auc(lo[i],yy[i])-fgm_auc(S[bb][i],yy[i]))
    ds=np.array(ds); pv=float((ds<=0).mean())
    print(f"{name:11s} {bb:11s} {a[bb]:9.3f} {af:8.3f} {dl:+8.3f}  [{np.percentile(ds,2.5):+.3f},{np.percentile(ds,97.5):+.3f}] {pv:11.3f}")
    out[name]=dict(best_block=bb,best_auc=float(a[bb]),fused=float(af),delta=float(dl),
                   ci=[float(np.percentile(ds,2.5)),float(np.percentile(ds,97.5))],p=pv,
                   per_block={k:float(v) for k,v in a.items()},events=int(yy.sum()),n=int(len(yy)))
rej,q=fgm_bh([out[k]["p"] for k in out],q=0.10)
for k,r_,q_ in zip(out,rej,q): out[k].update(bh_q=float(q_),bh_reject=bool(r_))
print("\nBH at q=0.10 across the 5 endpoints:")
for k in out: print(f"   {k:11s} p={out[k]['p']:.3f}  q={out[k]['bh_q']:.3f}  {'SURVIVES' if out[k]['bh_reject'] else 'no'}")
json.dump(out,open("/tmp/exp3_null.json","w"),indent=1)
