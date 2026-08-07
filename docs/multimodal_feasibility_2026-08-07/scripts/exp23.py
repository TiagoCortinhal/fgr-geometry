"""EXP 2 conditional dependence + EXP 3 decision fusion, per the prespec."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/Users/tiago/dev/fgr-geometry/tools/impact_fetal_panel")
from fgm_tools import *
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, RidgeCV
fgm_setup(); P=fgm_panel(); Z,cols,bl,fids=P["Z"],P["cols"],np.array(P["blocks"]),P["fids"]
NAMES=["growth","maternal","Doppler","cardiac"]
G={b:[i for i in range(len(cols)) if bl[i]==b] for b in NAMES}
# ---------- outcomes ----------
o=pd.read_csv("data/impact_outcomes.csv"); o["fid"]=pd.to_numeric(o.Cod,errors="coerce")
def flag(c):
    s=o.set_index("fid")[c]
    def m(v):
        if isinstance(v,str): return 1.0 if v.strip().lower() in ("yes","si","sí","1") else 0.0
        return float(v) if np.isfinite(v) else np.nan
    d={int(k):m(v) for k,v in s.items() if np.isfinite(k)}
    return np.array([d.get(int(f),np.nan) for f in fids])
SGA=flag("SGA_birth (<p10)"); SEV=flag("severeSGA (<p3)"); LGA=flag("LGA_birth (>p90)")
mer=pd.read_excel("data/IMPACT_merged_by_Cod.xlsx"); mer["fid"]=pd.to_numeric(mer["Cod"],errors="coerce")
nic=mer.set_index("fid")["NICU"] if "NICU" in mer.columns else None
NIC=np.array([float(nic.get(int(f),np.nan)) if nic is not None else np.nan for f in fids])
if not np.isfinite(NIC).any():
    c=[x for x in mer.columns if "nicu" in str(x).lower() or "UCI" in str(x)]
    print("NICU candidates:",c[:5])
AGA=np.where(np.isfinite(SGA)&np.isfinite(LGA),((SGA==0)&(LGA==0)).astype(float),np.nan)
END={"SGA":SGA,"severeSGA":SEV,"AGA":AGA,"LGA":LGA,"NICU":NIC}
print("ENDPOINT COUNTS:",{k:(int(np.nansum(v)),int(np.isfinite(v).sum())) for k,v in END.items()})
# ---------- EXP 3 decision fusion ----------
def block_scores(y,seed=0):
    """out-of-fold per-block logistic score, then combine"""
    m=np.isfinite(y); yy=y[m].astype(int)
    if yy.sum()<15 or (1-yy).sum()<15: return None
    S={}
    for b,ix in G.items():
        X=np.where(np.isfinite(Z[m][:,ix]),Z[m][:,ix],0.0)
        p=np.zeros(len(yy))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,yy):
            p[te]=LogisticRegression(max_iter=2000,C=1.0).fit(X[tr],yy[tr]).predict_proba(X[te])[:,1]
        S[b]=p
    Xall=np.where(np.isfinite(Z[m]),Z[m],0.0); pj=np.zeros(len(yy))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(Xall,yy):
        pj[te]=LogisticRegression(max_iter=2000,C=1.0).fit(Xall[tr],yy[tr]).predict_proba(Xall[te])[:,1]
    eps=1e-6
    lo=sum(np.log(np.clip(S[b],eps,1-eps))-np.log(np.clip(1-S[b],eps,1-eps)) for b in G)
    return yy,S,pj,lo
print("\n=== EXP 3 — DECISION-LEVEL FUSION ===")
print(f"{'endpoint':11s} {'n':>5s} {'ev':>4s} "+" ".join(f"{b[:7]:>8s}" for b in NAMES)+f" {'FUSED':>8s} {'joint':>8s} {'best1':>8s}")
res3={}
for name,y in END.items():
    r=block_scores(y)
    if r is None: print(f"{name:11s} skipped (too few events)"); continue
    yy,S,pj,lo=r
    a={b:fgm_auc(S[b],yy) for b in NAMES}
    af=fgm_auc(lo,yy); aj=fgm_auc(pj,yy); best=max(a.values())
    print(f"{name:11s} {len(yy):5d} {int(yy.sum()):4d} "+" ".join(f"{a[b]:8.3f}" for b in NAMES)+f" {af:8.3f} {aj:8.3f} {best:8.3f}")
    res3[name]=dict(n=int(len(yy)),events=int(yy.sum()),per_block=a,fused=float(af),joint=float(aj),
                    best_single=float(best),fused_minus_best=float(af-best))
json.dump(res3,open("/tmp/exp3.json","w"),indent=1)
# ---------- EXP 2 conditional dependence ----------
print("\n=== EXP 2 — CONDITIONAL DEPENDENCE (stratified on birthweight centile, OUTSIDE the panel) ===")
pc=flag("percentil_birth")
groups=[G[b] for b in NAMES]
def crossR2(Zs):
    out=[]
    for b,ix in G.items():
        other=[i for i in range(len(cols)) if bl[i]!=b]
        for j in ix:
            y=Zs[:,j]; m=np.isfinite(y)
            if m.sum()<60: continue
            A=np.where(np.isfinite(Zs[m][:,other]),Zs[m][:,other],0.0)
            out.append(fgm_cv_r2(y[m],A))
    return float(np.mean(out)) if out else np.nan
allidx=[i for g_ in groups for i in g_]
cc=np.isfinite(Z[:,allidx]).all(1)
q=np.nanquantile(pc[cc],[1/3,2/3])
print(f"  complete cases {int(cc.sum())} | centile tertile cuts {q.round(1)}")
res2={}
for lab,sel in (("all",cc),("low centile",cc&(pc<=q[0])),("mid",cc&(pc>q[0])&(pc<=q[1])),("high",cc&(pc>q[1]))):
    if sel.sum()<50: print(f"  {lab:14s} n={int(sel.sum())} too few"); continue
    r=fgm_omega_null(Z[sel][:,allidx],[[allidx.index(i) for i in g_] for g_ in groups],nperm=100)
    x=crossR2(Z[sel])
    print(f"  {lab:14s} n={int(sel.sum()):4d}  Omega={r['Omega']:+.4f} null[{r['null_lo']:+.4f},{r['null_hi']:+.4f}] {r['verdict']:12s} crossR2={x:+.4f}")
    res2[lab]=dict(n=int(sel.sum()),Omega=r["Omega"],null=[r["null_lo"],r["null_hi"]],verdict=r["verdict"],crossR2=x)
json.dump(res2,open("/tmp/exp2.json","w"),indent=1)
