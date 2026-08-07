"""Is the blood-pressure hit (r=0.215) independent of maternal habitus,
or does it collapse like everything else once BMI is removed from the images?"""
import numpy as np, pandas as pd, json, re, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
d=np.load("/tmp/card_sub.npz",allow_pickle=True)
IMG,GA,EFW=d["IMG"],d["GA"],d["EFW"]
Zp,cols,bl,fids=d["Zpanel"],[str(x) for x in d["cols"]],[str(x) for x in d["bl"]],[int(x) for x in d["fids"]]
def euro(s):
    v=pd.to_numeric(s,errors="coerce")
    if np.isfinite(v).mean()>0.5: return v.to_numpy(float)
    t=s.astype(str).str.strip()
    if t.str.match(r"^-?\d{1,3}(?:\.\d{3})*,\d+$|^-?\d+,\d+$").mean()<0.3: return None
    v2=pd.to_numeric(t.str.replace(".","",regex=False).str.replace(",",".",regex=False),errors="coerce")
    return v2.to_numpy(float) if np.isfinite(v2).mean()>0.3 else None
R=pd.read_excel("data/IMPACT_merged_by_Cod.xlsx"); key=pd.to_numeric(R["Cod"],errors="coerce")
pos={int(f):i for i,f in enumerate(key) if np.isfinite(f)}; idx=np.array([pos.get(f,-1) for f in fids])
def get(c):
    v=euro(R[c]); a=np.array([v[i] if i>=0 else np.nan for i in idx]); return (a-np.nanmean(a))/np.nanstd(a)
BP=get("Vis3TR_MAP"); BMI=get("BMIpre"); PRES=get("presentacion"); CAL=get("Calidad_ecocardio")
BIOi=[i for i in range(len(cols)) if bl[i]=="growth"]
def oof(y,X,COV,seed=0):
    m=np.isfinite(y)&np.isfinite(X).all(1)&np.isfinite(COV).all(1)
    if m.sum()<150: return np.nan,int(m.sum())
    A=COV[m]; yy=y[m]-A@np.linalg.lstsq(A,y[m],rcond=None)[0]
    Xs=X[m]-A@np.linalg.lstsq(A,X[m],rcond=None)[0]
    p=np.zeros_like(yy)
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(Xs):
        pc=PCA(10,random_state=0).fit(Xs[tr])
        p[te]=RidgeCV(alphas=np.logspace(-2,3,20)).fit(pc.transform(Xs[tr]),yy[tr]).predict(pc.transform(Xs[te]))
    return float(np.corrcoef(p,yy)[0,1]),int(m.sum())
n=len(IMG); one=np.ones((n,1))
BASE=np.column_stack([one,GA[:,None],np.where(np.isfinite(EFW),EFW,0.0)[:,None],
                      np.where(np.isfinite(Zp[:,BIOi]),Zp[:,BIOi],0.0)])
print(f"BP vs BMI correlation: r={pearsonr(BP[np.isfinite(BP)&np.isfinite(BMI)],BMI[np.isfinite(BP)&np.isfinite(BMI)])[0]:+.3f}\n")
print("IMAGE -> MEAN ARTERIAL PRESSURE, adding confounds to the adjustment:")
for lab,extra in (("GA+size+biometry",None),
                  ("  + maternal BMI",BMI),
                  ("  + BMI & presentation",np.column_stack([BMI,PRES])),
                  ("  + BMI, presentation, echo quality",np.column_stack([BMI,PRES,CAL]))):
    COV=BASE if extra is None else np.column_stack([BASE,np.where(np.isfinite(extra),extra,0.0)])
    r_,n_=oof(BP,IMG,COV); print(f"   {lab:38s} r={r_:+.3f}  n={n_}")
print("\nCONTROL -- same ladder for maternal BMI itself (should die when BMI is a covariate):")
for lab,extra in (("GA+size+biometry",None),("  + maternal BMI",BMI)):
    COV=BASE if extra is None else np.column_stack([BASE,np.where(np.isfinite(extra),extra,0.0)])
    r_,n_=oof(BMI,IMG,COV); print(f"   {lab:38s} r={r_:+.3f}  n={n_}")
print("\nPERMUTATION NULL for BP fully adjusted (BMI+presentation+quality), 400 perms:")
COV=np.column_stack([BASE,np.where(np.isfinite(np.column_stack([BMI,PRES,CAL])),np.column_stack([BMI,PRES,CAL]),0.0)])
obs,_=oof(BP,IMG,COV); rng=np.random.default_rng(0)
m=np.isfinite(BP)&np.isfinite(IMG).all(1)&np.isfinite(COV).all(1)
nl=[]
for b in range(400):
    y=BP.copy(); y[m]=rng.permutation(BP[m]); nl.append(oof(y,IMG,COV)[0])
nl=np.array(nl); p=(1+(nl>=obs).sum())/(1+len(nl))
print(f"   observed {obs:+.4f}  null mean {nl.mean():+.4f}  p95 {np.percentile(nl,95):+.4f}  p={p:.4f}")
json.dump(dict(obs=float(obs),p=float(p),null_p95=float(np.percentile(nl,95))),open("/tmp/mm_bp.json","w"),indent=1)
np.save("/tmp/_nullbp.npy",nl)
