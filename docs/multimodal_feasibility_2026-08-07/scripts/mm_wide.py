"""Image alignment against EVERY tabular variable individually -- 1013 numeric
registry columns, not the 4 defined blocks. Per-variable, GA/size/biometry
adjusted, with BH correction and the GA-leakage audit the cardiac failure taught."""
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
R=pd.read_excel("data/IMPACT_merged_by_Cod.xlsx")
key=pd.to_numeric(R["Cod"],errors="coerce")
pos={int(f):i for i,f in enumerate(key) if np.isfinite(f)}
idx=np.array([pos.get(f,-1) for f in fids])
adm=re.compile(r"fecha|date|_days$|FUR|^Cod|NHC|^Group|Protocol|^etnia|postal|LMP",re.I)
V={}
for c in R.columns:
    if c=="Cod" or adm.search(str(c)): continue
    v=euro(R[c])
    if v is None: continue
    a=np.array([v[i] if i>=0 else np.nan for i in idx])
    if np.isfinite(a).mean()<0.60 or np.nanstd(a)==0: continue
    if len(np.unique(a[np.isfinite(a)]))<3: continue        # drop binary flags
    V[str(c)]=(a-np.nanmean(a))/np.nanstd(a)
names=list(V); print(f"tabular variables tested individually: {len(names)}  (registry 1431 cols, 1013 numeric)")
BIOi=[i for i in range(len(cols)) if bl[i]=="growth"]
base=np.isfinite(IMG).all(1)&np.isfinite(GA)
g=GA[:,None]; e=np.where(np.isfinite(EFW),EFW,0.0)[:,None]
B=np.where(np.isfinite(Zp[:,BIOi]),Zp[:,BIOi],0.0)
COV=np.column_stack([np.ones(len(IMG)),g,e,B])
def oof_r(y,seed=0):
    m=np.isfinite(y)&base
    if m.sum()<150: return np.nan,int(m.sum())
    yy=y[m]; A=COV[m]
    yy=yy-A@np.linalg.lstsq(A,yy,rcond=None)[0]
    Xs=IMG[m]; Xs=Xs-A@np.linalg.lstsq(A,Xs,rcond=None)[0]
    p=np.zeros_like(yy)
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(Xs):
        pc=PCA(10,random_state=0).fit(Xs[tr])
        p[te]=RidgeCV(alphas=np.logspace(-2,3,20)).fit(pc.transform(Xs[tr]),yy[tr]).predict(pc.transform(Xs[te]))
    return float(np.corrcoef(p,yy)[0,1]),int(m.sum())
rows=[]
for nm in names:
    r_,n_=oof_r(V[nm])
    if not np.isfinite(r_): continue
    m=np.isfinite(V[nm])&np.isfinite(GA)
    leak=abs(pearsonr(V[nm][m],GA[m])[0])
    rows.append((nm,r_,n_,leak))
rows.sort(key=lambda x:-x[1])
print(f"\nTOP 25 by GA+size+biometry-adjusted held-out image->variable correlation:")
print(f"{'variable':40s} {'r':>7s} {'n':>5s} {'GAleak':>7s}")
for nm,r_,n_,lk in rows[:25]:
    print(f"  {nm[:38]:40s} {r_:+7.3f} {n_:5d} {lk:7.3f}")
json.dump([dict(var=a,r=b,n=c,ga_leak=d_) for a,b,c,d_ in rows],open("/tmp/mm_wide.json","w"),indent=1)
print(f"\nsaved {len(rows)} variables")
