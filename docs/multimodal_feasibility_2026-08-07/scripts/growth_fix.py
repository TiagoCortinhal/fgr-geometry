"""Growth arm redone WITHOUT self-adjustment. Biometry cannot be a covariate for
the biometry target, and EFW is itself a growth variable -- so the only valid
nuisance terms are GA and (for the confound test) maternal BMI."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
d=np.load("/tmp/card_sub.npz",allow_pickle=True)
Zp,cols,bl=d["Zpanel"],[str(x) for x in d["cols"]],[str(x) for x in d["bl"]]
IMG,GA=d["IMG"],d["GA"]
GI=[i for i in range(len(cols)) if bl[i]=="growth"]
BMI=Zp[:,cols.index("maternal_bmi")]
def cc(X,Y,npc=10,seed=0):
    o=[]
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(X):
        p=PCA(min(npc,X.shape[1],len(tr)-1),random_state=0).fit(X[tr])
        c=CCA(n_components=1,max_iter=2000).fit(p.transform(X[tr]),Y[tr])
        a,b=c.transform(p.transform(X[te]),Y[te]); o.append(np.corrcoef(a[:,0],b[:,0])[0,1])
    return float(np.mean(o))
def rs(X,cv):
    if not cv: return X
    A=np.column_stack([np.ones(len(X))]+[np.asarray(c).reshape(len(X),-1) for c in cv])
    return X-A@np.linalg.lstsq(A,X,rcond=None)[0]
sel=np.isfinite(IMG).all(1)&(np.isfinite(Zp[:,GI]).sum(1)>=4)&np.isfinite(GA)
Y=np.where(np.isfinite(Zp[sel][:,GI]),Zp[sel][:,GI],0.0)
Is=np.where(np.isfinite(IMG[sel]),IMG[sel],0.0)
g=GA[sel][:,None]; b=np.where(np.isfinite(BMI[sel]),BMI[sel],0.0)[:,None]
n=int(sel.sum()); print(f"growth arm, VALID adjustment only.  n={n}")
lad={}
for lab,cv in (("raw",[]),("GA",[g]),("GA+maternal BMI",[g,b])):
    lad[lab]=cc(rs(Is,cv),rs(Y,cv)); print(f"   {lab:18s} {lad[lab]:+.4f}")
Xa=rs(Is,[g,b]); Ya=rs(Y,[g,b]); obs=lad["GA+maternal BMI"]
rng=np.random.default_rng(0)
nl=np.array([cc(Xa,Ya[rng.permutation(n)]) for _ in range(400)])
p=(1+(nl>=obs).sum())/(1+len(nl))
print(f"   perm p={p:.4f}  null p95={np.percentile(nl,95):+.4f}")
print(f"\n   INVALID rung from the first run (target residualised on itself): -0.179, p=1.000")
json.dump(dict(n=n,ladder=lad,p=float(p),null_p95=float(np.percentile(nl,95))),
          open("/tmp/growth_fix.json","w"),indent=1)
