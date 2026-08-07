"""Cerebral images -> Doppler survives BH (q=0.096). Which Doppler variable, and is it
the CEREBRAL one (MCA/CPR) as mechanism would predict, or a placental one?"""
import numpy as np, json, warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/Users/tiago/dev/fgr-geometry/tools/impact_fetal_panel")
from fgm_tools import *
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
fgm_setup(); P=fgm_panel(); Z,cols,bl,fids=P["Z"],P["cols"],np.array(P["blocks"]),P["fids"]
GA=fgm_ga_at_echo(fids); BMI=Z[:,cols.index("maternal_bmi")]
PPL,cnt=fgm_image_pcs_by_plane(fids,n_pc=4,use_labelled=True)
CER=PPL["cerebral"]
DOP=[(c,cols.index(c)) for c in ("Percentil_ACM","Percentil_CPR","Percentil_AU","Percentil_UTA","Percentil_DV")]
def oof(y,X,COV,seed=0):
    m=np.isfinite(y)&np.isfinite(X).all(1)&np.isfinite(COV).all(1)
    if m.sum()<120: return np.nan,int(m.sum())
    A=COV[m]; yy=y[m]-A@np.linalg.lstsq(A,y[m],rcond=None)[0]
    Xs=X[m]-A@np.linalg.lstsq(A,X[m],rcond=None)[0]
    p=np.zeros_like(yy)
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(Xs):
        pc=PCA(min(4,Xs.shape[1],len(tr)-1),random_state=0).fit(Xs[tr])
        p[te]=RidgeCV(alphas=np.logspace(-2,3,20)).fit(pc.transform(Xs[tr]),yy[tr]).predict(pc.transform(Xs[te]))
    return float(np.corrcoef(p,yy)[0,1]),int(m.sum())
n=len(Z); one=np.ones((n,1))
COV=np.column_stack([one,GA[:,None],np.where(np.isfinite(BMI),BMI,0.0)[:,None]])
print("WHICH DOPPLER VARIABLE DOES THE CEREBRAL IMAGE BLOCK TRACK?")
print("  (MCA and CPR are CEREBRAL measurements; AU/UTA are PLACENTAL; DV is venous)\n")
rows=[]
for nm,ix in DOP:
    r_,n_=oof(Z[:,ix],CER,COV)
    site="CEREBRAL" if nm in ("Percentil_ACM","Percentil_CPR") else ("placental" if nm in ("Percentil_AU","Percentil_UTA") else "venous")
    print(f"   {nm:18s} [{site:9s}] r={r_:+.3f}  n={n_}")
    rows.append(dict(var=nm,site=site,r=float(r_),n=int(n_)))
print("\nCONTROL: the SAME test with ABDOMINAL and FEMUR image blocks")
for pl in ("abdominal","femur"):
    rr=[oof(Z[:,ix],PPL[pl],COV)[0] for nm,ix in DOP]
    print(f"   {pl:11s} "+"  ".join(f"{nm.split('_')[1][:4]}={v:+.3f}" for (nm,_),v in zip(DOP,rr)))
    rows.append(dict(block=pl,per_var={nm:float(v) for (nm,_),v in zip(DOP,rr)}))
print("\nPERMUTATION NULL for cerebral -> ACM and -> CPR (400 perms)")
rng=np.random.default_rng(0)
for nm,ix in DOP[:2]:
    y=Z[:,ix]; obs,_=oof(y,CER,COV)
    m=np.isfinite(y)&np.isfinite(CER).all(1)&np.isfinite(COV).all(1)
    nl=[]
    for _ in range(400):
        ys=y.copy(); ys[m]=rng.permutation(y[m]); nl.append(oof(ys,CER,COV)[0])
    nl=np.array(nl); p=(1+(nl>=obs).sum())/(1+len(nl))
    print(f"   {nm:18s} obs={obs:+.4f}  null p95={np.percentile(nl,95):+.4f}  p={p:.4f}")
    rows.append(dict(var=nm,perm_p=float(p),obs=float(obs),null_p95=float(np.percentile(nl,95))))
json.dump(rows,open("/tmp/cereb.json","w"),indent=1)
