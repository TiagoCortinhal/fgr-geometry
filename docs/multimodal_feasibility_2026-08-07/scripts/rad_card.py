"""Cardiac replicates on BOTH encoders (radiomics 0.169 p=0.003, USFM 0.111 p=0.040).
Does it localise to interpretable cardiac variables, or dissolve like cerebral->Doppler did?"""
import numpy as np, json, warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/Users/tiago/dev/fgr-geometry/tools/impact_fetal_panel")
from fgm_tools import *
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
fgm_setup(); P=fgm_panel(); Z,cols,bl,fids=P["Z"],P["cols"],np.array(P["blocks"]),P["fids"]
GA=fgm_ga_at_echo(fids); BMI=Z[:,cols.index("maternal_bmi")]
RAD,_=fgm_radiomics(fids,n_pc=12,artifact_path_fn=lambda v:"/tmp/_rad.parquet")
USFM,_=fgm_image_pcs(fids,n_pc=12)
CARD=[(cols[i],i) for i in range(len(cols)) if bl[i]=="cardiac"]
gg=np.where(np.isfinite(GA),GA,np.nanmean(GA)); bb=np.where(np.isfinite(BMI),BMI,0.0)
COV=np.column_stack([np.ones(len(Z)),gg[:,None],bb[:,None]])
def oof(y,X,seed=0):
    m=np.isfinite(y)&np.isfinite(X).all(1)
    if m.sum()<120: return np.nan,int(m.sum())
    A=COV[m]; yy=y[m]-A@np.linalg.lstsq(A,y[m],rcond=None)[0]
    Xs=X[m]-A@np.linalg.lstsq(A,X[m],rcond=None)[0]
    p=np.zeros_like(yy)
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(Xs):
        pc=PCA(min(8,Xs.shape[1],len(tr)-1),random_state=0).fit(Xs[tr])
        p[te]=RidgeCV(alphas=np.logspace(-2,3,20)).fit(pc.transform(Xs[tr]),yy[tr]).predict(pc.transform(Xs[te]))
    return float(np.corrcoef(p,yy)[0,1]),int(m.sum())
print("PER-CARDIAC-VARIABLE, both representations (GA+BMI adjusted)\n")
print(f"{'variable':28s} {'radiomics':>10s} {'USFM':>8s} {'n':>5s}")
rows=[]
for nm,ix in CARD:
    r1,n1=oof(Z[:,ix],RAD); r2,_=oof(Z[:,ix],USFM)
    print(f"   {nm:26s} {r1:+10.3f} {r2:+8.3f} {n1:5d}")
    rows.append(dict(var=nm,radiomics=float(r1),usfm=float(r2),n=int(n1)))
agree=[r for r in rows if r["radiomics"]>0.08 and r["usfm"]>0.08]
print(f"\n   variables where BOTH exceed 0.08: {[r['var'] for r in agree] or 'NONE'}")
rr=[r["radiomics"] for r in rows]; uu=[r["usfm"] for r in rows]
print(f"   agreement across the 11 variables: r={pearsonr(rr,uu)[0]:+.3f}")
print("\nCONTROL — is the cardiac block signal just ECHO IMAGE QUALITY?")
import pandas as pd
mer=pd.read_excel("data/IMPACT_merged_by_Cod.xlsx"); mer["fid"]=pd.to_numeric(mer["Cod"],errors="coerce")
q=mer.set_index("fid")["Calidad_ecocardio"] if "Calidad_ecocardio" in mer.columns else None
CAL=np.array([float(q.get(int(f),np.nan)) if q is not None else np.nan for f in fids])
print(f"   echo-quality coverage: {np.isfinite(CAL).mean():.2f}")
for nm,IM in (("radiomics",RAD),("USFM",USFM)):
    r_,n_=oof(CAL,IM)
    print(f"   {nm:10s} -> echo quality  r={r_:+.3f}  n={n_}")
    rows.append(dict(control=nm,to_echo_quality=float(r_)))
# add quality to the covariates and retest the block
print("\n   cardiac BLOCK with echo quality ADDED to the covariates:")
COV2=np.column_stack([COV,np.where(np.isfinite(CAL),CAL,0.0)[:,None]])
def oof2(y,X,COVx,seed=0):
    m=np.isfinite(y)&np.isfinite(X).all(1)
    A=COVx[m]; yy=y[m]-A@np.linalg.lstsq(A,y[m],rcond=None)[0]
    Xs=X[m]-A@np.linalg.lstsq(A,X[m],rcond=None)[0]
    p=np.zeros_like(yy)
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(Xs):
        pc=PCA(min(8,Xs.shape[1],len(tr)-1),random_state=0).fit(Xs[tr])
        p[te]=RidgeCV(alphas=np.logspace(-2,3,20)).fit(pc.transform(Xs[tr]),yy[tr]).predict(pc.transform(Xs[te]))
    return float(np.corrcoef(p,yy)[0,1])
CIX=[i for _,i in CARD]
Y1=np.nanmean(np.where(np.isfinite(Z[:,CIX]),Z[:,CIX],np.nan),axis=1)
for nm,IM in (("radiomics",RAD),("USFM",USFM)):
    print(f"     {nm:10s} cardiac-mean: GA+BMI {oof2(Y1,IM,COV):+.3f} -> +quality {oof2(Y1,IM,COV2):+.3f}")
    rows.append(dict(block_control=nm,ga_bmi=float(oof2(Y1,IM,COV)),plus_quality=float(oof2(Y1,IM,COV2))))
json.dump(rows,open("/tmp/rad_card.json","w"),indent=1)
