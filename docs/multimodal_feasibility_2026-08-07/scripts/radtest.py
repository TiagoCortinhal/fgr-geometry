"""RADIOMICS vs USFM: is the image null encoder-general or a USFM artefact?
Same protocol as the per-plane USFM run: GA+BMI adjusted, permutation nulls, BH."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/Users/tiago/dev/fgr-geometry/tools/impact_fetal_panel")
from fgm_tools import *
RADPQ="/tmp/_rad.parquet"
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
fgm_setup(); P=fgm_panel(); Z,cols,bl,fids=P["Z"],P["cols"],np.array(P["blocks"]),P["fids"]
GA=fgm_ga_at_echo(fids); BMI=Z[:,cols.index("maternal_bmi")]
RAD,rfeat=fgm_radiomics(fids,n_pc=12,artifact_path_fn=lambda v: RADPQ)
USFM,_=fgm_image_pcs(fids,n_pc=12)
print(f"radiomics features used: {len(rfeat)}  (families only, metadata excluded)")
print(f"  families: {sorted({f.split('_')[1] for f in rfeat if '_' in f})}")
print(f"  fetuses with radiomics: {int(np.isfinite(RAD).all(1).sum())} | with USFM: {int(np.isfinite(USFM).all(1).sum())}")
# GA leakage of the representation itself
for nm,R in (("radiomics",RAD),("USFM",USFM)):
    rr=[abs(pearsonr(R[np.isfinite(R[:,k])&np.isfinite(GA),k],GA[np.isfinite(R[:,k])&np.isfinite(GA)])[0]) for k in range(min(5,R.shape[1]))]
    print(f"  {nm:10s} |r| with GA, PC1-5: {[round(x,3) for x in rr]}")
def cc(X,Y,npc=10,seed=0):
    o=[]
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(X):
        k=min(npc,X.shape[1],len(tr)-1)
        p=PCA(k,random_state=0).fit(X[tr])
        c=CCA(n_components=1,max_iter=2000).fit(p.transform(X[tr]),Y[tr])
        a,b=c.transform(p.transform(X[te]),Y[te]); o.append(np.corrcoef(a[:,0],b[:,0])[0,1])
    return float(np.mean(o))
def rs(X,cv):
    if not cv: return X
    A=np.column_stack([np.ones(len(X))]+[np.asarray(c).reshape(len(X),-1) for c in cv])
    return X-A@np.linalg.lstsq(A,X,rcond=None)[0]
TB={b:[i for i in range(len(cols)) if bl[i]==b] for b in ("growth","maternal","Doppler","cardiac")}
print(f"\n{'representation':14s} {'target':10s} {'n':>5s} {'raw':>8s} {'GA':>8s} {'GA+BMI':>8s} {'p':>7s}")
res={}
for iname,IM in (("radiomics",RAD),("USFM",USFM)):
    for tname,ix in TB.items():
        Y=Z[:,ix]
        sel=np.isfinite(IM).all(1)&(np.isfinite(Y).sum(1)>=Y.shape[1]-1)&np.isfinite(GA)
        if sel.sum()<120: continue
        Ys=np.where(np.isfinite(Y[sel]),Y[sel],0.0); Is=IM[sel]
        g=GA[sel][:,None]; b=np.where(np.isfinite(BMI[sel]),BMI[sel],0.0)[:,None]
        lad={lb:cc(rs(Is,cv),rs(Ys,cv)) for lb,cv in (("raw",[]),("GA",[g]),("GA+BMI",[g,b]))}
        Xa=rs(Is,[g,b]); Ya=rs(Ys,[g,b]); obs=lad["GA+BMI"]
        rng=np.random.default_rng(0); n=len(Xa)
        nl=np.array([cc(Xa,Ya[rng.permutation(n)]) for _ in range(300)])
        p=(1+(nl>=obs).sum())/(1+len(nl))
        print(f"{iname:14s} {tname:10s} {int(sel.sum()):5d} {lad['raw']:+8.3f} {lad['GA']:+8.3f} {lad['GA+BMI']:+8.3f} {p:7.3f}")
        res[f"{iname}|{tname}"]=dict(n=int(sel.sum()),ladder=lad,p=float(p),null_p95=float(np.percentile(nl,95)))
# do the two representations agree with each other?
m=np.isfinite(RAD).all(1)&np.isfinite(USFM).all(1)
print(f"\nDO THE TWO REPRESENTATIONS SEE THE SAME THING? (n={int(m.sum())})")
print(f"   radiomics <-> USFM held-out cc = {cc(RAD[m],USFM[m]):+.3f}")
print(f"   radiomics -> maternal BMI (GA-adj) = {cc(rs(RAD[m],[GA[m][:,None]]),rs(np.where(np.isfinite(BMI[m]),BMI[m],0.0)[:,None],[GA[m][:,None]])):+.3f}")
json.dump(res,open("/tmp/radtest.json","w"),indent=1)
