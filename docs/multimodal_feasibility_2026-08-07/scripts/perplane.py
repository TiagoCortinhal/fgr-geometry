"""PER-PLANE image blocks vs the pooled double-averaged representation.
Does the image null survive when femur frames are not averaged into head frames?"""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/Users/tiago/dev/fgr-geometry/tools/impact_fetal_panel")
from fgm_tools import *
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
fgm_setup(); P=fgm_panel(); Z,cols,bl,fids=P["Z"],P["cols"],np.array(P["blocks"]),P["fids"]
GA=fgm_ga_at_echo(fids); EFW=Z[:,cols.index("efw_z_ig21")]; BMI=Z[:,cols.index("maternal_bmi")]
BIOi=[i for i in range(len(cols)) if bl[i]=="growth"]
BIO=np.where(np.isfinite(Z[:,BIOi]),Z[:,BIOi],0.0)
POOL,_=fgm_image_pcs(fids,n_pc=12)
for tag,lab in ((True,"LABELLED plane"),(False,"PROPAGATED plane_prop")):
    PP,cnt=fgm_image_pcs_by_plane(fids,n_pc=4,use_labelled=tag)
    print(f"\n=== {lab} ===")
    for p,c in cnt.items(): print(f"   {p:11s} frames={c['frames']:6d} fetuses={c['fetuses']:4d}")
    if tag: PPL,CNTL=PP,cnt
    else:   PPP,CNTP=PP,cnt
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
print("\n=== PER-PLANE IMAGE BLOCK -> TABULAR BLOCK (GA+BMI adjusted; biometry NOT used as its own covariate) ===")
print(f"{'image block':16s} {'target':10s} {'n':>5s} {'raw':>8s} {'GA':>8s} {'GA+BMI':>8s} {'p':>7s} {'null95':>8s}")
res={}
srcs=[("pooled (12 PC)",POOL)]+[(f"plane:{p}",PPL[p]) for p in PPL]
for iname,IM in srcs:
    for tname,ix in TB.items():
        Y=Z[:,ix]
        sel=np.isfinite(IM).all(1)&(np.isfinite(Y).sum(1)>=Y.shape[1]-1)&np.isfinite(GA)
        if sel.sum()<120: continue
        Ys=np.where(np.isfinite(Y[sel]),Y[sel],0.0); Is=np.where(np.isfinite(IM[sel]),IM[sel],0.0)
        g=GA[sel][:,None]; b=np.where(np.isfinite(BMI[sel]),BMI[sel],0.0)[:,None]
        lad={}
        for lb,cv in (("raw",[]),("GA",[g]),("GA+BMI",[g,b])):
            lad[lb]=cc(rs(Is,cv),rs(Ys,cv))
        Xa=rs(Is,[g,b]); Ya=rs(Ys,[g,b]); obs=lad["GA+BMI"]
        rng=np.random.default_rng(0); n=len(Xa)
        nl=np.array([cc(Xa,Ya[rng.permutation(n)]) for _ in range(300)])
        p=(1+(nl>=obs).sum())/(1+len(nl))
        print(f"{iname:16s} {tname:10s} {int(sel.sum()):5d} {lad['raw']:+8.3f} {lad['GA']:+8.3f} {lad['GA+BMI']:+8.3f} {p:7.3f} {np.percentile(nl,95):+8.3f}")
        res[f"{iname}|{tname}"]=dict(n=int(sel.sum()),ladder=lad,p=float(p),null_p95=float(np.percentile(nl,95)))
json.dump(dict(results=res,counts_labelled={k:v for k,v in CNTL.items()},
               counts_prop={k:v for k,v in CNTP.items()}),open("/tmp/perplane.json","w"),indent=1)
