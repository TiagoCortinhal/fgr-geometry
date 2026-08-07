"""Image<->cardiac with the RAW morphology params the archived finding used,
vs the canonical Percentil_* panel columns. Same protocol for both."""
import numpy as np, pandas as pd, json, re, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
d=np.load("/tmp/card_sub.npz",allow_pickle=True)
Cpan,IMG,GA,EFW=d["C"],d["IMG"],d["GA"],d["EFW"]
Zp,cols,bl,fids=d["Zpanel"],[str(x) for x in d["cols"]],d["bl"],[int(x) for x in d["fids"]]
def euro(s):
    v=pd.to_numeric(s,errors="coerce")
    if np.isfinite(v).mean()>0.5: return v.to_numpy(float)
    t=s.astype(str).str.strip()
    if t.str.match(r"^-?\d{1,3}(?:\.\d{3})*,\d+$|^-?\d+,\d+$").mean()<0.3: return None
    v2=pd.to_numeric(t.str.replace(".","",regex=False).str.replace(",",".",regex=False),errors="coerce")
    return v2.to_numpy(float) if np.isfinite(v2).mean()>0.3 else None
E=pd.read_excel("/Users/tiago/dev/fgr-geometry/data_local/IMPACT_ecocardio_zscores_corrected.xlsx")
# the ARCHIVED set: morphology geometry, excluding cervical length (not cardiac) and percentile duplicates
RAW=["Circunf_cardiaca","long_cardiac_left","long_cardiac_right","basal_cardiac_left",
     "basal_cardiac_right","LV_SI","RV_SI"]
key=pd.to_numeric(E["Cod"],errors="coerce")
M={}
for c in RAW:
    v=euro(E[c]); M[c]=dict(zip([int(x) for x in key[np.isfinite(key)]],v[np.isfinite(key)]))
Craw=np.array([[M[c].get(f,np.nan) for c in RAW] for f in fids])
Craw=(Craw-np.nanmean(Craw,0))/np.nanstd(Craw,0)
print(f"RAW morphology params ({len(RAW)}): {RAW}")
print(f"  coverage per param: {[int(np.isfinite(Craw[:,j]).sum()) for j in range(len(RAW))]}")
BIO=[i for i in range(len(cols)) if bl[i]=="growth"]
def cc_heldout(X,Y,npc=10,folds=5,seed=0):
    o=[]
    for tr,te in KFold(folds,shuffle=True,random_state=seed).split(X):
        p=PCA(min(npc,X.shape[1],len(tr)-1),random_state=0).fit(X[tr])
        c=CCA(n_components=1,max_iter=2000).fit(p.transform(X[tr]),Y[tr])
        a,b=c.transform(p.transform(X[te]),Y[te]); o.append(np.corrcoef(a[:,0],b[:,0])[0,1])
    return float(np.mean(o))
def resid(X,cv):
    if not cv: return X
    A=np.column_stack([np.ones(len(X))]+[np.asarray(c).reshape(len(X),-1) for c in cv])
    return X-A@np.linalg.lstsq(A,X,rcond=None)[0]
res={}
for tag,C in (("RAW morphology (7)",Craw),("canonical Percentil_* (11)",Cpan)):
    sel=np.isfinite(IMG).all(1)&(np.isfinite(C).sum(1)>=C.shape[1]-1)&np.isfinite(GA)
    Cs=np.where(np.isfinite(C[sel]),C[sel],0.0); Is=np.where(np.isfinite(IMG[sel]),IMG[sel],0.0)
    g=GA[sel][:,None]; e=np.where(np.isfinite(EFW[sel]),EFW[sel],0.0)[:,None]
    B=np.where(np.isfinite(Zp[sel][:,BIO]),Zp[sel][:,BIO],0.0)
    print(f"\n=== {tag}  n={int(sel.sum())} ===")
    lad={}
    for lab,cv in (("raw",[]),("GA",[g]),("GA+size",[g,e]),("GA+size+biometry",[g,e,B])):
        v=cc_heldout(resid(Is,cv),resid(Cs,cv)); lad[lab]=v
        print(f"   {lab:20s} {v:+.4f}")
    Xa=resid(Is,[g,e,B]); Ya=resid(Cs,[g,e,B]); obs=lad["GA+size+biometry"]
    rng=np.random.default_rng(0); n=len(Xa)
    nl=np.array([cc_heldout(Xa,Ya[rng.permutation(n)]) for _ in range(500)])
    p=(1+(nl>=obs).sum())/(1+len(nl))
    bs=np.array([cc_heldout(Xa[i],Ya[i]) for i in (rng.integers(0,n,n) for _ in range(200))])
    print(f"   perm p={p:.4f}  null p95={np.percentile(nl,95):+.4f}  boot CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
    res[tag]=dict(n=int(sel.sum()),ladder=lad,p=float(p),null_p95=float(np.percentile(nl,95)),
                  ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))])
    np.save(f"/tmp/_null_{tag.split()[0]}.npy",nl)
json.dump(res,open("/tmp/card2.json","w"),indent=1)
np.savez("/tmp/card_raw.npz",Craw=Craw,RAW=np.array(RAW))
