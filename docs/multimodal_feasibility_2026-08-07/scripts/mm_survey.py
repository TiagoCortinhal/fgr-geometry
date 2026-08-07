"""IN-DEPTH MULTIMODAL SURVEY: every tabular block vs images, full adjustment
ladder + GA-leakage audit + permutation null. The cardiac failure showed the
raw value is meaningless without the ladder."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
d=np.load("/tmp/card_sub.npz",allow_pickle=True)
Zp,cols,bl,fids=d["Zpanel"],[str(x) for x in d["cols"]],[str(x) for x in d["bl"]],[int(x) for x in d["fids"]]
IMG,GA,EFW=d["IMG"],d["GA"],d["EFW"]
dr=np.load("/tmp/card_raw.npz",allow_pickle=True); Craw,RAWN=dr["Craw"],[str(x) for x in dr["RAW"]]
BIOi=[i for i in range(len(cols)) if bl[i]=="growth"]
def cc(X,Y,npc=10,folds=5,seed=0):
    o=[]
    for tr,te in KFold(folds,shuffle=True,random_state=seed).split(X):
        p=PCA(min(npc,X.shape[1],Y.shape[1] if Y.shape[1]<npc else npc,len(tr)-1),random_state=0).fit(X[tr])
        c=CCA(n_components=1,max_iter=2000).fit(p.transform(X[tr]),Y[tr])
        a,b=c.transform(p.transform(X[te]),Y[te]); o.append(np.corrcoef(a[:,0],b[:,0])[0,1])
    return float(np.mean(o))
def rs(X,cv):
    if not cv: return X
    A=np.column_stack([np.ones(len(X))]+[np.asarray(c).reshape(len(X),-1) for c in cv])
    return X-A@np.linalg.lstsq(A,X,rcond=None)[0]
BLOCKS={b:[i for i in range(len(cols)) if bl[i]==b] for b in ("growth","maternal","Doppler","cardiac")}
res={}
print(f"{'block':22s} {'n':>4s} {'GAleak':>7s} {'raw':>7s} {'GA':>7s} {'+size':>7s} {'+biom':>7s} {'p':>7s} {'null95':>7s}")
for name,Y,names in ([(b,Zp[:,ix],[cols[i] for i in ix]) for b,ix in BLOCKS.items()]
                     +[("cardiac RAW morph",Craw,RAWN)]):
    sel=np.isfinite(IMG).all(1)&(np.isfinite(Y).sum(1)>=Y.shape[1]-1)&np.isfinite(GA)
    if sel.sum()<100: print(f"{name:22s} n={int(sel.sum())} too few"); continue
    Ys=np.where(np.isfinite(Y[sel]),Y[sel],0.0); Is=np.where(np.isfinite(IMG[sel]),IMG[sel],0.0)
    g=GA[sel][:,None]; e=np.where(np.isfinite(EFW[sel]),EFW[sel],0.0)[:,None]
    B=np.where(np.isfinite(Zp[sel][:,BIOi]),Zp[sel][:,BIOi],0.0)
    leak=np.nanmean([abs(pearsonr(Y[sel][np.isfinite(Y[sel][:,j]),j],GA[sel][np.isfinite(Y[sel][:,j])])[0]) for j in range(Y.shape[1])])
    lad={}
    for lab,cv in (("raw",[]),("GA",[g]),("GA+size",[g,e]),("+biom",[g,e,B])):
        lad[lab]=cc(rs(Is,cv),rs(Ys,cv))
    Xa=rs(Is,[g,e,B]); Ya=rs(Ys,[g,e,B]); obs=lad["+biom"]
    rng=np.random.default_rng(0); n=len(Xa)
    nl=np.array([cc(Xa,Ya[rng.permutation(n)]) for _ in range(400)])
    p=(1+(nl>=obs).sum())/(1+len(nl))
    print(f"{name:22s} {int(sel.sum()):4d} {leak:7.3f} {lad['raw']:+7.3f} {lad['GA']:+7.3f} {lad['GA+size']:+7.3f} {lad['+biom']:+7.3f} {p:7.3f} {np.percentile(nl,95):+7.3f}")
    res[name]=dict(n=int(sel.sum()),ga_leak=float(leak),ladder=lad,p=float(p),
                   null_p95=float(np.percentile(nl,95)),vars=names)
json.dump(res,open("/tmp/mm_survey.json","w"),indent=1)
