"""Image<->cardiac alignment: held-out regularised CCA with an adjustment ladder.
Reproduces the archived 0.248 with current tooling, then extends."""
import numpy as np, json, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
d=np.load("/tmp/card_sub.npz",allow_pickle=True)
C,IMG,GA,EFW,m=d["C"],d["IMG"],d["GA"],d["EFW"],d["m"]
Zp,cols,bl=d["Zpanel"],[str(x) for x in d["cols"]],d["bl"]
CARD=[str(x) for x in d["CARD"]]
sel=m&np.isfinite(GA)
Cs=C[sel]; Is=IMG[sel]; g=GA[sel]; e=EFW[sel]
# impute the 0-1 missing cardiac cells by column mean (>=10/11 present)
Cs=np.where(np.isfinite(Cs),Cs,0.0)
e=np.where(np.isfinite(e),e,0.0)
g=np.where(np.isfinite(g),g,np.nanmean(g))
BIO=[i for i in range(len(cols)) if bl[i]=="growth"]
B=np.where(np.isfinite(Zp[sel][:,BIO]),Zp[sel][:,BIO],0.0)
Is=np.where(np.isfinite(Is),Is,0.0)
n=len(Cs); print(f"n={n}  cardiac d={Cs.shape[1]}  image PCs d={Is.shape[1]}")

def resid(X,covars):
    if covars is None: return X
    A=np.column_stack([np.ones(len(X))]+covars)
    return X-A@np.linalg.lstsq(A,X,rcond=None)[0]

def heldout_cc(X,Y,ncomp=1,folds=5,seed=0,npc=10):
    """held-out canonical correlation: fit CCA on train, correlate projections on test"""
    out=[]
    for tr,te in KFold(folds,shuffle=True,random_state=seed).split(X):
        p=PCA(npc,random_state=0).fit(X[tr])          # image PCA fit on TRAIN only
        Xtr,Xte=p.transform(X[tr]),p.transform(X[te])
        c=CCA(n_components=ncomp,max_iter=2000).fit(Xtr,Y[tr])
        a,b=c.transform(Xte,Y[te])
        out.append(np.corrcoef(a[:,0],b[:,0])[0,1])
    return float(np.mean(out))

ladder=[("raw",None),("GA",[g[:,None]]),("GA+size",[g[:,None],e[:,None]]),
        ("GA+size+biometry",[g[:,None],e[:,None],B])]
print("\n=== ADJUSTMENT LADDER (held-out canonical correlation, 5-fold) ===")
res={}
for lab,cov in ladder:
    Xa=resid(Is,cov); Ya=resid(Cs,cov)
    cc=heldout_cc(Xa,Ya)
    res[lab]=float(cc); print(f"  {lab:20s} held-out cc = {cc:+.4f}")
print("  archived reference: 0.218@GA -> 0.225@GA+size -> 0.248@GA+size+biometry")

# permutation null on the fully-adjusted arm
print("\n=== PERMUTATION NULL (fully adjusted, 1000 perms) ===")
Xa=resid(Is,[g[:,None],e[:,None],B]); Ya=resid(Cs,[g[:,None],e[:,None],B])
obs=heldout_cc(Xa,Ya); rng=np.random.default_rng(0); nl=[]
for b_ in range(1000):
    nl.append(heldout_cc(Xa,Ya[rng.permutation(n)]))
nl=np.array(nl); p=(1+(nl>=obs).sum())/(1+len(nl))
print(f"  observed {obs:+.4f}  null mean {nl.mean():+.4f}  p95 {np.percentile(nl,95):+.4f}  p={p:.2e}")
# bootstrap CI
bs=[]
for b_ in range(300):
    i=rng.integers(0,n,n); bs.append(heldout_cc(Xa[i],Ya[i]))
print(f"  bootstrap CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
json.dump(dict(ladder=res,obs=float(obs),null_mean=float(nl.mean()),null_p95=float(np.percentile(nl,95)),
  p=float(p),ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],n=int(n)),
  open("/tmp/card1.json","w"),indent=1)
np.save("/tmp/_null1.npy",nl)
