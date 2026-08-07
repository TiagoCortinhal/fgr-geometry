"""TARGET-FREE information decomposition: O-information (Rosas et al. 2019).
Omega = TC - DTC.  Omega > 0 => REDUNDANCY-dominated;  Omega < 0 => SYNERGY-dominated.
Needs no outcome -- it is a property of the joint distribution of the blocks alone.
Gaussian closed form: entropies from log-dets, so it is exact for second-order structure."""
import numpy as np, pandas as pd, json, itertools, warnings; warnings.filterwarnings("ignore")
from fgm.wp2_canonical import assemble_canonical
from fgm.loadings_heatmap_wide import SENTINEL_Z
rng=np.random.default_rng(0)

def H(C):                       # gaussian entropy up to the shared constant
    s,ld=np.linalg.slogdet(C)
    return 0.5*ld if s>0 else np.nan
def decomp(Z,groups):
    """groups: list of index-lists. Returns TC, DTC, Omega over the MULTIPLET of groups."""
    n=len(groups); C=np.cov(Z,rowvar=False)
    allidx=[i for g in groups for i in g]
    Hall=H(C[np.ix_(allidx,allidx)])
    Hi=[H(C[np.ix_(g,g)]) for g in groups]
    Hmi=[]
    for k in range(n):
        rest=[i for j,g in enumerate(groups) if j!=k for i in g]
        Hmi.append(H(C[np.ix_(rest,rest)]))
    TC=sum(Hi)-Hall
    DTC=sum(Hmi)-(n-1)*Hall
    return TC,DTC,TC-DTC

print("=== ESTIMATOR CONTROLS (planted structure, same n and shape as the cohort) ===")
N=977
def mk(kind,n=N):
    if kind=="independent":
        return np.column_stack([rng.standard_normal(n) for _ in range(4)])
    if kind=="redundant":                       # 4 noisy copies of one common factor
        f=rng.standard_normal(n)
        return np.column_stack([0.8*f+0.6*rng.standard_normal(n) for _ in range(4)])
    if kind=="synergistic":                     # X4 = X1+X2+X3 : no pairwise info, joint determines it
        a,b,c=(rng.standard_normal(n) for _ in range(3))
        return np.column_stack([a,b,c,(a+b+c)/np.sqrt(3)+0.1*rng.standard_normal(n)])
for kind in ("independent","redundant","synergistic"):
    Xs=mk(kind); Xs=(Xs-Xs.mean(0))/Xs.std(0)
    tc,dtc,om=decomp(Xs,[[0],[1],[2],[3]])
    print(f"  {kind:14s} TC={tc:+.4f}  DTC={dtc:+.4f}  Omega={om:+.4f}   -> {'REDUNDANT' if om>0.01 else 'SYNERGISTIC' if om<-0.01 else 'neither'}")
print("  (expected: independent ~0, redundant Omega>0, synergistic Omega<0)\n")

# ---------------- the real cohort ----------------
X,cols,blocks,fids,*_=assemble_canonical(); cols=list(cols); bl=np.array(blocks)
Xc=X.copy()
for i,c in enumerate(cols):
    if c.endswith("_z_ig21"): Xc[np.abs(Xc[:,i])>SENTINEL_Z,i]=np.nan
# image block: per-fetus pooled USFM PC1-3
z=np.load("/Users/tiago/dev/fgr-geometry/results/img_align/emb_usfm_multilayer.npz",allow_pickle=True)
imp=(pd.Series(z["dataset_type"]).astype(str).values=="impact")
E=z["emb_l5"].astype(np.float32)[imp]
fi=pd.to_numeric(pd.Series(z["fetus_id"]).astype(str),errors="coerce").values[imp]
ok=np.isfinite(fi)
df=pd.DataFrame(E[ok]); df["fid"]=fi[ok].astype(int)
pf=df.groupby("fid").mean()
from sklearn.decomposition import PCA
IP=PCA(3,random_state=0).fit_transform(pf.values)
lut={int(f):IP[i] for i,f in enumerate(pf.index)}
IMG=np.array([lut.get(int(f),[np.nan]*3) for f in fids])
Xp=np.column_stack([Xc,IMG]); colsp=cols+["imgPC1","imgPC2","imgPC3"]; blp=list(bl)+["image"]*3
mu=np.nanmean(Xp,0); sd=np.nanstd(Xp,0); sd[sd==0]=1; Z=(Xp-mu)/sd
blA=np.array(blp)
cc=np.isfinite(Z).all(1)
print(f"complete cases across all {Z.shape[1]} variables: {int(cc.sum())} of {len(Z)}")
Zc=Z[cc]
names=["growth","maternal","Doppler","cardiac","image"]
groups=[[i for i in range(len(colsp)) if blA[i]==b] for b in names]
print("block sizes:",{n:len(g) for n,g in zip(names,groups)})
tc,dtc,om=decomp(Zc,groups)
print(f"\n=== COHORT, 5 BLOCKS (complete cases n={int(cc.sum())}) ===")
print(f"  TC (total correlation)      = {tc:+.4f}   total dependence among blocks")
print(f"  DTC (dual total corr)       = {dtc:+.4f}")
print(f"  OMEGA (O-information)       = {om:+.4f}   -> {'REDUNDANCY' if om>0 else 'SYNERGY'}-dominated")
# permutation null: shuffle each block independently -> destroys cross-block structure
nulls=[]
for _ in range(200):
    Zs=Zc.copy()
    for g in groups: Zs[:,g]=Zs[rng.permutation(len(Zs))][:,g]
    nulls.append(decomp(Zs,groups)[2])
print(f"  block-shuffle null: mean {np.mean(nulls):+.4f}  95% [{np.percentile(nulls,2.5):+.4f},{np.percentile(nulls,97.5):+.4f}]")
json.dump(dict(TC=float(tc),DTC=float(dtc),Omega=float(om),n=int(cc.sum()),
  null_mean=float(np.mean(nulls)),null_lo=float(np.percentile(nulls,2.5)),
  null_hi=float(np.percentile(nulls,97.5))),open("/tmp/oinfo.json","w"),indent=1)
np.save("/tmp/_Zc.npy",Zc); json.dump({"groups":groups,"names":names,"colsp":colsp},open("/tmp/_grp.json","w"))
