"""(2) SPLIT-HALF first -- assumption-free adjudication of ICC=0 vs rho=+0.21.
Then (1) heteroscedasticity-aware ICC."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, pearsonr
z=np.load("/Users/tiago/dev/fgr-geometry/results/img_align/_lag_seq_debiased.npz",allow_pickle=True)
lg,mk=z["lag_seq"],z["lag_mask"]; fd=[int(x) for x in z["fids"]]
rows=[(fd[i],t,lg[i,t]) for i in range(len(lg)) for t in np.where(mk[i])[0]]
D=pd.DataFrame(rows,columns=["fid","slot","lag"]); D=D[np.isfinite(D.lag)]
print(f"N obs = {len(D)}   fetuses = {D.fid.nunique()}")   # print it, don't infer it
print("\n=== (2) SPLIT-HALF RELIABILITY (no ANOVA assumptions) ===")
print("odd-numbered vs even-numbered images of the SAME fetus\n")
for kmin in (2,4,6):
    sub=D.groupby("fid").filter(lambda g: len(g)>=kmin)
    a=sub[sub.slot%2==0].groupby("fid").lag.mean()
    b=sub[sub.slot%2==1].groupby("fid").lag.mean()
    j=pd.concat([a,b],axis=1,keys=["even","odd"]).dropna()
    if len(j)<30: continue
    r=pearsonr(j.even,j.odd)[0]
    sb=2*r/(1+r) if r>-1 else np.nan            # Spearman-Brown: full-length reliability
    print(f"  >={kmin} images: n={len(j):3d}  half-half r={r:+.3f}   Spearman-Brown reliability={sb:+.3f}")
# random-half version (slot parity may confound with acquisition order)
rng=np.random.default_rng(0); rs=[]
sub=D.groupby("fid").filter(lambda g: len(g)>=4)
for b_ in range(50):
    A=[];B=[]
    for f,g in sub.groupby("fid"):
        idx=rng.permutation(len(g)); h=len(g)//2
        A.append(g.lag.values[idx[:h]].mean()); B.append(g.lag.values[idx[h:2*h]].mean())
    rs.append(pearsonr(A,B)[0])
r=np.mean(rs); print(f"  RANDOM halves (>=4 img, n={len(A)}, 50 draws): r={r:+.3f} [{np.percentile(rs,2.5):+.3f},{np.percentile(rs,97.5):+.3f}]"
      f"  -> SB reliability {2*r/(1+r):+.3f}")
print("\n=== (1) HETEROSCEDASTICITY-AWARE ICC ===")
def icc_of(Dx,tag):
    g=Dx.groupby("fid").lag; ni=g.size().values; mi=g.mean().values
    N=len(Dx); n_g=len(ni); gm=Dx.lag.mean()
    MSB=((ni*(mi-gm)**2).sum())/(n_g-1)
    MSW=(g.apply(lambda s:((s-s.mean())**2).sum()).sum())/(N-n_g)
    k0=(N-(ni**2).sum()/N)/(n_g-1)
    vb=max((MSB-MSW)/k0,0.0)
    print(f"  {tag:34s} N={N:5d} fet={n_g:4d}  MSB={MSB:6.3f} MSW={MSW:6.3f}  ICC={vb/(vb+MSW):.4f}")
    return vb/(vb+MSW)
icc_of(D,"all slots (as before)")
icc_of(D[D.slot>0],"drop slot 0 (SD 4.04)")
icc_of(D[D.slot>1],"drop slots 0-1")
icc_of(D[(D.slot>=2)&(D.slot<=5)],"slots 2-5 only (homogeneous SD)")
# variance-standardise each slot, then ICC
D2=D.copy(); D2["lag"]=D2.groupby("slot").lag.transform(lambda s:(s-s.mean())/s.std())
icc_of(D2,"per-slot variance-standardised")
json.dump({"note":"see log"},open("/tmp/split_done.json","w"))
