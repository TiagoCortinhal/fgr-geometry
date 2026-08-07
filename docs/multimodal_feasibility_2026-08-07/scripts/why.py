"""Split-half says reliability 0.27-0.53; ANOVA says ICC=0. Both cannot be right.
Split-half is assumption-free, so the ANOVA is the suspect. Where does it break?"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
z=np.load("/Users/tiago/dev/fgr-geometry/results/img_align/_lag_seq_debiased.npz",allow_pickle=True)
lg,mk=z["lag_seq"],z["lag_mask"]; fd=[int(x) for x in z["fids"]]
rows=[(fd[i],t,lg[i,t]) for i in range(len(lg)) for t in np.where(mk[i])[0]]
D=pd.DataFrame(rows,columns=["fid","slot","lag"]); D=D[np.isfinite(D.lag)]
ni=D.groupby("fid").size()
print("KEY: the ANOVA pools ALL fetuses; split-half only uses those with >=2 (or >=4) images.")
print(f"  fetuses with exactly 1 image: {(ni==1).sum()} of {len(ni)}  ({100*(ni==1).mean():.0f}%)")
print(f"  they contribute {int((ni==1).sum())} observations and ZERO within-fetus information,")
print("  but they DO inflate MSB's degrees of freedom and dilute the between-fetus signal.\n")
def icc_of(Dx,tag):
    g=Dx.groupby("fid").lag; nn=g.size().values; mi=g.mean().values
    N=len(Dx); n_g=len(nn); gm=Dx.lag.mean()
    MSB=((nn*(mi-gm)**2).sum())/(n_g-1)
    MSW=(g.apply(lambda s:((s-s.mean())**2).sum()).sum())/(N-n_g)
    k0=(N-(nn**2).sum()/N)/(n_g-1); vb=max((MSB-MSW)/k0,0.0)
    print(f"  {tag:36s} N={N:5d} fet={n_g:4d} MSB={MSB:6.3f} MSW={MSW:6.3f} ICC={vb/(vb+MSW):.4f}")
    return vb/(vb+MSW)
print("ICC RESTRICTED TO FETUSES WITH REPEATED MEASURES (the split-half population):")
for kmin in (2,3,4,6):
    icc_of(D.groupby("fid").filter(lambda g: len(g)>=kmin), f">={kmin} images")
print("\nPOSITIVE CONTROL on the >=4 subset, same unbalanced design:")
rng=np.random.default_rng(0)
sub=D.groupby("fid").filter(lambda g: len(g)>=4); nn=sub.groupby("fid").size()
for t_icc in (0.30,0.10):
    a=rng.standard_normal(len(nn))*np.sqrt(t_icc)
    sim=[(f,a[i]+rng.standard_normal()*np.sqrt(1-t_icc)) for i,(f,k) in enumerate(nn.items()) for _ in range(k)]
    S=pd.DataFrame(sim,columns=["fid","lag"])
    icc_of(S,f"  planted ICC={t_icc:.2f}")
