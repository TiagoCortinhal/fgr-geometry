"""Proper variance decomposition of the image lag. My earlier within/between comparison used
different populations and an already-averaged between-SD, which biases the ratio upward."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
z=np.load("/Users/tiago/dev/fgr-geometry/results/img_align/_lag_seq_debiased.npz",allow_pickle=True)
lg,mk=z["lag_seq"],z["lag_mask"]; fd=[int(x) for x in z["fids"]]
rows=[]
for i in range(len(lg)):
    for t in np.where(mk[i])[0]: rows.append((fd[i],t,lg[i,t]))
D=pd.DataFrame(rows,columns=["fid","slot","lag"])
D=D[np.isfinite(D.lag)]
k=D.groupby("fid").size()
print(f"observations {len(D)}  fetuses {D.fid.nunique()}  obs/fetus: median {k.median():.0f} mean {k.mean():.2f}")
# ---- one-way random-effects ANOVA: lag_ij = mu + a_i + e_ij
g=D.groupby("fid").lag
ni=g.size().values; mi=g.mean().values; N=len(D); n_g=len(ni); gm=D.lag.mean()
MSB=((ni*(mi-gm)**2).sum())/(n_g-1)
MSW=(D.groupby("fid").lag.apply(lambda s:((s-s.mean())**2).sum()).sum())/(N-n_g)
k0=(N-(ni**2).sum()/N)/(n_g-1)     # harmonic-ish mean group size for unbalanced data
var_b=max((MSB-MSW)/k0,0.0); var_w=MSW
icc=var_b/(var_b+var_w)
print(f"\nVARIANCE DECOMPOSITION (one-way random effects, unbalanced; k0={k0:.2f})")
print(f"  between-fetus variance  {var_b:.4f}   (SD {np.sqrt(var_b):.3f})")
print(f"  within-fetus  variance  {var_w:.4f}   (SD {np.sqrt(var_w):.3f})")
print(f"  ICC = {icc:.4f}   <- fraction of lag variance that is BETWEEN fetuses")
print(f"\n  my earlier claim compared within SD 1.784 (n>=4 fetuses, raw) with between SD 1.260")
print(f"  (per-fetus MEANS, already averaged) -- different populations, and the between term")
print(f"  was pre-shrunk by averaging. Corrected SDs: within {np.sqrt(var_w):.3f}, between {np.sqrt(var_b):.3f}")
# ---- reliability of the per-fetus mean (Spearman-Brown)
print(f"\nRELIABILITY OF THE PER-FETUS MEAN, by number of images k:")
for kk in (1,2,3,5,8,12):
    R=kk*icc/(1+(kk-1)*icc)
    print(f"   k={kk:<3} reliability {R:.3f}   (n fetuses with >={kk}: {(k>=kk).sum()})")
Rmean=np.mean([ki*icc/(1+(ki-1)*icc) for ki in ni])
print(f"   ACTUAL cohort mean reliability = {Rmean:.3f}")
# ---- does PLANE explain the within-fetus noise?  (the fixable part)
print(f"\nATTENUATION: an observed correlation r is capped at sqrt(reliability) = {np.sqrt(Rmean):.3f}")
print(f"  so a TRUE lag-tabular correlation of rho would appear as rho*{np.sqrt(Rmean):.2f}")
print(f"  the measured latent->lag R2 of 0.033 corresponds to a disattenuated R2 of {0.033/Rmean:.4f}")
json.dump(dict(icc=float(icc),var_between=float(var_b),var_within=float(var_w),
  reliability_by_k={int(kk):float(kk*icc/(1+(kk-1)*icc)) for kk in (1,2,3,5,8,12)},
  cohort_mean_reliability=float(Rmean),k0=float(k0),n_obs=int(N),n_fetuses=int(n_g)),
  open("/tmp/icc.json","w"),indent=1)
