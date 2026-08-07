"""Why does the ladder collapse? Test whether the raw 0.248 is GA-driven:
raw morphology params are UNSCALED sizes (cm), so they grow with GA -- and
image appearance encodes GA. That would produce a large raw cc that vanishes on adjustment."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
d=np.load("/tmp/card_sub.npz",allow_pickle=True); dr=np.load("/tmp/card_raw.npz",allow_pickle=True)
Craw,RAW=dr["Craw"],[str(x) for x in dr["RAW"]]; Cpan=d["C"]; GA=d["GA"]; IMG=d["IMG"]
CARD=[str(x) for x in d["CARD"]]
print("DO THE TWO CARDIAC REPRESENTATIONS TRACK GESTATIONAL AGE?\n")
print("  RAW morphology (unscaled measurements):")
for j,c in enumerate(RAW):
    m=np.isfinite(Craw[:,j])&np.isfinite(GA)
    print(f"    {c:24s} r with GA = {pearsonr(Craw[m,j],GA[m])[0]:+.3f}")
print("\n  canonical Percentil_* (already age-normalised by construction):")
for j,c in enumerate(CARD[:6]):
    m=np.isfinite(Cpan[:,j])&np.isfinite(GA)
    print(f"    {c:24s} r with GA = {pearsonr(Cpan[m,j],GA[m])[0]:+.3f}")
mr=np.nanmean([abs(pearsonr(Craw[np.isfinite(Craw[:,j])&np.isfinite(GA),j],GA[np.isfinite(Craw[:,j])&np.isfinite(GA)])[0]) for j in range(len(RAW))])
mp=np.nanmean([abs(pearsonr(Cpan[np.isfinite(Cpan[:,j])&np.isfinite(GA),j],GA[np.isfinite(Cpan[:,j])&np.isfinite(GA)])[0]) for j in range(len(CARD))])
print(f"\n  mean |r| with GA:  raw morphology {mr:.3f}   percentile-scored {mp:.3f}")
# and the image side
print("\n  image PC1-5 vs GA:")
for k in range(5):
    m=np.isfinite(IMG[:,k])&np.isfinite(GA)
    print(f"    imgPC{k+1}  r with GA = {pearsonr(IMG[m,k],GA[m])[0]:+.3f}")
json.dump(dict(mean_abs_r_GA_raw=float(mr),mean_abs_r_GA_pct=float(mp),
  raw_per=[float(pearsonr(Craw[np.isfinite(Craw[:,j])&np.isfinite(GA),j],GA[np.isfinite(Craw[:,j])&np.isfinite(GA)])[0]) for j in range(len(RAW))],
  RAW=RAW),open("/tmp/card3.json","w"),indent=1)
