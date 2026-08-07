"""Kalman with the image lag as a 5th measurement channel, variance sigma^2/k per visit.
Question: does an unreliable image channel add information to the biometry size state?"""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from scipy.optimize import minimize_scalar
from fgm.latent_kalman import kalman_filter, rts_smoother, independent_placement, build_extended, MEAS
from fgm.wp2_canonical import FactorAnalysisMissing
lo,_=build_extended("data/visits_long_z.csv","data/IMPACT_merged_by_Cod.xlsx",0)
print(f"N rows={len(lo)}  fetuses={lo._fid.nunique()}  nodes={sorted(lo.visit.unique())}")
Xz=lo[MEAS].to_numpy(float)
fa=FactorAnalysisMissing(n_factors=1,seed=0).fit(Xz)
b=fa.W_[:,0].copy(); Psi=fa.psi_.copy(); mu_b=fa.mu_.copy()
if b[MEAS.index("ac_z_ig21")]<0: b=-b
print("biometry channels:", {c:round(float(bb),3) for c,bb in zip(MEAS,b)})
# ---- per-fetus lag with its own k (number of images) ----
z=np.load("/Users/tiago/dev/fgr-geometry/results/img_align/_lag_seq_debiased.npz",allow_pickle=True)
lg,mk=z["lag_seq"],z["lag_mask"]; fdz=[int(x) for x in z["fids"]]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lagmean=np.nanmean(np.where(mk,lg,np.nan),axis=1)
kcount=mk.sum(1)
LAG=dict(zip(fdz,lagmean)); KC=dict(zip(fdz,kcount))
# reliability-derived measurement variance: var of the MEAN of k noisy images
ICC=0.062          # from the >=4-image restricted estimate (lag_reliability_corrected.json)
lagsd=np.nanstd(lagmean)
print(f"\nlag channel: ICC={ICC}, between-SD of per-fetus means={lagsd:.3f}")
def lag_var(k):
    """noise variance of a k-image mean, in units where the state has variance 1"""
    if k<1: return np.inf
    rel=k*ICC/(1+(k-1)*ICC)
    return max((1-rel)/max(rel,1e-6),1e-3)
for k in (1,2,4,8): print(f"   k={k}: reliability {k*ICC/(1+(k-1)*ICC):.3f} -> measurement variance {lag_var(k):.1f}")
# lag loading on the size state: regress standardised lag on the independent-placement size estimate
def per_fetus_ext(lo,mu_b,with_lag):
    for fid,g in lo.groupby("_fid"):
        g=g.sort_values("ga_weeks")
        gas=g["ga_weeks"].to_numpy(float); X=g[MEAS].to_numpy(float)-mu_b
        ok=~np.isnan(gas); gas=gas[ok]; X=X[ok]
        if with_lag:
            lv=LAG.get(int(fid),np.nan); kk=KC.get(int(fid),0)
            X=np.column_stack([X,np.full(len(X),(lv-np.nanmean(lagmean))/lagsd if np.isfinite(lv) else np.nan)])
            yield fid,gas,[X[i] for i in range(len(gas))],[~np.isnan(X[i]) for i in range(len(gas))],kk
        else:
            yield fid,gas,[X[i] for i in range(len(gas))],[~np.isnan(X[i]) for i in range(len(gas))],0
# estimate the lag's loading on the size state
base=list(per_fetus_ext(lo,mu_b,False))
def fitq(fets,bb,PP):
    def nll(lq):
        q=np.exp(lq); t=0.0
        for fid,gas,xs,ms,_ in fets:
            if len(gas)>=1: t+=kalman_filter(gas,xs,ms,bb,PP,q)[4]
        return -t
    r=minimize_scalar(nll,bounds=(np.log(1e-4),np.log(5.0)),method="bounded")
    return float(np.exp(r.x))
q0=fitq(base,b,Psi)
sz=[]; lgv=[]
for fid,gas,xs,ms,_ in base:
    mi,_=independent_placement(xs,ms,b,Psi)
    if len(mi) and int(fid) in LAG and np.isfinite(LAG[int(fid)]):
        sz.append(np.mean(mi)); lgv.append((LAG[int(fid)]-np.nanmean(lagmean))/lagsd)
from scipy.stats import pearsonr
r_sl=pearsonr(sz,lgv)[0]
print(f"\nlag vs size-state correlation: r={r_sl:+.3f}  -> loading b_lag={r_sl:+.3f} (n={len(sz)})")
print("\n=== COMPARISON: posterior SD of the size state, with and without the lag channel ===")
res={}
for tag,wl in (("biometry only",False),("+ image lag channel",True)):
    fets=list(per_fetus_ext(lo,mu_b,wl))
    if wl:
        bb=np.append(b,r_sl); 
    else:
        bb=b
    sds=[]; n_lag=0
    for fid,gas,xs,ms,kk in fets:
        if len(gas)==0: continue
        PP=np.append(Psi,lag_var(kk)) if wl else Psi
        mf,Pf,_,_,_=kalman_filter(gas,xs,ms,bb,PP,q0)
        msm,Ps=rts_smoother(gas,mf,Pf,q0)
        sds.append(np.sqrt(Ps).mean())
        if wl and kk>=1: n_lag+=1
    print(f"  {tag:22s} mean smoothed posterior SD = {np.mean(sds):.4f}  (n={len(sds)} fetuses)")
    res[tag]=float(np.mean(sds))
print(f"\n  narrowing from the lag channel: {100*(1-res['+ image lag channel']/res['biometry only']):.2f}%")
# and for the well-measured fetuses only
print("\n  restricted to fetuses with >=4 images (reliability ~0.41):")
for tag,wl in (("biometry only",False),("+ image lag",True)):
    fets=[f for f in per_fetus_ext(lo,mu_b,wl) if KC.get(int(f[0]),0)>=4]
    bb=np.append(b,r_sl) if wl else b
    sds=[]
    for fid,gas,xs,ms,kk in fets:
        if len(gas)==0: continue
        PP=np.append(Psi,lag_var(kk)) if wl else Psi
        mf,Pf,_,_,_=kalman_filter(gas,xs,ms,bb,PP,q0)
        _,Ps=rts_smoother(gas,mf,Pf,q0); sds.append(np.sqrt(Ps).mean())
    print(f"    {tag:18s} mean posterior SD = {np.mean(sds):.4f}  (n={len(sds)})")
    res[tag+" k>=4"]=float(np.mean(sds))
json.dump(dict(q=q0,b_lag=float(r_sl),icc=ICC,**res),open("/tmp/kal_lag.json","w"),indent=1)
