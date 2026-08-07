import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
z=np.load("/Users/tiago/dev/fgr-geometry/results/img_align/_lag_seq_debiased.npz",allow_pickle=True)
lg,mk=z["lag_seq"],z["lag_mask"]; fd=[int(x) for x in z["fids"]]
rows=[(fd[i],t,lg[i,t]) for i in range(len(lg)) for t in np.where(mk[i])[0]]
D=pd.DataFrame(rows,columns=["fid","slot","lag"]); D=D[np.isfinite(D.lag)]
g=D.groupby("fid").lag; ni=g.size().values; mi=g.mean().values; N=len(D); n_g=len(ni); gm=D.lag.mean()
MSB=((ni*(mi-gm)**2).sum())/(n_g-1)
MSW=(g.apply(lambda s:((s-s.mean())**2).sum()).sum())/(N-n_g)
print(f"MSB={MSB:.4f}  MSW={MSW:.4f}   MSB/MSW={MSB/MSW:.4f}")
print(f"  -> MSB < MSW, so the between-fetus variance component is NEGATIVE and clips to 0.")
print(f"     Fetuses differ LESS than images of the same fetus. This is not a small ICC; it is a null one.\n")
# SANITY: a POSITIVE CONTROL -- does the same estimator recover a known ICC?
rng=np.random.default_rng(0)
for true_icc in (0.30,0.05):
    a=rng.standard_normal(n_g)*np.sqrt(true_icc)
    sim=[]
    for i,(f,k) in enumerate(zip(g.size().index,ni)):
        sim += [(f,a[i]+rng.standard_normal()*np.sqrt(1-true_icc)) for _ in range(k)]
    S=pd.DataFrame(sim,columns=["fid","lag"]); gg=S.groupby("fid").lag
    nn=gg.size().values; mm=gg.mean().values; GM=S.lag.mean()
    B=((nn*(mm-GM)**2).sum())/(len(nn)-1)
    W=(gg.apply(lambda s:((s-s.mean())**2).sum()).sum())/(len(S)-len(nn))
    k0=(len(S)-(nn**2).sum()/len(S))/(len(nn)-1)
    est=max((B-W)/k0,0)/(max((B-W)/k0,0)+W)
    print(f"  POSITIVE CONTROL: planted ICC={true_icc:.2f} -> estimator recovers {est:.3f}")
# where does the within-fetus variance come from? SLOT (=visit/time) or residual?
print(f"\nIS THE WITHIN-FETUS VARIANCE STRUCTURED BY TIME SLOT?")
sv=D.groupby("slot").lag.agg(["mean","std","size"])
print(sv.head(8).round(3).to_string())
print(f"\n  between-SLOT SD of slot means: {sv['mean'].std():.3f}  vs overall lag SD {D.lag.std():.3f}")
# two-way: fetus + slot
import itertools
D["slot_c"]=D.slot.astype(str)
ss=D.groupby("slot").lag.transform("mean")
resid=D.lag-ss
gg2=resid.groupby(D.fid); ni2=gg2.size().values; mi2=gg2.mean().values
B2=((ni2*(mi2-resid.mean())**2).sum())/(len(ni2)-1)
W2=(gg2.apply(lambda s:((s-s.mean())**2).sum()).sum())/(len(resid)-len(ni2))
k02=(len(resid)-(ni2**2).sum()/len(resid))/(len(ni2)-1)
icc2=max((B2-W2)/k02,0)/(max((B2-W2)/k02,0)+W2)
print(f"  ICC after removing the slot mean: {icc2:.4f}")
json.dump(dict(MSB=float(MSB),MSW=float(MSW),ratio=float(MSB/MSW),icc_slot_adjusted=float(icc2)),
          open("/tmp/icc2.json","w"),indent=1)
