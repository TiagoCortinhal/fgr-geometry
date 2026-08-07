"""ARD control, NON-CIRCULAR: fit the latent on the 25 TABULAR outputs only, then fit ARD
lengthscales for the lag AND the nulls on that latent. Now all four are equally unseen."""
import numpy as np, torch, json, warnings; warnings.filterwarnings("ignore")
from fgm.wp2_canonical import assemble_canonical, FactorAnalysisMissing, varimax
from fgm.latent_risk_uncertainty import fa_posterior
torch.set_num_threads(10); torch.manual_seed(0)
d=np.load("/tmp/gplvm_fit.npz",allow_pickle=True)
Y,cols5,lag=d["Y"],[str(x) for x in d["cols5"]],d["lag"]
li=cols5.index("image_lag"); Yt=np.delete(Y,li,axis=1)
N,Dt=Yt.shape; Q=8
obs=[np.where(np.isfinite(Yt[:,j]))[0] for j in range(Dt)]
fa=FactorAnalysisMissing(Q,seed=0).fit(Yt); Wf=varimax(fa.W_)
X0,_,_=fa_posterior(Wf,fa.psi_,fa.mu_,Yt)
Xl=torch.tensor(X0.astype(np.float32),requires_grad=True)
lls=torch.zeros(Dt,Q,requires_grad=True); lsf=torch.zeros(Dt,requires_grad=True)
lsn=torch.full((Dt,),-1.0,requires_grad=True)
yt=[torch.tensor(Yt[o,j],dtype=torch.float32) for j,o in enumerate(obs)]
opt=torch.optim.Adam([Xl,lls,lsf,lsn],lr=0.05)
for s in range(250):
    opt.zero_grad(); nll=0.0
    for j,o in enumerate(obs):
        Xj=Xl[o]/torch.exp(lls[j]); d2=torch.cdist(Xj,Xj)**2
        K=torch.exp(2*lsf[j])*torch.exp(-0.5*d2)+(torch.exp(2*lsn[j])+1e-4)*torch.eye(len(o))
        L=torch.linalg.cholesky(K); a=torch.cholesky_solve(yt[j].unsqueeze(1),L)
        nll=nll+0.5*(yt[j]@a.squeeze())+torch.log(torch.diagonal(L)).sum()
    nll=nll+0.5*(Xl**2).sum(); nll.backward(); opt.step()
Xt=Xl.detach()
print("ARD ON A LAG-FREE LATENT — lag and nulls now equally unseen\n")
rng=np.random.default_rng(0); m=np.isfinite(lag)
cands={"image_lag":lag}
cands["noise_full"]=rng.standard_normal(N)
cands["noise_lagcov"]=np.where(m,rng.standard_normal(N),np.nan)
sh=lag.copy(); s2=lag[m].copy(); rng.shuffle(s2); sh[m]=s2; cands["lag_shuffled"]=sh
def fit_ard(yv,seed=0):
    torch.manual_seed(seed)
    o=np.where(np.isfinite(yv))[0]
    ys=(yv[o]-np.nanmean(yv))/np.nanstd(yv)
    ytt=torch.tensor(ys,dtype=torch.float32)
    L_=torch.zeros(Q,requires_grad=True); sf=torch.zeros(1,requires_grad=True); sn=torch.full((1,),-1.0,requires_grad=True)
    op=torch.optim.Adam([L_,sf,sn],lr=0.05); Xo=Xt[o]
    for s in range(250):
        op.zero_grad(); Xj=Xo/torch.exp(L_); d2=torch.cdist(Xj,Xj)**2
        K=torch.exp(2*sf)*torch.exp(-0.5*d2)+(torch.exp(2*sn)+1e-4)*torch.eye(len(o))
        Lc=torch.linalg.cholesky(K); a=torch.cholesky_solve(ytt.unsqueeze(1),Lc)
        nl=0.5*(ytt@a.squeeze())+torch.log(torch.diagonal(Lc)).sum(); nl.backward(); op.step()
    r=np.exp(-L_.detach().numpy()); return r/r.sum(), float(nl.item())
print(f"{'output':>16} {'max relevance':>14} {'nll':>10}")
out={}
for lab,yv in cands.items():
    rel,nl=fit_ard(yv); out[lab]=dict(max_rel=float(rel.max()),nll=nl)
    print(f"{lab:>16} {rel.max():>14.3f} {nl:>10.1f}")
print(f"\n  uniform = {1/Q:.3f}")
print("  CIRCULAR version (latent fitted WITH the lag): lag 0.464 vs nulls 0.188-0.204")
json.dump(out,open("/tmp/ard_clean.json","w"),indent=1)
