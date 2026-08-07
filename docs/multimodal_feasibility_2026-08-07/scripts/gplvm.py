"""Missing-data GPLVM (MAP, per-output ARD) as a NONLINEAR unsupervised latent.
Question: does the image lag find a home in a nonlinear latent that the linear FA denies it?
Marginalisation note: a GPLVM cannot do product-of-experts marginalisation in closed form,
so this is a DIAGNOSTIC, not a replacement for the paper's model."""
import numpy as np, pandas as pd, torch, json, time, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from fgm.wp2_canonical import assemble_canonical, FactorAnalysisMissing, varimax
from fgm.loadings_heatmap_wide import SENTINEL_Z
from fgm.latent_risk_uncertainty import fa_posterior
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(10)
Q=8   # latent dimension, matched to the paper's K=8

X,cols,blocks,fids,*_=assemble_canonical(); cols=list(cols); bl=np.array(blocks)
Xc=X.copy()
for i,c in enumerate(cols):
    if c.endswith("_z_ig21"): Xc[np.abs(Xc[:,i])>SENTINEL_Z,i]=np.nan
z=np.load("/Users/tiago/dev/fgr-geometry/results/img_align/_lag_seq_debiased.npz",allow_pickle=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pooled=np.nanmean(np.where(z["lag_mask"],z["lag_seq"],np.nan),axis=1)
lag=np.array([dict(zip([int(x) for x in z["fids"]],pooled)).get(int(f),np.nan) for f in fids])
X5=np.column_stack([Xc,lag]); cols5=cols+["image_lag"]; bl5=np.append(bl,"image")
mu=np.nanmean(X5,0); sd=np.nanstd(X5,0); sd[sd==0]=1
Y=(X5-mu)/sd
N,D=Y.shape
obs=[np.where(np.isfinite(Y[:,j]))[0] for j in range(D)]
print(f"GPLVM  N={N}  D={D}  Q={Q}  | observed per output: min {min(len(o) for o in obs)} max {max(len(o) for o in obs)}",flush=True)

def fit_gplvm(Yin, obs_idx, Xinit, steps=250, lr=0.05, tag=""):
    Xl=torch.tensor(Xinit,dtype=torch.float32,requires_grad=True)
    D_=Yin.shape[1]
    log_ls=torch.zeros(D_,Q,requires_grad=True)          # per-output ARD lengthscales
    log_sf=torch.zeros(D_,requires_grad=True)            # signal scale
    log_sn=torch.full((D_,),-1.0,requires_grad=True)     # noise
    yt=[torch.tensor(Yin[o,j],dtype=torch.float32) for j,o in enumerate(obs_idx)]
    opt=torch.optim.Adam([Xl,log_ls,log_sf,log_sn],lr=lr)
    t0=time.time()
    for s in range(steps):
        opt.zero_grad(); nll=0.0
        for j,o in enumerate(obs_idx):
            if len(o)<Q+5: continue
            Xj=Xl[o]/torch.exp(log_ls[j])
            d2=torch.cdist(Xj,Xj)**2
            K=torch.exp(2*log_sf[j])*torch.exp(-0.5*d2)+ \
              (torch.exp(2*log_sn[j])+1e-4)*torch.eye(len(o))
            L=torch.linalg.cholesky(K)
            a=torch.cholesky_solve(yt[j].unsqueeze(1),L)
            nll=nll+0.5*(yt[j]@a.squeeze()) + torch.log(torch.diagonal(L)).sum()
        nll=nll+0.5*(Xl**2).sum()                        # N(0,I) prior on the latent
        nll.backward(); opt.step()
        if s%50==0: print(f"   {tag} step {s:3d}  nll {nll.item():10.1f}  [{time.time()-t0:.0f}s]",flush=True)
    return Xl.detach().numpy(), log_ls.detach().numpy(), log_sf.detach().numpy(), log_sn.detach().numpy()

# init from linear FA so the comparison is like-for-like and optimisation starts sane
fa=FactorAnalysisMissing(Q,seed=0).fit(Y); W=varimax(fa.W_)
Ez,_,_=fa_posterior(W,fa.psi_,fa.mu_,Y)
Xg,lls,lsf,lsn=fit_gplvm(Y,obs,Ez.astype(np.float32),steps=250,tag="main")
np.savez("/tmp/gplvm_fit.npz",Xg=Xg,log_ls=lls,log_sf=lsf,log_sn=lsn,Y=Y,
         cols5=np.array(cols5),bl5=bl5,Ez=Ez,W=W,lag=lag,fids=fids)
# ARD relevance = 1/lengthscale, normalised per output -> the nonlinear analogue of a loading row
rel=np.exp(-lls); rel=rel/rel.sum(1,keepdims=True)
li=cols5.index("image_lag")
print("\nPER-OUTPUT ARD RELEVANCE (fraction of each output's dependence on each latent dim)")
print("   output                       "+"  ".join(f"q{q+1}" for q in range(Q)))
for j in list(range(0,D,4))+[li]:
    print(f"   {cols5[j]:26s} "+"  ".join(f"{rel[j,q]:.2f}" for q in range(Q)))
print(f"\n   image_lag max relevance {rel[li].max():.3f}  (uniform would be {1/Q:.3f})")
print(f"   tabular mean max relevance {np.mean([rel[j].max() for j in range(D-1)]):.3f}")
json.dump(dict(lag_rel=[float(x) for x in rel[li]],lag_max=float(rel[li].max()),
  tabular_mean_max=float(np.mean([rel[j].max() for j in range(D-1)])),
  uniform=1.0/Q),open("/tmp/gplvm_ard.json","w"),indent=1)
