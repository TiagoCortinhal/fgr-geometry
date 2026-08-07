"""CLEAN REFIT: the latent must never have seen the lag, not even through initialisation.
Three initialisations, and per-fold refitting so held-out fetuses are placed without their lag."""
import numpy as np, torch, json, time, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from fgm.wp2_canonical import assemble_canonical, FactorAnalysisMissing, varimax
from fgm.loadings_heatmap_wide import SENTINEL_Z
from fgm.latent_risk_uncertainty import fa_posterior
torch.set_num_threads(10)
d=np.load("/tmp/gplvm_fit.npz",allow_pickle=True)
Y,cols5,lag=d["Y"],[str(x) for x in d["cols5"]],d["lag"]
li=cols5.index("image_lag"); Yt=np.delete(Y,li,axis=1)   # 25 TABULAR only, forever
N,Dt=Yt.shape; Q=8
obs=[np.where(np.isfinite(Yt[:,j]))[0] for j in range(Dt)]
def fit(Xinit,steps=250,seed=0):
    torch.manual_seed(seed)
    Xl=torch.tensor(Xinit,dtype=torch.float32,requires_grad=True)
    lls=torch.zeros(Dt,Q,requires_grad=True); lsf=torch.zeros(Dt,requires_grad=True)
    lsn=torch.full((Dt,),-1.0,requires_grad=True)
    yt=[torch.tensor(Yt[o,j],dtype=torch.float32) for j,o in enumerate(obs)]
    opt=torch.optim.Adam([Xl,lls,lsf,lsn],lr=0.05)
    for s in range(steps):
        opt.zero_grad(); nll=0.0
        for j,o in enumerate(obs):
            Xj=Xl[o]/torch.exp(lls[j]); d2=torch.cdist(Xj,Xj)**2
            K=torch.exp(2*lsf[j])*torch.exp(-0.5*d2)+(torch.exp(2*lsn[j])+1e-4)*torch.eye(len(o))
            L=torch.linalg.cholesky(K); a=torch.cholesky_solve(yt[j].unsqueeze(1),L)
            nll=nll+0.5*(yt[j]@a.squeeze())+torch.log(torch.diagonal(L)).sum()
        nll=nll+0.5*(Xl**2).sum(); nll.backward(); opt.step()
    return Xl.detach().numpy()
def oof(y,A,seed=0):
    m=np.isfinite(y); yy=y[m]; AA=A[m]; p=np.zeros_like(yy)
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(AA):
        p[te]=RidgeCV(alphas=np.logspace(-2,3,20)).fit(AA[tr],yy[tr]).predict(AA[te])
    return 1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum()
print("CLEAN REFIT — latent never sees the lag, three independent initialisations\n")
Yf=np.where(np.isfinite(Yt),Yt,0.0)
fa=FactorAnalysisMissing(Q,seed=0).fit(Yt); Wf=varimax(fa.W_)
Ez_t,_,_=fa_posterior(Wf,fa.psi_,fa.mu_,Yt)
inits={"FA on 25 tabular":Ez_t.astype(np.float32),
       "PCA on mean-filled":PCA(Q).fit_transform(Yf).astype(np.float32),
       "random N(0,1)":np.random.default_rng(3).standard_normal((N,Q)).astype(np.float32)}
rng=np.random.default_rng(0); m=np.isfinite(lag)
res={}
for lab,X0 in inits.items():
    t0=time.time(); Xn=fit(X0)
    r=oof(lag,Xn)
    nulls=[]
    for b in range(20):
        sh=lag.copy(); sh[m]=rng.permutation(lag[m]); nulls.append(oof(sh,Xn))
    print(f"  {lab:22s} lag R2 = {r:+.4f}   shuffle p95 {np.percentile(nulls,95):+.4f}   [{time.time()-t0:.0f}s]")
    res[lab]=dict(r2=float(r),null95=float(np.percentile(nulls,95)))
print(f"\n  reference: linear FA latent (25 tabular) -> lag = {oof(lag,Ez_t):+.4f}")
print(f"             ridge on raw 25 tabular        -> lag = {oof(lag,Yf):+.4f}")
res["linear_FA_latent"]=float(oof(lag,Ez_t)); res["ridge_raw"]=float(oof(lag,Yf))
json.dump(res,open("/tmp/clean_refit.json","w"),indent=1)
