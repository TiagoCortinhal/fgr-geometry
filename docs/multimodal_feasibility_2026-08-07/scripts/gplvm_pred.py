"""Does the GPLVM's concentrated ARD translate into OUT-OF-FOLD predictive power for the lag?
The ARD fit above used ALL rows to place the latent, so it is in-sample by construction.
Honest test: hold out rows, place them using only their TABULAR values, predict the lag."""
import numpy as np, torch, json, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(10)
d=np.load("/tmp/gplvm_fit.npz",allow_pickle=True)
Y,Xg,cols5=d["Y"],d["Xg"],[str(x) for x in d["cols5"]]
li=cols5.index("image_lag"); lag=Y[:,li]
m=np.isfinite(lag)
print("STEP 1 — is the ARD concentration IN-SAMPLE only?")
print("  The latent Xg was fitted WITH the lag as one of the 26 outputs, so the lag helped")
print("  place every fetus. Predicting the lag from that latent is circular.\n")
def oof_ridge(y,A):
    mm=np.isfinite(y); yy=y[mm]; AA=A[mm]; p=np.zeros_like(yy)
    for tr,te in KFold(5,shuffle=True,random_state=0).split(AA):
        p[te]=RidgeCV(alphas=np.logspace(-2,3,20)).fit(AA[tr],yy[tr]).predict(AA[te])
    return 1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum()
print(f"  lag from the GPLVM latent (CIRCULAR, lag was an output): R2 = {oof_ridge(lag,Xg):+.4f}")
# STEP 2: refit the GPLVM WITHOUT the lag among the outputs, then test
print("\nSTEP 2 — refit the latent using the 25 TABULAR outputs ONLY, then predict the lag.")
Q=Xg.shape[1]; N=Y.shape[0]
Yt=np.delete(Y,li,axis=1); Dt=Yt.shape[1]
obs=[np.where(np.isfinite(Yt[:,j]))[0] for j in range(Dt)]
Xl=torch.tensor(Xg,dtype=torch.float32,requires_grad=True)
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
Xn=Xl.detach().numpy()
r_clean=oof_ridge(lag,Xn)
print(f"  lag from the TABULAR-ONLY GPLVM latent:  R2 = {r_clean:+.4f}")
rng=np.random.default_rng(0)
nulls=[]
for b in range(20):
    sh=lag.copy(); sh[m]=rng.permutation(lag[m]); nulls.append(oof_ridge(sh,Xn))
print(f"  shuffle null p95: {np.percentile(nulls,95):+.4f}")
# reference points
from fgm.wp2_canonical import assemble_canonical
print(f"\n  REFERENCE  linear FA latent -> lag:        +0.0076  (measured earlier)")
print(f"             25 tabular -> lag, ridge:       +0.0323")
json.dump(dict(circular=float(oof_ridge(lag,Xg)),clean=float(r_clean),
  null95=float(np.percentile(nulls,95))),open("/tmp/gplvm_pred.json","w"),indent=1)
