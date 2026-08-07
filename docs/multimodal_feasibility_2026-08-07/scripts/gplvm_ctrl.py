"""Controls for the GPLVM ARD result. A concentrated relevance means nothing until we know
what a PURE NOISE output looks like under the same fit."""
import numpy as np, torch, json, time, warnings; warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(10)
d=np.load("/tmp/gplvm_fit.npz",allow_pickle=True)
Y,Xg,cols5,bl5,lag=d["Y"],d["Xg"],[str(x) for x in d["cols5"]],d["bl5"],d["lag"]
Q=Xg.shape[1]; N,D=Y.shape
li=cols5.index("image_lag")
# --- CONTROL: add 3 pure-noise outputs and 1 SHUFFLED lag, refit ARD on the FIXED latent.
# Holding the latent fixed isolates the question: given this latent, how concentrated is a
# null output's relevance? If noise also concentrates ~0.45, the lag result is an artefact.
rng=np.random.default_rng(0)
extra={"noise_full":rng.standard_normal(N),
       "noise_lagcov":np.where(np.isfinite(lag),rng.standard_normal(N),np.nan),
       "lag_shuffled":np.full(N,np.nan)}
m=np.isfinite(lag); sh=lag[m].copy(); rng.shuffle(sh); extra["lag_shuffled"][m]=sh
names=list(extra); Ye=np.column_stack([Y]+[extra[k] for k in names])
cols_e=cols5+names
Xt=torch.tensor(Xg,dtype=torch.float32)   # FIXED latent
def fit_ard(yv):
    o=np.where(np.isfinite(yv))[0]
    yt=torch.tensor((yv[o]-np.nanmean(yv))/np.nanstd(yv),dtype=torch.float32)
    lls=torch.zeros(Q,requires_grad=True); lsf=torch.zeros(1,requires_grad=True)
    lsn=torch.full((1,),-1.0,requires_grad=True)
    opt=torch.optim.Adam([lls,lsf,lsn],lr=0.05)
    Xo=Xt[o]
    for s in range(250):
        opt.zero_grad()
        Xj=Xo/torch.exp(lls); d2=torch.cdist(Xj,Xj)**2
        K=torch.exp(2*lsf)*torch.exp(-0.5*d2)+(torch.exp(2*lsn)+1e-4)*torch.eye(len(o))
        L=torch.linalg.cholesky(K); a=torch.cholesky_solve(yt.unsqueeze(1),L)
        nll=0.5*(yt@a.squeeze())+torch.log(torch.diagonal(L)).sum()
        nll.backward(); opt.step()
    r=np.exp(-lls.detach().numpy()); return r/r.sum(), float(nll.item()), len(o)
print("ARD CONCENTRATION ON THE FIXED LATENT — real lag vs nulls\n")
print(f"{'output':>16} {'n':>5} {'max relevance':>14} {'nll':>10}")
out={}
for lab,yv in [("image_lag",Y[:,li])]+[(k,extra[k]) for k in names]:
    rel,nll,n=fit_ard(yv)
    print(f"{lab:>16} {n:>5} {rel.max():>14.3f} {nll:>10.1f}")
    out[lab]=dict(max_rel=float(rel.max()),rel=[float(x) for x in rel],nll=nll,n=n)
print(f"\n  uniform baseline = {1/Q:.3f}")
print("  => if the noise outputs also reach ~0.45, concentration is an ARTEFACT of ARD on a fixed latent.")
json.dump(out,open("/tmp/gplvm_ctrl.json","w"),indent=1)
