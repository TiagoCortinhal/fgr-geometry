"""Both arms get hyperparameter selection on HELD-OUT MASKED-CELL LIKELIHOOD (never on coverage),
then a single comparison at the selected point. Seeds reset per run."""
import numpy as np, torch, torch.nn as nn, json, time, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from scipy.stats import norm
from fgm.wp2_canonical import assemble_canonical, FactorAnalysisMissing, varimax
from fgm.loadings_heatmap_wide import SENTINEL_Z
from fgm.latent_risk_uncertainty import fa_posterior
torch.set_num_threads(10)
X,cols,blocks,fids,*_=assemble_canonical(); cols=list(cols); bl=list(blocks)
Xc=X.copy()
for i,c in enumerate(cols):
    if c.endswith("_z_ig21"): Xc[np.abs(Xc[:,i])>SENTINEL_Z,i]=np.nan
z=np.load("/Users/tiago/dev/fgr-geometry/results/img_align/_lag_seq_debiased.npz",allow_pickle=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pooled=np.nanmean(np.where(z["lag_mask"],z["lag_seq"],np.nan),axis=1)
lag=np.array([dict(zip([int(x) for x in z["fids"]],pooled)).get(int(f),np.nan) for f in fids])
X26=np.column_stack([Xc,lag]); cols26=cols+["image_lag"]; bl26=bl+["image"]
mu_=np.nanmean(X26,0); sd_=np.nanstd(X26,0); sd_[sd_==0]=1; Y=(X26-mu_)/sd_
N,D=Y.shape; dop=[i for i,b in enumerate(bl26) if b=="Doppler"]
class VAE(nn.Module):
    def __init__(s,D,K,h):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(2*D,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        s.mu=nn.Linear(h,K); s.lv=nn.Linear(h,K)
        s.dec=nn.Sequential(nn.Linear(K,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        s.om=nn.Linear(h,D); s.ov=nn.Linear(h,D)
    def forward(s,y,m,sample=True):
        hh=s.enc(torch.cat([y*m,m],1)); mu=s.mu(hh); lv=s.lv(hh).clamp(-6,3)
        zz=mu+torch.randn_like(mu)*torch.exp(0.5*lv) if sample else mu
        g=s.dec(zz); return s.om(g), s.ov(g).clamp(-6,3), mu, lv
def train_vae(Ytr,K,beta,h,ep,seed=0):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    yt=torch.tensor(np.nan_to_num(Ytr),dtype=torch.float32)
    ot=torch.tensor(np.isfinite(Ytr).astype(np.float32))
    net=VAE(Ytr.shape[1],K,h); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-4)
    for e in range(ep):
        p=rng.uniform(0.10,0.40)
        drop=(torch.rand_like(ot)<p).float()*ot; keep=ot*(1-drop)
        om,ov,m_,l_=net(yt,keep)
        nll=(0.5*(ov+(yt-om)**2/torch.exp(ov))*drop).sum()/drop.sum().clamp(min=1)
        kl=(-0.5*(1+l_-m_**2-l_.exp()).sum(1)).mean()/Ytr.shape[1]
        (nll+beta*kl).backward(); opt.step(); opt.zero_grad()
    return net
def held_out_ll(Ytr,Yva,kind,**kw):
    """SELECTION CRITERION: gaussian log-lik of randomly hidden OBSERVED cells. Not coverage."""
    rng=np.random.default_rng(99)
    hid=(rng.random(Yva.shape)<0.25)&np.isfinite(Yva)
    Yin=Yva.copy(); Yin[hid]=np.nan
    if kind=="vae":
        net=train_vae(Ytr,kw["K"],kw["beta"],kw["h"],kw["ep"])
        yv=torch.tensor(np.nan_to_num(Yin),dtype=torch.float32)
        kv=torch.tensor(np.isfinite(Yin).astype(np.float32))
        with torch.no_grad(): om,ov,_,_=net(yv,kv,sample=False)
        pm,pv=om.numpy(),np.exp(ov.numpy())
    else:
        fa=FactorAnalysisMissing(kw["K"],seed=0).fit(Ytr); W=varimax(fa.W_)
        Ez,G,_=fa_posterior(W,fa.psi_,fa.mu_,Yin)
        pm=Ez@W.T+fa.mu_
        pv=np.stack([np.array([W[j]@G[n]@W[j]+fa.psi_[j] for j in range(Yva.shape[1])]) for n in range(len(Yva))])
    ll=[]
    for n_,j_ in zip(*np.where(hid)):
        s=np.sqrt(max(pv[n_,j_],1e-9)); ll.append(norm.logpdf(Yva[n_,j_],pm[n_,j_],s))
    return float(np.mean(ll))
tr,va=next(KFold(4,shuffle=True,random_state=7).split(Y))
print("SELECTION on held-out masked-cell log-likelihood (NOT coverage)\n")
print("  VAE grid:")
best=(-1e9,None); t0=time.time()
for K in (4,8,16):
    for beta in (0.01,0.1,0.3):
        for h in (32,64):
            ll=held_out_ll(Y[tr],Y[va],"vae",K=K,beta=beta,h=h,ep=2000)
            if ll>best[0]: best=(ll,dict(K=K,beta=beta,h=h,ep=2000))
            print(f"    K={K:<3} beta={beta:<5} h={h:<3}  ll={ll:+.4f}",flush=True)
print(f"  -> selected VAE {best[1]}  ll={best[0]:+.4f}  [{time.time()-t0:.0f}s]")
print("\n  FA grid:")
bfa=(-1e9,None)
for K in (4,6,8,10,12):
    ll=held_out_ll(Y[tr],Y[va],"fa",K=K)
    if ll>bfa[0]: bfa=(ll,dict(K=K))
    print(f"    K={K:<3}  ll={ll:+.4f}")
print(f"  -> selected FA {bfa[1]}  ll={bfa[0]:+.4f}")
json.dump(dict(vae=best[1],vae_ll=best[0],fa=bfa[1],fa_ll=bfa[0]),open("/tmp/vae_sel.json","w"),indent=1)
