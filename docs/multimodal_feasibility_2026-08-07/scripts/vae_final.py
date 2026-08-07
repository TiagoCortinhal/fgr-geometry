"""Comparison at the SELECTED configurations, plus the nonlinear gate AT THOSE SETTINGS
(the gate my earlier run never executed at the settings it reported)."""
import numpy as np, torch, torch.nn as nn, json, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from scipy.stats import norm
from fgm.wp2_canonical import assemble_canonical, FactorAnalysisMissing, varimax
from fgm.loadings_heatmap_wide import SENTINEL_Z
from fgm.latent_risk_uncertainty import fa_posterior
torch.set_num_threads(10)
SEL=json.load(open("/tmp/vae_sel.json")); VK,VB,VH,VE=SEL["vae"]["K"],SEL["vae"]["beta"],SEL["vae"]["h"],SEL["vae"]["ep"]
FK=SEL["fa"]["K"]
print(f"selected: VAE K={VK} beta={VB} h={VH} ep={VE} (ll={SEL['vae_ll']:+.4f}) | FA K={FK} (ll={SEL['fa_ll']:+.4f})\n")
X,cols,blocks,fids,*_=assemble_canonical(); cols=list(cols); bl=list(blocks)
Xc=X.copy()
for i,c in enumerate(cols):
    if c.endswith("_z_ig21"): Xc[np.abs(Xc[:,i])>SENTINEL_Z,i]=np.nan
z=np.load("/Users/tiago/dev/fgr-geometry/results/img_align/_lag_seq_debiased.npz",allow_pickle=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pooled=np.nanmean(np.where(z["lag_mask"],z["lag_seq"],np.nan),axis=1)
lag=np.array([dict(zip([int(x) for x in z["fids"]],pooled)).get(int(f),np.nan) for f in fids])
X26=np.column_stack([Xc,lag]); bl26=bl+["image"]; cols26=cols+["image_lag"]
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
# --- GATE: nonlinear positive control AT THE SELECTED SETTINGS ---
print("GATE — nonlinear positive control at the SELECTED VAE settings:")
Zf=np.nan_to_num(Y)
sig=Zf[:,cols26.index("efw_z_ig21")]*Zf[:,cols26.index("Percentil_CPR")]+Zf[:,cols26.index("hc_z_ig21")]**2
sig=(sig-sig.mean())/sig.std()
for t in (0.10,0.30):
    Yp=np.column_stack([Y,np.sqrt(t)*sig+np.sqrt(1-t)*np.random.default_rng(0).standard_normal(N)])
    tr,te=next(KFold(5,shuffle=True,random_state=0).split(Yp))
    net=train_vae(Yp[tr],VK,VB,VH,VE)
    keep=np.isfinite(Yp).astype(np.float32); keep[:,-1]=0
    yv=torch.tensor(np.nan_to_num(Yp[te]),dtype=torch.float32)
    with torch.no_grad(): om,_,_,_=net(yv,torch.tensor(keep[te]),sample=False)
    yy=Yp[te,-1]; pr=om[:,-1].numpy()
    print(f"   planted nonlinear R2={t:.2f} -> recovers {1-((yy-pr)**2).sum()/((yy-yy.mean())**2).sum():+.4f}")
print()
res={}
for arm in ("VAE","FA (marginalisation)"):
    h95=[];h90=[]
    for tr,te in KFold(5,shuffle=True,random_state=0).split(Y):
        Ym=Y.copy(); Ym[np.ix_(te,dop)]=np.nan
        if arm=="VAE":
            net=train_vae(Ym,VK,VB,VH,VE)
            yv=torch.tensor(np.nan_to_num(Ym),dtype=torch.float32)
            kv=torch.tensor(np.isfinite(Ym).astype(np.float32))
            with torch.no_grad(): om,ov,_,_=net(yv,kv,sample=False)
            pm=om.numpy(); pv=np.exp(ov.numpy())
            for t in te:
                for j in dop:
                    if not np.isfinite(Y[t,j]): continue
                    s=np.sqrt(max(pv[t,j],1e-9))
                    h95.append(abs(Y[t,j]-pm[t,j])<=norm.ppf(0.975)*s); h90.append(abs(Y[t,j]-pm[t,j])<=norm.ppf(0.95)*s)
        else:
            fa=FactorAnalysisMissing(FK,seed=0).fit(Ym); W=varimax(fa.W_)
            Ez,G,_=fa_posterior(W,fa.psi_,fa.mu_,Ym)
            for t in te:
                for j in dop:
                    if not np.isfinite(Y[t,j]): continue
                    m_=W[j]@Ez[t]+fa.mu_[j]; s=np.sqrt(max(W[j]@G[t]@W[j]+fa.psi_[j],1e-9))
                    h95.append(abs(Y[t,j]-m_)<=norm.ppf(0.975)*s); h90.append(abs(Y[t,j]-m_)<=norm.ppf(0.95)*s)
    print(f"  {arm:22s} n={len(h95):5d}  95%: {np.mean(h95):.3f}  90%: {np.mean(h90):.3f}   |dev| {abs(np.mean(h95)-.95)+abs(np.mean(h90)-.90):.4f}")
    res[arm]=dict(n=len(h95),c95=float(np.mean(h95)),c90=float(np.mean(h90)))
json.dump(dict(sel=SEL,cov=res),open("/tmp/vae_final.json","w"),indent=1)
