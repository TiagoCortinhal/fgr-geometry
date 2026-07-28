#!/usr/bin/env python3
"""SPATIAL feature VQ-VAE on FROZEN FetalCLIP patch features.

Per image: patch tokens form a 16x16 x C feature map (C = 1024 last-layer, or 24*1024
all-layers, per-layer z-scored). A small CONV encoder over the grid -> VQ codebook (EMA,
dead-code reinit) -> conv decoder reconstructs the feature map. Codes therefore capture
NEIGHBOURHOOD context (a spatial visual vocabulary), and each code is DECODABLE back to a
feature patch. Trained minibatched STRAIGHT FROM SHARDS on GPU (no k-means, no memmap).

Outputs: trained model + per-image code grids (out_usfmae/vqvae_codes_<tag>.npz),
recon error, codes_used, code x position/plane/GA groundings, figure.

USAGE: python hpc_feature_vqvae.py --enc FetalCLIP --all-layers --K 256 --epochs 15
       [--last-layer for 1024-d input] [--zdim 64] [--down 2]
"""
import os, glob, json, argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"

# ---------------- data: stream shards -> (B,C,16,16) feature maps ----------------
class ShardFeeder:
    """iterate feature-map batches. Builds a ONE-TIME uncompressed memmap of z-scored feature
    maps (B,C,g,g) on beegfs so epochs read fast. MEMORY-SAFE: selects the chosen layers while
    still fp16 and processes each shard in small image SUB-CHUNKS (transient ~1GB, not 13GB).
    s.sel = list of layer indices to keep (e.g. 12 evenly-spaced of 24)."""
    def __init__(s, enc, all_layers, layer, n_layers=12, subchunk=64):
        s.fs=sorted(glob.glob(os.path.join(OUT,f"fulltok_{enc}","shard_*.npz"))); assert s.fs
        z0=np.load(s.fs[0],allow_pickle=True); _,s.L,s.T,s.D=z0["tokens"].shape; del z0
        s.all=all_layers; s.layer=layer; s.grid=int(round((s.T-1)**0.5)); s.sub=subchunk
        if all_layers:
            s.sel=list(np.linspace(0,s.L-1,min(n_layers,s.L)).round().astype(int))   # evenly-spaced layers
        else:
            s.sel=[s.layer if s.layer>=0 else s.L-1]
        s.C=len(s.sel)*s.D
        s.mu=s.sd=None
        lay_tag="Lall" if (all_layers and len(s.sel)==s.L) else (f"L{len(s.sel)}of{s.L}" if all_layers else f"L{layer}")
        s.cache=os.path.join(OUT,f"fmcache_{enc}_{lay_tag}.dat"); s.cmeta=s.cache.replace(".dat",".meta.npz")
        if all_layers: s._stats()
    def build_memmap(s, rebuild=False):
        if (not rebuild) and os.path.exists(s.cache) and os.path.exists(s.cmeta):
            M=np.load(s.cmeta,allow_pickle=True); s.n=int(M["n"]); print(f"  reuse fm cache {s.cache} ({s.n} imgs)",flush=True); return
        ntot=sum(np.load(f,allow_pickle=True)["ga"].shape[0] for f in s.fs); g=s.grid
        print(f"  building fm memmap: {ntot} imgs x {s.C} x {g}x{g} fp16 = {ntot*s.C*g*g*2/1e9:.0f} GB",flush=True)
        mm=np.lib.format.open_memmap(s.cache,mode="w+",dtype=np.float16,shape=(ntot,s.C,g,g))
        GA=[];PL=[];NM=[];row=0;t0=time.time()
        for i,f in enumerate(s.fs):
            z=np.load(f,allow_pickle=True); tok=z["tokens"]; ni=tok.shape[0]
            for j in range(0,ni,s.sub):                                  # sub-chunk: transient ~1GB
                fm=s._map(tok[j:j+s.sub]); mm[row:row+len(fm)]=fm.astype(np.float16); row+=len(fm); del fm
            GA.append(z["ga"]);PL.append(z["plane"]);NM.append(z["names"]); del z,tok
            if i%15==0: print(f"    cache shard {i+1}/{len(s.fs)} {time.time()-t0:.0f}s",flush=True)
        mm.flush(); s.n=ntot
        np.savez(s.cmeta,ga=np.concatenate(GA),plane=np.concatenate(PL),names=np.concatenate(NM),n=ntot,C=s.C,g=g)
        print(f"  fm cache built {time.time()-t0:.0f}s",flush=True)
    def cached_batches(s,bs,shuffle=True):
        mm=np.load(s.cache,mmap_mode="r"); M=np.load(s.cmeta,allow_pickle=True)
        ga=M["ga"];pl=M["plane"];nm=M["names"]; idx=np.random.permutation(s.n) if shuffle else np.arange(s.n)
        for i in range(0,s.n,bs):
            j=np.sort(idx[i:i+bs]); yield np.asarray(mm[j]).astype(np.float32), ga[j], pl[j], nm[j]
    def _stats(s):
        # stats only for SELECTED layers, sub-chunked, kept fp16 until the small reduction
        Ls=len(s.sel); n=0; a=np.zeros((Ls,s.D)); b=np.zeros((Ls,s.D)); t0=time.time()
        for i,f in enumerate(s.fs):
            tok=np.load(f,allow_pickle=True)["tokens"]          # (n,L,T,dim) fp16 (mmap-ish)
            ni=tok.shape[0]
            for j in range(0,ni,s.sub):
                pt=tok[j:j+s.sub][:,s.sel,1:,:].astype(np.float32)   # (b,Ls,Np,D) selected only
                x=pt.transpose(0,2,1,3).reshape(-1,Ls,s.D)
                a+=x.sum(0); b+=(x**2).sum(0); n+=len(x); del pt,x
            del tok
            if i%15==0: print(f"    stats shard {i+1}/{len(s.fs)} {time.time()-t0:.0f}s",flush=True)
        s.mu=(a/n).astype(np.float32); s.sd=(np.sqrt(np.maximum(b/n-(a/n)**2,1e-12))+1e-6).astype(np.float32)  # (Ls,D)
    def _map(s,tok):
        # tok slice (b,L,T,dim) -> (b,C,g,g) using ONLY selected layers, z-scored if all_layers
        pt=tok[:,s.sel,1:,:].astype(np.float32)                # (b,Ls,Np,D) selected
        b,Ls,Np,D=pt.shape; x=pt.transpose(0,2,1,3)            # (b,Np,Ls,D)
        if s.all: x=(x-s.mu[None,None])/s.sd[None,None]        # per-layer z-score
        fm=x.reshape(b,Np,Ls*D); g=int(round(Np**0.5))
        return fm.reshape(b,g,g,Ls*D).transpose(0,3,1,2)       # (b,C,g,g)
    def batches(s,bs,shuffle=True):
        order=np.random.permutation(len(s.fs)) if shuffle else range(len(s.fs))
        for si in order:
            z=np.load(s.fs[si],allow_pickle=True); fm=s._map(z["tokens"])
            ga=z["ga"]; pl=z["plane"]; nm=z["names"]; del z
            idx=np.random.permutation(len(fm)) if shuffle else np.arange(len(fm))
            for i in range(0,len(fm),bs):
                j=idx[i:i+bs]; yield fm[j], ga[j], pl[j], nm[j]

# ---------------- VQ-VAE ----------------
class VQ(nn.Module):
    def __init__(s,K,dim,beta=0.25,decay=0.99,eps=1e-5,reinit=1.0):
        super().__init__(); s.K=K; s.dim=dim; s.beta=beta; s.decay=decay; s.eps=eps; s.reinit=reinit
        e=torch.randn(K,dim); s.register_buffer("cb",e); s.register_buffer("cs",torch.zeros(K)); s.register_buffer("ew",e.clone())
    def forward(s,z):                         # z: (B,dim,H,W)
        B,C,H,W=z.shape; zf=z.permute(0,2,3,1).reshape(-1,C)     # (BHW,dim)
        d=(zf.pow(2).sum(1,keepdim=True)-2*zf@s.cb.t()+s.cb.pow(2).sum(1)); idx=d.argmin(1)
        zq=s.cb[idx]
        if s.training:
            with torch.no_grad():
                oh=torch.zeros(len(zf),s.K,device=zf.device); oh.scatter_(1,idx[:,None],1)
                n=oh.sum(0); s.cs.mul_(s.decay).add_(n,alpha=1-s.decay)
                s.ew.mul_(s.decay).add_(oh.t()@zf,alpha=1-s.decay)
                N=s.cs.sum(); cs=(s.cs+s.eps)/(N+s.K*s.eps)*N; s.cb.copy_(s.ew/cs[:,None])
                dead=s.cs<s.reinit
                if dead.any():
                    pick=zf[torch.randint(0,len(zf),(int(dead.sum()),),device=zf.device)]
                    s.cb[dead]=pick; s.ew[dead]=pick; s.cs[dead]=1.0
        loss=s.beta*((zq.detach()-zf)**2).mean()
        zq=zf+(zq-zf).detach()
        return zq.view(B,H,W,C).permute(0,3,1,2), idx.view(B,H,W), loss

class VQVAE(nn.Module):
    def __init__(s,C,zdim=64,K=256,down=2):
        super().__init__()
        s.enc=nn.Sequential(nn.Conv2d(C,256,1),nn.GELU(),nn.Conv2d(256,zdim,3,padding=1))
        s.vq=VQ(K,zdim)
        s.dec=nn.Sequential(nn.Conv2d(zdim,256,3,padding=1),nn.GELU(),nn.Conv2d(256,C,1))
    def forward(s,x):
        z=s.enc(x); zq,idx,vql=s.vq(z); xr=s.dec(zq); return xr,idx,vql

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--enc",default="FetalCLIP"); ap.add_argument("--layer",type=int,default=-1)
    ap.add_argument("--all-layers",action="store_true"); ap.add_argument("--last-layer",action="store_true")
    ap.add_argument("--n-layers",type=int,default=12,help="how many evenly-spaced layers when --all-layers")
    ap.add_argument("--subchunk",type=int,default=64,help="images per sub-chunk during cache build (memory)")
    ap.add_argument("--K",type=int,default=256); ap.add_argument("--zdim",type=int,default=64)
    ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--bs",type=int,default=32); ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--rebuild",action="store_true",help="rebuild the fm memmap cache")
    a=ap.parse_args()
    all_layers=a.all_layers and not a.last_layer
    tag=f"{a.enc}_{'Lall' if all_layers else 'L'+str(a.layer)}_vqvae_K{a.K}"
    fd=ShardFeeder(a.enc,all_layers,a.layer,n_layers=a.n_layers,subchunk=a.subchunk)
    print(f"[{tag}] feature-map C={fd.C} grid {fd.grid}x{fd.grid} | K={a.K} zdim={a.zdim} dev={DEV}",flush=True)
    fd.build_memmap(rebuild=a.rebuild)     # one-time decompress+z-score -> fast memmap
    m=VQVAE(fd.C,a.zdim,a.K).to(DEV)
    opt=torch.optim.Adam([p for n,p in m.named_parameters() if not n.startswith("vq.")],a.lr)
    t0=time.time()
    for ep in range(a.epochs):
        m.train(); tot=0; rec=0; nb=0
        for fm,_,_,_ in fd.cached_batches(a.bs):
            x=torch.from_numpy(fm).to(DEV).float(); opt.zero_grad()
            xr,idx,vql=m(x); rl=F.mse_loss(xr,x); (rl+vql).backward(); opt.step()
            rec+=rl.item(); tot+=len(x); nb+=1
        print(f"  ep{ep} recon_mse={rec/nb:.4f} used={int((m.vq.cs>=1).sum())}/{a.K} {time.time()-t0:.0f}s",flush=True)
    # assign pass -> code grids + meta + recon error
    m.eval(); codes=[]; GA=[]; PL=[]; NM=[]; rerr=[]
    with torch.no_grad():
        for fm,ga,pl,nm in fd.cached_batches(a.bs,shuffle=False):
            x=torch.from_numpy(fm).to(DEV).float(); xr,idx,_=m(x)
            codes.append(idx.cpu().numpy().astype(np.int16)); rerr.append(((xr-x)**2).mean((1,2,3)).cpu().numpy())
            GA.append(ga); PL.append(pl); NM.append(nm)
    codes=np.concatenate(codes); codes=codes.reshape(len(codes),-1)
    ga=np.concatenate(GA).astype(np.float32); plane=np.concatenate(PL).astype(str); names=np.concatenate(NM)
    rerr=np.concatenate(rerr); ni,npatch=codes.shape; grid=int(round(npatch**0.5)); used=len(set(codes.ravel()))
    print(f"  codes used {used}/{a.K} | mean recon mse {rerr.mean():.4f} | {ni} imgs",flush=True)
    # groundings
    rows=np.arange(npatch)//grid; cols=np.arange(npatch)%grid; code_pos={}
    for c in range(a.K):
        mask=(codes==c)
        if mask.sum()<20: continue
        occ=np.array([(codes[:,p]==c).sum() for p in range(npatch)],float); occ=occ/occ.sum() if occ.sum() else occ
        ent=-(occ[occ>0]*np.log(occ[occ>0])).sum()/np.log(npatch)
        code_pos[int(c)]={"row":float(np.repeat(rows[None,:],ni,0)[mask].mean()/grid),
                          "col":float(np.repeat(cols[None,:],ni,0)[mask].mean()/grid),
                          "spatial_conc":float(1-ent),"freq":int(mask.sum())}
    freq=np.stack([(codes==c).mean(1) for c in range(a.K)],1)
    code_ga={int(c):float(spearmanr(freq[:,c],ga)[0]) for c in range(a.K) if freq[:,c].std()>1e-6}
    strong=sorted(code_ga.items(),key=lambda kv:-abs(kv[1]))[:8]
    print("  top GA-shifting codes:",[(c,round(r,2)) for c,r in strong],flush=True)
    json.dump({"tag":tag,"n_img":int(ni),"grid":int(grid),"K":a.K,"codes_used":used,"recon_mse":float(rerr.mean()),
               "code_pos":code_pos,"code_ga_spearman":code_ga},open(os.path.join(OUTP,f"{tag}.json"),"w"),indent=2)
    np.savez(os.path.join(OUT,f"vqvae_codes_{tag}.npz"),codes=codes,names=names,ga=ga,plane=plane,
             codebook=m.vq.cb.cpu().numpy().astype(np.float32),grid=grid)
    torch.save(m.state_dict(),os.path.join(OUT,f"{tag}.pt"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(13,5)); cs=list(code_pos.values())
        sc=ax[0].scatter([d["col"] for d in cs],[1-d["row"] for d in cs],s=[np.sqrt(d["freq"]) for d in cs],c=[d["spatial_conc"] for d in cs],cmap="viridis")
        ax[0].set_title(f"VQ-VAE codes by grid position\n{tag} {used}/{a.K} recon={rerr.mean():.3f}"); plt.colorbar(sc,ax=ax[0]); ax[0].set_xlabel("col"); ax[0].set_ylabel("row(top=1)")
        gb=np.linspace(ga.min(),ga.max(),12)
        for c,r in strong[:5]: ax[1].plot((gb[:-1]+gb[1:])/2,[freq[(ga>=gb[i])&(ga<gb[i+1]),c].mean() for i in range(len(gb)-1)],"-o",ms=3,label=f"code {c} ρ={r:.2f}")
        ax[1].set_title("GA-shifting codes"); ax[1].set_xlabel("GA wk"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUTP,f"{tag}.png"),dpi=145,bbox_inches="tight"); print(f"  figure -> out_probe/{tag}.png",flush=True)
    except Exception as ex: print("  fig skipped:",ex,flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
