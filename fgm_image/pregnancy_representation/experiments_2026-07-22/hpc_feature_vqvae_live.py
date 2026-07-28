#!/usr/bin/env python3
"""SPATIAL feature VQ-VAE with FROZEN FetalCLIP run LIVE (no shard store, no memmap cache).

Loads raw PNGs, runs frozen FetalCLIP ViT-L/14 on the GPU each batch (hooks grab N selected
layers' patch tokens -> 16x16xC feature map), feeds a conv VQ-VAE (EMA codebook + dead-code
reinit) that reconstructs the feature map. Backprop ONLY through the VQ-VAE; FetalCLIP frozen.

No 128GB memmap, no OOM, no 2h build. Cost: frozen forward recomputed each epoch (~12min/epoch
at ~28 fr/s), ~3h for 15 epochs. Robust and simple.

Images: IMPACT preprocessed (ga_cnn_index.csv, 20413 frames). Weights: FetalCLIP_weights.pt.
USAGE: python hpc_feature_vqvae_live.py --n-layers 12 --K 256 --epochs 15
Outputs: out_probe/<tag>.json/.png + out_usfmae/vqvae_codes_<tag>.npz + <tag>.pt
"""
import os, glob, json, argparse, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae")); os.makedirs(OUT,exist_ok=True)
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
IMG_DIR="/mnt/beegfs/groups/collage/data/IMPACT_FULL/processed/IMPACT_FULL/preprocessed"
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")
WEIGHTS=os.path.join(ROOT,"FetalCLIP_weights.pt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
CLIP_MEAN=((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))
tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize(*CLIP_MEAN)])

def build_fetalclip():
    import open_clip
    model=open_clip.create_model("ViT-L-14",pretrained=None)
    ck=torch.load(WEIGHTS,map_location="cpu",weights_only=False)
    sd=ck.get("state_dict",ck.get("model",ck)); sd={k.replace("module.",""):v for k,v in sd.items()}
    vis={k[len("visual."):]:v for k,v in sd.items() if k.startswith("visual.")}
    model.visual.load_state_dict(vis,strict=False)
    vt=model.visual.to(DEV).eval()
    for p in vt.parameters(): p.requires_grad_(False)
    return vt

@torch.no_grad()
def clip_layer_maps(vt, x, sel):
    """frozen FetalCLIP forward; return (B, len(sel), Npatch, D) selected-layer patch tokens."""
    v=vt; h=v.conv1(x); h=h.reshape(h.shape[0],h.shape[1],-1).permute(0,2,1)
    cls=v.class_embedding.to(h.dtype)+torch.zeros(h.shape[0],1,h.shape[-1],device=h.device,dtype=h.dtype)
    h=torch.cat([cls,h],1)+v.positional_embedding.to(h.dtype); h=v.ln_pre(h); h=h.permute(1,0,2)
    out=[]
    for i,blk in enumerate(v.transformer.resblocks,1):
        h=blk(h)
        if i in sel: out.append(h.permute(1,0,2)[:,1:,:])     # drop CLS -> (B,Npatch,D)
    return torch.stack(out,1)                                  # (B,len(sel),Npatch,D)

class VQ(nn.Module):
    def __init__(s,K,dim,beta=0.25,decay=0.99,eps=1e-5,reinit=1.0):
        super().__init__(); s.K=K; s.beta=beta; s.decay=decay; s.eps=eps; s.reinit=reinit
        e=torch.randn(K,dim); s.register_buffer("cb",e); s.register_buffer("cs",torch.zeros(K)); s.register_buffer("ew",e.clone())
    def forward(s,z):
        B,C,H,W=z.shape; zf=z.permute(0,2,3,1).reshape(-1,C)
        d=(zf.pow(2).sum(1,keepdim=True)-2*zf@s.cb.t()+s.cb.pow(2).sum(1)); idx=d.argmin(1); zq=s.cb[idx]
        if s.training:
            with torch.no_grad():
                oh=torch.zeros(len(zf),s.K,device=zf.device); oh.scatter_(1,idx[:,None],1)
                s.cs.mul_(s.decay).add_(oh.sum(0),alpha=1-s.decay); s.ew.mul_(s.decay).add_(oh.t()@zf,alpha=1-s.decay)
                N=s.cs.sum(); cs=(s.cs+s.eps)/(N+s.K*s.eps)*N; s.cb.copy_(s.ew/cs[:,None])
                dead=s.cs<s.reinit
                if dead.any():
                    pk=zf[torch.randint(0,len(zf),(int(dead.sum()),),device=zf.device)]; s.cb[dead]=pk; s.ew[dead]=pk; s.cs[dead]=1.0
        loss=s.beta*((zq.detach()-zf)**2).mean(); zq=zf+(zq-zf).detach()
        return zq.view(B,H,W,C).permute(0,3,1,2), idx.view(B,H,W), loss

class VQVAE(nn.Module):
    def __init__(s,C,zdim=64,K=256):
        super().__init__()
        s.enc=nn.Sequential(nn.Conv2d(C,256,1),nn.GELU(),nn.Conv2d(256,zdim,3,padding=1))
        s.vq=VQ(K,zdim); s.dec=nn.Sequential(nn.Conv2d(zdim,256,3,padding=1),nn.GELU(),nn.Conv2d(256,C,1))
    def forward(s,x):
        z=s.enc(x); zq,idx,vql=s.vq(z); return s.dec(zq),idx,vql

def frame_table():
    df=pd.read_csv(INDEX).copy()
    df["img"]=df["new_filename"].astype(str).apply(lambda n: os.path.join(IMG_DIR,n if n.endswith(".png") else n+".png"))
    return df[df["img"].apply(os.path.exists)].reset_index(drop=True)

def load_batch(paths):
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-layers",type=int,default=12); ap.add_argument("--K",type=int,default=256)
    ap.add_argument("--zdim",type=int,default=64); ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--bs",type=int,default=64); ap.add_argument("--lr",type=float,default=2e-3)
    a=ap.parse_args()
    df=frame_table(); vt=build_fetalclip()
    # infer dims + selected layers from one batch
    nblk=len(vt.transformer.resblocks); sel=sorted(set(np.linspace(1,nblk,min(a.n_layers,nblk)).round().astype(int)))
    x0=load_batch(df["img"].iloc[:2]).to(DEV); m0=clip_layer_maps(vt,x0,set(sel))
    Ls,Np,D=m0.shape[1],m0.shape[2],m0.shape[3]; g=int(round(Np**0.5)); C=Ls*D
    tag=f"FetalCLIP_live_L{Ls}of{nblk}_vqvae_K{a.K}"
    print(f"[{tag}] {len(df)} imgs | layers {sel} | map C={C} grid {g}x{g} | dev {DEV}",flush=True)
    # per-layer z-score stats: one quick frozen pass (mean/std over patches)
    print("  computing per-layer z-score stats (1 frozen pass) ...",flush=True); t0=time.time()
    ssum=torch.zeros(Ls,D,device=DEV); ssq=torch.zeros(Ls,D,device=DEV); ntok=0
    for i in range(0,len(df),a.bs):
        x=load_batch(df["img"].iloc[i:i+a.bs]).to(DEV); mp=clip_layer_maps(vt,x,set(sel))  # (b,Ls,Np,D)
        f=mp.reshape(-1,Ls,D); ssum+=f.sum(0); ssq+=(f**2).sum(0); ntok+=f.shape[0]
        if i% (a.bs*40)==0: print(f"    stats {i}/{len(df)} {time.time()-t0:.0f}s",flush=True)
    mu=(ssum/ntok); sd=(ssq/ntok-mu**2).clamp(min=1e-12).sqrt()+1e-6           # (Ls,D)
    def to_map(mp):  # (b,Ls,Np,D)->(b,C,g,g) z-scored
        b=mp.shape[0]; f=((mp-mu[None,:,None])/sd[None,:,None]).permute(0,2,1,3).reshape(b,Np,C)
        return f.reshape(b,g,g,C).permute(0,3,1,2)
    net=VQVAE(C,a.zdim,a.K).to(DEV)
    opt=torch.optim.Adam([p for n,p in net.named_parameters() if not n.startswith("vq.")],a.lr)
    for ep in range(a.epochs):
        net.train(); order=np.random.permutation(len(df)); rec=0; nb=0
        for i in range(0,len(df),a.bs):
            idx=order[i:i+a.bs]; x=load_batch(df["img"].iloc[idx]).to(DEV)
            fmap=to_map(clip_layer_maps(vt,x,set(sel)))
            xr,_,vql=net(fmap); rl=F.mse_loss(xr,fmap); opt.zero_grad(); (rl+vql).backward(); opt.step()
            rec+=rl.item(); nb+=1
        print(f"  ep{ep} recon_mse={rec/nb:.4f} used={int((net.vq.cs>=1).sum())}/{a.K} {time.time()-t0:.0f}s",flush=True)
    # assign pass
    net.eval(); codes=[]; GA=[]; PL=[]; NM=[]; rerr=[]
    with torch.no_grad():
        for i in range(0,len(df),a.bs):
            sl=df.iloc[i:i+a.bs]; x=load_batch(sl["img"]).to(DEV); fmap=to_map(clip_layer_maps(vt,x,set(sel)))
            xr,ids,_=net(fmap); codes.append(ids.reshape(len(sl),-1).cpu().numpy().astype(np.int16))
            rerr.append(((xr-fmap)**2).mean((1,2,3)).cpu().numpy())
            GA.append(sl["ga_weeks_recovered"].values); PL.append(sl["plane_prop"].values); NM.append(sl["new_filename"].values)
    codes=np.concatenate(codes); ga=np.concatenate(GA).astype(np.float32); plane=np.concatenate(PL).astype(str); names=np.concatenate(NM)
    rerr=np.concatenate(rerr); ni,npatch=codes.shape; used=len(set(codes.ravel()))
    print(f"  codes used {used}/{a.K} | recon mse {rerr.mean():.4f} | {ni} imgs",flush=True)
    rows=np.arange(npatch)//g; cols=np.arange(npatch)%g; code_pos={}
    for c in range(a.K):
        mask=(codes==c)
        if mask.sum()<20: continue
        occ=np.array([(codes[:,p]==c).sum() for p in range(npatch)],float); occ=occ/occ.sum() if occ.sum() else occ
        ent=-(occ[occ>0]*np.log(occ[occ>0])).sum()/np.log(npatch)
        code_pos[int(c)]={"row":float(np.repeat(rows[None,:],ni,0)[mask].mean()/g),"col":float(np.repeat(cols[None,:],ni,0)[mask].mean()/g),"spatial_conc":float(1-ent),"freq":int(mask.sum())}
    freq=np.stack([(codes==c).mean(1) for c in range(a.K)],1)
    code_ga={int(c):float(spearmanr(freq[:,c],ga)[0]) for c in range(a.K) if freq[:,c].std()>1e-6}
    strong=sorted(code_ga.items(),key=lambda kv:-abs(kv[1]))[:8]
    print("  top GA-shifting codes:",[(c,round(r,2)) for c,r in strong],flush=True)
    json.dump({"tag":tag,"n_img":int(ni),"grid":int(g),"K":a.K,"codes_used":used,"recon_mse":float(rerr.mean()),
               "layers":sel,"code_pos":code_pos,"code_ga_spearman":code_ga},open(os.path.join(OUTP,f"{tag}.json"),"w"),indent=2)
    np.savez(os.path.join(OUT,f"vqvae_codes_{tag}.npz"),codes=codes,names=names,ga=ga,plane=plane,codebook=net.vq.cb.cpu().numpy(),grid=g)
    torch.save(net.state_dict(),os.path.join(OUT,f"{tag}.pt"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(13,5)); cs=list(code_pos.values())
        sc=ax[0].scatter([d["col"] for d in cs],[1-d["row"] for d in cs],s=[np.sqrt(d["freq"]) for d in cs],c=[d["spatial_conc"] for d in cs],cmap="viridis")
        ax[0].set_title(f"{tag} {used}/{a.K} recon={rerr.mean():.3f}"); plt.colorbar(sc,ax=ax[0]); ax[0].set_xlabel("col"); ax[0].set_ylabel("row(top=1)")
        gb=np.linspace(ga.min(),ga.max(),12)
        for c,r in strong[:5]: ax[1].plot((gb[:-1]+gb[1:])/2,[freq[(ga>=gb[i])&(ga<gb[i+1]),c].mean() for i in range(len(gb)-1)],"-o",ms=3,label=f"code {c} ρ={r:.2f}")
        ax[1].set_title("GA-shifting codes"); ax[1].set_xlabel("GA wk"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUTP,f"{tag}.png"),dpi=145,bbox_inches="tight"); print(f"  figure -> out_probe/{tag}.png",flush=True)
    except Exception as ex: print("  fig skipped:",ex,flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
