#!/usr/bin/env python3
"""FACTORISED (shared + private) VQ-VAE on FROZEN FetalCLIP, run LIVE.

Motivation: concatenating 12 layers into ONE codebook forces every discrete word to describe
all depths at once, wasting the fact that shallow layers carry texture and deep layers carry
semantics. Here the vocabulary is FACTORISED by depth:

  12 selected layers -> 3 depth GROUPS (shallow / mid / deep), each 4 layers x 1024 = 4096-d
  SHARED part   = depth-invariant content (mean across groups)  -> ONE shared codebook
  PRIVATE part  = each group's DEVIATION from shared            -> ONE codebook PER group

The split is an EXPLICIT decomposition (mean + residual), not two free encoders competing, so
it cannot collapse onto one side (the leakage failure mode of factorised VAEs). Each group is
reconstructed from (shared code + its private code), so we can measure directly:
  - recon from SHARED only        -> how much appearance is depth-invariant
  - recon from shared + PRIVATE   -> how much is depth-specific
  - codes used / GA-shift per codebook -> is the shared or a private vocabulary GA-structured

All codebooks are EMA-updated with dead-code reinit (no collapse). FetalCLIP frozen (no grads).
USAGE: python hpc_factorised_vqvae_live.py --n-layers 12 --groups 3 --K-shared 128 --K-private 64 --epochs 15
"""
import os, json, argparse, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae")); os.makedirs(OUT,exist_ok=True)
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
# COHORT switch. IMPACT is a narrow late window (GA 26-39wk); the CLINICAL set spans GA
# 6.2-41.6wk (544 fetuses) and is the cohort the pregnancy-representation thesis needs.
# Both indices share columns: nid,new_filename,ga_weeks_recovered,plane_prop.
COHORTS={
 "impact":  (os.environ.get("IMPACT_IMG_DIR","/mnt/beegfs/groups/collage/data/IMPACT_FULL/processed/IMPACT_FULL/preprocessed"),
             os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")),
 "clinical":(os.environ.get("CLINICAL_IMG_DIR","/mnt/beegfs/groups/collage/data/IMPACT_CLINICAL/processed/grouped/IMPACT_CLINICAL/preprocessed"),
             os.path.join(HERE,"clinical_index.csv")),
}
IMG_DIR,INDEX=COHORTS["impact"]        # overridden in main() by --cohort
WEIGHTS=os.path.join(ROOT,"FetalCLIP_weights.pt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
CLIP_NORM=((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))
tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize(*CLIP_NORM)])

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
def clip_layer_maps(vt,x,sel):
    v=vt; h=v.conv1(x); h=h.reshape(h.shape[0],h.shape[1],-1).permute(0,2,1)
    cls=v.class_embedding.to(h.dtype)+torch.zeros(h.shape[0],1,h.shape[-1],device=h.device,dtype=h.dtype)
    h=torch.cat([cls,h],1)+v.positional_embedding.to(h.dtype); h=v.ln_pre(h); h=h.permute(1,0,2)
    out=[]
    for i,blk in enumerate(v.transformer.resblocks,1):
        h=blk(h)
        if i in sel: out.append(h.permute(1,0,2)[:,1:,:])
    return torch.stack(out,1)                       # (B,Ls,Npatch,D)

class VQ(nn.Module):
    """EMA codebook + dead-code reinit."""
    def __init__(s,K,dim,beta=0.25,decay=0.99,eps=1e-5,reinit=1.0):
        super().__init__(); s.K=K; s.beta=beta; s.decay=decay; s.eps=eps; s.reinit=reinit
        e=torch.randn(K,dim); s.register_buffer("cb",e); s.register_buffer("cs",torch.zeros(K)); s.register_buffer("ew",e.clone())
    def forward(s,z):                               # z (B,dim,H,W)
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

class FactorisedVQVAE(nn.Module):
    """shared codebook on the depth-INVARIANT part; one private codebook per depth GROUP on
    that group's DEVIATION. Each group decoded from concat(shared_q, private_q)."""
    def __init__(s,Cg,ngroup,K_shared=128,K_priv=64,zs=64,zp=32):
        super().__init__(); s.ngroup=ngroup
        s.enc_s=nn.Sequential(nn.Conv2d(Cg,256,1),nn.GELU(),nn.Conv2d(256,zs,3,padding=1))
        s.vq_s=VQ(K_shared,zs)
        s.enc_p=nn.ModuleList([nn.Sequential(nn.Conv2d(Cg,256,1),nn.GELU(),nn.Conv2d(256,zp,3,padding=1)) for _ in range(ngroup)])
        s.vq_p=nn.ModuleList([VQ(K_priv,zp) for _ in range(ngroup)])
        s.dec=nn.ModuleList([nn.Sequential(nn.Conv2d(zs+zp,256,3,padding=1),nn.GELU(),nn.Conv2d(256,Cg,1)) for _ in range(ngroup)])
    def forward(s,groups):                          # groups: list of (B,Cg,H,W), len=ngroup
        shared_in=torch.stack(groups,0).mean(0)     # depth-INVARIANT part (mean across groups)
        zs=s.enc_s(shared_in); zsq,ids,l_s=s.vq_s(zs)
        recons=[]; idp=[]; l_p=0.
        for g in range(s.ngroup):
            dev=groups[g]-shared_in                 # this group's DEVIATION from shared
            zp=s.enc_p[g](dev); zpq,idg,lp=s.vq_p[g](zp); l_p=l_p+lp
            recons.append(s.dec[g](torch.cat([zsq,zpq],1))); idp.append(idg)
        return recons, ids, idp, l_s+l_p, zsq
    @torch.no_grad()
    def shared_only(s,groups):
        """recon using shared code + ZERO private -> measures depth-invariant share."""
        shared_in=torch.stack(groups,0).mean(0); zs=s.enc_s(shared_in); zsq,_,_=s.vq_s(zs)
        zero=torch.zeros(zsq.shape[0],s.vq_p[0].cb.shape[1],*zsq.shape[2:],device=zsq.device)
        return [s.dec[g](torch.cat([zsq,zero],1)) for g in range(s.ngroup)]

def frame_table():
    df=pd.read_csv(INDEX).copy()
    df["img"]=df["new_filename"].astype(str).apply(lambda n: os.path.join(IMG_DIR,n if n.endswith(".png") else n+".png"))
    return df[df["img"].apply(os.path.exists)].reset_index(drop=True)

def load_batch(paths): return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-layers",type=int,default=12); ap.add_argument("--groups",type=int,default=3)
    ap.add_argument("--K-shared",type=int,default=128); ap.add_argument("--K-private",type=int,default=64)
    ap.add_argument("--zs",type=int,default=64); ap.add_argument("--zp",type=int,default=32)
    ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--bs",type=int,default=64); ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--cohort",default="impact",choices=["impact","clinical"])
    a=ap.parse_args()
    global IMG_DIR,INDEX
    IMG_DIR,INDEX=COHORTS[a.cohort]
    df=frame_table()
    if len(df)==0:
        raise SystemExit(f"0 frames resolved for cohort={a.cohort} at {IMG_DIR}\n"
                         f"  index={INDEX}\n  -> set {'CLINICAL_IMG_DIR' if a.cohort=='clinical' else 'IMPACT_IMG_DIR'} to the right dir")
    print(f"cohort={a.cohort} | index={os.path.basename(INDEX)} | frames resolved {len(df)} | "
          f"GA {df.ga_weeks_recovered.min():.1f}-{df.ga_weeks_recovered.max():.1f} SD {df.ga_weeks_recovered.std():.2f} | "
          f"fetuses {df.nid.nunique()}",flush=True)
    vt=build_fetalclip()
    nblk=len(vt.transformer.resblocks)
    sel=sorted(set(int(x) for x in np.linspace(1,nblk,min(a.n_layers,nblk)).round().astype(int)))
    x0=load_batch(df["img"].iloc[:2]).to(DEV); m0=clip_layer_maps(vt,x0,set(sel))
    Ls,Np,D=m0.shape[1],m0.shape[2],m0.shape[3]; g=int(round(Np**0.5))
    assert Ls%a.groups==0, f"{Ls} layers not divisible by {a.groups} groups"
    per=Ls//a.groups; Cg=per*D
    tag=f"FetalCLIP_factVQ_{a.cohort}_L{Ls}g{a.groups}_Ks{a.K_shared}_Kp{a.K_private}"
    # BIT BUDGET: the flat single-codebook baseline is log2(K_flat) bits/patch. Factorised is
    # log2(K_shared) + ngroup*log2(K_private). Print both so the recon comparison is only made
    # against a bit-MATCHED baseline (extra capacity, not factorisation, can lower recon).
    bits=np.log2(a.K_shared)+a.groups*np.log2(a.K_private)
    print(f"[{tag}] {len(df)} imgs | layers {sel} | {a.groups} groups x {per} layers x {D}d = Cg {Cg} | grid {g}x{g} | dev {DEV}",flush=True)
    print(f"  BIT BUDGET {bits:.1f} bits/patch (shared {np.log2(a.K_shared):.0f} + {a.groups}x{np.log2(a.K_private):.0f}) "
          f"-- compare ONLY against a flat baseline with K=2^{bits:.0f}; else the recon gap is capacity not factorisation",flush=True)
    # per-layer z-score stats (one frozen pass)
    print("  z-score stats (1 frozen pass) ...",flush=True); t0=time.time()
    ssum=torch.zeros(Ls,D,device=DEV); ssq=torch.zeros(Ls,D,device=DEV); nt=0
    for i in range(0,len(df),a.bs):
        mp=clip_layer_maps(vt,load_batch(df["img"].iloc[i:i+a.bs]).to(DEV),set(sel)); f=mp.reshape(-1,Ls,D)
        ssum+=f.sum(0); ssq+=(f**2).sum(0); nt+=f.shape[0]
        if i%(a.bs*40)==0: print(f"    stats {i}/{len(df)} {time.time()-t0:.0f}s",flush=True)
    mu=ssum/nt; sd=(ssq/nt-mu**2).clamp(min=1e-12).sqrt()+1e-6
    def to_groups(mp):
        b=mp.shape[0]; z=(mp-mu[None,:,None])/sd[None,:,None]           # (b,Ls,Np,D) z-scored
        out=[]
        for gi in range(a.groups):
            blk=z[:,gi*per:(gi+1)*per]                                   # (b,per,Np,D)
            f=blk.permute(0,2,1,3).reshape(b,Np,Cg)
            out.append(f.reshape(b,g,g,Cg).permute(0,3,1,2))             # (b,Cg,g,g)
        return out
    net=FactorisedVQVAE(Cg,a.groups,a.K_shared,a.K_private,a.zs,a.zp).to(DEV)
    opt=torch.optim.Adam([p for n,p in net.named_parameters() if ".vq_" not in n and not n.startswith("vq_")],a.lr)
    for ep in range(a.epochs):
        net.train(); rec=0; nb=0
        order=np.random.permutation(len(df))
        for i in range(0,len(df),a.bs):
            gs=to_groups(clip_layer_maps(vt,load_batch(df["img"].iloc[order[i:i+a.bs]]).to(DEV),set(sel)))
            recons,ids,idp,vql,_=net(gs)
            rl=sum(F.mse_loss(r,t) for r,t in zip(recons,gs))/a.groups
            opt.zero_grad(); (rl+vql).backward(); opt.step(); rec+=rl.item(); nb+=1
        us=int((net.vq_s.cs>=1).sum()); up=[int((v.cs>=1).sum()) for v in net.vq_p]
        print(f"  ep{ep} recon={rec/nb:.4f} shared_used={us}/{a.K_shared} priv_used={up} {time.time()-t0:.0f}s",flush=True)
    # eval: full vs shared-only recon (the factorisation answer) + code maps
    net.eval(); Cs=[]; Cp=[]; GA=[]; PL=[]; NM=[]; rfull=[]; rshared=[]
    with torch.no_grad():
        for i in range(0,len(df),a.bs):
            sl=df.iloc[i:i+a.bs]
            gs=to_groups(clip_layer_maps(vt,load_batch(sl["img"]).to(DEV),set(sel)))
            recons,ids,idp,_,_=net(gs); sonly=net.shared_only(gs)
            rfull.append(np.mean([((r-t)**2).mean((1,2,3)).cpu().numpy() for r,t in zip(recons,gs)],0))
            rshared.append(np.mean([((r-t)**2).mean((1,2,3)).cpu().numpy() for r,t in zip(sonly,gs)],0))
            Cs.append(ids.reshape(len(sl),-1).cpu().numpy().astype(np.int16))
            Cp.append(np.stack([q.reshape(len(sl),-1).cpu().numpy().astype(np.int16) for q in idp],1))
            GA.append(sl["ga_weeks_recovered"].values); PL.append(sl["plane_prop"].values); NM.append(sl["new_filename"].values)
    Cs=np.concatenate(Cs); Cp=np.concatenate(Cp); ga=np.concatenate(GA).astype(np.float32)
    plane=np.concatenate(PL).astype(str); names=np.concatenate(NM)
    rfull=np.concatenate(rfull); rshared=np.concatenate(rshared)
    print(f"  FULL recon {rfull.mean():.4f} | SHARED-ONLY recon {rshared.mean():.4f} "
          f"-> private adds {rshared.mean()-rfull.mean():+.4f} (depth-specific share)",flush=True)
    # SAVE FIRST (irreplaceable)
    np.savez(os.path.join(OUT,f"factvq_codes_{tag}.npz"),codes_shared=Cs,codes_private=Cp,names=names,ga=ga,plane=plane,
             cb_shared=net.vq_s.cb.cpu().numpy(),cb_private=np.stack([v.cb.cpu().numpy() for v in net.vq_p]),grid=g)
    torch.save(net.state_dict(),os.path.join(OUT,f"{tag}.pt")); print("  saved model + code maps",flush=True)
    # GA structure: shared vs private vocabularies
    def ga_shift(codes,K):
        fr=np.stack([(codes==c).mean(1) for c in range(K)],1)
        return {int(c):float(spearmanr(fr[:,c],ga)[0]) for c in range(K) if fr[:,c].std()>1e-6}
    gs_shared=ga_shift(Cs,a.K_shared)
    gs_priv=[ga_shift(Cp[:,gi,:],a.K_private) for gi in range(a.groups)]
    top=lambda d,n=5: sorted(d.items(),key=lambda kv:-abs(kv[1]))[:n]
    print("  shared codes GA-shift:",[(c,round(r,2)) for c,r in top(gs_shared)],flush=True)
    for gi in range(a.groups):
        print(f"  private g{gi} GA-shift:",[(c,round(r,2)) for c,r in top(gs_priv[gi])],flush=True)
    json.dump({"tag":str(tag),"n_img":int(len(ga)),"grid":int(g),"layers":[int(x) for x in sel],"groups":int(a.groups),
               "K_shared":int(a.K_shared),"K_private":int(a.K_private),"bits_per_patch":float(bits),
               "recon_full":float(rfull.mean()),"recon_shared_only":float(rshared.mean()),
               "private_gain":float(rshared.mean()-rfull.mean()),
               "shared_used":int((net.vq_s.cs>=1).sum()),"private_used":[int((v.cs>=1).sum()) for v in net.vq_p],
               "ga_shift_shared":gs_shared,"ga_shift_private":gs_priv},
              open(os.path.join(OUTP,f"{tag}.json"),"w"),indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(13,5))
        ax[0].bar(["shared\nonly","shared+\nprivate"],[rshared.mean(),rfull.mean()],color=["#8899bb","#33aa66"])
        ax[0].set_ylabel("feature recon MSE (lower better)"); ax[0].set_title(f"factorisation: private adds {rshared.mean()-rfull.mean():+.3f}\n{tag}")
        allga=[("shared",gs_shared)]+[(f"priv g{gi}",gs_priv[gi]) for gi in range(a.groups)]
        ax[1].bar([n for n,_ in allga],[max((abs(v) for v in d.values()),default=0) for _,d in allga],color="#cc7744")
        ax[1].set_ylabel("max |Spearman| code-freq vs GA"); ax[1].set_title("GA structure per vocabulary")
        fig.tight_layout(); fig.savefig(os.path.join(OUTP,f"{tag}.png"),dpi=145,bbox_inches="tight")
        print(f"  figure -> out_probe/{tag}.png",flush=True)
    except Exception as ex: print("  fig skipped:",ex,flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
