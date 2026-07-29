#!/usr/bin/env python3
"""CROSS-ENCODER factorised VQ: SHARED vocabulary across 4 frozen encoders + PRIVATE per encoder.

Question: is a visual word a property of FETAL ULTRASOUND, or of one encoder's training? A code
that survives across FetalCLIP / USF-MAE / USFM / DINOv2 is the former; one that only exists in
FetalCLIP is the latter. Prior code work used FetalCLIP ALONE, so this is the missing control.

Two obstacles and how they are handled honestly:

 (1) DIFFERENT DIMS AND GRIDS. FetalCLIP ViT-L/14 = 24 layers x 1024d, 16x16 patches;
     USF-MAE / USFM = 12 x 768, 14x14; DINOv2 ViT-g/14 = 40 x 1536, 16x16. Patch tokens cannot
     simply be concatenated. Reconciliation = a FIXED, SEEDED RANDOM PROJECTION per encoder
     (all-layer concat -> common width), applied incrementally layer-by-layer so the huge concat
     (DINOv2: 40*1536 = 61440d/patch) is never materialised. Random projection is chosen over
     PCA (rejected by design) and over a LEARNED adapter: a learned adapter is also the
     reconstruction TARGET here, so it could trivially collapse the target to make recon look
     good. A fixed projection cannot cheat and approximately preserves distances (JL lemma).
     Grids are resampled to the common 14x14 (DOWN-sample only; never invents detail).

 (2) COLLAPSE OF THE SHARED/PRIVATE SPLIT. shared = MEAN across encoders of the projected maps
     (arithmetic, not learned); private = each encoder's DEVIATION from that mean. Every encoder
     must be reconstructed from concat(shared_code, its_private_code), so encoder-specific
     content in the shared code hurts the other three -- that pressure, not a loss weight, is
     what keeps shared common.

Measures: FULL vs SHARED-ONLY recon per encoder (how much appearance is encoder-general), codes
used per codebook, and GA-shift of the shared vs each private vocabulary. All encoders FROZEN
(unfreezing on GA would make any GA structure in the codes circular).

COST: 4 live encoders. --fit-frames subsets the codebook FIT (assignment still covers all frames)
because K=128 does not need 20k images; default 6000. DINOv2 ViT-g dominates runtime.

USAGE: python hpc_crossenc_factvq.py --epochs 8 --K-shared 128 --K-private 64 [--fit-frames 6000]
       python hpc_crossenc_factvq.py --check     # shapes only, no training
"""
import os, sys, json, argparse, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
OUT=os.environ.get("GA_OUT_DIR",os.path.join(HERE,"out_usfmae")); os.makedirs(OUT,exist_ok=True)
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
GRID=14          # common grid: the min across encoders (16->14 is a DOWN-sample)
CW=256           # common projected width per patch
ENCS=["FetalCLIP","USF-MAE","USFM","DINOv2"]

from hpc_extract_4encoders import BUILDERS, frame_table   # reuse verified weight loading

# ---- patch-token forwards (the shipped builders return pooled CLS+mean-patch, not the grid) ----
@torch.no_grad()
def patch_tokens(name, mdl, x):
    """-> (B, Lblk, Npatch, dim) patch tokens for every block, CLS/registers dropped."""
    if name=="FetalCLIP":
        v=mdl; h=v.conv1(x); h=h.reshape(h.shape[0],h.shape[1],-1).permute(0,2,1)
        cls=v.class_embedding.to(h.dtype)+torch.zeros(h.shape[0],1,h.shape[-1],device=h.device,dtype=h.dtype)
        h=torch.cat([cls,h],1)+v.positional_embedding.to(h.dtype); h=v.ln_pre(h); h=h.permute(1,0,2)
        out=[]
        for blk in v.transformer.resblocks:
            h=blk(h); out.append(h.permute(1,0,2)[:,1:,:])
        return torch.stack(out,1)
    if name=="USF-MAE":
        s=mdl; h=s.patch_embed(x)+s.pos_embed[:,1:,:]
        cls=s.cls_token+s.pos_embed[:,:1,:]; h=torch.cat([cls.expand(x.shape[0],-1,-1),h],1)
        out=[]
        for b in s.blocks: h=b(h); out.append(h[:,1:])
        return torch.stack(out,1)
    if name=="USFM":
        h=mdl.patch_embed(x); cls=mdl.cls_token.expand(x.shape[0],-1,-1); h=torch.cat([cls,h],1)
        rpb=mdl.rel_pos_bias() if getattr(mdl,"rel_pos_bias",None) is not None else None
        out=[]
        for b in mdl.blocks:
            h=b(h,rel_pos_bias=rpb) if rpb is not None else b(h)
            out.append(h[:,1:])
        return torch.stack(out,1)
    if name=="DINOv2":
        feats=mdl.get_intermediate_layers(x,n=mdl.n_blocks,return_class_token=True)
        return torch.stack([p for p,_ in feats],1)          # registers already excluded
    raise KeyError(name)

class Reconciler:
    """per-encoder: layerwise z-score -> FIXED random projection (incremental) -> (B,CW,GRID,GRID)."""
    def __init__(s,name,L,D,npatch,seed=0):
        s.name=name; s.L=L; s.D=D; s.g=int(round(npatch**0.5))
        gen=torch.Generator(device="cpu").manual_seed(seed+abs(hash(name))%10000)
        # one (D,CW) block per layer; sum of per-layer projections == projection of the concat
        s.R=(torch.randn(L,D,CW,generator=gen)/np.sqrt(L*D)).to(DEV)
        s.mu=None; s.sd=None
    def fit_stats(s,acc_sum,acc_sq,n):
        s.mu=(acc_sum/n); s.sd=((acc_sq/n-s.mu**2).clamp(min=1e-12).sqrt()+1e-6)
    def __call__(s,tok):                                   # tok (B,L,Np,D)
        B=tok.shape[0]
        z=(tok-s.mu[None,:,None,:])/s.sd[None,:,None,:]
        p=torch.einsum("blnd,ldc->bnc",z,s.R)              # (B,Np,CW) incremental over layers
        p=p.permute(0,2,1).reshape(B,CW,s.g,s.g)
        if s.g!=GRID: p=F.adaptive_avg_pool2d(p,(GRID,GRID))
        return p

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
                    pk=zf[torch.randint(0,len(zf),(int(dead.sum()),),device=zf.device)]
                    s.cb[dead]=pk; s.ew[dead]=pk; s.cs[dead]=1.0
        loss=s.beta*((zq.detach()-zf)**2).mean(); zq=zf+(zq-zf).detach()
        return zq.view(B,H,W,C).permute(0,3,1,2), idx.view(B,H,W), loss

class CrossEncVQ(nn.Module):
    def __init__(s,nenc,Ks=128,Kp=64,zs=64,zp=32):
        super().__init__(); s.n=nenc
        s.enc_s=nn.Sequential(nn.Conv2d(CW,256,1),nn.GELU(),nn.Conv2d(256,zs,3,padding=1)); s.vq_s=VQ(Ks,zs)
        s.enc_p=nn.ModuleList([nn.Sequential(nn.Conv2d(CW,256,1),nn.GELU(),nn.Conv2d(256,zp,3,padding=1)) for _ in range(nenc)])
        s.vq_p=nn.ModuleList([VQ(Kp,zp) for _ in range(nenc)])
        s.dec=nn.ModuleList([nn.Sequential(nn.Conv2d(zs+zp,256,3,padding=1),nn.GELU(),nn.Conv2d(256,CW,1)) for _ in range(nenc)])
    def forward(s,maps):
        shared_in=torch.stack(maps,0).mean(0)
        zsq,ids,l=s.vq_s(s.enc_s(shared_in)); rec=[]; idp=[]
        for e in range(s.n):
            zpq,ie,lp=s.vq_p[e](s.enc_p[e](maps[e]-shared_in)); l=l+lp
            rec.append(s.dec[e](torch.cat([zsq,zpq],1))); idp.append(ie)
        return rec,ids,idp,l
    @torch.no_grad()
    def shared_only(s,maps):
        shared_in=torch.stack(maps,0).mean(0); zsq,_,_=s.vq_s(s.enc_s(shared_in))
        z=torch.zeros(zsq.shape[0],s.vq_p[0].cb.shape[1],*zsq.shape[2:],device=zsq.device)
        return [s.dec[e](torch.cat([zsq,z],1)) for e in range(s.n)]

def load_imgs(paths,tf): return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--K-shared",type=int,default=128); ap.add_argument("--K-private",type=int,default=64)
    ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--bs",type=int,default=24)
    ap.add_argument("--lr",type=float,default=2e-3); ap.add_argument("--fit-frames",type=int,default=6000)
    ap.add_argument("--check",action="store_true")
    a=ap.parse_args()
    df=frame_table(); print(f"frames {len(df)}",flush=True)
    models={}; 
    for e in ENCS:
        m,tf,_=BUILDERS[e](); models[e]=(m,tf); print(f"  loaded {e}",flush=True)
    # probe shapes
    x0={e:load_imgs(df["img"].iloc[:2],models[e][1]).to(DEV) for e in ENCS}
    shapes={}
    for e in ENCS:
        t=patch_tokens(e,models[e][0],x0[e]); shapes[e]=tuple(t.shape[1:]); print(f"  {e}: L={t.shape[1]} Np={t.shape[2]} D={t.shape[3]}",flush=True)
    if a.check: print("CHECK OK"); return
    # shapes[e] is (L, Npatch, D) -- pass by NAME: positional *shapes[e] swapped D and npatch,
    # which built the random matrix as (L,Npatch,CW) and blew up the einsum.
    rec={e:Reconciler(e,L=shapes[e][0],D=shapes[e][2],npatch=shapes[e][1]) for e in ENCS}
    # layerwise z-score stats (one frozen pass over the FIT subset)
    fit=df.sample(min(a.fit_frames,len(df)),random_state=0).reset_index(drop=True)
    t0=time.time()
    # cache the ~9min stats pass: a later crash must not cost it again
    SCACHE=os.path.join(OUT,f"crossenc_zstats_{len(fit)}.npz")
    if os.path.exists(SCACHE):
        z=np.load(SCACHE,allow_pickle=True)
        for e in ENCS:
            rec[e].mu=torch.tensor(z[f"{e}_mu"],device=DEV); rec[e].sd=torch.tensor(z[f"{e}_sd"],device=DEV)
        print(f"  reused z-score stats {SCACHE}",flush=True)
        acc=None
    else:
        acc={e:[torch.zeros(shapes[e][0],shapes[e][2],device=DEV),torch.zeros(shapes[e][0],shapes[e][2],device=DEV),0] for e in ENCS}
        print(f"  z-score stats on {len(fit)} fit frames ...",flush=True)
    if acc is not None:
        for i in range(0,len(fit),a.bs):
            for e in ENCS:
                t=patch_tokens(e,models[e][0],load_imgs(fit["img"].iloc[i:i+a.bs],models[e][1]).to(DEV))
                f=t.permute(0,2,1,3).reshape(-1,shapes[e][0],shapes[e][2])
                acc[e][0]+=f.sum(0); acc[e][1]+=(f**2).sum(0); acc[e][2]+=f.shape[0]
            if i%(a.bs*40)==0: print(f"    stats {i}/{len(fit)} {time.time()-t0:.0f}s",flush=True)
        for e in ENCS: rec[e].fit_stats(*acc[e])
        np.savez(SCACHE,**{f"{e}_mu":rec[e].mu.cpu().numpy() for e in ENCS},
                        **{f"{e}_sd":rec[e].sd.cpu().numpy() for e in ENCS})
        print(f"  saved z-score stats -> {SCACHE}",flush=True)
    # shape assertion: projection must be (L, D, CW) with D the FEATURE width, not Npatch
    for e in ENCS:
        assert rec[e].R.shape==(shapes[e][0],shapes[e][2],CW), f"{e} proj {tuple(rec[e].R.shape)} != {(shapes[e][0],shapes[e][2],CW)}"
        assert rec[e].mu.shape==(shapes[e][0],shapes[e][2]), f"{e} mu {tuple(rec[e].mu.shape)}"
    print("  projection shapes OK",flush=True)
    net=CrossEncVQ(len(ENCS),a.K_shared,a.K_private).to(DEV)
    opt=torch.optim.Adam([p for n,p in net.named_parameters() if ".vq_" not in n and not n.startswith("vq_")],a.lr)
    for ep in range(a.epochs):
        net.train(); tot=0; nb=0; order=np.random.permutation(len(fit))
        for i in range(0,len(fit),a.bs):
            sl=fit["img"].iloc[order[i:i+a.bs]]
            maps=[rec[e](patch_tokens(e,models[e][0],load_imgs(sl,models[e][1]).to(DEV))) for e in ENCS]
            r,ids,idp,vql=net(maps)
            rl=sum(F.mse_loss(x,y) for x,y in zip(r,maps))/len(ENCS)
            opt.zero_grad(); (rl+vql).backward(); opt.step(); tot+=rl.item(); nb+=1
        print(f"  ep{ep} recon={tot/nb:.4f} shared={int((net.vq_s.cs>=1).sum())}/{a.K_shared} "
              f"priv={[int((v.cs>=1).sum()) for v in net.vq_p]} {time.time()-t0:.0f}s",flush=True)
    # assign over ALL frames + FULL vs SHARED-ONLY per encoder
    net.eval(); CS=[]; CP=[]; GA=[]; NM=[]; rf=[[] for _ in ENCS]; rs=[[] for _ in ENCS]
    with torch.no_grad():
        for i in range(0,len(df),a.bs):
            sl=df.iloc[i:i+a.bs]
            maps=[rec[e](patch_tokens(e,models[e][0],load_imgs(sl["img"],models[e][1]).to(DEV))) for e in ENCS]
            r,ids,idp,_=net(maps); so=net.shared_only(maps)
            for k in range(len(ENCS)):
                rf[k].append(((r[k]-maps[k])**2).mean((1,2,3)).cpu().numpy())
                rs[k].append(((so[k]-maps[k])**2).mean((1,2,3)).cpu().numpy())
            CS.append(ids.reshape(len(sl),-1).cpu().numpy().astype(np.int16))
            CP.append(np.stack([q.reshape(len(sl),-1).cpu().numpy().astype(np.int16) for q in idp],1))
            GA.append(sl["ga_weeks_recovered"].values); NM.append(sl["new_filename"].values)
            if i%(a.bs*60)==0: print(f"    assign {i}/{len(df)}",flush=True)
    CS=np.concatenate(CS); CP=np.concatenate(CP); ga=np.concatenate(GA).astype(np.float32); nm=np.concatenate(NM)
    res={"encoders":ENCS,"shapes":{e:list(shapes[e]) for e in ENCS},"grid":GRID,"common_width":CW,
         "K_shared":a.K_shared,"K_private":a.K_private,"n_frames":int(len(ga)),"fit_frames":int(len(fit)),
         "codes_used_shared":int(len(set(CS.ravel()))),
         "codes_used_private":{e:int(len(set(CP[:,k,:].ravel()))) for k,e in enumerate(ENCS)},
         "recon_full":{e:float(np.concatenate(rf[k]).mean()) for k,e in enumerate(ENCS)},
         "recon_shared_only":{e:float(np.concatenate(rs[k]).mean()) for k,e in enumerate(ENCS)}}
    def gash(codes,K):
        fr=np.stack([(codes==c).mean(1) for c in range(K)],1)
        d={int(c):float(spearmanr(fr[:,c],ga)[0]) for c in range(K) if fr[:,c].std()>1e-6}
        return sorted(d.items(),key=lambda kv:-abs(kv[1]))[:5]
    res["ga_shift_shared_top5"]=gash(CS,a.K_shared)
    res["ga_shift_private_top5"]={e:gash(CP[:,k,:],a.K_private) for k,e in enumerate(ENCS)}
    np.savez(os.path.join(OUT,"crossenc_codes.npz"),codes_shared=CS,codes_private=CP,ga=ga,names=nm,
             cb_shared=net.vq_s.cb.cpu().numpy(),cb_private=np.stack([v.cb.cpu().numpy() for v in net.vq_p]),
             encoders=np.array(ENCS),grid=GRID)
    torch.save(net.state_dict(),os.path.join(OUT,"crossenc_factvq.pt")); print("  saved codes + model",flush=True)
    json.dump(res,open(os.path.join(OUTP,"crossenc_factvq.json"),"w"),indent=2)
    for e in ENCS:
        print(f"  {e:10s} FULL {res['recon_full'][e]:.4f} | SHARED-ONLY {res['recon_shared_only'][e]:.4f} "
              f"-> private adds {res['recon_shared_only'][e]-res['recon_full'][e]:+.4f}",flush=True)
    print("  shared GA-shift:",[(c,round(r,2)) for c,r in res["ga_shift_shared_top5"]],flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
