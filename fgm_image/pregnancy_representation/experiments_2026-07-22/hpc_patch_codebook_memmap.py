#!/usr/bin/env python3
"""Frozen patch-token CODEBOOK via ONE-TIME MEMMAP cache — raw all-layers, no PCA, GPU k-means.

The streaming version re-decompressed all 71 shards EVERY k-means iteration (~40 min/pass ->
40 h). Fix: decompress ONCE into a flat uncompressed memmap on beegfs (fp16), record per-layer
mean/std in the same pass; then k-means + assign read the memmap with fast random access (no
decompression) for all iterations. Full raw all-layers (24576-d), no PCA, no lighter rep.

Pass 1 (once, ~40 min): stream shards -> write raw patches to memmap + accumulate stats.
k-means: chunk-read memmap, z-score on GPU, Lloyd iterations (fast).
assign: chunk-read once -> per-image code grids.

Cache: <OUT>/patchcache_<enc>[_Lall].dat (+ .meta.npz). Reused if present (--rebuild to redo).
USAGE: python hpc_patch_codebook_memmap.py --enc FetalCLIP --all-layers --K 64
"""
import os, glob, json, argparse, numpy as np, torch
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"

def build_cache(enc, all_layers, layer, cache, meta):
    fs=sorted(glob.glob(os.path.join(OUT,f"fulltok_{enc}","shard_*.npz"))); assert fs
    z0=np.load(fs[0],allow_pickle=True); _,L,T,D=z0["tokens"].shape; del z0
    npatch=T-1; dim=(L*D) if all_layers else D
    # total images
    ntot=0
    for f in fs: ntot+=np.load(f,allow_pickle=True)["ga"].shape[0]
    print(f"  building memmap cache: {ntot} imgs x {npatch} patches x {dim}d fp16 "
          f"= {ntot*npatch*dim*2/1e9:.0f} GB -> {cache}",flush=True)
    mm=np.lib.format.open_memmap(cache,mode="w+",dtype=np.float16,shape=(ntot*npatch,dim))
    GA=[]; PL=[]; NM=[]; s=None; s2=None; row=0; import time; t0=time.time()
    for i,f in enumerate(fs):
        z=np.load(f,allow_pickle=True); tok=z["tokens"]
        if all_layers:
            pt=tok[:,:,1:,:].astype(np.float32); n,L,Np,D=pt.shape
            pt=pt.transpose(0,2,1,3).reshape(n*Np,L,D)          # (n*Np, L, D) raw
            if s is None: s=np.zeros((L,D)); s2=np.zeros((L,D))
            s+=pt.sum(0); s2+=(pt**2).sum(0)
            flat=pt.reshape(n*Np,L*D)
        else:
            pt=tok[:,layer,1:,:].astype(np.float32); n=pt.shape[0]; flat=pt.reshape(n*npatch,D)
        mm[row:row+len(flat)]=flat.astype(np.float16); row+=len(flat)
        GA.append(z["ga"]); PL.append(z["plane"]); NM.append(z["names"]); del tok,z,pt,flat
        if i%10==0: print(f"    cached shard {i+1}/{len(fs)} {time.time()-t0:.0f}s",flush=True)
    mm.flush()
    mu=(s/(ntot*npatch)) if all_layers else None
    sd=(np.sqrt(np.maximum(s2/(ntot*npatch)-mu**2,1e-12))+1e-6) if all_layers else None
    np.savez(meta, ga=np.concatenate(GA), plane=np.concatenate(PL), names=np.concatenate(NM),
             npatch=npatch, dim=dim, L=L if all_layers else 0, D=D,
             mu=(mu if mu is not None else np.array([])), sd=(sd if sd is not None else np.array([])))
    print(f"  cache built {time.time()-t0:.0f}s",flush=True)

def zscore_gpu(Xb, mu_t, sd_t, all_layers, L, D):
    if not all_layers: return Xb
    b=Xb.shape[0]; return ((Xb.view(b,L,D)-mu_t)/sd_t).view(b,L*D)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--enc",default="FetalCLIP"); ap.add_argument("--layer",type=int,default=-1)
    ap.add_argument("--all-layers",action="store_true"); ap.add_argument("--K",type=int,default=64)
    ap.add_argument("--iters",type=int,default=60); ap.add_argument("--rebuild",action="store_true")
    ap.add_argument("--chunk",type=int,default=200000)
    a=ap.parse_args()
    tag=f"{a.enc}_{'Lall' if a.all_layers else 'L'+str(a.layer)}_K{a.K}"
    cache=os.path.join(OUT,f"patchcache_{a.enc}{'_Lall' if a.all_layers else ''}.dat")
    meta=cache.replace(".dat",".meta.npz")
    if a.rebuild or not (os.path.exists(cache) and os.path.exists(meta)):
        build_cache(a.enc,a.all_layers,a.layer,cache,meta)
    M=np.load(meta,allow_pickle=True); npatch=int(M["npatch"]); dim=int(M["dim"]); L=int(M["L"]); D=int(M["D"])
    ga=M["ga"].astype(np.float32); plane=M["plane"].astype(str); names=M["names"]; ni=len(ga)
    grid=int(round(npatch**0.5))
    mm=np.load(cache,mmap_mode="r"); N=mm.shape[0]
    mu_t=torch.from_numpy(M["mu"]).to(DEV).float()[None] if a.all_layers else None
    sd_t=torch.from_numpy(M["sd"]).to(DEV).float()[None] if a.all_layers else None
    print(f"[{tag}] cache {N} patches x {dim}d | {ni} imgs {npatch}/img | dev {DEV}",flush=True)
    # init centroids from a random chunk
    g=torch.Generator().manual_seed(0)
    ridx=np.sort(np.random.default_rng(0).choice(N,min(a.K*50,N),replace=False))
    seed=torch.from_numpy(np.asarray(mm[ridx])).to(DEV).float()
    seed=zscore_gpu(seed,mu_t,sd_t,a.all_layers,L,D); C=seed[torch.randperm(len(seed),generator=g)[:a.K]].clone()
    import time; t0=time.time()
    for it in range(a.iters):
        Csum=torch.zeros(a.K,dim,device=DEV); Ccnt=torch.zeros(a.K,device=DEV)
        for i in range(0,N,a.chunk):
            Xb=torch.from_numpy(np.asarray(mm[i:i+a.chunk])).to(DEV).float(); Xb=zscore_gpu(Xb,mu_t,sd_t,a.all_layers,L,D)
            d=(Xb.pow(2).sum(1,keepdim=True)-2*Xb@C.t()+C.pow(2).sum(1)); lab=d.argmin(1)
            Csum.index_add_(0,lab,Xb); Ccnt.index_add_(0,lab,torch.ones(len(Xb),device=DEV)); del Xb,d,lab
        dead=Ccnt<1; Cnew=torch.where(dead[:,None],C,Csum/Ccnt.clamp(min=1)[:,None])
        if dead.any():
            ri=np.sort(np.random.default_rng(it+1).choice(N,int(dead.sum()),replace=False))
            rr=zscore_gpu(torch.from_numpy(np.asarray(mm[ri])).to(DEV).float(),mu_t,sd_t,a.all_layers,L,D); Cnew[dead]=rr
        shift=(Cnew-C).pow(2).sum(1).mean().item(); C=Cnew
        if it%5==0: print(f"  kmeans it{it} shift={shift:.4f} used={int((Ccnt>=1).sum())}/{a.K} {time.time()-t0:.0f}s",flush=True)
        if shift<1e-5 and it>5: break
    # assign
    codes=np.empty((ni,npatch),np.int16)
    for i in range(0,N,a.chunk):
        Xb=torch.from_numpy(np.asarray(mm[i:i+a.chunk])).to(DEV).float(); Xb=zscore_gpu(Xb,mu_t,sd_t,a.all_layers,L,D)
        d=(Xb.pow(2).sum(1,keepdim=True)-2*Xb@C.t()+C.pow(2).sum(1)); lab=d.argmin(1).cpu().numpy()
        flatpos=np.arange(i,min(i+a.chunk,N)); codes.reshape(-1)[flatpos]=lab; del Xb,d
    used=len(set(codes.ravel())); print(f"  codes used {used}/{a.K}",flush=True)
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
    json.dump({"tag":tag,"n_img":int(ni),"n_patch":int(npatch),"grid":int(grid),"K":a.K,"codes_used":used,
               "code_pos":code_pos,"code_ga_spearman":code_ga},open(os.path.join(OUTP,f"patch_codebook_{tag}.json"),"w"),indent=2)
    np.savez(os.path.join(OUT,f"patch_codes_{tag}.npz"),codes=codes,names=names,ga=ga,plane=plane,centroids=C.cpu().numpy().astype(np.float32),grid=grid)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(13,5)); cs=list(code_pos.values())
        sc=ax[0].scatter([d["col"] for d in cs],[1-d["row"] for d in cs],s=[np.sqrt(d["freq"]) for d in cs],c=[d["spatial_conc"] for d in cs],cmap="viridis")
        ax[0].set_title(f"codes by grid position\n{tag} {used}/{a.K}"); plt.colorbar(sc,ax=ax[0]); ax[0].set_xlabel("col"); ax[0].set_ylabel("row(top=1)")
        gb=np.linspace(ga.min(),ga.max(),12)
        for c,r in strong[:5]: ax[1].plot((gb[:-1]+gb[1:])/2,[freq[(ga>=gb[i])&(ga<gb[i+1]),c].mean() for i in range(len(gb)-1)],"-o",ms=3,label=f"code {c} ρ={r:.2f}")
        ax[1].set_title("GA-shifting codes"); ax[1].set_xlabel("GA wk"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUTP,f"patch_codebook_{tag}.png"),dpi=145,bbox_inches="tight"); print(f"  figure -> out_probe/patch_codebook_{tag}.png",flush=True)
    except Exception as ex: print("  fig skipped:",ex,flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
