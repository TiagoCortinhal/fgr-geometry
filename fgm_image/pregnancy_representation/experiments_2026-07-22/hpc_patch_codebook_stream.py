#!/usr/bin/env python3
"""Frozen patch-token CODEBOOK, STREAMING (memory-safe) — raw all-layers, no PCA, GPU k-means.

The non-streaming version OOM'd: raw all-layers is ~25 MB/image and loading many at once
blows host RAM. This version NEVER holds more than one shard's patches: it streams shards
through (a) a per-layer mean/std pass, (b) k-means iterations (centroid sums accumulated on
GPU), (c) a final assign pass. Host RAM only ever holds one shard (~few GB).

Each image -> 16x16 grid of code IDs. Grounds codes by position / plane / GA (same as before).
Reads out_usfmae/fulltok_<enc>/shard_*.npz. Outputs out_probe/patch_codebook_<tag>.{json,png}
+ out_usfmae/patch_codes_<tag>.npz (code maps + centroids + per-layer stats).

USAGE: python hpc_patch_codebook_stream.py --enc FetalCLIP --all-layers --K 64
       [--layer -1 for single layer] [--max_shards N to subsample] [--iters 60]
"""
import os, glob, json, argparse, numpy as np, torch
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"

def shard_list(enc,max_shards):
    fs=sorted(glob.glob(os.path.join(OUT,f"fulltok_{enc}","shard_*.npz")))
    assert fs, f"no shards for {enc}"
    return fs[:max_shards] if max_shards else fs

def iter_patches(f, layer, all_layers, stats=None):
    """yield (patches (Npatch_total, dim), meta per image) for ONE shard. Applies per-layer
    z-score if stats=(mu,sd) given. Returns raw dim patches, dropping CLS."""
    z=np.load(f,allow_pickle=True); tok=z["tokens"]        # (n,L,T,dim) fp16
    if all_layers:
        pt=tok[:,:,1:,:].astype(np.float32); n,L,Np,D=pt.shape
        pt=pt.transpose(0,2,1,3).reshape(n,Np,L,D)         # (n,Np,L,D)
        if stats is not None:
            mu,sd=stats; pt=(pt-mu)/sd
        pt=pt.reshape(n,Np,L*D)
    else:
        pt=tok[:,layer,1:,:].astype(np.float32); n,Np,_=pt.shape; pt=pt  # (n,Np,D)
    del tok
    return pt, z["ga"].astype(np.float32), z["plane"].astype(str), z["names"].astype(str), z

def compute_layer_stats(fs):
    """streaming per-layer mean/std over all patches (one shard at a time)."""
    n=0; s=None; s2=None; L=D=None
    for f in fs:
        z=np.load(f,allow_pickle=True); pt=z["tokens"][:,:,1:,:].astype(np.float32)  # (n,L,Np,D)
        ni,L,Np,D=pt.shape; x=pt.transpose(0,2,1,3).reshape(-1,L,D)  # (n*Np,L,D)
        if s is None: s=np.zeros((L,D)); s2=np.zeros((L,D))
        s+=x.sum(0); s2+=(x**2).sum(0); n+=x.shape[0]; del z,pt,x
    mu=s/n; sd=np.sqrt(np.maximum(s2/n-mu**2,1e-12))+1e-6
    return mu[None].astype(np.float32), sd[None].astype(np.float32)   # (1,L,D)

def gpu_kmeans_stream(fs, layer, all_layers, stats, K, dim, iters=60, seed=0):
    g=torch.Generator().manual_seed(seed)
    # init centroids from first shard's random patches
    pt0,_,_,_,_=iter_patches(fs[0],layer,all_layers,stats); pt0=pt0.reshape(-1,dim)
    C=torch.from_numpy(pt0[torch.randperm(len(pt0),generator=g)[:K].numpy()]).to(DEV).float(); del pt0
    for it in range(iters):
        Csum=torch.zeros(K,dim,device=DEV); Ccnt=torch.zeros(K,device=DEV); tot=0
        for f in fs:
            pt,_,_,_,_=iter_patches(f,layer,all_layers,stats); X=torch.from_numpy(pt.reshape(-1,dim)).to(DEV).float(); del pt
            d=(X.pow(2).sum(1,keepdim=True)-2*X@C.t()+C.pow(2).sum(1)); lab=d.argmin(1)
            Csum.index_add_(0,lab,X); Ccnt.index_add_(0,lab,torch.ones(len(X),device=DEV)); tot+=len(X); del X,d,lab
        dead=Ccnt<1; Cnew=torch.where(dead[:,None],C,Csum/Ccnt.clamp(min=1)[:,None])
        if dead.any():
            pt,_,_,_,_=iter_patches(fs[0],layer,all_layers,stats); pool=pt.reshape(-1,dim)
            idx=torch.randint(0,len(pool),(int(dead.sum()),),generator=g)
            Cnew[dead]=torch.from_numpy(pool[idx.numpy()]).to(DEV).float(); del pt,pool
        shift=(Cnew-C).pow(2).sum(1).mean().item(); C=Cnew
        if it%10==0: print(f"  kmeans it{it} shift={shift:.4f} used={int((Ccnt>=1).sum())}/{K} (n={tot})",flush=True)
        if shift<1e-5 and it>5: break
    return C

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--enc",default="FetalCLIP"); ap.add_argument("--layer",type=int,default=-1)
    ap.add_argument("--all-layers",action="store_true"); ap.add_argument("--K",type=int,default=64)
    ap.add_argument("--max_shards",type=int,default=None); ap.add_argument("--iters",type=int,default=60)
    a=ap.parse_args()
    tag=f"{a.enc}_{'Lall' if a.all_layers else 'L'+str(a.layer)}_K{a.K}"
    fs=shard_list(a.enc,a.max_shards)
    # dim + grid from first shard
    z0=np.load(fs[0],allow_pickle=True); _,L,T,D=z0["tokens"].shape; del z0
    npatch=T-1; grid=int(round(npatch**0.5)); dim=(L*D) if a.all_layers else D
    print(f"[{tag}] {len(fs)} shards | {npatch} patches ({grid}x{grid}) | dim {dim} | dev {DEV}",flush=True)
    stats=None
    if a.all_layers:
        print("  computing per-layer mean/std (streaming) ...",flush=True); stats=compute_layer_stats(fs)
        print("  per-layer z-score stats ready (no PCA, raw dim kept)",flush=True)
    C=gpu_kmeans_stream(fs,a.layer,a.all_layers,stats,a.K,dim,a.iters)
    # assign pass -> code maps + meta
    codes=[]; GA=[]; PL=[]; NM=[]
    for f in fs:
        pt,ga,pl,nm,_=iter_patches(f,a.layer,a.all_layers,stats); n=pt.shape[0]
        X=torch.from_numpy(pt.reshape(-1,dim)).to(DEV).float(); del pt
        d=(X.pow(2).sum(1,keepdim=True)-2*X@C.t()+C.pow(2).sum(1)); lab=d.argmin(1).cpu().numpy().reshape(n,npatch)
        codes.append(lab.astype(np.int16)); GA.append(ga); PL.append(pl); NM.append(nm); del X,d
    codes=np.concatenate(codes); ga=np.concatenate(GA); plane=np.concatenate(PL); names=np.concatenate(NM)
    ni=codes.shape[0]; used=len(set(codes.ravel()))
    print(f"  codes used {used}/{a.K} over {ni} images",flush=True)
    # groundings (same as before)
    rows=np.arange(npatch)//grid; cols=np.arange(npatch)%grid
    code_pos={}
    for c in range(a.K):
        mask=(codes==c)
        if mask.sum()<20: continue
        occ=np.array([(codes[:,p]==c).sum() for p in range(npatch)],float); occ=occ/occ.sum() if occ.sum() else occ
        ent=-(occ[occ>0]*np.log(occ[occ>0])).sum()/np.log(npatch)
        pr=np.repeat(rows[None,:],ni,0)[mask].mean()/grid; pc=np.repeat(cols[None,:],ni,0)[mask].mean()/grid
        code_pos[int(c)]={"row":float(pr),"col":float(pc),"spatial_conc":float(1-ent),"freq":int(mask.sum())}
    freq=np.stack([(codes==c).mean(1) for c in range(a.K)],1)
    code_ga={int(c):float(spearmanr(freq[:,c],ga)[0]) for c in range(a.K) if freq[:,c].std()>1e-6}
    strong=sorted(code_ga.items(),key=lambda kv:-abs(kv[1]))[:8]
    print("  top GA-shifting codes:",[(c,round(r,2)) for c,r in strong],flush=True)
    json.dump({"tag":tag,"n_img":int(ni),"n_patch":int(npatch),"grid":int(grid),"K":a.K,"codes_used":used,
               "code_pos":code_pos,"code_ga_spearman":code_ga},open(os.path.join(OUTP,f"patch_codebook_{tag}.json"),"w"),indent=2)
    np.savez(os.path.join(OUT,f"patch_codes_{tag}.npz"),codes=codes,names=names,ga=ga,plane=plane,
             centroids=C.cpu().numpy().astype(np.float32),grid=grid,layer_stats=(stats if stats is not None else np.array([])))
    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(13,5)); cs=list(code_pos.values())
        sc=ax[0].scatter([d["col"] for d in cs],[1-d["row"] for d in cs],s=[np.sqrt(d["freq"]) for d in cs],
                         c=[d["spatial_conc"] for d in cs],cmap="viridis")
        ax[0].set_title(f"codes by grid position (color=concentration)\n{tag} {used}/{a.K}"); plt.colorbar(sc,ax=ax[0])
        ax[0].set_xlabel("col"); ax[0].set_ylabel("row(top=1)")
        gb=np.linspace(ga.min(),ga.max(),12)
        for c,r in strong[:5]:
            ax[1].plot((gb[:-1]+gb[1:])/2,[freq[(ga>=gb[i])&(ga<gb[i+1]),c].mean() for i in range(len(gb)-1)],"-o",ms=3,label=f"code {c} ρ={r:.2f}")
        ax[1].set_title("GA-shifting codes: patch-freq vs GA"); ax[1].set_xlabel("GA wk"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUTP,f"patch_codebook_{tag}.png"),dpi=145,bbox_inches="tight")
        print(f"  figure -> out_probe/patch_codebook_{tag}.png",flush=True)
    except Exception as ex: print("  fig skipped:",ex,flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
