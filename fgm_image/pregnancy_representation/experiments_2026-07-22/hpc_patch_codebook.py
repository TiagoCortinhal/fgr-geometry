#!/usr/bin/env python3
"""Frozen patch-token CODEBOOK — a discrete spatial visual vocabulary from the extracted
full-token store. Each image -> grid of code indices (a map of 'visual words'); we test
whether codes are anatomically/positionally grounded and whether their usage shifts with GA.

This is the FAST, non-reconstructive first look (Stage 1). Reconstructive VQ-VAE from pixels
is Stage 2 (separate, heavier). No pixel decode here — codes explain the encoder's patch
features, and we ground them by position + plane + GA.

Reads full-token shards out_usfmae/fulltok_<enc>/shard_*.npz (tokens (n,L,Ntok,dim) fp16).
Uses ONE layer's patch tokens (drop CLS). Fits MiniBatchKMeans (robust, no collapse) as the
codebook over a sample of patches, assigns all, then:
  - code x position: which grid cells each code occupies (spatial grounding)
  - code x plane   : is a code plane-specific
  - code x GA      : does code usage shift early->late (maturation vocabulary)
Outputs: out_probe/patch_codebook_<enc>_L<layer>_K<K>.{json,png}

USAGE:
  python hpc_patch_codebook.py --enc FetalCLIP --layer -1 --K 64
  python hpc_patch_codebook.py --enc USF-MAE  --layer -1 --K 64
"""
import os, glob, json, argparse, numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import spearmanr
import torch
DEV="cuda" if torch.cuda.is_available() else "cpu"

def gpu_kmeans(flat, K, iters=60, bs=100000, seed=0):
    """Minibatch Lloyd's k-means on GPU for high-dim patches (e.g. 24576-d). Memory-bounded:
    processes patches in chunks, never materialises the full (N,K) distance matrix at once.
    Returns (centroids np (K,D), labels np (N,)) plus dead-code reinit each iter."""
    g=torch.Generator(device="cpu").manual_seed(seed)
    N,D=flat.shape
    init=torch.from_numpy(flat[torch.randperm(N,generator=g)[:K].numpy()]).to(DEV).float()
    C=init.clone()
    def assign(Xb):                             # (b,) labels for a batch on DEV
        d=(Xb.pow(2).sum(1,keepdim=True)-2*Xb@C.t()+C.pow(2).sum(1)); return d.argmin(1)
    for it in range(iters):
        Csum=torch.zeros(K,D,device=DEV); Ccnt=torch.zeros(K,device=DEV)
        for i in range(0,N,bs):
            Xb=torch.from_numpy(flat[i:i+bs]).to(DEV).float(); lab=assign(Xb)
            Csum.index_add_(0,lab,Xb); Ccnt.index_add_(0,lab,torch.ones(len(Xb),device=DEV))
        dead=Ccnt<1
        Cnew=torch.where(dead[:,None],C,Csum/Ccnt.clamp(min=1)[:,None])
        if dead.any():                          # reseed dead centroids to random patches
            idx=torch.randint(0,N,(int(dead.sum()),),generator=g)
            Cnew[dead]=torch.from_numpy(flat[idx.numpy()]).to(DEV).float()
        shift=(Cnew-C).pow(2).sum(1).mean().item(); C=Cnew
        if it%10==0: print(f"    kmeans it{it} shift={shift:.4f} used={int((Ccnt>=1).sum())}/{K}",flush=True)
        if shift<1e-5 and it>5: break
    lab=np.empty(N,np.int32)
    for i in range(0,N,bs):
        lab[i:i+bs]=assign(torch.from_numpy(flat[i:i+bs]).to(DEV).float()).cpu().numpy()
    return C.cpu().numpy(), lab

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")   # IMPACT index has plane_prop + GA

def load_patch_layer(enc, layer, max_imgs=4000, all_layers=False, pca_dim=None):
    """stream shards, take patch tokens (drop CLS token 0). NO PCA — raw features.
    layer mode: ONE layer -> (n_img, n_patch, dim).
    all_layers: concat ALL layers per patch -> (n_img, n_patch, L*dim), then PER-LAYER
    z-score (each layer block standardized) so all layers contribute to distance and the
    high-norm late layers don't dominate. Raw dimensionality kept (e.g. 24*1024=24576)."""
    d=os.path.join(OUT,f"fulltok_{enc}")
    shards=sorted(glob.glob(os.path.join(d,"shard_*.npz")))
    assert shards, f"no shards at {d}"
    P=[]; GA=[]; PL=[]; NM=[]; got=0; L=None; D=None
    for f in shards:
        z=np.load(f,allow_pickle=True); tok=z["tokens"]        # (n,L,Ntok,dim)
        if all_layers:
            pt=tok[:,:,1:,:].astype(np.float32); n,L,Np,D=pt.shape
            pt=pt.transpose(0,2,1,3).reshape(n,Np,L*D)         # (n,Npatch, L*dim) RAW
        else:
            pt=tok[:,layer,1:,:].astype(np.float32)
        P.append(pt); GA.append(z["ga"]); PL.append(z["plane"]); NM.append(z["names"])
        got+=len(pt); del tok,z
        if got>=max_imgs: break
    P=np.concatenate(P)[:max_imgs]; GA=np.concatenate(GA)[:max_imgs]
    PL=np.concatenate(PL)[:max_imgs]; NM=np.concatenate(NM)[:max_imgs]
    if all_layers:
        # per-layer standardize: z-score each layer block over all (img,patch), keep raw dim
        n,Np,LD=P.shape; flat=P.reshape(n*Np,L,D)              # (n*Np, L, D)
        mu=flat.mean(0,keepdims=True); sd=flat.std(0,keepdims=True)+1e-6   # (1,L,D)
        P=((flat-mu)/sd).reshape(n,Np,L*D).astype(np.float32)
        print(f"  all-layers RAW concat {L*D}d, per-layer z-scored (no PCA)",flush=True)
    return P, GA.astype(np.float32), PL.astype(str), NM.astype(str)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--enc",default="FetalCLIP"); ap.add_argument("--layer",type=int,default=-1)
    ap.add_argument("--all-layers",action="store_true",help="concat ALL layers per patch then PCA")
    ap.add_argument("--pca-dim",type=int,default=256)
    ap.add_argument("--K",type=int,default=64); ap.add_argument("--max_imgs",type=int,default=4000)
    a=ap.parse_args()
    # RAW all-layers concat = 24*1024*256 patches ~= 25 MB/img in host RAM. Default 1500 imgs
    # -> ~38 GB loaded (fine on the HPC node). Override --max_imgs if the node has more RAM.
    # GPU k-means handles the COMPUTE at full 24576-d; the load is the only RAM bound.
    max_imgs=min(a.max_imgs,1500) if a.all_layers else a.max_imgs
    tag=f"{a.enc}_{'Lall' if a.all_layers else 'L'+str(a.layer)}_K{a.K}"
    P,ga,plane,names=load_patch_layer(a.enc,a.layer,max_imgs,all_layers=a.all_layers,pca_dim=a.pca_dim)
    ni,npatch,dim=P.shape; grid=int(round(npatch**0.5))
    print(f"[{tag}] {ni} imgs x {npatch} patches ({grid}x{grid}) x {dim}d | GA {ga.min():.0f}-{ga.max():.0f}",flush=True)
    # fit codebook on flattened patches — GPU minibatch k-means (raw high-dim, no PCA)
    flat=P.reshape(-1,dim)
    print(f"  GPU k-means on {flat.shape[0]} patches x {dim}d, K={a.K}, dev={DEV}",flush=True)
    centroids,labels=gpu_kmeans(flat,a.K)
    codes=labels.reshape(ni,npatch)                            # (n_img, n_patch) code map
    used=len(set(codes.ravel())); print(f"  codes used {used}/{a.K}",flush=True)
    # 1) spatial grounding: for each code, mean grid position + positional concentration
    pos=np.arange(npatch); rows=pos//grid; cols=pos%grid
    code_pos={}
    for c in range(a.K):
        mask=(codes==c)
        if mask.sum()<20: continue
        pr=np.repeat(rows[None,:],ni,0)[mask].mean()/grid
        pc=np.repeat(cols[None,:],ni,0)[mask].mean()/grid
        # concentration = 1 - normalized spatial entropy of this code's position usage
        occ=np.zeros(npatch)
        for p in range(npatch): occ[p]=(codes[:,p]==c).sum()
        occ=occ/occ.sum() if occ.sum() else occ
        ent=-(occ[occ>0]*np.log(occ[occ>0])).sum()/np.log(npatch)
        code_pos[c]={"row":float(pr),"col":float(pc),"spatial_conc":float(1-ent),"freq":int(mask.sum())}
    # 2) code x plane: is a code plane-specific? (per-image code presence vs plane)
    planes=sorted(set(plane)); code_plane={}
    for c in range(a.K):
        present=(codes==c).any(1)
        if present.sum()<20: continue
        by={pl:float((present&(plane==pl)).sum()/max((plane==pl).sum(),1)) for pl in planes}
        code_plane[c]=by
    # 3) code x GA: does per-image code FREQUENCY shift with GA? (maturation vocabulary)
    freq=np.stack([(codes==c).mean(1) for c in range(a.K)],1)   # (n_img,K) fraction of patches
    code_ga_rho={}
    for c in range(a.K):
        if freq[:,c].std()>1e-6:
            code_ga_rho[c]=float(spearmanr(freq[:,c],ga)[0])
    strong=sorted(code_ga_rho.items(),key=lambda kv:-abs(kv[1]))[:8]
    print("  top GA-shifting codes (code: spearman freq~GA):",[(c,round(r,2)) for c,r in strong],flush=True)
    res={"tag":tag,"n_img":ni,"n_patch":npatch,"grid":grid,"K":a.K,"codes_used":used,
         "code_pos":code_pos,"code_plane":code_plane,"code_ga_spearman":code_ga_rho}
    json.dump(res,open(os.path.join(OUTP,f"patch_codebook_{tag}.json"),"w"),indent=2)
    # figure: (a) spatial-concentration map of codes, (b) top GA-shifting codes' freq vs GA
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(13,5))
        # a: scatter each code at its mean grid pos, size=freq, color=spatial_conc
        cs=[code_pos[c] for c in code_pos]
        sc=ax[0].scatter([d["col"] for d in cs],[1-d["row"] for d in cs],
                         s=[np.sqrt(d["freq"]) for d in cs],c=[d["spatial_conc"] for d in cs],cmap="viridis")
        ax[0].set_title(f"codes by mean grid position (color=spatial concentration)\n{tag}, {used}/{a.K} used")
        ax[0].set_xlabel("col (norm)"); ax[0].set_ylabel("row (norm, top=1)"); plt.colorbar(sc,ax=ax[0],label="concentration")
        # b: freq vs GA for top GA-shifting codes
        gb=np.linspace(ga.min(),ga.max(),12)
        for c,r in strong[:5]:
            binned=[freq[(ga>=gb[i])&(ga<gb[i+1]),c].mean() for i in range(len(gb)-1)]
            ax[1].plot((gb[:-1]+gb[1:])/2,binned,"-o",ms=3,label=f"code {c} (ρ={r:.2f})")
        ax[1].set_title("GA-shifting codes: patch-frequency vs GA"); ax[1].set_xlabel("GA (wk)"); ax[1].set_ylabel("frac of patches"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUTP,f"patch_codebook_{tag}.png"),dpi=145,bbox_inches="tight")
        print(f"  figure -> out_probe/patch_codebook_{tag}.png",flush=True)
    except Exception as ex: print("  fig skipped:",ex,flush=True)
    # save the code maps for overlay-on-image later
    np.savez(os.path.join(OUT,f"patch_codes_{tag}.npz"),codes=codes.astype(np.int16),
             names=names,ga=ga,plane=plane,centroids=centroids.astype(np.float32),grid=grid)
    print(f"  code maps -> out_usfmae/patch_codes_{tag}.npz  DONE",flush=True)

if __name__=="__main__": main()
