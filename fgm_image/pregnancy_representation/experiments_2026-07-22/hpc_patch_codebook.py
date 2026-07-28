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

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")   # IMPACT index has plane_prop + GA

def load_patch_layer(enc, layer, max_imgs=4000, all_layers=False, pca_dim=256):
    """stream shards, take patch tokens (drop CLS token 0).
    layer mode: ONE layer -> (n_img, n_patch, dim).
    all_layers: concat ALL layers per patch (dim*L) then PCA-reduce to pca_dim (fit on a
    patch sample) so k-means stays tractable (24*1024=24576-d -> pca_dim). Returns same shape."""
    d=os.path.join(OUT,f"fulltok_{enc}")
    shards=sorted(glob.glob(os.path.join(d,"shard_*.npz")))
    assert shards, f"no shards at {d}"
    P=[]; GA=[]; PL=[]; NM=[]; got=0
    for f in shards:
        z=np.load(f,allow_pickle=True); tok=z["tokens"]        # (n,L,Ntok,dim)
        if all_layers:
            pt=tok[:,:,1:,:].astype(np.float32)                # (n,L,Npatch,dim)
            n,L,Np,D=pt.shape
            pt=pt.transpose(0,2,1,3).reshape(n,Np,L*D)         # (n,Npatch, L*dim)
        else:
            pt=tok[:,layer,1:,:].astype(np.float32)            # (n,Npatch,dim)
        P.append(pt); GA.append(z["ga"]); PL.append(z["plane"]); NM.append(z["names"])
        got+=len(pt); del tok,z
        if got>=max_imgs: break
    P=np.concatenate(P)[:max_imgs]; GA=np.concatenate(GA)[:max_imgs]
    PL=np.concatenate(PL)[:max_imgs]; NM=np.concatenate(NM)[:max_imgs]
    if all_layers and P.shape[-1]>pca_dim:
        from sklearn.decomposition import PCA
        n,Np,D=P.shape; flat=P.reshape(-1,D)
        samp=flat[np.random.default_rng(0).choice(len(flat),min(200000,len(flat)),replace=False)]
        pca=PCA(pca_dim,random_state=0).fit(samp)
        P=pca.transform(flat).reshape(n,Np,pca_dim).astype(np.float32)
        print(f"  all-layers concat {D}d -> PCA {pca_dim}d (var kept {pca.explained_variance_ratio_.sum():.2f})",flush=True)
    return P, GA.astype(np.float32), PL.astype(str), NM.astype(str)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--enc",default="FetalCLIP"); ap.add_argument("--layer",type=int,default=-1)
    ap.add_argument("--all-layers",action="store_true",help="concat ALL layers per patch then PCA")
    ap.add_argument("--pca-dim",type=int,default=256)
    ap.add_argument("--K",type=int,default=64); ap.add_argument("--max_imgs",type=int,default=4000)
    a=ap.parse_args()
    # all-layers concat is ~25MB/img (FetalCLIP 24*1024*256) -> cap images to bound RAM
    max_imgs=min(a.max_imgs,1200) if a.all_layers else a.max_imgs
    tag=f"{a.enc}_{'Lall' if a.all_layers else 'L'+str(a.layer)}_K{a.K}"
    P,ga,plane,names=load_patch_layer(a.enc,a.layer,max_imgs,all_layers=a.all_layers,pca_dim=a.pca_dim)
    ni,npatch,dim=P.shape; grid=int(round(npatch**0.5))
    print(f"[{tag}] {ni} imgs x {npatch} patches ({grid}x{grid}) x {dim}d | GA {ga.min():.0f}-{ga.max():.0f}",flush=True)
    # fit codebook on flattened patches (MiniBatchKMeans = robust, no collapse)
    flat=P.reshape(-1,dim)
    km=MiniBatchKMeans(a.K,batch_size=4096,n_init=3,max_iter=200,random_state=0).fit(flat)
    codes=km.predict(flat).reshape(ni,npatch)                  # (n_img, n_patch) code map
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
             names=names,ga=ga,plane=plane,centroids=km.cluster_centers_.astype(np.float32),grid=grid)
    print(f"  code maps -> out_usfmae/patch_codes_{tag}.npz  DONE",flush=True)

if __name__=="__main__": main()
