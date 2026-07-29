#!/usr/bin/env python3
"""STEP 1 -- THE CHEAP KILL TEST. Within-encoder across-seed codebook agreement.

THIS IS THE ONLY THING THAT TRAINS in the WP2 annotation plan: K=16 codebooks, 4 encoders x 5 seeds
= 20 fits. It runs BEFORE any cross-encoder analysis, because it can end the design.

THE LOGIC. The four-experts design treats cross-encoder agreement as evidence that a visual word is
a property of fetal ultrasound rather than of one model. That inference is only valid if agreement
between DIFFERENT encoders is meaningfully higher than agreement between two fits of the SAME encoder
that differ only by random seed. Paulo & Belrose (arXiv:2501.16615) found SAEs differing ONLY in
initialisation shared ~30% of features on identical model and data. If our seed-to-seed agreement is
also ~0.3, then a cross-encoder agreement of 0.3 is EXACTLY ZERO EVIDENCE, and the whole design
fails -- for the price of some k-means instead of the full analysis.

    within-encoder across-seed agreement IS THE CEILING.
    every cross-encoder number must later be reported as a FRACTION of it.

KILL THRESHOLD, fixed here before the numbers are seen:
    mean within-encoder across-seed AMI < 0.50  ->  DESIGN FAILS, stop, do not run the annotation.
    0.50-0.70  -> proceed but every cross-encoder claim is bounded by this ceiling and must say so.
    >= 0.70    -> the ceiling is high enough that cross-encoder agreement is interpretable.

STATISTIC. Adjusted Mutual Information between the two seeds' patch-level assignments over the SAME
patches. AMI is chance-corrected, so it does not reward agreement that two random 16-way partitions
would produce anyway -- which a raw overlap rate does. Reported alongside Hungarian-matched accuracy
for readability. The permutation reference permutes WHOLE IMAGES (not individual patches): patches
within a frame are highly correlated, so a per-patch shuffle produces a null that is too easy to
beat and would flatter every number here.

WHAT THIS SCRIPT DOES NOT DO. It does not touch the WP2 axis, outcomes, GA, or sex. It is a pure
representation-stability measurement, so there is nothing here that can be tuned toward a result.

USAGE:
  python hpc_seed_ceiling.py --check              # shapes + patch budget only, no fitting
  python hpc_seed_ceiling.py --K 16 --seeds 5     # the kill test
Outputs: out_probe/seed_ceiling_K16.json, out_usfmae/wp2codes_<ENC>_K16.npz (seed 0 kept for step 2)
"""
import os, sys, json, time, argparse, numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.metrics import adjusted_mutual_info_score as ami
from scipy.optimize import linear_sum_assignment
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
OUT=os.environ.get("GA_OUT_DIR",os.path.join(HERE,"out_usfmae")); os.makedirs(OUT,exist_ok=True)
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
ENCS=["FetalCLIP","USF-MAE","USFM","DINOv2"]
FIT_FRAMES=4000      # frames used to FIT the codebook (k-means converges long before 20k frames)
BATCH=32
KILL_AMI=0.50        # fixed before seeing numbers
from hpc_extract_4encoders import BUILDERS, frame_table

def cohort_table(cohort):
    """IMPACT via the shipped frame_table; CLINICAL via the prefix-tolerant resolver.

    The clinical store nests under processed/grouped/IMPACT_CLINICAL/<machine>/<subdir>/ and its
    on-disk filenames carry a LEADING NUMERIC PREFIX that clinical_index.csv lacks
    (disk: 60813_IMP0469_20171213_...OBMBFET.png vs index: IMP0469_20171213_...OBMBFET), which is why
    a flat basename join found 0 of 30,257. clinical_paths.resolve() walks the tree and matches on the
    prefix-stripped stem, and excludes debug_inpainting/ artefacts."""
    if cohort=="impact": return frame_table()
    import clinical_paths
    df,_look=clinical_paths.resolve(os.environ.get("CLINICAL_ROOT",      # returns (df, lookup)
        "/mnt/beegfs/groups/collage/data/IMPACT_CLINICAL/processed"))
    df=df[df["img"].astype(str).str.len()>0].reset_index(drop=True)      # drop unresolved rows
    assert len(df)>1000, (f"clinical resolver matched only {len(df)} frames of the 30,257 index rows -- "
        "check CLINICAL_ROOT and that the container binds .../IMPACT_CLINICAL/processed (the clinical "
        "images were invisible inside Apptainer until that bind was added)")
    return df
from hpc_crossenc_factvq import patch_tokens          # verified for all four encoders

def gpu_kmeans(X,K,seed,iters=40):
    """minibatch Lloyd's on GPU. X: (N,D) torch on DEV. Returns centroids (K,D)."""
    g=torch.Generator(device="cpu").manual_seed(seed)
    idx=torch.randperm(X.shape[0],generator=g)[:K]
    C=X[idx.to(X.device)].clone()
    for _ in range(iters):
        # assign in chunks to bound memory
        lab=torch.empty(X.shape[0],dtype=torch.long,device=X.device)
        for a in range(0,X.shape[0],200_000):
            b=X[a:a+200_000]
            lab[a:a+200_000]=torch.cdist(b,C).argmin(1)
        for k in range(K):
            m=lab==k
            if m.any(): C[k]=X[m].mean(0)
            else: C[k]=X[torch.randint(0,X.shape[0],(1,),generator=g).to(X.device)][0]  # dead-code reinit
    return C

def assign(X,C,chunk=200_000):
    lab=torch.empty(X.shape[0],dtype=torch.long,device=X.device)
    for a in range(0,X.shape[0],chunk):
        lab[a:a+chunk]=torch.cdist(X[a:a+chunk],C).argmin(1)
    return lab.cpu().numpy()

def hungarian_acc(a,b,K):
    M=np.zeros((K,K),int)
    for i,j in zip(a,b): M[i,j]+=1
    r,c=linear_sum_assignment(-M)
    return float(M[r,c].sum()/max(len(a),1))

def collect_tokens(enc,df,layer_frac=0.75):
    """patch tokens at ONE layer (fixed fraction of depth, not tuned) -> (n_frames*n_patch, D) on DEV."""
    m,tf,_=BUILDERS[enc]()
    feats=[]; t0=time.time()
    for b0 in range(0,len(df),BATCH):
        bs=df.iloc[b0:b0+BATCH]
        x=torch.stack([tf(Image.open(p).convert("RGB")) for p in bs["img"]]).to(DEV)
        t=patch_tokens(enc,m,x)                     # (B, L, Np, D)
        L=t.shape[1]; li=int(round(layer_frac*(L-1)))
        feats.append(t[:,li].reshape(-1,t.shape[-1]).float().cpu())
        if (b0//BATCH)%25==0: print(f"    {enc} {b0}/{len(df)} {time.time()-t0:.0f}s",flush=True)
    del m; torch.cuda.empty_cache() if DEV=="cuda" else None
    X=torch.cat(feats).to(DEV)
    X=(X-X.mean(0))/(X.std(0)+1e-6)                 # per-dim standardise; no PCA
    return X, li, L

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--K",type=int,default=16); ap.add_argument("--seeds",type=int,default=5)
    ap.add_argument("--cohort",default="impact",choices=["impact","clinical"])
    ap.add_argument("--check",action="store_true"); ap.add_argument("--fit-frames",type=int,default=FIT_FRAMES)
    a=ap.parse_args()
    df=cohort_table(a.cohort)
    rng=np.random.default_rng(0)
    # fit frames: sampled across fetuses so no fetus dominates the vocabulary
    per=max(1,a.fit_frames//df.nid.nunique())
    fit=df.groupby("nid",group_keys=False).apply(lambda g: g.sample(min(len(g),per),random_state=0))
    fit=fit.sample(min(len(fit),a.fit_frames),random_state=0).reset_index(drop=True)
    res={"K":a.K,"seeds":a.seeds,"kill_threshold_mean_AMI":KILL_AMI,
         "fit_frames":int(len(fit)),"fit_fetuses":int(fit.nid.nunique()),"all_frames":int(len(df)),
         "statistic":"Adjusted Mutual Information between two seeds' patch assignments (chance-corrected)",
         "note":"this measures representation stability only; no axis, outcome, GA or sex is touched"}
    print(f"  fit set: {len(fit)} frames / {fit.nid.nunique()} fetuses (of {len(df)} frames)",flush=True)
    if a.check:
        m,tf,_=BUILDERS["FetalCLIP"]()
        x=torch.stack([tf(Image.open(p).convert("RGB")) for p in fit["img"].head(2)]).to(DEV)
        t=patch_tokens("FetalCLIP",m,x)
        res["probe_FetalCLIP_shape"]=list(t.shape)
        print(f"  CHECK FetalCLIP patch tokens {tuple(t.shape)} (B,L,Np,D)",flush=True)
        json.dump(res,open(os.path.join(OUTP,f"seed_ceiling_K{a.K}{'' if a.cohort=='impact' else '_clin'}.json"),"w"),indent=2)
        print("  CHECK ONLY -> wrote json, no fitting",flush=True); return

    res["per_encoder"]={}
    for enc in ENCS:
        try:
            X,li,L=collect_tokens(enc,fit)
            labs=[]; cents=[]
            for s in range(a.seeds):
                C=gpu_kmeans(X,a.K,seed=s); cents.append(C.cpu().numpy()); labs.append(assign(X,C))
                used=len(np.unique(labs[-1]))
                print(f"    {enc} seed{s}: {used}/{a.K} codes used",flush=True)
            pair_ami=[]; pair_acc=[]
            for i in range(a.seeds):
                for j in range(i+1,a.seeds):
                    pair_ami.append(float(ami(labs[i],labs[j])))
                    pair_acc.append(hungarian_acc(labs[i],labs[j],a.K))
            # whole-IMAGE permutation reference (patches within a frame are correlated)
            npatch=len(labs[0])//len(fit)
            perm=rng.permutation(len(fit))
            shuf=labs[1].reshape(len(fit),npatch)[perm].reshape(-1)
            res["per_encoder"][enc]={"layer_index":int(li),"n_layers":int(L),
              "codes_used_per_seed":[int(len(np.unique(l))) for l in labs],
              "mean_pairwise_AMI":float(np.mean(pair_ami)),"min_pairwise_AMI":float(np.min(pair_ami)),
              "mean_hungarian_acc":float(np.mean(pair_acc)),
              "whole_image_permutation_AMI":float(ami(labs[0],shuf))}
            np.savez(os.path.join(OUT,f"wp2codes_{enc}_K{a.K}{'' if a.cohort=='impact' else '_clin'}.npz"),
                     codes=labs[0].reshape(len(fit),npatch).astype(np.int16),
                     centroids=cents[0], nid=fit["nid"].astype(str).values,
                     plane=fit["plane_prop"].values, ga=fit["ga_weeks_recovered"].values,
                     names=fit["new_filename"].values, layer_index=li, seeds_ami=np.array(pair_ami))
            print(f"  [{enc}] mean across-seed AMI {np.mean(pair_ami):.3f} (min {np.min(pair_ami):.3f}) | "
                  f"hungarian {np.mean(pair_acc):.3f} | whole-image-perm ref {ami(labs[0],shuf):.3f}",flush=True)
            del X; torch.cuda.empty_cache() if DEV=="cuda" else None
        except Exception as ex:
            import traceback; traceback.print_exc()
            res["per_encoder"][enc]={"error":f"{type(ex).__name__}: {ex}"}
    vals=[v["mean_pairwise_AMI"] for v in res["per_encoder"].values() if "mean_pairwise_AMI" in v]
    if vals:
        mean_ceiling=float(np.mean(vals)); res["mean_ceiling_across_encoders"]=mean_ceiling
        if mean_ceiling<KILL_AMI:
            res["VERDICT"]=(f"DESIGN FAILS -- mean within-encoder across-seed AMI {mean_ceiling:.3f} < {KILL_AMI}. "
              "Two fits of the SAME encoder differing only by seed do not agree, so cross-encoder agreement "
              "cannot be evidence about fetal ultrasound. STOP: do not run the WP2 annotation on this vocabulary.")
        elif mean_ceiling<0.70:
            res["VERDICT"]=(f"PROCEED WITH A BOUND -- ceiling {mean_ceiling:.3f}. Every cross-encoder agreement "
              "must be reported as a FRACTION of this ceiling and the bound stated in the text.")
        else:
            res["VERDICT"]=(f"CEILING ADEQUATE ({mean_ceiling:.3f}) -- cross-encoder agreement is interpretable; "
              "still report it as a fraction of this ceiling.")
        print(f"\n  MEAN CEILING {mean_ceiling:.3f}\n  {res['VERDICT']}",flush=True)
    json.dump(res,open(os.path.join(OUTP,f"seed_ceiling_K{a.K}{'' if a.cohort=='impact' else '_clin'}.json"),"w"),indent=2)
    print(f"saved out_probe/seed_ceiling_K{a.K}.json\nDONE",flush=True)

if __name__=="__main__": main()
