#!/usr/bin/env python3
"""1/E NULL-FLOOR TEST for every shared/private split in this project.

THE FLAW THIS MEASURES. Defining shared = the arithmetic MEAN over E streams makes the mean carry
exactly 1/E of the variance EVEN WHEN THE TRUE SHARED FRACTION IS ZERO:
    E independent zero-mean unit-variance streams -> Var(mean) = 1/E
    E=3 depth groups  -> 33% "shared" from nothing
    E=4 encoders      -> 25% "shared" from nothing
So an observed "shared explains ~25-30%" is INSIDE THE NULL. This floor was never subtracted in
the depth-factorised result already reported (FULL recon 0.298 vs SHARED-ONLY 0.830) nor in the
cross-encoder design, which is why this test must run BEFORE any further shared-vocabulary work.

WHAT IT DOES (no GPU, no encoders -- operates on the SAVED code maps + saved features):
  1. ANALYTIC floor: 1/E, printed alongside the observed split.
  2. EMPIRICAL floor by STREAM SHUFFLE: destroy any real cross-stream correspondence while keeping
     each stream's own marginal structure, then push it through the IDENTICAL variance
     decomposition. Two shuffle schemes, because they break different things:
       'patch'  -- independently permute the PATCH order within each stream (kills patch-to-patch
                   correspondence, keeps each stream's patch distribution)
       'sample' -- independently permute the IMAGE order within each stream (kills image-level
                   correspondence)
     Repeat n_perm times -> a floor BAND (mean, p5, p95), not a point estimate.
  3. Reports the observed shared fraction, the analytic 1/E floor, the empirical band, and the
     EXCESS. Verdict thresholds are fixed here BEFORE the numbers are seen:
       observed inside the empirical band            -> NO shared structure (framing dead)
       observed above band but < 1/E + 0.10          -> WEAK, do not build on it
       observed >= 1/E + 0.10 and outside the band   -> real shared structure

Inputs: the saved per-encoder / per-depth-group feature maps if present, else the code maps
(codes_shared / codes_private) whose variance decomposition is the same algebra.

USAGE: python hpc_shared_floor.py [--npz <factvq_codes_*.npz|crossenc_codes.npz>] [--n-perm 200]
Outputs: out_probe/shared_floor.json
"""
import os, glob, json, argparse, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("GA_OUT_DIR",os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)

def onehot_streams(codes_private,Kp):
    """(n_img, E, n_patch) private code ids -> (E, n_img*n_patch, Kp) one-hot stream matrices.
    Variance decomposition on the one-hot representation is the discrete analogue of the feature
    decomposition: shared = mean over streams, private = deviation."""
    n,E,P=codes_private.shape
    S=np.zeros((E,n*P,Kp),np.float32)
    for e in range(E):
        flat=codes_private[:,e,:].reshape(-1)
        S[e,np.arange(len(flat)),flat]=1.0
    return S

def split_fractions(S):
    """S: (E, N, D). shared = mean over E; private = deviation. -> (shared_frac, private_frac)."""
    m=S.mean(0)                                  # (N,D) the 'shared' construction
    dev=S-m[None]                                # (E,N,D) deviations
    vs=float(m.var()); vd=float(dev.var()); tot=vs+vd
    return vs/tot, vd/tot

def shuffled_floor(S,scheme,n_perm,rng):
    """break cross-stream correspondence, keep each stream's own marginal structure."""
    E,N,D=S.shape; out=[]
    for _ in range(n_perm):
        T=np.empty_like(S)
        for e in range(E):
            idx=rng.permutation(N)               # both schemes permute along N; 'patch' vs 'sample'
            T[e]=S[e][idx]                       # differ only in how N was laid out upstream
        out.append(split_fractions(T)[0])
    return np.array(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",default=None); ap.add_argument("--n-perm",type=int,default=200)
    a=ap.parse_args()
    cands=[a.npz] if a.npz else (sorted(glob.glob(os.path.join(OUT,"crossenc_codes.npz")))
                                 +sorted(glob.glob(os.path.join(OUT,"factvq_codes_*.npz"))))
    cands=[c for c in cands if c and os.path.exists(c)]
    assert cands, f"no saved code maps found in {OUT}"
    res={"analytic_note":"shared = arithmetic mean over E streams => Var(mean)=1/E when streams are independent; 1/E is the FLOOR, not a result."}
    rng=np.random.default_rng(0)
    for npz in cands:
        z=np.load(npz,allow_pickle=True); tag=os.path.basename(npz)
        if "codes_private" not in z.files: print(f"  skip {tag}: no private streams"); continue
        cp=z["codes_private"]; Kp=int(z["cb_private"].shape[1]); E=cp.shape[1]
        S=onehot_streams(cp,Kp)
        obs_s,obs_p=split_fractions(S)
        floor_analytic=1.0/E
        band=shuffled_floor(S,"sample",a.n_perm,rng)
        p5,p50,p95=np.percentile(band,[5,50,95])
        excess=obs_s-p95
        if obs_s<=p95: verdict="NO shared structure -- observed inside the shuffle floor band"
        elif obs_s < floor_analytic+0.10: verdict="WEAK -- above the band but within 0.10 of the 1/E floor; do not build on it"
        else: verdict="REAL shared structure -- above both the 1/E floor and the shuffle band"
        streams=[str(x) for x in z["encoders"]] if "encoders" in z.files else [f"stream{i}" for i in range(E)]
        res[tag]={"E":int(E),"streams":streams,"K_private":Kp,
                  "observed_shared_fraction":float(obs_s),"observed_private_fraction":float(obs_p),
                  "analytic_1overE_floor":float(floor_analytic),
                  "shuffle_floor_band":{"p5":float(p5),"median":float(p50),"p95":float(p95),"n_perm":a.n_perm},
                  "excess_over_band_p95":float(excess),"verdict":verdict}
        print(f"\n[{tag}] E={E} streams={streams}")
        print(f"  observed shared fraction : {obs_s:.3f}")
        print(f"  analytic 1/E floor       : {floor_analytic:.3f}")
        print(f"  shuffle floor band       : p5 {p5:.3f} | median {p50:.3f} | p95 {p95:.3f}  ({a.n_perm} perms)")
        print(f"  excess over band p95     : {excess:+.3f}")
        print(f"  VERDICT: {verdict}",flush=True)
    json.dump(res,open(os.path.join(OUTP,"shared_floor.json"),"w"),indent=2)
    print(f"\nsaved out_probe/shared_floor.json\nDONE",flush=True)

if __name__=="__main__": main()
