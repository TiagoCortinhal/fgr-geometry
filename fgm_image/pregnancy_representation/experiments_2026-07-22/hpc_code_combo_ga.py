#!/usr/bin/env python3
"""WHICH COMBINATION OF CODES PREDICTS GA? — honest sparse selection over the VQ vocabulary.

Motivation: single-code GA correlations were weak (|rho| 0.10-0.15), but that is a UNIVARIATE
test of one visual word. The real question is which COMBINATION of codes carries gestational
age, and how close that gets to the continuous full-embedding clock (r~0.435 on IMPACT).

STATISTICAL DISCIPLINE (the point of this script):
  * features  = per-image code-FREQUENCY histograms (fraction of the 256 patches per code),
                built from the SAVED code maps (no re-extraction).
  * CV        = GroupKFold by FETUS (no fetus in train and test).
  * selection = performed INSIDE the training folds ONLY (nested); the reported r is measured
                on held-out fetuses that had no influence on which codes were selected.
                Searching for the best combination and scoring it in-sample would inflate r.
  * stability = how often each code is selected across folds (5/5 = real, 1/5 = noise).
  * null      = the IDENTICAL selection pipeline run on SHUFFLED GA, so we know what r the
                selection procedure produces by chance.
  * blocks    = shared-only / each private-only / all, to localise WHICH DEPTH carries GA.

Reads out_usfmae/factvq_codes_<tag>.npz (shared+private) and/or vqvae_codes_<tag>.npz (flat).
USAGE: python hpc_code_combo_ga.py            # auto-finds the saved npz files
       python hpc_code_combo_ga.py --npz <path> --sizes 1,2,3,5,10,20,50
"""
import os, glob, json, argparse, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")

def hist_from_codes(codes,K):
    """(n_img,n_patch) code ids -> (n_img,K) frequency histogram."""
    n=codes.shape[0]; H=np.zeros((n,K),np.float32)
    for c in range(K): H[:,c]=(codes==c).mean(1)
    return H

def load_blocks(npz):
    z=np.load(npz,allow_pickle=True); keys=set(z.files)
    ga=z["ga"].astype(np.float32); names=z["names"].astype(str); blocks={}
    if "codes_shared" in keys:                     # factorised model
        Ks=z["cb_shared"].shape[0]; blocks["shared"]=hist_from_codes(z["codes_shared"],Ks)
        cp=z["codes_private"]; Kp=z["cb_private"].shape[1]
        for gi in range(cp.shape[1]): blocks[f"private_g{gi}"]=hist_from_codes(cp[:,gi,:],Kp)
    elif "codes" in keys:                           # flat model
        K=z["codebook"].shape[0]; blocks["flat"]=hist_from_codes(z["codes"],K)
    else: raise SystemExit(f"unrecognised npz keys: {sorted(keys)}")
    return blocks, ga, names

def fetus_ids(names):
    """map frame filenames -> fetus id via the index (grouping key for CV)."""
    df=pd.read_csv(INDEX)
    col="nid" if "nid" in df.columns else ("fetus_id" if "fetus_id" in df.columns else None)
    assert col, f"no fetus column in {INDEX}: {list(df.columns)[:12]}"
    m=dict(zip(df["new_filename"].astype(str),df[col].astype(str)))
    g=np.array([m.get(n,m.get(n.replace(".png",""),"NA")) for n in names])
    return g

def oof_ridge(X,y,groups,alpha=10.0):
    """out-of-fold predictions, fetus-grouped."""
    pred=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,groups=groups):
        sc=StandardScaler().fit(X[tr])
        pred[te]=Ridge(alpha=alpha).fit(sc.transform(X[tr]),y[tr]).predict(sc.transform(X[te]))
    return pred

def nested_forward(X,y,groups,sizes,alpha=10.0,cand_cap=64):
    """greedy forward selection done INSIDE each training fold; the returned r for each size is
    measured on the held-out fold. Also returns per-fold selections for stability."""
    sizes=sorted(sizes); maxk=max(sizes)
    pred={k:np.zeros(len(y)) for k in sizes}; picks=[]
    for tr,te in GroupKFold(5).split(X,groups=groups):
        sc=StandardScaler().fit(X[tr]); Xtr=sc.transform(X[tr]); Xte=sc.transform(X[te])
        ytr=y[tr]
        # cheap pre-screen INSIDE the fold: keep the top-|corr| candidates to bound cost
        r_scr=np.abs(np.array([pearsonr(Xtr[:,j],ytr)[0] if Xtr[:,j].std()>1e-9 else 0.0
                               for j in range(Xtr.shape[1])]))
        cand=list(np.argsort(-r_scr)[:cand_cap])
        sel=[]; inner=GroupKFold(3)
        gtr=groups[tr]
        while len(sel)<maxk and cand:
            best=None; bestr=-9
            for j in cand:
                cols=sel+[j]; p=np.zeros(len(ytr))
                for a,b in inner.split(Xtr[:,cols],groups=gtr):
                    p[b]=Ridge(alpha=alpha).fit(Xtr[np.ix_(a,cols)],ytr[a]).predict(Xtr[np.ix_(b,cols)])
                rr=pearsonr(p,ytr)[0]
                if rr>bestr: bestr=rr; best=j
            sel.append(best); cand.remove(best)
            if len(sel) in pred:
                cols=sel
                pred[len(sel)][te]=Ridge(alpha=alpha).fit(Xtr[:,cols],ytr).predict(Xte[:,cols])
        picks.append(list(sel))
    out={k:float(pearsonr(pred[k],y)[0]) for k in sizes}
    return out, picks

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",default=None); ap.add_argument("--sizes",default="1,2,3,5,10,20")
    ap.add_argument("--alpha",type=float,default=10.0); ap.add_argument("--cand-cap",type=int,default=64)
    a=ap.parse_args()
    sizes=[int(x) for x in a.sizes.split(",")]
    files=[a.npz] if a.npz else (sorted(glob.glob(os.path.join(OUT,"factvq_codes_*.npz")))
                                 +sorted(glob.glob(os.path.join(OUT,"vqvae_codes_*.npz"))))
    files=[f for f in files if f]; assert files, f"no saved code maps in {OUT}"
    allres={}
    for npz in files:
        blocks,ga,names=load_blocks(npz); groups=fetus_ids(names)
        ok=np.isfinite(ga)&(ga>=6)&(ga<=42)&(groups!="NA")
        ga=ga[ok]; groups=groups[ok]; blocks={k:v[ok] for k,v in blocks.items()}
        tag=os.path.basename(npz).replace(".npz","")
        print(f"\n=== {tag} | {len(ga)} frames {len(set(groups))} fetuses | GA {ga.min():.1f}-{ga.max():.1f} "
              f"| blocks { {k:v.shape[1] for k,v in blocks.items()} }",flush=True)
        res={"n_frames":int(len(ga)),"n_fetuses":int(len(set(groups))),
             "blocks":{k:int(v.shape[1]) for k,v in blocks.items()}}
        # 1) ALL codes per block, and all blocks together (the multivariate ceiling)
        for name,X in list(blocks.items())+[("ALL",np.concatenate(list(blocks.values()),1))]:
            r=float(pearsonr(oof_ridge(X,ga,groups,a.alpha),ga)[0])
            res.setdefault("all_codes_r",{})[name]=r
            print(f"  ALL codes [{name:11s}] d={X.shape[1]:4d}  OOF r={r:.3f}",flush=True)
        # 2) which COMBINATION — nested forward selection on the pooled feature set
        X=np.concatenate(list(blocks.values()),1)
        offs={}; o=0
        for k,v in blocks.items(): offs[k]=(o,o+v.shape[1]); o+=v.shape[1]
        rs,picks=nested_forward(X,ga,groups,sizes,a.alpha,a.cand_cap)
        res["combo_r_by_size"]=rs; res["fold_selections"]=[[int(j) for j in p] for p in picks]
        print("  COMBINATION (nested, held-out):",{k:round(v,3) for k,v in rs.items()},flush=True)
        # stability: how often each feature is selected across folds + which block it came from
        from collections import Counter
        cnt=Counter(j for p in picks for j in p)
        def blk(j):
            for k,(s,e) in offs.items():
                if s<=j<e: return f"{k}#{j-s}"
            return str(j)
        stab=[(blk(j),c) for j,c in cnt.most_common(12)]
        res["selection_stability"]=[[b,int(c)] for b,c in stab]
        print("  stability (feature, folds_selected/5):",stab[:8],flush=True)
        # 3) GA-SHUFFLE NULL through the identical pipeline
        rng=np.random.default_rng(0); ga_s=rng.permutation(ga)
        rs_null,_=nested_forward(X,ga_s,groups,[max(sizes)],a.alpha,a.cand_cap)
        r_all_null=float(pearsonr(oof_ridge(X,ga_s,groups,a.alpha),ga_s)[0])
        res["null_shuffled_GA"]={"all_codes_r":r_all_null,f"combo_r_k{max(sizes)}":rs_null[max(sizes)]}
        print(f"  NULL (shuffled GA): all-codes r={r_all_null:.3f} | combo k={max(sizes)} r={rs_null[max(sizes)]:.3f}",flush=True)
        allres[tag]=res
    allres["reference"]={"full_embedding_clock_IMPACT_r":0.435,
        "note":"IMPACT GA window is narrow (26-41wk); the clinical wide-GA FetalCLIP clock reached r=0.906 on 30257 clinical frames. Compare code-histogram r against the IMPACT 0.435 ceiling, NOT against 0.906."}
    p=os.path.join(OUTP,"code_combo_ga.json"); json.dump(allres,open(p,"w"),indent=2)
    print(f"\nsaved {p}",flush=True)

if __name__=="__main__": main()
