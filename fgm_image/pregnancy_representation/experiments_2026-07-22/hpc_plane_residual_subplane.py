#!/usr/bin/env python3
"""PLANE-RESIDUAL vocabulary + UNSUPERVISED SUBPLANE DISCOVERY (CPU, saved codes only).

Two questions, both label-light:

(A) PLANE-RESIDUAL. Imaging plane is the dominant organiser of this vocabulary (topic eta^2 up
    to 0.56), so it is the main nuisance axis. Rather than trust the kNN-propagated plane_prop
    label (audited as unreliable on clinical: 3 of 6 wrong at conf=1.00), remove plane
    LABEL-FREE-ish and ask what survives:
      raw            : code histogram as-is
      within-plane z : each code z-scored inside its plane_prop stratum (uses the label only as
                       a stratifier, not as a feature -- noise degrades power, cannot invent signal)
      plane-regressed: plane one-hot regressed out of every code column
    then measure held-out GA r for each. If GA survives plane removal, maturation is not merely
    a plane-composition artefact.

(B) SUBPLANE DISCOVERY. No brain-subplane labels exist anywhere in this dataset (only the 3
    coarse planes). So instead of conditioning on subplanes we cannot see, test whether the
    VOCABULARY DISCOVERS them: cluster CEREBRAL frames by code composition and ask whether the
    sub-groups are (i) internally coherent, (ii) better than a shuffled-histogram null at the
    same k, and (iii) exportable as exemplar frame lists so the sub-views can be eyeballed.
    Honest framing: coherent sub-groups are a HYPOTHESIS about sub-views, not a validated label
    -- confirmation requires looking at the exemplars.

Controls: GroupKFold-by-fetus everywhere; GA-shuffle null; matched-capacity shuffled-histogram
null for the clustering; silhouette + dip for continuum-vs-clusters.

USAGE: python hpc_plane_residual_subplane.py [--npz <factvq_codes_*.npz>] [--k 2,3,4,5,6]
Outputs: out_probe/plane_residual_subplane.json (+ exemplar frame lists per discovered group)
"""
import os, glob, json, argparse, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")

def hist_from_codes(codes,K):
    n=codes.shape[0]; H=np.zeros((n,K),np.float32)
    for c in range(K): H[:,c]=(codes==c).mean(1)
    return H

def load(npz):
    z=np.load(npz,allow_pickle=True); parts=[]; offs={}; o=0
    Ks=z["cb_shared"].shape[0]; parts.append(hist_from_codes(z["codes_shared"],Ks)); offs["shared"]=(o,o+Ks); o+=Ks
    cp=z["codes_private"]; Kp=z["cb_private"].shape[1]
    for gi in range(cp.shape[1]):
        parts.append(hist_from_codes(cp[:,gi,:],Kp)); offs[f"private_g{gi}"]=(o,o+Kp); o+=Kp
    H=np.concatenate(parts,1)
    df=pd.read_csv(INDEX); key="nid" if "nid" in df.columns else "fetus_id"
    m=dict(zip(df["new_filename"].astype(str),df[key].astype(str)))
    names=z["names"].astype(str); fid=np.array([m.get(n,m.get(n.replace(".png",""),"NA")) for n in names])
    ga=z["ga"].astype(np.float32); plane=z["plane"].astype(str)
    ok=(fid!="NA")&np.isfinite(ga)&(ga>=6)&(ga<=42)
    return H[ok],ga[ok],plane[ok],fid[ok],names[ok],offs

def oof_r(X,y,g,alpha=10.0):
    p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,groups=g):
        sc=StandardScaler().fit(X[tr]); p[te]=Ridge(alpha=alpha).fit(sc.transform(X[tr]),y[tr]).predict(sc.transform(X[te]))
    return float(pearsonr(p,y)[0])

def within_plane_z(H,plane):
    Z=H.copy()
    for pl in np.unique(plane):
        m=plane==pl
        if m.sum()<10: continue
        Z[m]=(H[m]-H[m].mean(0))/(H[m].std(0)+1e-8)
    return Z

def plane_regressed(H,plane):
    D=pd.get_dummies(plane).values.astype(np.float32)
    B=np.linalg.lstsq(D,H,rcond=None)[0]
    return H-D@B

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",default=None); ap.add_argument("--k",default="2,3,4,5,6")
    a=ap.parse_args(); ks=[int(x) for x in a.k.split(",")]
    npz=a.npz or sorted(glob.glob(os.path.join(OUT,"factvq_codes_*.npz")))[0]
    H,ga,plane,fid,names,offs=load(npz)
    tag=os.path.basename(npz).replace("factvq_codes_","").replace(".npz","")
    print(f"[{tag}] {H.shape} frames x codes | {len(set(fid))} fetuses | GA {ga.min():.1f}-{ga.max():.1f} "
          f"| planes {dict(zip(*np.unique(plane,return_counts=True)))}",flush=True)
    res={"tag":tag,"n_frames":int(H.shape[0]),"n_codes":int(H.shape[1]),"n_fetuses":int(len(set(fid))),
         "planes":{k:int(v) for k,v in zip(*np.unique(plane,return_counts=True))}}
    # ---------- (A) plane-residual ----------
    rng=np.random.default_rng(0)
    variants={"raw":H,"within_plane_z":within_plane_z(H,plane),"plane_regressed":plane_regressed(H,plane)}
    res["A_plane_residual"]={}
    for nm,X in variants.items():
        r=oof_r(X,ga,fid); r_null=oof_r(X,rng.permutation(ga),fid)
        res["A_plane_residual"][nm]={"GA_r":r,"GA_shuffle_null":r_null}
        print(f"  A {nm:16s} held-out GA r={r:+.3f} | GA-shuffle null {r_null:+.3f}",flush=True)
    keep=res["A_plane_residual"]["plane_regressed"]["GA_r"]/max(res["A_plane_residual"]["raw"]["GA_r"],1e-9)
    res["A_fraction_of_GA_surviving_plane_removal"]=float(keep)
    print(f"  A -> {100*keep:.0f}% of the raw GA correlation survives regressing plane out",flush=True)
    # ---------- (B) subplane discovery on cerebral ----------
    ce=plane=="cerebral"
    res["B_subplane_discovery"]={"n_cerebral":int(ce.sum()),"note":
      "NO brain-subplane labels exist in this dataset; these groups are a HYPOTHESIS about "
      "sub-views discovered from code composition, not validated labels. Confirm by viewing exemplars."}
    if ce.sum()>=200:
        Xc=within_plane_z(H,plane)[ce]                  # remove the coarse-plane mean first
        Xs=Xc.copy()
        for j in range(Xs.shape[1]): Xs[:,j]=rng.permutation(Xs[:,j])   # matched-capacity null
        gac=ga[ce]; fidc=fid[ce]; nmc=names[ce]
        for k in ks:
            lab=KMeans(k,n_init=10,random_state=0).fit_predict(Xc)
            labn=KMeans(k,n_init=10,random_state=0).fit_predict(Xs)
            sil=float(silhouette_score(Xc,lab)); siln=float(silhouette_score(Xs,labn))
            # do groups differ in GA? (a sub-view should NOT be a GA proxy if it is anatomical)
            gmeans=[float(gac[lab==c].mean()) for c in range(k)]
            sizes=[int((lab==c).sum()) for c in range(k)]
            res["B_subplane_discovery"][f"k{k}"]={"silhouette":sil,"silhouette_shuffled_null":siln,
                "excess_over_null":float(sil-siln),"group_sizes":sizes,"group_GA_mean":gmeans}
            print(f"  B k={k} silhouette={sil:.3f} (shuffled null {siln:.3f}, excess {sil-siln:+.3f}) "
                  f"sizes={sizes} GA means={[round(x,1) for x in gmeans]}",flush=True)
        # exemplars at the k with the largest excess over null, for eyeballing
        best=max(ks,key=lambda k: res["B_subplane_discovery"][f"k{k}"]["excess_over_null"])
        lab=KMeans(best,n_init=10,random_state=0).fit_predict(Xc)
        cen=KMeans(best,n_init=10,random_state=0).fit(Xc).cluster_centers_
        ex={}
        for c in range(best):
            idx=np.where(lab==c)[0]
            d=((Xc[idx]-cen[c])**2).sum(1)
            ex[str(c)]=[str(x) for x in nmc[idx[np.argsort(d)[:12]]]]   # 12 most typical frames
        res["B_subplane_discovery"]["exemplars_best_k"]={"k":int(best),"frames_per_group":ex}
        print(f"  B exemplars saved for k={best} (12 most typical frames per group) -> view these to "
              f"judge whether the groups are real sub-views",flush=True)
    else:
        print("  B skipped: too few cerebral frames",flush=True)
    p=os.path.join(OUTP,"plane_residual_subplane.json"); json.dump(res,open(p,"w"),indent=2)
    print(f"saved {p}\nDONE",flush=True)

if __name__=="__main__": main()
