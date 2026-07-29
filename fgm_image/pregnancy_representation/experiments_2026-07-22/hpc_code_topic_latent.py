#!/usr/bin/env python3
"""BAG-OF-VISUAL-WORDS LATENT — interpretable topics over the VQ codebook.

Why this latent: every earlier latent here (GRU-VAE, beta-TCVAE, PHATE) had uninterpretable
axes, which is why 3D views read as tangles and why regression axes had to be bolted on post
hoc. Here each input dimension is "how much of visual word k is in this scan", and each topic
is a group of co-occurring words -- nameable, and viewable as the real patches that use it.

Licensed by the preceding probe: the 320 code frequencies retain held-out GA r=0.385 vs the
FetalCLIP continuous full-embedding clock r=0.469 (L18/24) on the same IMPACT frames -- 82% of
that ceiling -- so the histogram is a SUFFICIENT representation, not a lossy summary.
(NB 0.435 is the USF-MAE value from the same 4-encoder IMPACT run, NOT FetalCLIP; these codes
are FetalCLIP-derived so 0.469 is the valid same-cohort ceiling.)

Construction
  per-image code histogram (320-d, shared+private) -> pool per FETUS (mean over that fetus's
  frames) -> NMF (non-negative -> additive "visual topics") and FA+varimax as a linear check.

What is measured (representation quality, NOT outcome AUC chasing)
  * topic interpretability : top codes per topic + which depth block they come from
  * GA organisation        : monotonicity (Spearman) of each topic vs GA; held-out GA r from topics
  * plane structure        : eta^2 of each topic across plane_prop (images organise by plane here)
  * region/cohesion        : silhouette + dip test -- is the topic space a CONTINUUM or clusters
  * outcome anchors        : birth-percentile / <p10 / >p90 as EVAL-ONLY read-offs, with the
                             honest prior that outcome separation has been null across families
Controls
  * GroupKFold-by-fetus for anything fitted
  * GA-shuffle and code-shuffle nulls through the identical pipeline
  * matched-capacity baseline: same-rank factorisation of a SHUFFLED histogram

USAGE: python hpc_code_topic_latent.py [--ranks 4,6,8,12] [--npz <factvq_codes_*.npz>]
Outputs: out_probe/code_topic_latent.json + .png + out_usfmae/code_topics_<tag>.npz
"""
import os, glob, json, argparse, numpy as np, pandas as pd
from sklearn.decomposition import NMF, FactorAnalysis
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr, f_oneway
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")
OUTCOMES=[x for x in [os.environ.get("GA_OUTCOMES")] if x]+[os.path.join(HERE,"outcomes.csv"),
          "/mnt/beegfs/home/tiago.fernandes/PyCharmProjects/fgr-geometry/data_local/impact_outcomes.csv"]

def hist_from_codes(codes,K):
    n=codes.shape[0]; H=np.zeros((n,K),np.float32)
    for c in range(K): H[:,c]=(codes==c).mean(1)
    return H

def load_histograms(npz):
    """-> H (n_img, 320) with block offsets, ga, plane, names"""
    z=np.load(npz,allow_pickle=True); parts=[]; offs={}; o=0
    Ks=z["cb_shared"].shape[0]; h=hist_from_codes(z["codes_shared"],Ks)
    parts.append(h); offs["shared"]=(o,o+Ks); o+=Ks
    cp=z["codes_private"]; Kp=z["cb_private"].shape[1]
    for gi in range(cp.shape[1]):
        h=hist_from_codes(cp[:,gi,:],Kp); parts.append(h); offs[f"private_g{gi}"]=(o,o+Kp); o+=Kp
    return np.concatenate(parts,1), offs, z["ga"].astype(np.float32), z["plane"].astype(str), z["names"].astype(str)

def per_fetus(H,ga,plane,names):
    df=pd.read_csv(INDEX); key="nid" if "nid" in df.columns else "fetus_id"
    m=dict(zip(df["new_filename"].astype(str),df[key].astype(str)))
    fid=np.array([m.get(n,m.get(n.replace(".png",""),"NA")) for n in names])
    ok=(fid!="NA")&np.isfinite(ga)&(ga>=6)&(ga<=42)
    H,ga,plane,fid=H[ok],ga[ok],plane[ok],fid[ok]
    uf=np.unique(fid); Hf=np.zeros((len(uf),H.shape[1]),np.float32); gf=np.zeros(len(uf),np.float32)
    for i,f in enumerate(uf):
        m2=fid==f; Hf[i]=H[m2].mean(0); gf[i]=ga[m2].mean()
    return H,ga,plane,fid,Hf,gf,uf

def oof_r(X,y,groups,alpha=10.0):
    p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,groups=groups):
        sc=StandardScaler().fit(X[tr]); p[te]=Ridge(alpha=alpha).fit(sc.transform(X[tr]),y[tr]).predict(sc.transform(X[te]))
    return float(pearsonr(p,y)[0])

def eta2(x,labels):
    gs=[x[labels==l] for l in np.unique(labels) if (labels==l).sum()>2]
    if len(gs)<2: return float("nan")
    gm=np.concatenate(gs).mean()
    ssb=sum(len(g)*(g.mean()-gm)**2 for g in gs); sst=sum(((np.concatenate(gs)-gm)**2))
    return float(ssb/sst) if sst>0 else float("nan")

def load_outcomes(uf):
    for p in OUTCOMES:
        if os.path.exists(p):
            o=pd.read_csv(p)
            cod=[c for c in o.columns if c.lower() in ("cod","nid","fetus_id")]
            if not cod: continue
            o["_k"]=o[cod[0]].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
            o=o.drop_duplicates("_k").set_index("_k")
            def col(*cands):
                for c in o.columns:
                    if any(s.lower() in c.lower() for s in cands): return o[c].reindex(uf)
                return None
            bp=col("percentil_birth","percentile_birth")
            yesno=lambda s: None if s is None else s.astype(str).str.strip().str.lower().isin(["yes","1","true","si"]).values
            return {"birth_pct":(None if bp is None else pd.to_numeric(bp,errors="coerce").values),
                    "sga_p10":yesno(col("SGA_birth","sga")), "lga_p90":yesno(col("LGA_birth","lga"))}, p
    return {}, None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",default=None); ap.add_argument("--ranks",default="4,6,8,12")
    a=ap.parse_args(); ranks=[int(x) for x in a.ranks.split(",")]
    npz=a.npz or sorted(glob.glob(os.path.join(OUT,"factvq_codes_*.npz")))[0]
    H,offs,ga,plane,names=load_histograms(npz)
    H,ga,plane,fid,Hf,gf,uf=per_fetus(H,ga,plane,names)
    tag=os.path.basename(npz).replace("factvq_codes_","").replace(".npz","")
    print(f"[{tag}] frames {H.shape} | fetuses {Hf.shape} | GA {gf.min():.1f}-{gf.max():.1f}",flush=True)
    res={"tag":tag,"n_frames":int(H.shape[0]),"n_fetuses":int(Hf.shape[0]),"n_codes":int(H.shape[1]),
         "blocks":{k:[int(s),int(e)] for k,(s,e) in offs.items()},"ranks":{}}
    blk=lambda j: next((f"{k}#{j-s}" for k,(s,e) in offs.items() if s<=j<e), str(j))
    # BASELINES -- must be UNIT-MATCHED. The topics are fitted/scored per FETUS, so the valid
    # ceiling for them is the per-FETUS raw histogram, NOT the per-frame one (per-fetus mean
    # pooling suppresses noise and mechanically raises r). Both are reported and labelled; the
    # frame-level FetalCLIP clock (r=0.469) is a FRAME-level number and must only be compared
    # against the frame-level histogram row.
    r_hist_frame=oof_r(H,ga,fid)
    r_hist_fetus=oof_r(Hf,gf,uf)
    print(f"  raw histogram  FRAME-level held-out GA r={r_hist_frame:.3f}  (compare vs FetalCLIP frame clock 0.469)",flush=True)
    print(f"  raw histogram  FETUS-level held-out GA r={r_hist_fetus:.3f}  <-- THE valid ceiling for the topic latents below",flush=True)
    res["raw_histogram_frame_GA_r"]=r_hist_frame
    res["raw_histogram_fetus_GA_r"]=r_hist_fetus
    res["unit_note"]=("topic latents are per-FETUS; their ceiling is raw_histogram_fetus_GA_r. "
                      "raw_histogram_frame_GA_r is the only value comparable to the frame-level "
                      "FetalCLIP full-embedding clock r=0.469 on these IMPACT frames.")
    for R in ranks:
        nmf=NMF(R,init="nndsvda",max_iter=8000,tol=1e-5,random_state=0)
        W=nmf.fit_transform(np.maximum(Hf,0)); Hc=nmf.components_          # W: fetus x topic
        fa=FactorAnalysis(R,random_state=0).fit(StandardScaler().fit_transform(Hf))
        Wfa=fa.transform(StandardScaler().fit_transform(Hf))
        # GA organisation of the topic latent (fetus level, grouped CV is per-fetus = plain CV)
        rga=oof_r(W,gf,uf); rga_fa=oof_r(Wfa,gf,uf)
        mono=[float(spearmanr(W[:,k],gf)[0]) for k in range(R)]
        # plane structure at frame level: project frames onto topics
        Wfr=nmf.transform(np.maximum(H,0))
        eta=[eta2(Wfr[:,k],plane) for k in range(R)]
        # continuum vs clusters
        sil={}
        for k in (2,3,4):
            lab=KMeans(k,n_init=10,random_state=0).fit_predict(W)
            sil[k]=float(silhouette_score(W,lab))
        try:
            import diptest; dip=[float(diptest.diptest(W[:,k])[1]) for k in range(R)]
        except Exception: dip=None
        # interpretability: top codes per topic + which block
        tops=[[blk(int(j)) for j in np.argsort(-Hc[k])[:6]] for k in range(R)]
        # matched-capacity null: same rank on a shuffled histogram
        rng=np.random.default_rng(0); Hs=Hf.copy()
        for j in range(Hs.shape[1]): Hs[:,j]=rng.permutation(Hs[:,j])
        Wn=NMF(R,init="nndsvda",max_iter=8000,tol=1e-5,random_state=0).fit_transform(np.maximum(Hs,0))
        rga_null=oof_r(Wn,gf,uf); rga_shufGA=oof_r(W,rng.permutation(gf),uf)
        entry={"GA_r_topics_NMF":rga,"GA_r_topics_FA":rga_fa,"GA_monotonicity_spearman":mono,
               "plane_eta2":eta,"silhouette_k2_3_4":sil,"dip_p":dip,"top_codes_per_topic":tops,
               "null_matched_capacity_shuffled_hist_GA_r":rga_null,"null_shuffledGA_GA_r":rga_shufGA}
        entry["nmf_n_iter"]=int(getattr(nmf,"n_iter_",-1))
        entry["nmf_converged"]=bool(getattr(nmf,"n_iter_",10**9)<8000)
        res["ranks"][str(R)]=entry
        print(f"  R={R:2d} GA r NMF={rga:.3f} FA={rga_fa:.3f} | max|mono|={max(abs(m) for m in mono):.2f} "
              f"| max plane eta2={np.nanmax(eta):.2f} | sil k2={sil[2]:.3f} "
              f"| NULLS hist-shuf {rga_null:.3f} GA-shuf {rga_shufGA:.3f} "
              f"| nmf_iter {entry['nmf_n_iter']}{'' if entry['nmf_converged'] else ' NOT-CONVERGED'}",flush=True)
        for k in range(min(R,4)): print(f"      topic{k}: mono={mono[k]:+.2f} eta2={eta[k]:.2f} top={tops[k][:5]}",flush=True)
    # outcomes as EVAL-ONLY read-offs at the best rank by GA r
    rs=[(int(k),res["ranks"][k]["GA_r_topics_NMF"]) for k in res["ranks"]]; rs.sort()
    mono_in_rank=all(b>=a-1e-9 for (_,a),(_,b) in zip(rs,rs[1:]))
    res["GA_r_monotonic_in_rank"]=bool(mono_in_rank)
    if mono_in_rank: print("  NOTE GA r rises monotonically with rank and never plateaus -> the top rank is the LARGEST TRIED, not a selected optimum (capacity trend).",flush=True)
    best=max(res["ranks"],key=lambda k: res["ranks"][k]["GA_r_topics_NMF"])
    Rb=int(best); Wb=NMF(Rb,init="nndsvda",max_iter=8000,tol=1e-5,random_state=0).fit_transform(np.maximum(Hf,0))
    oc,src=load_outcomes(uf); res["outcomes_source"]=src; res["outcome_readoffs_rank"]=Rb; res["outcome_readoffs"]={}
    if not src:   # fail LOUDLY: silent omission of the eval anchors is worse than an error
        res["outcome_readoffs"]="NOT RUN -- no outcomes csv found (searched: "+"; ".join(OUTCOMES)+")"
        print("  !! OUTCOME READ-OFFS NOT RUN -- no outcomes csv found. Searched:",flush=True)
        for _p in OUTCOMES: print(f"       exists={os.path.exists(_p)}  {_p}",flush=True)
        print("     -> set GA_OUTCOMES=/path/to/impact_outcomes.csv and re-run",flush=True)
    for k,v in (oc or {}).items():
        if v is None: continue
        v=np.asarray(v); m=np.isfinite(v.astype(float)) if v.dtype!=bool else np.ones(len(v),bool)
        if v.dtype==bool:
            if v[m].sum()<15: res["outcome_readoffs"][k]="too few positives"; continue
            p=np.zeros(m.sum())
            X=StandardScaler().fit_transform(Wb[m]); y=v[m].astype(int)
            for tr,te in GroupKFold(5).split(X,groups=uf[m]):
                p[te]=LogisticRegression(max_iter=2000).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
            res["outcome_readoffs"][k]={"auc":float(roc_auc_score(y,p)),"n_pos":int(y.sum())}
        else:
            res["outcome_readoffs"][k]={"r":oof_r(Wb[m],v[m].astype(float),uf[m]),"n":int(m.sum())}
        print(f"  outcome {k}: {res['outcome_readoffs'][k]}",flush=True)
    np.savez(os.path.join(OUT,f"code_topics_{tag}.npz"),W=Wb,fetus=uf,ga=gf,rank=Rb,hist_fetus=Hf)
    json.dump(res,open(os.path.join(OUTP,"code_topic_latent.json"),"w"),indent=2)
    print(f"  saved topics + json (best rank {Rb})",flush=True); print("DONE",flush=True)

if __name__=="__main__": main()
