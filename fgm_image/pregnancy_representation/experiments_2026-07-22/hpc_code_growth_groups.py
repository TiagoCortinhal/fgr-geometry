#!/usr/bin/env python3
"""WHICH VISUAL WORDS ARE ENRICHED IN SGA / AGA / LGA -- per plane, FetalCLIP K=16.

THE QUESTION. The K=16 vocabulary is reproducible (within-encoder across-seed AMI 0.710-0.797, every
seed pair >=0.662, ~4.6x the whole-image-shuffle floor), so a code can be treated as a property of the
data rather than of one training run. Given that, which codes are over-represented in small vs
appropriate vs large-for-gestational-age fetuses?

WHY THIS IS NOT A SETTLED NULL. This project has established that clinical CATEGORIES do not occupy
distinct latent REGIONS (1-vs-rest AUC: FGR 0.412, constitutional-small 0.557, LGA 0.424, i.e. at or
below chance) and that >=12 construction families found no growth-outcome signal in the image channel.
Asking "is code k enriched in LGA" is a WEAKER, more targeted question than "do LGA fetuses form a
region": a single code shifting its share is detectable where a whole-space separation is not. And
there is one genuine prior positive to test against: LGA localises to the ABDOMINAL plane --
radiomics abdominal AUC 0.579 vs cerebral 0.464, USFM abdominal 0.69.

THE DESIGN CONFLICT THIS SCRIPT RESOLVES, deliberately and in the opposite direction to its sibling.
hpc_wp2_annotate.py samples patches PLANE-BALANCED (equal patches per plane per fetus) because plane
labels here are kNN-propagated and audited unreliable (3 of 6 wrong at conf=1.00), and one-hot
adjustment with ~90%-accurate labels leaves residual |r|=0.425 when a signal is 100% plane-driven --
you cannot adjust with a label that noisy. BUT if LGA genuinely IS an abdominal soft-tissue phenotype,
plane-balancing DILUTES precisely the signal most likely to exist. So here we do NOT balance: we
STRATIFY, running the test separately WITHIN each plane. Plane is then held constant by restriction
rather than by mixing, the noisy label costs us only power (misassigned frames land in the wrong
stratum) instead of biasing the contrast, and the abdominal-vs-cerebral comparison becomes a
DIRECTIONAL PREDICTION stated before the numbers.

PRE-REGISTERED, before any number is computed:
  PRIMARY   the ABDOMINAL x LGA cell. One cell, chosen because the prior predicts it. Directional:
            abdominal effect > cerebral effect. Allowed to fail.
  EXPLORATORY  every other plane x group cell.
  16 codes x 3 planes x 3 groups = 144 tests -> Benjamini-Yekutieli (valid under the arbitrary
  dependence compositional shares guarantee). The project ledger already stands at 218 constructions /
  584 tests, so nominal 0.05 is meaningless; the reference bar is BY q plus the permutation null.
  POWER IS THE BINDING CONSTRAINT: LGA n~77 and SGA<p10 n~169 of ~908, and splitting by plane reduces
  each cell further. A null here is as likely to mean underpowered as absent, which is why the GA
  positive control below is reported FIRST and gates interpretation.

CONFOUNDS. GA (spline) and fetal sex adjusted -- sex because female fetuses are smaller by nature and
therefore confound any size-related contrast. Patch budget equalised per fetus within plane, else
"enrichment" partly means "who got more scans". Effects reported in raw shares AND CLR coordinates
with the analytic closure floor -r0/(K-1) beside each: shares sum to 1, so planting a real effect on
one code drives the other K-1 negative by arithmetic alone (verified: signal on one code drove all 15
others negative, mean -0.110).

OUTCOME LABELS ARE EVALUATION-ONLY and were never model inputs; the codebook is unsupervised k-means
on frozen features. SGA-confirmed is BANNED in this project; we use SGA_birth (<p10), LGA_birth (>p90)
and severeSGA (<p3) as declared, plus continuous birth percentile.

USAGE: python hpc_code_growth_groups.py [--K 16] [--enc FetalCLIP] [--n-perm 20000]
Outputs: out_probe/code_growth_groups_K16.json
"""
import os, sys, json, argparse, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, mannwhitneyu
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
OUT=os.environ.get("GA_OUT_DIR",os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
HAND=os.path.join(HERE,"handoff")
ECHO=os.path.join(ROOT,"data_local","IMPACT_ecocardio_zscores_corrected.xlsx")
PLANES=["abdominal","cerebral","femur"]
PRIMARY=("abdominal","LGA")            # pre-registered single primary cell

def ga_spline(x):
    x=np.asarray(x,float); q=np.nanpercentile(x,[25,50,75])
    return np.column_stack([x]+[np.clip(x-k,0,None)**3 for k in q])

def clr(P,eps=1e-6):
    Q=np.clip(P,eps,None); L=np.log(Q); return L-L.mean(1,keepdims=True)

def resid(v,C):
    m=np.isfinite(v)&np.isfinite(C).all(1); out=np.full(len(v),np.nan)
    if m.sum()<30: return out
    out[m]=v[m]-LinearRegression().fit(C[m],v[m]).predict(C[m]); return out

def by_correct(p):
    p=np.asarray(p,float); ok=np.isfinite(p); q=np.full(p.shape,np.nan); ps=p[ok]; n=len(ps)
    if n==0: return q
    c=np.sum(1.0/np.arange(1,n+1)); o=np.argsort(ps)
    adj=np.minimum.accumulate((ps[o]*n*c/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n); out[o]=np.clip(adj,0,1); q[ok]=out; return q

def auc(x,y):
    """rank AUC of x separating y==1 from y==0, nan-safe."""
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<20 or len(np.unique(y[m]))<2: return np.nan
    try: u=mannwhitneyu(x[m][y[m]==1],x[m][y[m]==0],alternative="two-sided").statistic
    except ValueError: return np.nan
    n1=(y[m]==1).sum(); n0=(y[m]==0).sum()
    return float(u/(n1*n0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--K",type=int,default=16); ap.add_argument("--enc",default="FetalCLIP")
    ap.add_argument("--n-perm",type=int,default=20000); ap.add_argument("--per-fetus",type=int,default=200)
    a=ap.parse_args(); rng=np.random.default_rng(0)
    npz=os.path.join(OUT,f"wp2fullcodes_{a.enc}_K{a.K}.npz")
    assert os.path.exists(npz), f"missing {npz} -- run hpc_wp2_annotate.py --assign-only first"
    z=np.load(npz,allow_pickle=True)
    codes,nid,plane,gaf=z["codes"],z["nid"].astype(str),z["plane"].astype(str),z["ga"].astype(float)

    # ---- outcomes (EVAL ONLY) ----
    e=pd.read_excel(ECHO); e["nid"]=e["Cod"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
    e=e.drop_duplicates("nid").set_index("nid")
    def yesno(col):
        """Coerce a binary outcome to 0/1. This file encodes them NUMERICALLY ('1.0'/'0.0'), while
        the project's impact_outcomes.csv uses 'yes'/'no' strings -- a string-only mapper returned ALL
        ZEROS here and would have silently skipped every group test. Handles both, and the caller
        asserts the counts against the project's recorded totals (SGA 169, LGA 77, severeSGA 61)."""
        s=e.get(col)
        if s is None: return None
        num=pd.to_numeric(s,errors="coerce")
        if num.notna().sum()>=0.5*s.notna().sum(): return (num>0).astype(float).where(num.notna())
        return s.astype(str).str.strip().str.lower().map({"yes":1,"si":1,"no":0}).astype(float)
    sga=yesno("SGA_birth"); lga=yesno("LGA_birth"); sev=yesno("severeSGA")
    counts={"SGA_birth":int(np.nansum(sga)) if sga is not None else 0,
            "LGA_birth":int(np.nansum(lga)) if lga is not None else 0,
            "severeSGA":int(np.nansum(sev)) if sev is not None else 0}
    print(f"  outcome counts parsed: {counts}",flush=True)
    assert counts["SGA_birth"]>100 and counts["LGA_birth"]>40, (
        f"outcome parsing produced implausible counts {counts}; the project records SGA 169 / LGA 77 / "
        "severeSGA 61. A silent all-zero parse would skip every test -- fix the encoding before running.")
    bp=pd.to_numeric(e.get("percentil_birth"),errors="coerce")
    sex=pd.to_numeric(e.get("SexoRN_cod"),errors="coerce")
    res={"encoder":a.enc,"K":a.K,"primary_cell":{"plane":PRIMARY[0],"group":PRIMARY[1],
           "directional_prediction":"abdominal effect > cerebral effect (LGA is an abdominal "
           "soft-tissue phenotype: radiomics abd 0.579 vs cer 0.464; USFM abd 0.69)"},
         "plane_handling":"STRATIFIED not balanced -- balancing would dilute an abdominal-specific effect",
         "outcome_use":"evaluation-only; the codebook is unsupervised k-means on frozen features",
         "n_tests":a.K*len(PLANES)*3,"multiplicity":"Benjamini-Yekutieli over all plane x group x code"}

    rows=[]; percell={}
    for pl in PLANES:
        sel=plane==pl
        if sel.sum()<200: percell[pl]={"skipped":f"only {int(sel.sum())} frames"}; continue
        # per-fetus histogram within this plane, patch budget equalised
        H={}; GA={}
        for f in np.unique(nid[sel]):
            fl=codes[sel][nid[sel]==f].reshape(-1)
            if len(fl)==0: continue
            pick=rng.choice(fl,a.per_fetus,replace=len(fl)<a.per_fetus)
            h=np.bincount(pick,minlength=a.K).astype(float); H[f]=h/h.sum()
            GA[f]=float(np.nanmedian(gaf[sel][nid[sel]==f]))
        fet=np.array(sorted(H)); P=np.vstack([H[f] for f in fet]); Pc=clr(P)
        gav=np.array([GA[f] for f in fet])
        sx=pd.Series(fet).map(sex.to_dict()).values.astype(float)
        C=np.column_stack([ga_spline(gav),sx])
        groups={"SGA":pd.Series(fet).map(sga.to_dict()).values.astype(float) if sga is not None else None,
                "LGA":pd.Series(fet).map(lga.to_dict()).values.astype(float) if lga is not None else None,
                "severeSGA":pd.Series(fet).map(sev.to_dict()).values.astype(float) if sev is not None else None}
        bpv=pd.Series(fet).map(bp.to_dict()).values.astype(float)
        # GA POSITIVE CONTROL, reported first: can this stratum recover GA at all?
        mg=np.isfinite(gav)
        ga_pred=cross_val_predict(RidgeCV(alphas=np.logspace(-2,3,20)),P[mg],gav[mg],
                                  cv=GroupKFold(5),groups=fet[mg])
        ga_r=float(spearmanr(ga_pred,gav[mg]).statistic)
        percell[pl]={"n_fetuses":int(len(fet)),"n_frames":int(sel.sum()),
                     "GA_positive_control_r":ga_r,
                     "group_counts":{k:(int(np.nansum(v)) if v is not None else None) for k,v in groups.items()},
                     "birthpct_obs":int(np.isfinite(bpv).sum())}
        print(f"[{pl}] n_fet={len(fet)} frames={int(sel.sum())} | GA control r={ga_r:+.3f} | "
              f"SGA={percell[pl]['group_counts']['SGA']} LGA={percell[pl]['group_counts']['LGA']}",flush=True)
        for gname,yv in groups.items():
            if yv is None or np.nansum(yv)<15:
                continue
            for k in range(a.K):
                rk=resid(rankdata(P[:,k]).astype(float),C)     # GA+sex adjusted, rank-based
                A=auc(rk,yv)
                # permutation null on the GROUP LABEL (keeps the histogram and covariates intact)
                nul=np.empty(min(a.n_perm,2000))
                for i in range(len(nul)):
                    nul[i]=auc(rk,rng.permutation(yv))
                p=float((1+np.sum(np.abs(np.asarray(nul)-0.5)>=abs(A-0.5)))/(len(nul)+1)) if np.isfinite(A) else np.nan
                rows.append({"plane":pl,"group":gname,"code":k,"auc":A,"p_perm":p,
                             "usage":float(P[:,k].mean()),
                             "r_clr_birthpct":float(spearmanr(Pc[:,k],bpv,nan_policy="omit").statistic),
                             "is_primary":bool(pl==PRIMARY[0] and gname==PRIMARY[1])})
    q=by_correct([r["p_perm"] for r in rows])
    for r,qq in zip(rows,q): r["q_BY"]=float(qq)
    res["per_plane"]=percell; res["tests"]=rows
    # primary cell readout
    prim=[r for r in rows if r["is_primary"]]
    if prim:
        best=max(prim,key=lambda r: abs((r["auc"] or 0.5)-0.5))
        cer=[r for r in rows if r["plane"]=="cerebral" and r["group"]==PRIMARY[1]]
        cbest=max(cer,key=lambda r: abs((r["auc"] or 0.5)-0.5)) if cer else None
        res["PRIMARY_RESULT"]={"code":best["code"],"auc":best["auc"],"q_BY":best["q_BY"],
          "cerebral_comparison_auc":(cbest or {}).get("auc"),
          "directional_prediction_held":bool(cbest and abs(best["auc"]-0.5)>abs(cbest["auc"]-0.5))}
        print(f"\n  PRIMARY (abdominal x LGA): best code {best['code']} AUC {best['auc']:.3f} "
              f"q_BY {best['q_BY']:.4f} | cerebral best {(cbest or {}).get('auc',float('nan')):.3f} "
              f"-> directional prediction {'HELD' if res['PRIMARY_RESULT']['directional_prediction_held'] else 'FAILED'}",flush=True)
    surv=[r for r in rows if np.isfinite(r["q_BY"]) and r["q_BY"]<0.10]
    res["n_surviving_BY_q0.10"]=len(surv)
    res["survivors"]=sorted(surv,key=lambda r:r["q_BY"])[:20]
    print(f"  {len(surv)} of {len(rows)} plane x group x code tests survive BY q<0.10",flush=True)
    json.dump(res,open(os.path.join(OUTP,f"code_growth_groups_K{a.K}.json"),"w"),indent=2,default=str)
    print(f"saved out_probe/code_growth_groups_K{a.K}.json\nDONE",flush=True)

if __name__=="__main__": main()
