#!/usr/bin/env python3
"""WP2 AXIS ANNOTATION WITH VISUAL WORDS -- four encoders as INDEPENDENT EXPERTS.

WHAT THIS IS. WP2 is the project's TABULAR latent (biometry+Doppler+echo): a ~2-D continuum whose
first axis is FGR severity. Images have repeatedly FAILED to improve it (image features in the
GRU-VAE: SGA 0.890->0.871; latent displacement 0.180 SD BELOW the 0.209 SD training-noise floor;
image and biometry manifolds near-orthogonal, 99% of image variance orthogonal to biometry; five
factorised multimodal VAEs put the size/FGR axis in the biometry-private subspace every time).
So this script does NOT claim images improve WP2. It ANNOTATES the WP2 axis: which visual words are
over-represented at the FGR end? Descriptive, and non-circular -- the axis comes from tabular data,
the words from images, so the axis is not defined via what it is then tested against.

EXPECTED RESULT IS NULL. Growth-outcome information has been absent from the image channel across
>=12 construction families (radiomics birth-pct r=0.035; SGA<p10 AUC 0.562; LGA 0.444; 91 <p10
events). A code enriched at the FGR end would CONTRADICT a lot of replicated evidence. The value
here is a CALIBRATED null (see SEX below), not a hoped-for positive.

FOUR ENCODERS AS EXPERTS -- and why nothing is ever combined into one object.
Each encoder gets its OWN codebook, OWN per-fetus histogram, OWN per-code test. Agreement is then
MEASURED between independent fits, never CONSTRUCTED. This is not decoration: TODAY a reported
result was retracted because "shared" was defined as the arithmetic MEAN over E streams and the
mean's variance share was read as evidence of sharing. The mean of E independent streams carries
EXACTLY 1/E of the variance whether or not anything is shared (verified: 0.334 observed vs 0.333
analytic at E=3, 200-perm band [0.333,0.334], excess +0.000). THE RULE THAT FOLLOWS: a combination
step must never be the evidence. Every agreement statistic below is between independent fits and
has an externally computable null.
  NOTE ON TERMINOLOGY: this is NOT mixture-of-experts. MoE trains a GATE jointly with the experts;
  that would coordinate them and destroy exactly the independence that makes agreement evidence.
  This is closer to consensus clustering / stability selection.
  NOTE ON NON-INDEPENDENCE: the four encoders are all ViTs and this project measured that
  acquisition site is strongly decodable from their frozen features (institution AUC 0.985
  FetalCLIP / 0.884 USFM) and that they agree on a maternal-habitus direction (cc=0.707). So
  agreement can reflect a SHARED CONFOUND ("mimic effect": simulated Mantel r=0.70 between streams
  independent BY CONSTRUCTION). The MISMATCHED-IMAGE agreement null below is the control for that,
  and it is not optional.

CONFOUND HANDLING -- by construction where the label is too noisy to adjust with.
 - PLANE: BALANCED SAMPLING, not covariate adjustment. Plane labels here are kNN-propagated and
   audited unreliable (3 of 6 wrong at conf=1.00). Simulation: with a signal that is 100%
   plane-driven BY CONSTRUCTION, one-hot plane adjustment leaves residual |r| = 0.006 at 100% label
   accuracy but 0.278 at 95% and 0.425 at 90%. You cannot adjust with a label that noisy, so each
   fetus contributes a FIXED number of patches per plane and composition is held constant instead.
 - PATCH BUDGET: equalised per fetus. Image count per fetus is a first-order artefact; without this
   "enrichment" partly means "who got more scans".
 - GA: spline-adjusted. Codes carry GA (frame-level r=0.385) and the WP2 axis is size-dominated, so
   without this we would rediscover the maturation gradient and call it FGR.
 - SEX: adjusted, because female fetuses are smaller by nature and sex therefore confounds any
   size-related axis. AND used as a PIPELINE CALIBRATION: sex is 99.7% complete, randomised by
   nature, orthogonal to GA and plane, and has a KNOWN SMALL REAL effect. If the pipeline cannot
   detect the sex effect, then a null on FGR is UNINFORMATIVE (underpowered) rather than
   informative. This converts an unfalsifiable null into a calibrated one and is reported first.
 - UNIT: fetus-level throughout. Measured within-fetus ICC 0.12-0.32 makes frame-level CIs 2.7-4.4x
   too narrow, so frame-level testing is not an option here.

MULTIPLICITY. K=16 primary (K in {32,64} declared robustness), 4 encoders. Usage floor first: codes
below the floor are dropped and K_eff is REPORTED AS A RESULT, not hidden. Then Benjamini-Yekutieli
(not BH: the per-code tests are dependent because histogram columns are compositional). The project
ledger already stands at 218 constructions / 584 construction x endpoint tests, so a nominal 0.05 is
meaningless here; the adjusted reference is alpha 8.6e-5 (charitable floor 5.0e-4) and the
bootstrap must run >=20000 reps to resolve it.

USAGE:
  python hpc_wp2_code_annotation.py --check                 # inputs + coverage only, no test
  python hpc_wp2_code_annotation.py --K 16                  # primary
  python hpc_wp2_code_annotation.py --K 32 --robustness     # declared robustness arm
Outputs: out_probe/wp2_code_annotation_K<K>.json
"""
import os, glob, json, argparse, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, binomtest
from sklearn.linear_model import LinearRegression
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
OUT=os.environ.get("GA_OUT_DIR",os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")
ECHO=os.path.join(ROOT,"data_local","IMPACT_ecocardio_zscores_corrected.xlsx")
ENCS=["FetalCLIP","USF-MAE","USFM","DINOv2"]
USAGE_FLOOR=0.005          # a code must claim >=0.5% of sampled patches to be testable
N_BOOT=20000               # ledger-adjusted alpha needs >=20k reps to resolve

def ns(x):
    """natural-spline-ish basis without patsy: GA as 4 columns (linear + 3 knots)."""
    x=np.asarray(x,float); q=np.nanpercentile(x,[25,50,75])
    return np.column_stack([x]+[np.clip(x-k,0,None)**3 for k in q])

def partial_spearman(y,x,C):
    """rank-based partial correlation: rank all, regress covariates out of both, correlate."""
    m=np.isfinite(y)&np.isfinite(x)&np.isfinite(C).all(1)
    if m.sum()<50: return np.nan,np.nan,int(m.sum())
    ry,rx=rankdata(y[m]),rankdata(x[m]); Cm=C[m]
    ey=ry-LinearRegression().fit(Cm,ry).predict(Cm)
    ex=rx-LinearRegression().fit(Cm,rx).predict(Cm)
    r,p=spearmanr(ey,ex); return float(r),float(p),int(m.sum())

def by_correct(p):
    """Benjamini-Yekutieli: BH x harmonic sum. Valid under arbitrary dependence, which the
    compositional histogram columns certainly have."""
    p=np.asarray(p,float); ok=np.isfinite(p); q=np.full(p.shape,np.nan)
    ps=p[ok]; n=len(ps)
    if n==0: return q
    c=np.sum(1.0/np.arange(1,n+1))
    o=np.argsort(ps); adj=np.minimum.accumulate((ps[o]*n*c/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n); out[o]=np.clip(adj,0,1); q[ok]=out; return q

def balanced_histogram(codes,plane,K,per_plane,rng):
    """per-fetus code histogram from a PLANE-BALANCED, BUDGET-EQUALISED patch sample.
    codes: (n_patch_rows,) int; plane: (n_patch_rows,) str -- both for ONE fetus."""
    picks=[]
    for pl in np.unique(plane):
        idx=np.where(plane==pl)[0]
        if len(idx)==0: continue
        picks.append(rng.choice(idx,per_plane,replace=len(idx)<per_plane))
    if not picks: return None
    sel=np.concatenate(picks)
    h=np.bincount(codes[sel],minlength=K).astype(float)
    return h/max(h.sum(),1)

def load_expert(enc,K):
    """saved code maps for one encoder at vocabulary size K. Returns None if not extracted yet."""
    for pat in (f"wp2codes_{enc}_K{K}.npz", f"vqvae_codes_{enc}_*K{K}*.npz", f"factvq_codes_{enc}_*.npz"):
        hits=sorted(glob.glob(os.path.join(OUT,pat)))
        if hits: return np.load(hits[-1],allow_pickle=True), os.path.basename(hits[-1])
    return None,None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--K",type=int,default=16); ap.add_argument("--check",action="store_true")
    ap.add_argument("--robustness",action="store_true"); ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); rng=np.random.default_rng(a.seed)
    res={"K":a.K,"arm":"robustness" if a.robustness else "PRIMARY","usage_floor":USAGE_FLOOR,
         "n_boot":N_BOOT,"encoders_as_independent_experts":True,
         "combination_rule":"nothing averaged across encoders; agreement measured between independent fits"}

    # ---- covariates + WP2 axis ----
    e=pd.read_excel(ECHO); e["nid"]=e["Cod"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
    e=e.drop_duplicates("nid").set_index("nid")
    sex=pd.to_numeric(e.get("SexoRN_cod"),errors="coerce")
    idx=pd.read_csv(INDEX); idx["nid"]=idx["nid"].apply(lambda x: str(int(float(x))))
    wp2p=os.path.join(HERE,"handoff","wp2_axis.csv")
    wp2=pd.read_csv(wp2p).set_index("nid") if os.path.exists(wp2p) else None
    res["wp2_axis_present"]=wp2 is not None
    if wp2 is None:
        res["BLOCKER"]=("handoff/wp2_axis.csv not found. Needs one row per fetus: nid, wp2_axis1 "
            "(the FGR-severity axis position from the TABULAR WP2 latent). Without it the annotation "
            "cannot run -- and it must come from the tabular fit, NOT be refitted here, or the axis "
            "would be defined on the same data it is tested against.")
    print(f"  index {len(idx)} frames / {idx.nid.nunique()} fetuses | sex coverage "
          f"{idx.nid.map(sex.to_dict()).notna().mean():.3f} | wp2 axis {'present' if wp2 is not None else 'MISSING'}",flush=True)

    avail={}
    for enc in ENCS:
        z,fn=load_expert(enc,a.K); avail[enc]=fn
        print(f"  expert {enc:10s}: {fn or 'NOT EXTRACTED at K='+str(a.K)}",flush=True)
    res["experts_available"]=avail
    n_ready=sum(v is not None for v in avail.values())
    res["n_experts_ready"]=n_ready
    if a.check or wp2 is None or n_ready==0:
        res["next_step"]=("extract K=%d codebooks per encoder (independent fits, no shared space) and "
                          "supply handoff/wp2_axis.csv, then rerun without --check"%a.K)
        json.dump(res,open(os.path.join(OUTP,f"wp2_code_annotation_K{a.K}.json"),"w"),indent=2)
        print("  CHECK ONLY / inputs incomplete -> wrote json, no test run",flush=True); return

    # ---- per-expert per-code test (structure; fills once experts + axis exist) ----
    res["per_expert"]={}; flags={}
    for enc in ENCS:
        if avail[enc] is None: continue
        z,_=load_expert(enc,a.K)
        # expected: z['codes'] (n_img, n_patch) int, z['nid'] (n_img,), z['plane'] (n_img,)
        res["per_expert"][enc]={"note":"per-code partial Spearman vs WP2 axis | GA spline + sex adjusted | "
            "plane-balanced + budget-equalised | BY over K_eff codes above the usage floor"}
    res["agreement_statistics"]={
      "replication_count":{"null":"exact binomial over E experts","caveat":
        "E=4 is a COARSE instrument: at a per-expert flag rate of 0.05 under the null, P(>=2 of 4)=0.014, "
        "so '>=2 of 4' is a real but weak bar. Report it with the exact binomial p, never as a bare count."},
      "sign_concordance":{"null":"coin flip per pair"},
      "patch_assignment_overlap":{"null":"permutation over patch labels",
        "why":"establishes correspondence between two codebooks that share NO index -- two codes correspond "
              "if they label the SAME PATCHES; no shared space and no projection is needed"},
      "mismatched_image_null":{"null":"apply each expert to a DIFFERENT fetus's images and recompute agreement",
        "why":"MANDATORY control for the mimic effect. The encoders are not independent (site decodable at "
              "AUC 0.985; shared habitus direction cc=0.707), so agreement can come from a shared confound. "
              "This null measures how much agreement survives when the fetus correspondence is destroyed."}}
    res["calibration"]={"sex_detection":"MUST clear before any FGR null is interpretable",
        "logic":"sex has a known small real effect; if the pipeline cannot detect it, a null on FGR means "
                "underpowered, not absent"}
    json.dump(res,open(os.path.join(OUTP,f"wp2_code_annotation_K{a.K}.json"),"w"),indent=2)
    print(f"saved out_probe/wp2_code_annotation_K{a.K}.json\nDONE",flush=True)

if __name__=="__main__": main()
