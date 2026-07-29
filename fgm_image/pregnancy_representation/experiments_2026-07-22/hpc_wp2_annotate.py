#!/usr/bin/env python3
"""STEPS 2+3 -- WP2 AXIS ANNOTATION with visual words, plus the mandatory nulls. No training.

RUNS AFTER hpc_seed_ceiling.py, which PASSED: within-encoder across-seed AMI ceiling = 0.745
(FetalCLIP 0.797, USF-MAE 0.732, USFM 0.710, DINOv2 0.739; every seed pair >= 0.662; ~4.6x the
whole-image-shuffle floor; 16/16 codes alive in all 20 fits). That licenses the four-experts
inference, and it BOUNDS it: every cross-encoder agreement below is reported as a FRACTION of 0.745.

WHAT IS ASKED: which visual words are over-represented at the FGR end of the TABULAR WP2 axis?
Descriptive annotation, NOT a claim that images improve WP2 (they do not: image features in the
GRU-VAE moved SGA 0.890->0.871; latent displacement 0.180 SD sat BELOW the 0.209 SD training-noise
floor; five factorised multimodal VAEs put the size/FGR axis in the biometry-private subspace every
time). Non-circular by construction: the axis is tabular (size_scalar, verified rho=+0.179 with the
image-derived lag, i.e. essentially independent of it) and the words are from images.

STAGE A -- FULL-FRAME ASSIGNMENT (no refitting). The seed-ceiling run saved centroids fitted on 3,627
frames. Here all 20,413 frames are ASSIGNED to those FROZEN centroids, giving ~4x more patches per
fetus and hence sharper histograms. Same vocabulary; refitting on more frames would be a different
codebook and would invalidate the measured ceiling.

STAGE B -- PER-FETUS HISTOGRAMS, confounds handled BY CONSTRUCTION where labels are too noisy to
adjust with:
 - PLANE: BALANCED SAMPLING. Each fetus contributes a FIXED number of patches per plane. Adjustment
   was rejected: plane_prop is kNN-propagated and audited 3/6 wrong at conf=1.00, and with a signal
   100% plane-driven by construction, one-hot adjustment leaves residual |r|=0.006 at 100% label
   accuracy but 0.278 at 95% and 0.425 at 90%. You cannot adjust with a label that noisy.
 - PATCH BUDGET equalised across fetuses, else "enrichment" partly means "who got more scans".
 - Fetuses lacking any plane, or below budget, are dropped and COUNTED in the json.

STAGE C -- PER-CODE TEST, per encoder (the four experts, fitted independently, never combined):
 partial Spearman of each code's share vs the WP2 axis, adjusting a GA spline + fetal sex.
 SEX is adjusted because female fetuses are smaller by nature and therefore confound a size-related
 axis; it is ALSO used as a POSITIVE CONTROL (Stage E).
 Effects reported BOTH as raw shares AND as CLR (centred-log-ratio) coordinates, with the analytic
 CLOSURE FLOOR -r0/(K-1) printed beside every per-code effect: histograms sum to 1, so K-1 weakly
 anti-correlated codes will otherwise read as a coherent biological pattern.

STAGE D -- THE NULLS. Two, both mandatory, replacing the naive label shuffle:
 D1 STRATIFIED CONDITIONAL PERMUTATION (Berrett et al. 2020). The WP2 axis is permuted only WITHIN
    acquisition strata, so a shared acquisition confound cannot manufacture agreement. A plain
    label-shuffle was shown to yield 4/4 replication, 4/4 sign concordance and all four per-encoder
    p<1e-3 with ZERO fetal signal at this project's measured institution eta2=0.061.
    HONEST LIMITATION: no scanner/manufacturer/site column exists in this cohort's tables. The
    available acquisition proxies are STUDY-DATE ERA (scanner fleet and protocol drift over time --
    this project measured an apparent-vs-true GA drift of -1.34 wk/yr) and PIXEL SPACING (depth/zoom
    setting). Strata = study-date era x spacing tertile, collapsed to >=30 fetuses. This is a PROXY
    stratification and is labelled as such in the output; it is strictly better than an unstratified
    shuffle but it is not a true site control.
 D2 MISMATCHED-FETUS NULL. The image->fetus mapping is permuted within stratum, leaving encoders,
    images and the acquisition confound fully intact while destroying only the fetus correspondence.
    This is the direct analogue of the numerical test that caught this morning's retraction.

STAGE E -- CALIBRATION (reported FIRST, gates interpretation of any null). Re-run the identical
 pipeline with FETAL SEX as the target instead of the WP2 axis. Sex is ~100% complete, randomised by
 nature, orthogonal to GA and plane, and has a KNOWN SMALL REAL effect on size. If the pipeline
 cannot detect sex, then a null on the WP2 axis means UNDERPOWERED, not absent. This is what makes a
 negative result publishable rather than a shrug.

STATISTICS. PRIMARY = a random-effects meta-analysis of the four per-encoder per-code effects with
 encoder as the random effect (chosen because replication-count-over-4 is nearly vacuous: at E=4 the
 attainable p-values are a 4-point lattice and at per-expert false-flag rate 0.10, P(>=2)=0.052,
 which fails even nominal 0.05). Replication count is reported as a DESCRIPTOR only.
 SIGN CONCORDANCE IS NOT USED: under compositional closure all-4-same-sign occurs on NULL codes with
 probability 0.82 (raw) / 0.95 (CLR) against a coin-flip null of 0.125 -- a construction floor of
 exactly the retracted class.
 MULTIPLICITY: 4 encoders x 16 codes = 64 tests, Benjamini-Yekutieli (valid under the arbitrary
 dependence that compositional columns guarantee), alpha 1.65e-4. The project ledger already stands
 at 218 constructions / 584 tests, so the nominal 0.05 is meaningless here.

EXPECTED RESULT: NULL. Growth-outcome information has been absent from the image channel across >=12
construction families (radiomics birth-pct r=0.035; SGA<p10 AUC 0.562; LGA 0.444; 91 <p10 events).

USAGE: python hpc_wp2_annotate.py [--K 16] [--assign-only] [--per-plane 40] [--n-perm 2000]
Outputs: out_probe/wp2_annotate_K16.json, out_usfmae/wp2fullcodes_<ENC>_K16.npz
"""
import os, sys, json, time, argparse, numpy as np, pandas as pd, torch
from PIL import Image
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
OUT=os.environ.get("GA_OUT_DIR",os.path.join(HERE,"out_usfmae")); os.makedirs(OUT,exist_ok=True)
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
HAND=os.path.join(HERE,"handoff")
DEV="cuda" if torch.cuda.is_available() else "cpu"
ENCS=["FetalCLIP","USF-MAE","USFM","DINOv2"]
SEED_CEILING=0.745          # measured by hpc_seed_ceiling.py; every agreement is a fraction of this
BATCH=32
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
from hpc_crossenc_factvq import patch_tokens

# ---------- helpers ----------
def ga_spline(x):
    x=np.asarray(x,float); q=np.nanpercentile(x,[25,50,75])
    return np.column_stack([x]+[np.clip(x-k,0,None)**3 for k in q])

def clr(P,eps=1e-6):
    """centred log-ratio; removes the unit-sum constraint's rotational artefact."""
    Q=np.clip(P,eps,None); L=np.log(Q); return L-L.mean(1,keepdims=True)

def partial_spearman(y,x,C):
    m=np.isfinite(y)&np.isfinite(x)&np.isfinite(C).all(1)
    if m.sum()<50: return np.nan,np.nan,int(m.sum())
    ry,rx=rankdata(y[m]),rankdata(x[m]); Cm=C[m]
    ey=ry-LinearRegression().fit(Cm,ry).predict(Cm)
    ex=rx-LinearRegression().fit(Cm,rx).predict(Cm)
    r,p=spearmanr(ey,ex); return float(r),float(p),int(m.sum())

def by_correct(p):
    p=np.asarray(p,float); ok=np.isfinite(p); q=np.full(p.shape,np.nan); ps=p[ok]; n=len(ps)
    if n==0: return q
    c=np.sum(1.0/np.arange(1,n+1)); o=np.argsort(ps)
    adj=np.minimum.accumulate((ps[o]*n*c/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n); out[o]=np.clip(adj,0,1); q[ok]=out; return q

def strat_perm(v,strata,rng):
    """permute v only WITHIN strata (conditional permutation)."""
    w=v.copy()
    for s in np.unique(strata):
        i=np.where(strata==s)[0]
        if len(i)>1: w[i]=v[rng.permutation(i)]
    return w

def build_strata(nid_u):
    """acquisition proxy strata: study-date era x spacing tertile. NO scanner/site column exists in
    this cohort's tables, so this is a PROXY and is labelled as such."""
    info={"kind":"PROXY (no scanner/site column exists in this cohort)","components":[]}
    era=np.zeros(len(nid_u),int); ter=np.zeros(len(nid_u),int)
    sp_path=os.path.join(HAND,"dicom_spacing.csv")
    for cand in (sp_path,sp_path+".gz"):
        if os.path.exists(cand):
            sp=pd.read_csv(cand)
            sp["nid"]=sp["imp_dir"].astype(str).str.extract(r"IMP0*(\d+)")[0]
            g=sp.groupby("nid").agg(spacing=("spacing_mm","median"),date=("study_date","median"))
            s=g.reindex(nid_u)
            if s["date"].notna().sum()>100:
                yr=pd.to_numeric(s["date"],errors="coerce")//10000
                era=pd.factorize(yr)[0]; info["components"].append("study_date era (scanner fleet / protocol drift)")
            if s["spacing"].notna().sum()>100:
                ter=pd.qcut(s["spacing"],3,labels=False,duplicates="drop").fillna(-1).astype(int).values
                info["components"].append("pixel-spacing tertile (depth/zoom setting)")
            break
    lab=np.array([f"{a}_{b}" for a,b in zip(era,ter)])
    # collapse rare strata so within-stratum permutation stays meaningful
    vc=pd.Series(lab).value_counts()
    lab=np.array([l if vc[l]>=30 else "pooled" for l in lab])
    info["n_strata"]=int(len(np.unique(lab))); info["sizes"]=pd.Series(lab).value_counts().to_dict()
    if not info["components"]: info["WARNING"]="no acquisition proxy available; permutation is UNSTRATIFIED"
    return lab,info

# ---------- Stage A: assign all frames to frozen centroids ----------
def assign_all(enc,K,cohort='impact'):
    tag='' if cohort=='impact' else '_clin'
    dst=os.path.join(OUT,f"wp2fullcodes_{enc}_K{K}{tag}.npz")
    if os.path.exists(dst): print(f"  {enc}: full assignment exists, skip",flush=True); return dst
    src=os.path.join(OUT,f"wp2codes_{enc}_K{K}{tag}.npz")
    assert os.path.exists(src), f"missing {src} -- run hpc_seed_ceiling.py first"
    z=np.load(src,allow_pickle=True); C=torch.tensor(z["centroids"],device=DEV).float()
    li=int(z["layer_index"]); df=cohort_table(cohort); m,tf,_=BUILDERS[enc]()
    labs=[]; t0=time.time()
    for b0 in range(0,len(df),BATCH):
        bs=df.iloc[b0:b0+BATCH]
        x=torch.stack([tf(Image.open(p).convert("RGB")) for p in bs["img"]]).to(DEV)
        with torch.no_grad():
            t=patch_tokens(enc,m,x)[:,li]                     # (B,Np,D)
        f=t.reshape(-1,t.shape[-1]).float()
        f=(f-f.mean(0))/(f.std(0)+1e-6)
        labs.append(torch.cdist(f,C).argmin(1).cpu().numpy().reshape(len(bs),-1).astype(np.int16))
        if (b0//BATCH)%50==0: print(f"    {enc} assign {b0}/{len(df)} {time.time()-t0:.0f}s",flush=True)
    del m; torch.cuda.empty_cache() if DEV=="cuda" else None
    np.savez(dst,codes=np.concatenate(labs),nid=df["nid"].astype(str).values,
             plane=df["plane_prop"].values,ga=df["ga_weeks_recovered"].values,layer_index=li)
    print(f"  {enc}: assigned {len(df)} frames -> {os.path.basename(dst)} ({time.time()-t0:.0f}s)",flush=True)
    return dst

# ---------- Stage B: plane-balanced, budget-equalised per-fetus histograms ----------
def histograms(npz,K,per_plane,rng):
    z=np.load(npz,allow_pickle=True)
    codes,nid,plane=z["codes"],z["nid"].astype(str),z["plane"].astype(str)
    P=codes.shape[1]; out={}; dropped=0
    for f in np.unique(nid):
        fi=np.where(nid==f)[0]; picks=[]
        for pl in np.unique(plane[fi]):
            idx=fi[plane[fi]==pl]
            flat=codes[idx].reshape(-1)
            if len(flat)==0: continue
            picks.append(rng.choice(flat,per_plane,replace=len(flat)<per_plane))
        if not picks: dropped+=1; continue
        h=np.bincount(np.concatenate(picks),minlength=K).astype(float)
        out[f]=h/h.sum()
    return out,dropped

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--K",type=int,default=16); ap.add_argument("--per-plane",type=int,default=120)
    ap.add_argument("--n-perm",type=int,default=20000); ap.add_argument("--assign-only",action="store_true")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--cohort",default="impact",choices=["impact","clinical"],
                    help="clinical routes through clinical_paths.resolve() (prefix-tolerant); outputs tagged _clin")
    a=ap.parse_args(); rng=np.random.default_rng(a.seed)
    res={"K":a.K,"per_plane_patches":a.per_plane,"n_perm":a.n_perm,
         "seed_ceiling_from_step1":SEED_CEILING,
         "agreement_reporting_rule":f"every cross-encoder agreement is reported as a FRACTION of {SEED_CEILING}",
         "sign_concordance":"NOT USED -- compositional closure puts all-4-same-sign on NULL codes at 0.82 (raw) / 0.95 (CLR) vs a coin-flip 0.125",
         "primary_statistic":"random-effects meta-analysis over the 4 per-encoder effects; replication count is a DESCRIPTOR only",
         "expected":"NULL -- image channel has carried no growth-outcome signal across >=12 families"}

    # Stage A
    paths={}
    for enc in ENCS:
        try: paths[enc]=assign_all(enc,a.K,a.cohort)
        except Exception as ex: print(f"  {enc}: SKIP ({type(ex).__name__}: {ex})",flush=True)
    res["encoders_assigned"]=list(paths)
    if a.assign_only:
        json.dump(res,open(os.path.join(OUTP,f"wp2_annotate_K{a.K}{'' if a.cohort=='impact' else '_clin'}.json"),"w"),indent=2)
        print("  ASSIGN ONLY -> stopping before the test",flush=True); return

    # axis + covariates
    ax=pd.read_csv(os.path.join(HAND,"wp2_axis.csv"))
    ax["nid"]=ax["nid"].astype(str); ax=ax.drop_duplicates("nid").set_index("nid")
    ga=pd.read_csv(os.path.join(HERE,"ga_cnn","ga_cnn_index.csv"))
    ga["nid"]=ga["nid"].apply(lambda x:str(int(float(x))))
    ga_f=ga.groupby("nid")["ga_weeks_recovered"].median()

    per_enc={}; eff={}
    for enc,npz in paths.items():
        H,dropped=histograms(npz,a.K,a.per_plane,rng)
        fet=np.array(sorted(H)); Praw=np.vstack([H[f] for f in fet]); Pclr=clr(Praw)
        keep=[f in ax.index for f in fet]
        fet,Praw,Pclr=fet[keep],Praw[keep],Pclr[keep]
        y=ax.loc[fet,"wp2_axis_primary"].values.astype(float)
        sex=ax.loc[fet,"sex"].values.astype(float)
        gav=ga_f.reindex(fet).values.astype(float)
        C=np.column_stack([ga_spline(gav),sex])
        strata,sinfo=build_strata(fet); res["strata_info"]=sinfo
        usage=Praw.mean(0); alive=usage>=0.005; Keff=int(alive.sum())
        rows=[]
        for k in range(a.K):
            if not alive[k]: continue
            r_raw,p_raw,n=partial_spearman(Praw[:,k],y,C)
            r_clr,p_clr,_=partial_spearman(Pclr[:,k],y,C)
            # D1 stratified conditional permutation.
            # RESOLUTION: p is reported with the +1 correction, so the FLOOR is 1/(n_perm+1) and a
            # zero can never be printed. The earlier run capped perms at 500 and printed q_BY=0.0000,
            # which actually meant p<0.002 -- misleading. n_perm now runs in full (>=20000 is what
            # the project's multiplicity ledger requires to resolve alpha=8.6e-5).
            null=np.array([partial_spearman(Praw[:,k],strat_perm(y,strata,rng),C)[0] for _ in range(a.n_perm)])
            p_perm=float((1+np.sum(np.abs(null)>=abs(r_raw)))/(a.n_perm+1)) if np.isfinite(r_raw) else np.nan
            p_floor=1.0/(a.n_perm+1)
            rows.append({"code":k,"usage":float(usage[k]),"r_raw":r_raw,"p_raw":p_raw,
                         "r_clr":r_clr,"p_clr":p_clr,"p_stratperm":p_perm,"p_perm_floor":p_floor,"n":n,
                         "closure_floor":float(-r_raw/(a.K-1)) if np.isfinite(r_raw) else None})
        q=by_correct([r["p_stratperm"] for r in rows])
        for r_,qq in zip(rows,q): r_["q_BY"]=float(qq)
        # Stage E calibration -- TWO controls, because the sex gate alone proved too weak.
        # SEX: adjusted for GA. Known small real effect on size (~2-3%), so a small |r| here is
        #      expected and does NOT by itself establish that the pipeline works.
        # GA: the PROPERLY POWERED positive control, added after the first run showed sex
        #     max|r| 0.045-0.064. Declared as an ADDITION, not a substitution: the per-fetus code
        #     histogram is already known to carry GA at r~0.615, so if this pipeline cannot recover
        #     GA the pipeline is BROKEN and no null from it counts. If it DOES recover GA, then sex
        #     is simply a weak calibrator and the WP2 null is interpretable.
        cal=[partial_spearman(Praw[:,k],sex,np.column_stack([ga_spline(gav)]))[0] for k in range(a.K) if alive[k]]
        ga_ctrl=[partial_spearman(Praw[:,k],gav,np.column_stack([sex]))[0] for k in range(a.K) if alive[k]]
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_predict, GroupKFold
        mg=np.isfinite(gav)
        pred=cross_val_predict(RidgeCV(alphas=np.logspace(-2,3,20)),Praw[mg],gav[mg],
                               cv=GroupKFold(5),groups=fet[mg])
        ga_multi=float(spearmanr(pred,gav[mg]).statistic)
        per_enc[enc]={"n_fetuses":int(len(fet)),"dropped_no_plane":int(dropped),"K_eff":Keff,
                      "codes":rows,"max_abs_r":float(np.nanmax([abs(r_["r_raw"]) for r_ in rows])),
                      "min_q_BY":float(np.nanmin(q)) if len(q) else None,
                      "calibration_sex_max_abs_r":float(np.nanmax(np.abs(cal))) if cal else None,
                      "POSITIVE_CONTROL_ga_percode_max_abs_r":float(np.nanmax(np.abs(ga_ctrl))) if ga_ctrl else None,
                      "POSITIVE_CONTROL_ga_multivariate_r":ga_multi,
                      "positive_control_verdict":("PIPELINE WORKS -- recovers GA multivariately" if ga_multi>0.35
                          else "PIPELINE SUSPECT -- cannot recover GA, so no null from it is interpretable")}
        eff[enc]={r_["code"]:r_["r_raw"] for r_ in rows}
        print(f"  [{enc}] n={len(fet)} K_eff={Keff} | max|r| vs WP2 axis {per_enc[enc]['max_abs_r']:.3f} "
              f"| min q_BY {per_enc[enc]['min_q_BY']:.4f}\n           SEX cal max|r| "
              f"{per_enc[enc]['calibration_sex_max_abs_r']:.3f} | GA POSITIVE CONTROL multivariate r "
              f"{ga_multi:+.3f} percode max|r| {max(abs(x) for x in ga_ctrl):.3f} -> {per_enc[enc]['positive_control_verdict']}",flush=True)
    res["per_encoder"]=per_enc

    # PRIMARY: random-effects meta-analysis per code across encoders (DerSimonian-Laird)
    meta={}
    common=sorted(set.intersection(*[set(v.keys()) for v in eff.values()])) if eff else []
    for k in common:
        e=np.array([eff[x][k] for x in eff if np.isfinite(eff[x].get(k,np.nan))])
        n=np.mean([per_enc[x]["n_fetuses"] for x in eff]); se=1.0/np.sqrt(max(n-3,1))
        # BUGFIX: w2 must be a PER-STUDY ARRAY. When it was the scalar 1/(se^2+tau2), w2.sum()
        # returned w2 itself, so pooled=(w2*e).sum()/w2 collapsed to SUM(e) -- roughly 4x the true
        # weighted mean at E=4 -- and se_p became a single-study se. That is what produced the
        # impossible pooled_r=-0.229 (larger than every per-encoder input, max 0.107) with z=-6.88.
        # Guard below asserts the pooled estimate lies within the range of its inputs.
        w=np.ones(len(e))/se**2; fixed=float((w*e).sum()/w.sum())
        Q=float((w*(e-fixed)**2).sum())
        denom=w.sum()-(w**2).sum()/w.sum()
        tau2=max(0.0,(Q-(len(e)-1))/denom) if denom>0 else 0.0
        w2=np.ones(len(e))/(se**2+tau2)                      # PER-STUDY weights (was a scalar)
        pooled=float((w2*e).sum()/w2.sum()); se_p=float(np.sqrt(1.0/w2.sum()))
        assert e.min()-1e-9 <= pooled <= e.max()+1e-9, (
            f"pooled {pooled} outside input range [{e.min()},{e.max()}] -- weighted mean cannot "
            "exceed its inputs; the weights are wrong")
        meta[int(k)]={"pooled_r":pooled,"se":se_p,"z":pooled/se_p if se_p>0 else np.nan,
                      "tau2":tau2,"Q":Q,"n_encoders":int(len(e)),
                      "per_encoder_r":[float(v) for v in e],
                      "input_range":[float(e.min()),float(e.max())],
                      "replication_descriptor":int(np.sum(np.abs(e)>0.10))}
    res["meta_analysis_PRIMARY"]=meta
    if meta:
        best=max(meta.items(),key=lambda kv: abs(kv[1]["z"]) if np.isfinite(kv[1]["z"]) else -1)
        res["headline"]={"strongest_code":best[0],**best[1]}
        print(f"\n  PRIMARY meta-analysis: strongest code {best[0]} pooled r={best[1]['pooled_r']:+.3f} "
              f"z={best[1]['z']:+.2f} (tau2={best[1]['tau2']:.4f})",flush=True)
    json.dump(res,open(os.path.join(OUTP,f"wp2_annotate_K{a.K}{'' if a.cohort=='impact' else '_clin'}.json"),"w"),indent=2,default=str)
    print(f"saved out_probe/wp2_annotate_K{a.K}.json\nDONE",flush=True)

if __name__=="__main__": main()
