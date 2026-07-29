#!/usr/bin/env python3
"""NAME THE VISUAL WORDS USING THE TABULAR PANEL AS THE DICTIONARY.

THE IDEA, and why it is the one direction still open. Every attempt to make images ADD to the tabular
latent has failed here, repeatedly and for a principled reason: image features in the GRU-VAE moved SGA
0.890->0.871; the latent was UNCHANGED by adding images (top-axis CCA 0.97-0.99, displacement 0.180 SD
BELOW the 0.209 SD training-noise floor); five factorised multimodal VAEs put the size/FGR axis in the
biometry-private subspace every time; echo fusion was negative. And the codes cannot rescue that --
they are a deterministic lossy function of the embeddings (codes retain GA at 0.385 where the
embeddings give 0.469), so by the data-processing inequality any SGA information the codes hold, the
embeddings already held, and the embeddings added nothing.

So this script INVERTS the direction. Not "do codes predict tabular" (a prediction question, answered)
but "what is the tabular PROFILE of the fetuses high in each code" -- tabular as a LABEL SOURCE for
naming the vocabulary. That matters because naming is the unsolved problem: there are only 3 coarse
plane labels here, audited unreliable (3 of 6 wrong at conf=1.00), NO subplane labels exist anywhere,
and expert labelling is the expensive bottleneck. Meanwhile a rich panel (Doppler, cardiac, biometry,
maternal, demographic) sits unused for naming. Descriptive by construction, so it needs no image->outcome
signal to be worth having -- which is precisely why it survives all the nulls above.

FOUR TRAPS, each handled explicitly:

 1. GA WOULD DOMINATE EVERY NAME. Codes carry GA (up to r=0.40 per plane), so every GA-correlated
    tabular variable lights up for every code and all 16 "names" collapse to the same name. GA is
    adjusted out with a spline. The PRE-adjustment version is reported too, as the positive control
    that the method has power at all.

 2. ADJUSTING FOR A VARIABLE MAKES IT UNNAMEABLE. If sex is adjusted out, no code can ever be named
    "male-associated". So the adjustment set is deliberately MINIMAL -- GA spline + plane composition,
    the two structural confounds -- and sex, size, Doppler, cardiac and maternal stay NAMEABLE. This
    asymmetry is a declared design choice, not an oversight.

 3. COMPOSITIONAL CLOSURE. Shares sum to 1, so a real effect on one code drives the other K-1 negative
    by arithmetic alone (measured: planting signal on one code drove all 15 others negative, mean
    -0.110, against a predicted floor -r0/(K-1)). Profiles are computed on CLR coordinates and the
    closure floor is reported beside every effect.

 4. MULTIPLICITY. 16 codes x ~O(100) variables is thousands of tests. Benjamini-Yekutieli (valid under
    the arbitrary dependence a correlated panel guarantees), and results are reported as VARIABLE
    FAMILIES rather than single winners, because the panel is internally correlated -- a profile should
    read "this code sits with the placental-Doppler family", not "variable #47".

THE PART THAT MAKES IT PUBLISHABLE -- NAMING RELIABILITY, already paid for. The 5 seeds from
hpc_seed_ceiling.py are on disk. Match codes ACROSS seeds (Hungarian on centroids), then ask whether a
matched code receives the SAME tabular profile. A code is only NAMED if its profile replicates. The
interpretability literature almost never has a naming-reliability criterion -- it is normally "here are
top-activating patches, trust us" -- and we can do better because the vocabulary itself is reproducible
(across-seed AMI 0.710-0.797).

CONTROLS: positive = GA recovers pre-adjustment; negative = shuffle the fetus<->code assignment and
profiles must vanish; reliability = profile reproduces across matched codes in >=4 of 5 seeds.

USAGE: python hpc_name_codes_tabular.py [--K 16] [--enc FetalCLIP] [--cohort impact]
Outputs: out_probe/code_names_K16.json  (+ per-code top-loading families)
"""
import os, sys, json, argparse, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
OUT=os.environ.get("GA_OUT_DIR",os.path.join(HERE,"out_usfmae"))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
ECHO=os.path.join(ROOT,"data_local","IMPACT_ecocardio_zscores_corrected.xlsx")
PLANES=["abdominal","cerebral","femur"]
MIN_OBS=200          # a variable must have this many observed fetuses to be nameable
RELIABLE_SEEDS=4     # a profile must replicate in >=4 of 5 seeds to count as a NAME

def eu_numeric(s):
    """MANY numeric columns in this file are EUROPEAN-DECIMAL STRINGS ('25,97055516'). A plain
    to_numeric silently returns NaN for all but a handful -- that is how the Hadlock GA target came
    out as n=162 when the true coverage is 989, a 6x understatement I reported before catching it.
    Apply per column and KEEP THE PARSE THAT RECOVERS MORE VALUES WITHOUT CHANGING MAGNITUDE: naively
    stripping '.' as a thousands separator multiplies genuine decimals by 10 (BPD 83 -> 830)."""
    plain=pd.to_numeric(s,errors="coerce")
    swapped=pd.to_numeric(s.astype(str).str.strip().str.replace(",",".",regex=False),errors="coerce")
    if swapped.notna().sum()>1.5*max(plain.notna().sum(),1): return swapped
    return plain

# ORDER MATTERS and substrings are dangerous: an earlier draft matched "et_" inside "percentil" and
# filed percentil_birth as CARDIAC. Outcome/biometry patterns are therefore checked FIRST, and the
# cardiac timing intervals use anchored tokens (ictms/etms/irtms) rather than bare fragments.
FAMILY_PATTERNS=[("outcome",("percentil","peso_rn","birth","sga","lga","apgar","ph_","nicu")),
                 ("biometry",("dbp_","_pc_","pc_eco","_pa_","pa_eco","_lf_","lf_eco","efw","talla_f")),
                 ("doppler",("uta","_au","au_","acm","cpr","_dv","dv_","ithsm","umbil","uterin")),
                 ("cardiac",("tapse","mapse","sapse","mpi","ictms","etms","irtms","cardiac_area",
                             "septum","_lv","lv_","_rv","rv_","longitudinal","basal")),
                 ("maternal",("bmi","imc","map","edad","tension","habito","fuma","paridad","peso_m",
                              "visnutri")),
                 ("demographic",("nivel","estudi","trabaj","etnia","social","econom","civil"))]
def family_of(name):
    n=str(name).lower()
    for fam,pats in FAMILY_PATTERNS:
        if any(p in n for p in pats): return fam
    return "other"

def ga_spline(x):
    x=np.asarray(x,float); q=np.nanpercentile(x,[25,50,75])
    return np.column_stack([x]+[np.clip(x-k,0,None)**3 for k in q])

def clr(P,eps=1e-6):
    Q=np.clip(P,eps,None); L=np.log(Q); return L-L.mean(1,keepdims=True)

def partial_spear(y,x,C,min_n=MIN_OBS):
    m=np.isfinite(y)&np.isfinite(x)&np.isfinite(C).all(1)
    if m.sum()<min_n: return np.nan,np.nan,int(m.sum())
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

def per_fetus_shares(codes,nid,plane,K,per_plane,rng):
    """plane-stratified, budget-equalised per-fetus code shares + the plane composition vector
    (kept as a COVARIATE, since plane composition is a structural confound but plane itself must not
    be adjusted so hard that plane-specific codes become unnameable)."""
    H={}; PC={}
    for f in np.unique(nid):
        fi=np.where(nid==f)[0]; picks=[]; comp=np.zeros(len(PLANES))
        for pi,pl in enumerate(PLANES):
            idx=fi[plane[fi]==pl]
            comp[pi]=len(idx)
            if len(idx)==0: continue
            flat=codes[idx].reshape(-1)
            picks.append(rng.choice(flat,per_plane,replace=len(flat)<per_plane))
        if not picks: continue
        h=np.bincount(np.concatenate(picks),minlength=K).astype(float)
        H[f]=h/h.sum(); PC[f]=comp/max(comp.sum(),1)
    return H,PC

def profile_matrix(P,T,C,cols):
    """(K x n_vars) profile of partial Spearman effects, plus p-values."""
    K=P.shape[1]; R=np.full((K,len(cols)),np.nan); Pv=np.full((K,len(cols)),np.nan)
    for k in range(K):
        for j,c in enumerate(cols):
            R[k,j],Pv[k,j],_=partial_spear(P[:,k],T[:,j],C)
    return R,Pv

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--K",type=int,default=16); ap.add_argument("--enc",default="FetalCLIP")
    ap.add_argument("--cohort",default="impact"); ap.add_argument("--per-plane",type=int,default=120)
    a=ap.parse_args(); rng=np.random.default_rng(0)
    tag='' if a.cohort=="impact" else '_clin'
    src=os.path.join(OUT,f"wp2codes_{a.enc}_K{a.K}{tag}.npz")
    full=os.path.join(OUT,f"wp2fullcodes_{a.enc}_K{a.K}{tag}.npz")
    use=full if os.path.exists(full) else src
    assert os.path.exists(use), f"missing {use} -- run hpc_seed_ceiling.py (and optionally --assign-only)"
    z=np.load(use,allow_pickle=True)
    codes,nid,plane,gaf=z["codes"],z["nid"].astype(str),z["plane"].astype(str),z["ga"].astype(float)
    res={"source":os.path.basename(use),"K":a.K,"encoder":a.enc,"cohort":a.cohort,
         "direction":"tabular as the DICTIONARY for naming codes (not codes as features to beat tabular)",
         "adjustment_set":"GA spline + plane composition ONLY -- sex/size/Doppler/cardiac/maternal stay NAMEABLE",
         "min_obs_per_variable":MIN_OBS,"reliability_rule":f"profile must replicate in >={RELIABLE_SEEDS} of 5 seeds"}

    # ---- wide tabular panel, european-decimal aware ----
    e=pd.read_excel(ECHO); e["nid"]=e["Cod"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
    e=e.drop_duplicates("nid").set_index("nid")
    num={}
    for c in e.columns:
        if str(c)=="nid": continue
        v=eu_numeric(e[c])
        if v.notna().sum()>=MIN_OBS and v.nunique(dropna=True)>2:
            # OUTCOMES ARE EVAL-ONLY in this project and must never become part of a code's NAME --
            # naming a code by birth percentile would smuggle the outcome into the description.
            if family_of(c)=="outcome": continue
            num[str(c)]=v
    panel=pd.DataFrame(num)
    res["n_tabular_variables"]=int(panel.shape[1])
    fams=pd.Series({c:family_of(c) for c in panel.columns})
    res["variables_per_family"]=fams.value_counts().to_dict()
    res["outcomes_excluded_from_naming"]=True
    print(f"  tabular panel: {panel.shape[1]} numeric variables with >={MIN_OBS} obs "
          f"| families {res['variables_per_family']}",flush=True)

    # ---- per-fetus shares (seed 0) + covariates ----
    H,PC=per_fetus_shares(codes,nid,plane,a.K,a.per_plane,rng)
    fet=np.array(sorted(H)); Praw=np.vstack([H[f] for f in fet]); Pc=clr(Praw)
    comp=np.vstack([PC[f] for f in fet])
    gav=np.array([np.nanmedian(gaf[nid==f]) for f in fet])
    T=panel.reindex(fet).values.astype(float); cols=list(panel.columns)
    C_adj=np.column_stack([ga_spline(gav),comp[:,:2]])          # GA + plane composition
    C_raw=np.zeros((len(fet),1))                                # unadjusted (positive control)
    print(f"  fetuses {len(fet)} | shares {Praw.shape}",flush=True)

    # ---- positive control: does GA itself come out pre-adjustment? ----
    ga_eff=[partial_spear(Praw[:,k],gav,C_raw)[0] for k in range(a.K)]
    res["positive_control_GA_percode_max_abs_r"]=float(np.nanmax(np.abs(ga_eff)))
    print(f"  POSITIVE CONTROL (unadjusted GA): per-code max|r| {np.nanmax(np.abs(ga_eff)):.3f}",flush=True)

    # ---- profiles, adjusted ----
    R,Pv=profile_matrix(Pc,T,C_adj,cols)
    q=by_correct(Pv.reshape(-1)).reshape(Pv.shape)
    res["n_tests"]=int(np.isfinite(Pv).sum())
    # ---- negative control: shuffle fetus<->code assignment ----
    perm=rng.permutation(len(fet))
    Rn,_=profile_matrix(Pc[perm],T,C_adj,cols)
    res["negative_control_shuffled_max_abs_r"]=float(np.nanmax(np.abs(Rn)))
    print(f"  NEGATIVE CONTROL (shuffled fetus<->code): max|r| {np.nanmax(np.abs(Rn)):.3f} "
          f"vs real {np.nanmax(np.abs(R)):.3f}",flush=True)

    # ---- naming reliability across the 5 saved seeds ----
    rel={}
    if "codes_all" in z.files and "centroids_all" in z.files:
        CA=z["centroids_all"]; KA=z["codes_all"]
        base=CA[0]/ (np.linalg.norm(CA[0],axis=1,keepdims=True)+1e-9)
        prof_by_seed={0:R}
        for sN in range(1,KA.shape[0]):
            cn=CA[sN]/(np.linalg.norm(CA[sN],axis=1,keepdims=True)+1e-9)
            r_,c_=linear_sum_assignment(-(base@cn.T))            # Hungarian match seed sN -> seed 0
            Hs,_=per_fetus_shares(KA[sN],nid,plane,a.K,a.per_plane,np.random.default_rng(sN))
            fs=np.array(sorted(Hs)); Ps=clr(np.vstack([Hs[f] for f in fs]))
            # align fetus order and code order to seed 0
            idxf=pd.Index(fs).get_indexer(fet); ok=idxf>=0
            Ps=Ps[idxf[ok]][:,c_]
            Rs,_=profile_matrix(Ps,T[ok],C_adj[ok],cols)
            prof_by_seed[sN]=Rs
        for k in range(a.K):
            cors=[]
            for sN,Rs in prof_by_seed.items():
                if sN==0: continue
                m=np.isfinite(R[k])&np.isfinite(Rs[k])
                if m.sum()>10: cors.append(float(spearmanr(R[k][m],Rs[k][m]).statistic))
            rel[k]={"profile_corr_per_seed":cors,
                    "n_seeds_ge_0.5":int(np.sum(np.array(cors)>=0.5)) if cors else 0,
                    "median_corr":float(np.median(cors)) if cors else np.nan}
        n_named=sum(1 for k in rel if rel[k]["n_seeds_ge_0.5"]>=RELIABLE_SEEDS-1)
        res["naming_reliability"]=rel; res["n_codes_with_reliable_profile"]=n_named
        print(f"  RELIABILITY: {n_named} of {a.K} codes have a profile reproducing in "
              f">={RELIABLE_SEEDS-1} of {KA.shape[0]-1} other seeds",flush=True)
    else:
        res["naming_reliability"]="unavailable -- rerun hpc_seed_ceiling.py to save centroids_all/codes_all"
        print("  RELIABILITY: SKIPPED (npz predates all-seed saving)",flush=True)

    # ---- the names ----
    names={}
    for k in range(a.K):
        ordr=np.argsort(-np.abs(np.nan_to_num(R[k])))
        top=[{"variable":cols[j],"family":family_of(cols[j]),"r":float(R[k,j]),"q_BY":float(q[k,j]),
              "closure_floor":float(-R[k,j]/(a.K-1))} for j in ordr[:8] if np.isfinite(R[k,j])]
        famscore={}
        for j in range(len(cols)):
            if np.isfinite(R[k,j]): famscore.setdefault(family_of(cols[j]),[]).append(abs(R[k,j]))
        famrank=sorted(((f,float(np.mean(v))) for f,v in famscore.items()),key=lambda t:-t[1])
        names[k]={"usage":float(Praw[:,k].mean()),"top_variables":top,
                  "family_ranking":famrank,
                  "reliable":bool(rel.get(k,{}).get("n_seeds_ge_0.5",0)>=RELIABLE_SEEDS-1) if rel else None,
                  "n_survive_BY_q0.10":int(np.nansum(q[k]<0.10))}
        fam0=famrank[0][0] if famrank else "?"
        print(f"    code {k:2d} usage {Praw[:,k].mean():.3f} | top family {fam0:12s} | "
              f"top var {top[0]['variable'][:28] if top else '-':30s} r={top[0]['r'] if top else float('nan'):+.3f} | "
              f"BY survivors {int(np.nansum(q[k]<0.10)):3d}"
              +(" | RELIABLE" if names[k]["reliable"] else ""),flush=True)
    res["code_names"]=names
    res["n_surviving_BY_q0.10_total"]=int(np.nansum(q<0.10))
    print(f"\n  {res['n_surviving_BY_q0.10_total']} of {res['n_tests']} code x variable tests survive BY q<0.10",flush=True)
    json.dump(res,open(os.path.join(OUTP,f"code_names_K{a.K}{tag}.json"),"w"),indent=2,default=str)
    print(f"saved out_probe/code_names_K{a.K}{tag}.json\nDONE",flush=True)

if __name__=="__main__": main()
