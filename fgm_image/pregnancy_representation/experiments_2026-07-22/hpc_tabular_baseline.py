#!/usr/bin/env python3
"""TABULAR BASELINE (Hadlock GA from biometry) + COVARIATE RESIDUALISATION HOOK.

Two jobs:

(A) HADLOCK BASELINE -- the reference the image clock must be judged against. Hadlock (1984)
    regression estimates GA from BPD/HC/AC/FL; clinically it is ~+-1-2 weeks in the third
    trimester, so it may well BEAT our image clock. That comparison has never been made here and
    without it "image clock r=0.469" has no clinical yardstick.
    Columns in IMPACT_ecocardio_zscores_corrected.xlsx (Spanish): DBP=BPD, PC=HC, PA=AC, LF=FL,
    EG_ecoIMPACT = GA at that visit (weeks).
    !! DATING CIRCULARITY: if GA was originally established by early-ultrasound biometry, then
    "biometry predicts GA" is partly tautological -- the same measurement family that SET the
    dating is being used to recover it. This script reports the check (agreement between
    EG_reclut/recruitment GA and EG_ecoIMPACT, and the implied dating source) and prints a WARNING
    with the Hadlock numbers so they are never quoted as a clean independent baseline.

(B) COVARIATE RESIDUALISATION -- pixel spacing + fetal sex (extensible) as COVARIATES, not as
    codebook inputs. Codes stay image-only ("what the tissue looks like"), and the covariates are
    removed from BOTH sides before the readout, so what survives is appearance-specific. Feeding
    them INTO the codebook would make a code mean "appearance + machine setting + sex", which
    breaks the interpretability claim.
    Writes covariates_residual.csv = per-frame covariate matrix + a residualiser other scripts
    import: residualise(X, C) -> X with C's linear contribution removed (fit on train folds only).

USAGE: python hpc_tabular_baseline.py [--echo <xlsx>] [--spacing <csv>]
Outputs: out_probe/tabular_baseline.json, handoff/covariates.csv
"""
import os, json, argparse, numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
HAND=os.path.join(HERE,"handoff"); os.makedirs(HAND,exist_ok=True)
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")
ECHO=os.path.join(ROOT,"data_local","IMPACT_ecocardio_zscores_corrected.xlsx")

# ---- Hadlock 1984 four-parameter regression: GA(weeks) from BPD,HC,AC,FL in CM ----
def hadlock_ga(bpd_cm,hc_cm,ac_cm,fl_cm):
    """Hadlock et al. 1984 'Estimating fetal age: computer-assisted analysis of multiple fetal
    growth parameters'. Four-parameter form (BPD+HC+AC+FL), inputs in CENTIMETRES."""
    return (10.85 + 0.060*hc_cm*fl_cm + 0.6700*bpd_cm + 0.1680*ac_cm)

def residualise(X, C, groups=None, folds=5):
    """remove C's linear contribution from X. If groups given, fit per training fold (no leakage)."""
    X=np.asarray(X,float); C=np.asarray(C,float)
    keep=np.isfinite(C).all(1)
    R=X.copy()
    if groups is None:
        lr=LinearRegression().fit(C[keep],X[keep]); R[keep]=X[keep]-lr.predict(C[keep]); return R
    for tr,te in GroupKFold(folds).split(X,groups=groups):
        tr=tr[keep[tr]]; te2=te[keep[te]]
        if len(tr)<10 or len(te2)==0: continue
        lr=LinearRegression().fit(C[tr],X[tr]); R[te2]=X[te2]-lr.predict(C[te2])
    return R

def oof(X,y,g,alpha=10.0):
    p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,groups=g):
        sc=StandardScaler().fit(X[tr]); p[te]=Ridge(alpha=alpha).fit(sc.transform(X[tr]),y[tr]).predict(sc.transform(X[te]))
    return p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--echo",default=ECHO); ap.add_argument("--spacing",default=os.path.join(HAND,"fetus_spacing.csv"))
    a=ap.parse_args()
    res={}
    e=pd.read_excel(a.echo)
    e["nid"]=e["Cod"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
    e=e.drop_duplicates("nid").set_index("nid")
    B={"bpd":"DBP_ecoIMPACT","hc":"PC_ecoIMPACT","ac":"PA_ecoIMPACT","fl":"LF_ecoIMPACT"}
    for k,v in B.items(): e[k]=pd.to_numeric(e[v],errors="coerce")
    ga_eco=pd.to_numeric(e["EG_ecoIMPACT"],errors="coerce")
    # units: mm vs cm -- decide from the median BPD (third-trimester BPD ~ 85mm = 8.5cm)
    med=e["bpd"].median(); unit="mm" if med>25 else "cm"
    f=0.1 if unit=="mm" else 1.0
    res["biometry_units_detected"]=unit; res["median_BPD_raw"]=float(med)
    print(f"  biometry units detected: {unit} (median BPD raw {med:.1f})",flush=True)
    had=hadlock_ga(e["bpd"]*f,e["hc"]*f,e["ac"]*f,e["fl"]*f)
    m=np.isfinite(had)&np.isfinite(ga_eco)&(ga_eco>=6)&(ga_eco<=42)
    r,_=pearsonr(had[m],ga_eco[m]); mae=float(np.abs(had[m]-ga_eco[m]).mean())
    res["hadlock_vs_scanGA"]={"r":float(r),"MAE_wk":mae,"MAE_days":mae*7,"n":int(m.sum())}
    print(f"  HADLOCK (tabular only) vs scan GA: r={r:.3f} MAE={mae:.2f}wk ({mae*7:.1f} days) n={m.sum()}",flush=True)
    # learned tabular baseline: ridge on the 4 raw measures (upper bound for tabular)
    X=e.loc[m,["bpd","hc","ac","fl"]].values*f; y=ga_eco[m].values; g=np.arange(m.sum())
    pr=oof(X,y,g); rr,_=pearsonr(pr,y)
    res["ridge_biometry_vs_scanGA"]={"r":float(rr),"MAE_wk":float(np.abs(pr-y).mean()),"n":int(m.sum())}
    print(f"  RIDGE on 4 raw measures    vs scan GA: r={rr:.3f} MAE={np.abs(pr-y).mean():.2f}wk",flush=True)
    # ---- dating-circularity check ----
    ga_rec=pd.to_numeric(e.get("EG_reclut"),errors="coerce")
    if ga_rec is not None:
        mm=np.isfinite(ga_rec)&np.isfinite(ga_eco)
        res["dating_check"]={"corr_EGreclut_vs_EGecoIMPACT":float(pearsonr(ga_rec[mm],ga_eco[mm])[0]) if mm.sum()>10 else None,
          "n":int(mm.sum()),
          "WARNING":("If GA was dated by EARLY-ULTRASOUND BIOMETRY then Hadlock-vs-GA is PARTLY "
                     "CIRCULAR: the same measurement family that set the dating is recovering it. "
                     "These numbers are a CLINICAL YARDSTICK, not an independent baseline. Confirm "
                     "the dating source (LMP vs early scan) before quoting them in the paper.")}
        print("  !! dating circularity: see WARNING in json -- Hadlock is a yardstick, not an independent baseline",flush=True)
    # ---- (B) covariates: spacing + sex, per FRAME ----
    idx=pd.read_csv(INDEX); idx["nid"]=idx["nid"].apply(lambda x: str(int(float(x))))
    cov=idx[["nid","new_filename","ga_weeks_recovered","plane_prop"]].copy()
    sex=pd.to_numeric(e.get("SexoRN_cod"),errors="coerce")
    cov["sex"]=cov["nid"].map(sex.to_dict())
    if os.path.exists(a.spacing):
        sp=pd.read_csv(a.spacing); c0=sp.columns[0]
        sp[c0]=sp[c0].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
        spc=[c for c in sp.columns if "spac" in c.lower()]
        if spc: cov["spacing"]=cov["nid"].map(dict(zip(sp[c0],pd.to_numeric(sp[spc[0]],errors="coerce"))))
    have=[c for c in ["spacing","sex"] if c in cov.columns]
    res["covariates"]={c:{"coverage":float(cov[c].notna().mean()),"n":int(cov[c].notna().sum())} for c in have}
    cov.to_csv(os.path.join(HAND,"covariates.csv"),index=False)
    print(f"  covariates written ({have}) coverage "
          f"{ {c:round(cov[c].notna().mean(),3) for c in have} } -> handoff/covariates.csv",flush=True)
    res["covariate_policy"]=("spacing and sex are COVARIATES, not codebook inputs: residualise(X,C) "
        "removes their linear contribution from the code features (fit on TRAIN folds only) before "
        "the readout, so surviving signal is appearance-specific. Feeding them into the codebook "
        "would make a code mean 'appearance + machine setting + sex'.")
    json.dump(res,open(os.path.join(OUTP,"tabular_baseline.json"),"w"),indent=2)
    print("saved out_probe/tabular_baseline.json\nDONE",flush=True)

if __name__=="__main__": main()
