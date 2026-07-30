#!/usr/bin/env python3
"""IS THE FROZEN ENCODER'S BIRTH-PERCENTILE SIGNAL SIZE-FREE? The stronger adjustment.

THE CLAIM UNDER TEST. In the LoRA run the FROZEN FetalCLIP embedding predicted birth percentile at
out-of-fold rho +0.262 raw and +0.213 after "size-orthogonalisation" (SGA 0.196 -> 0.163, LGA
0.155 -> 0.108). That +0.213 is the only unexplained image -> growth-outcome signal in this project,
against a long registry of nulls (>=12 construction families; 1-vs-rest AUCs FGR 0.412, LGA 0.424;
0 of 144 per-plane per-code enrichment tests surviving BY q<0.10).

WHY THAT +0.213 IS NOT YET TRUSTWORTHY, and it is my own adjustment that is weak. It orthogonalised the
appearance score against the EMBEDDING'S OWN PREDICTED EFW. Predicted size is a lossy, noisy proxy for
size: whatever size information the embedding holds NON-linearly, or holds but predicts poorly, stays in
the residual. Removing a noisy proxy removes only part of the confound and leaves an attenuated version
of exactly the settled size finding -- which would then read as a new discovery. The honest test adjusts
for MEASURED biometry.

WHAT THIS SCRIPT DOES, in increasing strictness:
  A. raw            no adjustment (reference)
  B. self-predicted the original weak version, reproduced here so the comparison is like-for-like
  C. measured-EFW   partial out measured EFW
  D. measured-ALL   partial out BPD, HC, AC, FL, EFW jointly -- the real test
  E. + GA           D plus a GA spline, since GA drives both appearance and size
  F. GA-only        adjust for GA but NOT size, to separate "is it size" from "is it maturation"
All adjustments are RANK-based partial correlations with the nuisance model FITTED INSIDE TRAINING FOLDS
(GroupKFold by fetus), never on the test fold.

CONTROLS:
  - a SHUFFLE null (permute the outcome across fetuses) at every adjustment level, to confirm the
    machinery returns ~0 when there is nothing to find;
  - a POSITIVE control (predict measured EFW itself), to confirm the readout has power at all;
  - bootstrap CIs over FETUSES (2000 draws), because this project measured within-fetus ICC 0.12-0.32
    and frame-level CIs were 2.7-4.4x too narrow.

INTERPRETATION FIXED IN ADVANCE, so the result cannot be read opportunistically:
  * if D/E collapse to within the shuffle band -> the signal WAS size, the null registry stands, and the
    +0.213 was an artefact of adjusting for a noisy proxy. This is the expected outcome.
  * if D/E survive with a CI excluding zero -> the frozen encoder carries growth-outcome information
    that measured biometry does not explain. That is a genuine positive and the first in this project.
  * F distinguishes maturation from size: if F is large while D/E are null, the signal is GA.

USAGE: python hpc_size_orthogonal_check.py [--n-boot 2000]
Output: out_probe/size_orthogonal_check.json
"""
import os, sys, json, argparse, numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import GroupKFold
from scipy.stats import rankdata, spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
HAND=os.path.join(HERE,"handoff")
DEV="cuda" if torch.cuda.is_available() else "cpu"
from hpc_extract_4encoders import BUILDERS, frame_table
BIOM=["DBP_ecoIMPACT","PC_ecoIMPACT","PA_ecoIMPACT","LF_ecoIMPACT","EFW_ecoIMPACT"]
EVAL_CAP=8

def eu(s):
    plain=pd.to_numeric(s,errors="coerce")
    sw=pd.to_numeric(s.astype(str).str.strip().str.replace(",",".",regex=False),errors="coerce")
    return sw if sw.notna().sum()>1.5*max(plain.notna().sum(),1) else plain

def ga_spline(x):
    x=np.asarray(x,float); q=np.nanpercentile(x,[25,50,75])
    return np.column_stack([x]+[np.clip(x-k,0,None)**3 for k in q])

def oof_predict(Z,y,folds):
    yy=np.asarray(y,float); pred=np.full(len(yy),np.nan)
    for k in np.unique(folds):
        tr=(folds!=k)&np.isfinite(yy); te=(folds==k)
        if tr.sum()<50 or te.sum()<5: continue
        pred[te]=RidgeCV(alphas=np.logspace(-2,4,25)).fit(Z[tr],yy[tr]).predict(Z[te])
    return pred

def partial_rho(a,y,C,folds):
    """rank partial correlation with the nuisance model fitted INSIDE training folds only."""
    ra=np.full(len(a),np.nan); ry=np.full(len(a),np.nan)
    ok=np.isfinite(a)&np.isfinite(y)&(np.isfinite(C).all(1) if C is not None else True)
    if ok.sum()<60: return np.nan,0
    A=rankdata(a[ok]).astype(float); Y=rankdata(y[ok]).astype(float)
    if C is None: return float(spearmanr(A,Y).statistic),int(ok.sum())
    Cc=C[ok]; f=folds[ok]
    ea=np.full(ok.sum(),np.nan); ey=np.full(ok.sum(),np.nan)
    for k in np.unique(f):
        tr=f!=k; te=f==k
        if tr.sum()<40 or te.sum()<3: continue
        ma=LinearRegression().fit(Cc[tr],A[tr]); my=LinearRegression().fit(Cc[tr],Y[tr])
        ea[te]=A[te]-ma.predict(Cc[te]); ey[te]=Y[te]-my.predict(Cc[te])
    m=np.isfinite(ea)&np.isfinite(ey)
    if m.sum()<60: return np.nan,int(m.sum())
    return float(spearmanr(ea[m],ey[m]).statistic),int(m.sum())

def boot_ci(a,y,C,folds,n_boot,rng):
    """bootstrap over FETUSES -- frame-level CIs were 2.7-4.4x too narrow at the measured ICC."""
    n=len(a); out=[]
    for _ in range(n_boot):
        ii=rng.integers(0,n,n)
        r,_=partial_rho(a[ii],y[ii],None if C is None else C[ii],folds[ii])
        if np.isfinite(r): out.append(r)
    if len(out)<50: return (np.nan,np.nan)
    return (float(np.percentile(out,2.5)),float(np.percentile(out,97.5)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n-boot",type=int,default=2000)
    ap.add_argument("--folds",type=int,default=5); a=ap.parse_args()
    rng=np.random.default_rng(0)
    # ---- frozen embeddings, per fetus (median over up to EVAL_CAP frames) ----
    df=frame_table(); m,tf,_=BUILDERS["FetalCLIP"](); m=m.to(DEV).eval()
    cache=os.path.join(HAND,"frozen_fetus_embed.npz")
    if os.path.exists(cache):
        z=np.load(cache,allow_pickle=True); fe=z["fid"].astype(str); Z=z["Z"]
        print(f"  loaded cached frozen embeddings {Z.shape}",flush=True)
    else:
        idx=df.groupby("nid").indices; fids=sorted(idx); rows=[]
        with torch.no_grad():
            for i,f in enumerate(fids):
                ii=idx[f][:EVAL_CAP]
                x=torch.stack([tf(Image.open(df["img"].iloc[j]).convert("RGB")) for j in ii]).to(DEV)
                h=m(x) if not hasattr(m,"encode_image") else m.encode_image(x)
                if isinstance(h,(tuple,list)): h=h[0]
                rows.append(h.float().cpu().numpy().mean(0) if h.ndim==2 else h.float().cpu().numpy())
                if i%100==0: print(f"    embed {i}/{len(fids)}",flush=True)
        fe=np.array(fids); Z=np.stack(rows)
        np.savez(cache,fid=fe,Z=Z); print(f"  cached {Z.shape} -> {cache}",flush=True)
    # ---- targets ----
    e=pd.read_excel(os.path.join(ROOT,"data_local","IMPACT_ecocardio_zscores_corrected.xlsx"))
    e["nid"]=e["Cod"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
    e=e.drop_duplicates("nid").set_index("nid")
    B=pd.DataFrame({c:eu(e[c]) for c in BIOM}).reindex(fe)
    ga=eu(e["EG_ecoIMPACT"]).reindex(fe).values.astype(float)
    bp=eu(e["percentil_birth"]).reindex(fe).values.astype(float)
    sga=pd.to_numeric(e["SGA_birth"],errors="coerce").reindex(fe)
    sga=(sga>0).astype(float).where(sga.notna()).values
    assert np.isfinite(bp).sum()>500 and np.nansum(sga)>100, \
        f"outcome parse implausible: bp={np.isfinite(bp).sum()} sga={np.nansum(sga)}"
    folds=np.zeros(len(fe),dtype=int)
    for k,(_,te) in enumerate(GroupKFold(a.folds).split(Z,groups=fe)): folds[te]=k
    print(f"  fetuses {len(fe)} | birth_pct n={int(np.isfinite(bp).sum())} SGA={int(np.nansum(sga))}",flush=True)

    res={"n_fetuses":int(len(fe)),"claim_under_test":
         "frozen FetalCLIP -> birth percentile, rho +0.262 raw / +0.213 after self-predicted-size adjustment",
         "why":"self-predicted EFW is a NOISY PROXY for size; removing it leaves an attenuated copy of "
               "the settled size finding, which would read as a new discovery",
         "n_boot":a.n_boot,"levels":{}}
    appear=oof_predict(Z,bp,folds)                      # appearance score for birth percentile
    size_self=oof_predict(Z,B["EFW_ecoIMPACT"].values,folds)
    Bm=B.values.astype(float)
    LEVELS={"A_raw":None,
            "B_self_predicted_EFW":size_self[:,None],
            "C_measured_EFW":Bm[:,[4]],
            "D_measured_ALL_biometry":Bm,
            "E_measured_ALL_plus_GA":np.column_stack([Bm,ga_spline(ga)]),
            "F_GA_only":ga_spline(ga)}
    for name,C in LEVELS.items():
        r,n=partial_rho(appear,bp,C,folds)
        lo,hi=boot_ci(appear,bp,C,folds,a.n_boot,rng)
        # shuffle null at THIS adjustment level
        nul=[]
        for _ in range(200):
            r0,_=partial_rho(appear,rng.permutation(bp),C,folds)
            if np.isfinite(r0): nul.append(abs(r0))
        res["levels"][name]={"rho":r,"n":n,"ci95":[lo,hi],
                             "shuffle_null_p95_abs":float(np.percentile(nul,95)) if nul else np.nan}
        print(f"  {name:26s} rho {r:+.3f} CI[{lo:+.3f},{hi:+.3f}] "
              f"shuffle|r|p95 {res['levels'][name]['shuffle_null_p95_abs']:.3f} n={n}",flush=True)
    # positive control: can this readout recover measured EFW at all?
    pc,_=partial_rho(oof_predict(Z,B["EFW_ecoIMPACT"].values,folds),B["EFW_ecoIMPACT"].values,None,folds)
    res["positive_control_predict_measured_EFW"]=pc
    print(f"  POSITIVE CONTROL (predict measured EFW): rho {pc:+.3f}",flush=True)
    # same battery for SGA
    res["sga_levels"]={}
    ap_sga=oof_predict(Z,sga,folds)
    for name in ("A_raw","D_measured_ALL_biometry","E_measured_ALL_plus_GA"):
        r,n=partial_rho(ap_sga,sga,LEVELS[name],folds)
        lo,hi=boot_ci(ap_sga,sga,LEVELS[name],folds,a.n_boot,rng)
        res["sga_levels"][name]={"rho":r,"n":n,"ci95":[lo,hi]}
        print(f"  [SGA] {name:26s} rho {r:+.3f} CI[{lo:+.3f},{hi:+.3f}]",flush=True)
    D=res["levels"]["D_measured_ALL_biometry"]; E=res["levels"]["E_measured_ALL_plus_GA"]
    def survives(d): return np.isfinite(d["rho"]) and abs(d["rho"])>d["shuffle_null_p95_abs"] and \
                            np.isfinite(d["ci95"][0]) and d["ci95"][0]*d["ci95"][1]>0
    if survives(D) and survives(E):
        res["VERDICT"]=(f"SURVIVES measured-biometry adjustment (D rho {D['rho']:+.3f} "
          f"CI[{D['ci95'][0]:+.3f},{D['ci95'][1]:+.3f}], E rho {E['rho']:+.3f}) -- the frozen encoder "
          "carries growth-outcome information that measured biometry and GA do not explain. First "
          "genuine image->outcome positive in this project; needs encoder replication before use.")
    else:
        res["VERDICT"]=(f"DOES NOT SURVIVE (D rho {D['rho']:+.3f}, E rho {E['rho']:+.3f} vs shuffle p95 "
          f"{D['shuffle_null_p95_abs']:.3f}) -- the +0.213 was an artefact of adjusting for a NOISY "
          "PROXY (self-predicted EFW) instead of measured biometry. The size/null registry stands.")
    print(f"\n  {res['VERDICT']}",flush=True)
    json.dump(res,open(os.path.join(OUTP,"size_orthogonal_check.json"),"w"),indent=2,default=str)
    print("saved out_probe/size_orthogonal_check.json\nDONE",flush=True)

if __name__=="__main__": main()
