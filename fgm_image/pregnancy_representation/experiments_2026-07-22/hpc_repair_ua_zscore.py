#!/usr/bin/env python3
"""REPAIR the umbilical-artery Doppler z-score. Writes handoff/ua_zscore_recomputed.csv.

WHY THE SHIPPED COLUMN CANNOT BE USED. `Zscore_AU` in the IMPACT tabular file is labelled as an
umbilical-artery z-score but behaves like a gestational-age index:

    spearman with GA              +0.978        (a UA z-score must be ~0 -- z-scoring REMOVES the
                                                 physiological GA trend by construction)
    spearman with its OWN raw     -0.066        (UA_ecoIMPACT, the raw UA PI it should be derived from)
    SD                            0.235         (a z-score has SD ~1 in its reference population)
    distinct values               57 over 981 fetuses, heavy ties (69 fetuses share one value)
    Percentil_AU                  SD 4.06, MINIMUM at the 34.4th centile -- no fetus below it

The two columns are internally consistent (Percentil_AU == Phi(Zscore_AU)*100 to 0.00 precision), so the
percentile conversion is fine and the compressed z is the broken INPUT. The clean sibling Percentil_UTA
spans 0-100 with SD 33.4, as a percentile should. Two values >=6 hold ~90% of the column's variance (one
6.009 alone holds 38.9%), which is what made an earlier variance decomposition read 49% GA-explained when
the bulk figure is ~0.97-0.99.

Most likely identity: a GA-indexed reference-curve value (the chart's EXPECTED value at that GA, not the
deviation), plausibly broadcast by a merge on GA-at-visit rather than on fetus -- which explains 57
distinct values across 55 GA levels and the exact ties.

DOWNSTREAM CONSEQUENCE, already established: this project's one surviving image-tabular positive was a
held-out CCA 'placental direction' whose tabular side loaded -0.96 on AU, i.e. this column WAS the entire
tabular canonical variate. A variable that does not track its own raw measurement carries no
fetus-specific placental information, so the INTERPRETATION as placental is dead independent of any
further check. Whether the CORRELATION survives is a separate question, testable by re-running that CCA
with the repaired column.

WHAT THIS SCRIPT COMPUTES, and its honest limitation. log(UA_ecoIMPACT) is residualised on a GA spline
and MAD-scaled. UA PI is right-skewed, so the log is taken first; the MAD scale is robust to the handful
of genuinely extreme fetuses. An earlier attempt let the SD itself vary with GA and produced z=42.3 for
one fetus, because the fitted SD collapsed to 0.001 at GA 39.3 where a single observation sits -- a
constant robust scale is used instead.

  LIMITATION, to state in any paper: this is an INTERNAL COHORT reference -- deviation from THIS
  cohort's own GA-conditional mean. It answers 'is this fetus's UA high for its gestational age within
  this cohort', NOT 'where does it sit on a published normative curve'. A normative z would need the
  actual coefficients from a published reference (e.g. Acharya 2005, Arduini & Rizzo 1990, fitted per
  Royston & Wright 1998); those are not reconstructed here from memory.

USAGE: python hpc_repair_ua_zscore.py
Output: handoff/ua_zscore_recomputed.csv  (nid, ua, ga, z_rob, z_old)
"""
import os, sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
HAND=os.path.join(HERE,"handoff"); os.makedirs(HAND,exist_ok=True)
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
# raw UA lives in the wide merged workbook, not the echo file
CANDIDATES=[os.path.join(ROOT,"..","fetal_growth_mechanism","data","master_merged.xlsx"),
            os.path.join(ROOT,"data_local","master_merged.xlsx"),
            os.path.join(ROOT,"..","fetal_growth_mechanism","data","IMPACT_merged_by_Cod.xlsx")]

def eu_num(s):
    """european-decimal aware and MAGNITUDE-PRESERVING: comma->period only. Stripping '.' as a thousands
    separator multiplies genuine decimals by 10 (BPD 83 -> 830), so it is never done here."""
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",",".",regex=False),errors="coerce")

def main():
    src=next((p for p in CANDIDATES if os.path.exists(p)),None)
    assert src, ("cannot find master_merged.xlsx / IMPACT_merged_by_Cod.xlsx -- the RAW UA column "
                 "(UA_ecoIMPACT / ua_pi) is not in the echo file, only the broken Zscore_AU is.\n"
                 f"looked in: {[os.path.abspath(p) for p in CANDIDATES]}")
    print(f"  source: {os.path.abspath(src)}",flush=True)
    head=pd.read_excel(src,nrows=1); allc=[str(c) for c in head.columns]
    raw=next((c for c in ("UA_ecoIMPACT","ua_pi") if c in allc),None)
    gac=next((c for c in ("EG_ecoIMPACT",) if c in allc),None)
    idc=next((c for c in ("Cod.1","Cod","record_id") if c in allc),None)
    assert raw and gac and idc, f"missing needed columns: raw={raw} ga={gac} id={idc}"
    keep=[c for c in {idc,raw,gac,"Zscore_AU"} if c in allc]
    D=pd.read_excel(src,usecols=keep)
    D["nid"]=D[idc].apply(lambda x: str(int(float(x)))
                          if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else None)
    D["ua"]=eu_num(D[raw]); D["ga"]=eu_num(D[gac])
    D["z_old"]=eu_num(D["Zscore_AU"]) if "Zscore_AU" in D.columns else np.nan
    D=D.dropna(subset=["nid"]).drop_duplicates("nid")
    d=D.dropna(subset=["ua","ga"]).copy()
    assert len(d)>500, f"only {len(d)} fetuses with raw UA + GA"
    # sanity: the RAW column must look like a UA PI and must NOT track GA strongly
    r_raw_ga=spearmanr(d.ua,d.ga).statistic
    print(f"  raw {raw}: n={len(d)} median {d.ua.median():.3f} SD {d.ua.std():.3f} | spearman with GA {r_raw_ga:+.3f}",flush=True)
    assert 0.3<d.ua.median()<2.0, f"raw UA median {d.ua.median():.3f} is not a plausible UA PI"

    def spl(x):
        x=np.asarray(x,float); q=np.nanpercentile(x,[33,66])
        return np.column_stack([x]+[np.clip(x-k,0,None)**2 for k in q])
    X=spl(d.ga.values); lg=np.log(d.ua.values)
    resid=lg-LinearRegression().fit(X,lg).predict(X)
    mad=np.median(np.abs(resid-np.median(resid)))*1.4826       # constant ROBUST scale (see docstring)
    assert mad>1e-6, "MAD scale collapsed"
    d["z_rob"]=(resid-np.median(resid))/mad

    r_ga=spearmanr(d.z_rob,d.ga).statistic; r_ua=spearmanr(d.z_rob,d.ua).statistic
    stats={"n":int(len(d)),"raw_column":raw,"source":os.path.abspath(src),
           "z_new":{"mean":float(d.z_rob.mean()),"sd":float(d.z_rob.std()),
                    "spearman_GA":float(r_ga),"spearman_raw_UA":float(r_ua),
                    "frac_abs_gt3":float((d.z_rob.abs()>3).mean())},
           "reference":"INTERNAL cohort GA-conditional (log UA residual, MAD-scaled) -- NOT normative"}
    if np.isfinite(d.z_old).any():
        mo=np.isfinite(d.z_old)
        stats["z_old_shipped"]={"sd":float(d.z_old[mo].std()),
            "spearman_GA":float(spearmanr(d.z_old[mo],d.ga[mo]).statistic),
            "spearman_raw_UA":float(spearmanr(d.z_old[mo],d.ua[mo]).statistic)}
    print(f"  REPAIRED z: mean {d.z_rob.mean():+.3f} SD {d.z_rob.std():.3f} | "
          f"with GA {r_ga:+.3f} (want ~0) | with raw UA {r_ua:+.3f} (want high)",flush=True)
    if "z_old_shipped" in stats:
        o=stats["z_old_shipped"]
        print(f"  shipped z:  SD {o['sd']:.3f} | with GA {o['spearman_GA']:+.3f} | "
              f"with raw UA {o['spearman_raw_UA']:+.3f}  <- tracks GA, not its own measurement",flush=True)
    # the repair must satisfy both conditions, else refuse to write a target we would train on
    assert abs(r_ga)<0.15, f"repaired z still tracks GA at {r_ga:+.3f} -- refusing to emit a GA proxy"
    assert abs(r_ua)>0.80, f"repaired z does not track its own raw UA ({r_ua:+.3f}) -- repair failed"
    out=os.path.join(HAND,"ua_zscore_recomputed.csv")
    d[["nid","ua","ga","z_rob","z_old"]].to_csv(out,index=False)
    json.dump(stats,open(os.path.join(OUTP,"ua_zscore_repair.json"),"w"),indent=2,default=str)
    print(f"  wrote {out} ({len(d)} fetuses)\nsaved out_probe/ua_zscore_repair.json\nDONE",flush=True)

if __name__=="__main__": main()
