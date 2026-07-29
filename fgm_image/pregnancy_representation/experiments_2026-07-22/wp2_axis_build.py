#!/usr/bin/env python3
"""STEP 0 -- build the TABULAR WP2 axis. No training, no images, CPU, seconds.

WHY THIS SCRIPT EXISTS. The annotation asks "which visual words are enriched at the FGR end of the
WP2 axis". For that to be non-circular the axis must come from TABULAR data only. The obvious
candidate on disk is NOT usable: final_perparam_latent.npz is the config-B GRU-VAE latent, which
takes the IMAGE-DERIVED appearance lag as an input -- its maturation axis correlates rho=+0.818 with
mean_lag and even its head-sparing axis +0.653. Annotating that with image codes would be
image-derived input tested against image-derived words.

TWO AXES, both purely tabular:
  PRIMARY   size_scalar -- the observed-biometry mean z-score already stored per fetus. Simplest,
            least contestable, needs no refit, and cannot contain image information.
  ROBUSTNESS fa1 -- axis 1 of a Factor Analysis refit of the static tabular panel (biometry visit
            z-scores + Doppler). Declared robustness, not primary, because a refit invites choices.

SIGN CONVENTION: both axes are oriented so that HIGHER = MORE FGR-LIKE (smaller fetus). Verified
against birth percentile: the correlation with birth_pct must be NEGATIVE after orientation. This is
checked and asserted, not assumed -- a sign flip would invert every enrichment claim downstream.

USAGE: python wp2_axis_build.py
Output: handoff/wp2_axis.csv  (nid, wp2_axis_primary, wp2_axis_fa1, birth_pct, sex)
"""
import os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
HAND=os.path.join(HERE,"handoff"); os.makedirs(HAND,exist_ok=True)
ECHO=os.path.join(ROOT,"data_local","IMPACT_ecocardio_zscores_corrected.xlsx")
# per-fetus tabular scalars saved by the earlier per-parameter latent work; we take ONLY the
# tabular columns from it (size_scalar, birth_pct) and deliberately IGNORE Z / mean_lag.
PERPARAM=os.environ.get("WP2_PERPARAM",os.path.join(HAND,"final_perparam_latent.npz"))

def orient(v, birth):
    """flip so HIGHER = MORE FGR-LIKE, i.e. NEGATIVE correlation with birth percentile."""
    m=np.isfinite(v)&np.isfinite(birth)
    r=spearmanr(v[m],birth[m]).statistic
    return (-v if r>0 else v), float(r)

def main():
    res={}
    assert os.path.exists(PERPARAM), (
        f"missing {PERPARAM}. It holds the per-fetus TABULAR scalars (size_scalar, birth_pct). "
        "Fetch it from the artifact store (final_perparam_latent.npz) into handoff/ first.")
    z=np.load(PERPARAM,allow_pickle=True)
    nid=np.array([str(int(x)) for x in z["fids"]])
    size=np.asarray(z["size_scalar"],float); birth=np.asarray(z["birth_pct"],float)
    # PROVENANCE GUARD: refuse to proceed if we were handed an image-contaminated axis by mistake.
    if "mean_lag" in z.files:
        ml=np.asarray(z["mean_lag"],float); m=np.isfinite(size)&np.isfinite(ml)
        r_lag=float(spearmanr(size[m],ml[m]).statistic)
        res["size_scalar_vs_mean_lag_rho"]=r_lag
        print(f"  provenance check: size_scalar vs image-derived mean_lag rho={r_lag:+.3f}",flush=True)
        assert abs(r_lag)<0.5, (f"size_scalar correlates {r_lag:+.3f} with the IMAGE-DERIVED lag -- too high "
                               "to treat as a purely tabular axis. Stop and inspect provenance.")
    prim,r_raw=orient(size,birth)
    res["primary"]={"source":"size_scalar (observed biometry mean z, tabular only)",
                    "raw_rho_vs_birth_pct":r_raw,"oriented":"higher = more FGR-like",
                    "oriented_rho_vs_birth_pct":float(spearmanr(prim[np.isfinite(prim)&np.isfinite(birth)],
                                                     birth[np.isfinite(prim)&np.isfinite(birth)]).statistic)}
    print(f"  PRIMARY size_scalar: rho vs birth_pct {r_raw:+.3f} raw -> "
          f"{res['primary']['oriented_rho_vs_birth_pct']:+.3f} oriented (must be NEGATIVE)",flush=True)
    assert res["primary"]["oriented_rho_vs_birth_pct"]<0, "orientation failed"

    # ---- robustness axis: FA on the static tabular panel ----
    fa1=np.full(len(nid),np.nan)
    try:
        from sklearn.decomposition import FactorAnalysis
        from sklearn.impute import SimpleImputer
        e=pd.read_excel(ECHO)
        e["nid"]=e["Cod"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
        e=e.drop_duplicates("nid").set_index("nid")
        cols=[c for c in e.columns if str(c).startswith("Zscore_")]
        dop=[c for c in cols if any(k in str(c) for k in ("UTA","AU","ACM","CPR","DV","Ithsmus"))]
        panel=e.reindex(nid)[dop].apply(pd.to_numeric,errors="coerce")
        res["fa_panel_cols"]=dop; res["fa_panel_obs_rate"]=float(panel.notna().mean().mean())
        if panel.notna().mean().mean()>0.5 and panel.shape[1]>=3:
            X=SimpleImputer(strategy="median").fit_transform(panel.values)
            X=(X-X.mean(0))/(X.std(0)+1e-9)
            f=FactorAnalysis(n_components=2,random_state=0).fit_transform(X)[:,0]
            fa1,r_fa=orient(f,birth)
            res["robustness"]={"source":f"FA axis1 over {len(dop)} Doppler z-cols","raw_rho_vs_birth_pct":r_fa,
                "rho_vs_primary":float(spearmanr(fa1,prim,nan_policy="omit").statistic)}
            print(f"  ROBUSTNESS fa1: rho vs birth_pct {r_fa:+.3f} | rho vs primary "
                  f"{res['robustness']['rho_vs_primary']:+.3f}",flush=True)
        else:
            res["robustness"]={"skipped":"Doppler panel too sparse or too few columns"}
            print("  ROBUSTNESS fa1: SKIPPED (panel too sparse)",flush=True)
    except Exception as ex:
        res["robustness"]={"error":f"{type(ex).__name__}: {ex}"}
        print(f"  ROBUSTNESS fa1: ERROR {type(ex).__name__}: {ex}",flush=True)

    sex=pd.to_numeric(pd.read_excel(ECHO).assign(
        nid=lambda d: d["Cod"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
        ).drop_duplicates("nid").set_index("nid").get("SexoRN_cod"),errors="coerce")
    out=pd.DataFrame({"nid":nid,"wp2_axis_primary":prim,"wp2_axis_fa1":fa1,
                      "birth_pct":birth,"sex":pd.Series(nid).map(sex.to_dict()).values})
    out.to_csv(os.path.join(HAND,"wp2_axis.csv"),index=False)
    res["n_fetuses"]=int(len(out)); res["sex_coverage"]=float(out["sex"].notna().mean())
    json.dump(res,open(os.path.join(HAND,"wp2_axis_provenance.json"),"w"),indent=2)
    print(f"  wrote handoff/wp2_axis.csv  n={len(out)} sex coverage {out['sex'].notna().mean():.3f}\nDONE",flush=True)

if __name__=="__main__": main()
