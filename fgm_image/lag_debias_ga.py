"""
Fix the GA bias in the appearance-lag clock, two ways, and export a bias-free lag.

Context: the USFM GA clock (Ridge on emb_l5) is a shrinkage regressor -> its predictions
regress toward the cohort-mean GA, so lag = pred_GA - true_GA carries a spurious
-0.26 wk-lag/wk-GA slope (young over-predicted, old under-predicted). This is NOT
maturation. We fix it two ways and show the outcome gradient survives.

Fixes
  (2) GA-balanced training: inverse-GA-bin-density sample weights -> uniform GA.
      Flattens the sparse tails but LOWERS clock r (0.847 -> 0.754). Not recommended.
  (1) Post-hoc calibration: nested-OOF linear recalibration (true ~ pred), inverted to
      de-shrink predictions. Barely helps -- hits the regression-dilution FLOOR
      -(1 - r^2) = -0.28: correcting with the NOISY predicted GA can only remove r^2
      of the bias.
  Canonical fix: GA-residual lag = lag - E[lag | true GA]. Uses TRUE GA, so it removes
      the bias EXACTLY (slope 0). No retrain needed. This is the recommended scalar.

Key result: SGA<normal<LGA gradient + birth-pct correlation survive ALL methods
  (p<1e-6, rho~0.22) -> the outcome signal is not an artifact of the age bias.

Inputs (read-only):
  results/img_align/emb_usfm_multilayer.npz   (emb_l5, fetus_id, study_date, new_filename)
  CiTUS master xlsx                            (FUReco/LMP dating anchor)
  results/img_align/_fetus_lag_{IMPACT,clinical}.csv   (outcome group + birth pct)
Output:
  results/img_align/_fetus_lag_debiased.csv    (fid, lag_debiased, lag_raw, lag_balanced, ga, grp, bpct)
  results/img_align/lag_debias.json            (all metrics)
Run:  python fgm_image/lag_debias_ga.py
"""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold
from scipy.stats import kruskal, spearmanr

ROOT = "/Users/tiago/dev/fgr-geometry"
IMG  = f"{ROOT}/results/img_align"
CITUS = "/Users/tiago/Documents/CiTUS/Dataset/Impact_longitudinal_161025_all_merged.xlsx"
GA_MIN, GA_MAX, ALPHA = 6.0, 42.0, 50.0

def load_inputs():
    cit = pd.read_excel(CITUS, sheet_name="IMPACT_Final_All_V18")[["Cod","LMP","FUReco"]].copy()
    cit["LMP"] = pd.to_datetime(cit.LMP, errors="coerce")
    cit["FUReco"] = pd.to_datetime(cit.FUReco, errors="coerce")
    anchor = cit.assign(a=cit.FUReco.fillna(cit.LMP)).set_index("Cod").a.to_dict()
    z = np.load(f"{IMG}/emb_usfm_multilayer.npz", allow_pickle=True)
    fid = pd.to_numeric(pd.Series(z["fetus_id"]), errors="coerce").astype("Int64")
    sd  = pd.to_numeric(pd.Series(z["study_date"]), errors="coerce")
    sd_dt = pd.to_datetime(sd.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")
    anc = pd.to_datetime(pd.Series(fid.map(anchor)), errors="coerce")
    ga = ((sd_dt - anc.values).dt.days / 7.0).values
    valid = pd.Series(ga).between(GA_MIN, GA_MAX).values & fid.notna().values
    return z["emb_l5"].astype(np.float32)[valid], ga[valid], fid.values[valid].astype(int)

def oof(E, y, g, weight_fn=None):
    o = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(E, y, g):
        w = weight_fn(y[tr]) if weight_fn else None
        o[te] = Ridge(alpha=ALPHA).fit(E[tr], y[tr], sample_weight=w).predict(E[te])
    return o

def ga_balance_weights(y, nbins=16):
    edges = np.linspace(GA_MIN, GA_MAX, nbins + 1)
    b = np.clip(np.digitize(y, edges) - 1, 0, nbins - 1)
    cnt = np.bincount(b, minlength=nbins).astype(float); cnt[cnt == 0] = np.nan
    w = 1.0 / cnt[b]; w = w / np.nanmean(w); w[np.isnan(w)] = 0
    return w

def calibrate(E, y, g, base_oof):
    """nested-OOF linear recalibration of predicted GA (de-shrink)."""
    o = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(E, y, g):
        pin = np.zeros(len(tr))
        for itr, ite in GroupKFold(4).split(E[tr], y[tr], g[tr]):
            pin[ite] = Ridge(alpha=ALPHA).fit(E[tr][itr], y[tr][itr]).predict(E[tr][ite])
        cal = LinearRegression().fit(pin.reshape(-1, 1), y[tr])
        o[te] = cal.predict(base_oof[te].reshape(-1, 1))
    return o

def per_fetus(o, ga, fid, gmap):
    pf = pd.DataFrame({"fid": fid, "lag": o - ga, "ga": ga}).groupby("fid").agg(
        lag=("lag","mean"), ga=("ga","mean")).join(gmap)
    return pf

def main():
    E, ga, fid = load_inputs()
    imp = pd.read_csv(f"{IMG}/_fetus_lag_IMPACT.csv")[["fid","grp","bpct"]]
    cli = pd.read_csv(f"{IMG}/_fetus_lag_clinical.csv")[["fid","grp","bpct"]]
    gmap = pd.concat([imp, cli]).drop_duplicates("fid").set_index("fid")

    o_base = oof(E, ga, fid)
    o_bal  = oof(E, ga, fid, weight_fn=ga_balance_weights)
    o_cal  = calibrate(E, ga, fid, o_base)

    pf_base = per_fetus(o_base, ga, fid, gmap)
    pf_bal  = per_fetus(o_bal,  ga, fid, gmap)
    pf_res  = pf_base.copy()
    pf_res["lag"] = pf_base.lag - LinearRegression().fit(pf_base[["ga"]], pf_base.lag).predict(pf_base[["ga"]])

    def summ(pf):
        g = {k: pf[pf.grp == k].lag for k in ["SGA","normal","LGA"]}
        ok = pf.bpct.notna()
        return dict(SGA=float(g["SGA"].mean()), normal=float(g["normal"].mean()), LGA=float(g["LGA"].mean()),
                    p=float(kruskal(*[g[k].dropna() for k in ["SGA","normal","LGA"]])[1]),
                    bpct_rho=float(spearmanr(pf.lag[ok], pf.bpct[ok])[0]),
                    ga_slope=float(np.polyfit(pf.ga, pf.lag, 1)[0]))

    out = pf_res.reset_index()[["fid","lag","ga","grp","bpct"]].rename(columns={"lag":"lag_debiased"})
    out["lag_raw"] = pf_base.reset_index().lag.values
    out["lag_balanced"] = pf_bal.reindex(out.fid).lag.values
    out.to_csv(f"{IMG}/_fetus_lag_debiased.csv", index=False)

    pf_cal = per_fetus(o_cal, ga, fid, gmap)
    r2 = np.corrcoef(o_base, ga)[0, 1] ** 2
    res = {
        "goal": "fix the clock GA bias via (fix1) calibration and (fix2) GA-balanced training; export bias-free lag",
        "clock": {"model": "Ridge(alpha=50) on USFM emb_l5, GroupKFold(5) by fetus",
                  "baseline_r": float(np.corrcoef(o_base, ga)[0,1])},
        "fix2_GA_balanced": {"clock_r": float(np.corrcoef(o_bal, ga)[0,1])},
        "regression_dilution_floor": {"r2": float(r2), "floor_minus_1_minus_r2": float(-(1 - r2))},
        "GA_residual_lag": {"GAbias_slope_perfetus": float(summ(pf_res)["ga_slope"])},
        "outcome_gradient_survives": {"baseline": summ(pf_base), "GA_balanced": summ(pf_bal),
                                      "calibrated": summ(pf_cal), "GA_residual": summ(pf_res)},
    }
    json.dump(res, open(f"{IMG}/lag_debias.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
