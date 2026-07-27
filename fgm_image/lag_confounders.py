"""Appearance-age lag — confounder and dataset-source analysis.

Tests whether the appearance-age lag (apparent GA from the block-6 USFM mean
clock, minus true/recovered GA) carries a real SGA/LGA signal or is an artifact
of (a) scanner machine, (b) calendar/study date, or (c) pooling the IMPACT and
clinical image sets, which occupy different GA ranges.

Inputs (produced earlier in the pipeline):
  results/img_align/emb_usfm_multilayer.npz   block-6 mean features + GA/fetus meta
  results/img_align/_lag_device_tags.csv      DICOM device tags per lag image
                                              (from harvest_device_tags.py)
  results/img_align/image_mapping_all.csv     new_filename -> raw DICOM path
  fetal_growth_mechanism/data/impact_outcomes.csv   SGA/LGA/percentile labels

Outputs (results/img_align/):
  _lag_device_merged.csv         per-image lag + device + date + dataset_type
  _fetus_lag_{IMPACT,clinical,both}.csv   per-fetus lag + outcome group
  lag_machine_stratification.json         machine + date confounder results
  lag_confounder_summary.json             consolidated verdict

Key findings (IMPACT cohort, 541 fetuses):
  * Machine explains R2=0.006 of fetus-lag; independent of outcome (p=0.85/0.40);
    lag-outcome survives machine adjustment (SGA -0.158, LGA +0.169).
  * Date explains R2=0.19 of fetus-lag but is a mechanical identity:
    lag_drift = apparent_drift(+2.33) - true_drift(+3.67) = -1.34 wk/yr,
    driven by regression-to-mean on a cohort scanned at rising GA over time.
    Outcome is independent of date (p=0.19/0.32); lag-outcome survives (partial
    SGA -0.149, LGA +0.158).
  * SGA<normal<LGA ordering holds in IMPACT-only (28-41 wk, p=0.0008) AND
    clinical-only (6-42 wk, p=0.0001) -> not a pooling artifact.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mannwhitneyu, chi2_contingency
from sklearn.linear_model import Ridge

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = f"{WS}/results/img_align"
DATA = f"{os.path.dirname(WS)}/fetal_growth_mechanism/data"


def _partial_corr(x, y, z):
    """corr(x, y | z) via residuals of linear fits on z."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return pearsonr(rx, ry)


def build_per_image_lag():
    """Per-image apparent GA (block-6 mean clock) minus true GA, + device/date."""
    z = np.load(f"{IMG}/emb_usfm_multilayer.npz", allow_pickle=True)
    E6 = z["emb_l5"].astype(np.float32)
    fn = pd.Series(z["new_filename"]).values
    ga = pd.to_numeric(pd.Series(z["ga_weeks_recovered"]), errors="coerce").values
    fid = pd.to_numeric(pd.Series(z["fetus_id"]), errors="coerce").values
    ok = np.isfinite(ga) & (ga >= 6) & (ga <= 42)
    clk = Ridge(alpha=50.0).fit(E6[ok], ga[ok])          # the MEAN GA clock, r=0.85
    app = clk.predict(E6)
    di = pd.DataFrame({"new_filename": fn, "true_ga": ga, "app_ga": app, "fid": fid})
    di = di[ok].drop_duplicates("new_filename")
    di["img_lag"] = di.app_ga - di.true_ga
    dev = pd.read_csv(f"{IMG}/_lag_device_tags.csv", low_memory=False)
    d = dev.merge(di, on="new_filename", how="inner")
    d["date"] = pd.to_datetime(d.StudyDate, format="%Y%m%d", errors="coerce")
    d.to_csv(f"{IMG}/_lag_device_merged.csv", index=False)
    return d


def fetus_lag(sub, out):
    """Aggregate per-image lag to per-fetus + attach SGA/LGA/percentile labels."""
    fl = sub.groupby("fid").agg(lag=("img_lag", "mean"), n=("img_lag", "size"),
                                date=("date", "median")).reset_index()
    fl["SGA"] = fl.fid.map(out["SGA_birth (<p10)"].eq("yes")).fillna(False).astype(int)
    fl["LGA"] = fl.fid.map(out["LGA_birth (>p90)"].eq("yes")).fillna(False).astype(int)
    fl["bpct"] = fl.fid.map(out["percentil_birth"])
    fl["grp"] = np.where(fl.SGA == 1, "SGA", np.where(fl.LGA == 1, "LGA", "normal"))
    return fl


def analyze():
    d = build_per_image_lag()
    d["fid"] = pd.to_numeric(d.fid, errors="coerce")
    out = pd.read_csv(f"{DATA}/impact_outcomes.csv").set_index("Cod")
    res = {"n_images": int(len(d)), "n_serials": int(d.DeviceSerialNumber.nunique()),
           "models": d.ManufacturerModelName.value_counts().to_dict()}

    # ---- machine confounder ----
    fl = fetus_lag(d, out)
    fl["serial"] = d.groupby("fid").DeviceSerialNumber.agg(
        lambda s: s.mode().iloc[0]).reindex(fl.fid).values
    big = fl.serial.value_counts().index[:6]
    fb = fl[fl.serial.isin(big)].copy()
    fb["lag_adj"] = fb.lag - fb.groupby("serial").lag.transform("mean")
    gm = fb.groupby("serial").lag.transform("mean")
    R2m = ((gm - fb.lag.mean()) ** 2).sum() / ((fb.lag - fb.lag.mean()) ** 2).sum()
    res["machine"] = {
        "R2_of_lag": float(R2m),
        "chi2_serial_SGA_p": float(chi2_contingency(pd.crosstab(fl.serial, fl.SGA))[1]),
        "chi2_serial_LGA_p": float(chi2_contingency(pd.crosstab(fl.serial, fl.LGA))[1]),
        "lag_SGA_raw": float(pearsonr(fl.lag, fl.SGA)[0]),
        "lag_SGA_adj": float(pearsonr(fb.lag_adj, fb.SGA)[0]),
        "lag_LGA_raw": float(pearsonr(fl.lag, fl.LGA)[0]),
        "lag_LGA_adj": float(pearsonr(fb.lag_adj, fb.LGA)[0])}

    # ---- date confounder (mechanical drift identity) ----
    dd = d.dropna(subset=["date"]).copy()
    dd["t"] = (dd.date - dd.date.min()).dt.days / 365.25
    mo = dd.groupby(dd.date.dt.to_period("M").dt.start_time).agg(
        true_ga=("true_ga", "mean"), app_ga=("app_ga", "mean"),
        lag=("img_lag", "mean"), t=("t", "mean"), n=("img_lag", "size")).reset_index()
    mo = mo[mo.n >= 20]
    fld = fl.copy(); fld["t"] = (fld.date - fld.date.min()).dt.days
    res["date"] = {
        "true_ga_drift_wk_yr": float(np.polyfit(mo.t, mo.true_ga, 1)[0]),
        "apparent_drift_wk_yr": float(np.polyfit(mo.t, mo.app_ga, 1)[0]),
        "lag_drift_wk_yr": float(np.polyfit(mo.t, mo.lag, 1)[0]),
        "date_SGA_r": float(pearsonr(fld.t, fld.SGA)[0]),
        "date_LGA_r": float(pearsonr(fld.t, fld.LGA)[0]),
        "lag_SGA_date_partial": float(_partial_corr(fld.lag.values, fld.SGA.values.astype(float), fld.t.values)[0]),
        "lag_LGA_date_partial": float(_partial_corr(fld.lag.values, fld.LGA.values.astype(float), fld.t.values)[0])}

    # ---- dataset-source split ----
    res["by_dataset"] = {}
    for tag, sub in [("IMPACT", d[d.dataset_type == "impact"]),
                     ("clinical", d[d.dataset_type == "clinical"]),
                     ("both", d)]:
        f = fetus_lag(sub, out)
        f.to_csv(f"{IMG}/_fetus_lag_{tag}.csv", index=False)
        s, l = f[f.grp == "SGA"].lag, f[f.grp == "LGA"].lag
        res["by_dataset"][tag] = {
            "ga_min": float(sub.true_ga.min()), "ga_max": float(sub.true_ga.max()),
            "n_images": int(len(sub)),
            "SGA_mean": float(s.mean()), "normal_mean": float(f[f.grp == "normal"].lag.mean()),
            "LGA_mean": float(l.mean()),
            "SGA_vs_LGA_p": float(mannwhitneyu(s, l)[1]),
            "lag_birthpct_r": float(pearsonr(f.lag, f.bpct.fillna(f.bpct.median()))[0])}

    json.dump(res, open(f"{IMG}/lag_confounder_summary.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    r = analyze()
    print(json.dumps(r, indent=2))
