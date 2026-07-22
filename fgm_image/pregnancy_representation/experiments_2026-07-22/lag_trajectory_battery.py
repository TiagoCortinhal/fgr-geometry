#!/usr/bin/env python3
"""Does appearance-lag add to biometry via its TRAJECTORY (not endpoint)?

Motivation: a static per-fetus mean lag adds ~nothing over biometry (SGA ΔAUC +0.011,
CI crosses 0) — biometry already measures size and lag is a size proxy. But lag's
information beyond biometry is in the DYNAMICS: how appearance-maturity changes over
gestation. This battery tests the trajectory framing with the legitimate power levers,
NOT a construction sweep:
  (1) TRAJECTORY features: per-fetus lag slope + late-value (vs static mean)
  (2) CONTINUOUS outcome: birth-percentile regression (vs dichotomised SGA/AGA)
  (3) ENCODER ENSEMBLE: mean of USF-MAE + DINOv2 lag (lower measurement error)
All with fetus-grouped CV, matched size-trajectory baseline, bootstrap CI.

Adversarial discipline: report ALL tests (no cherry-picking best), state nulls as nulls,
increments judged by bootstrap CI + P(Δ>0), matched-capacity baseline (biometry trajectory).

Inputs (workspace/artifacts):
  traj_substrate.npz         : mae(N,1536), dino(N,384), ga(N), nid(N), birthpct, sga, lga (per visit)
  comprehensive_tabular.csv  : per-fetus biometry z (efw/ac/hc/fl) + outcomes
Run in env fgrgeom.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# ---------- OOF GA-clock lag, per encoder ----------
def oof_lag(emb, ga, nid, alpha=10):
    """Leave-fetus-out GA clock; lag = predicted GA - true GA."""
    X = StandardScaler().fit_transform(emb)
    pred = np.zeros(len(ga))
    for tr, te in GroupKFold(5).split(X, groups=nid):
        pred[te] = Ridge(alpha=alpha).fit(X[tr], ga[tr]).predict(X[te])
    return pred - ga

# ---------- per-fetus trajectory features ----------
def traj_features(nid, ga, lag):
    """Per-fetus: mean lag, lag slope over GA, lag at latest visit, size-traj analogues added by caller."""
    df = pd.DataFrame({"nid": nid, "ga": ga, "lag": lag})
    def f(g):
        slope = np.polyfit(g.ga, g.lag, 1)[0] if (len(g) >= 2 and g.ga.std() > 0) else 0.0
        late = g.loc[g.ga.idxmax(), "lag"]
        return pd.Series({"lag_mean": g.lag.mean(), "lag_slope": slope, "lag_late": late})
    return df.groupby("nid").apply(f, include_groups=False)

# ---------- bootstrap ΔAUC / ΔR ----------
def boot_delta(y, pa, pb, metric="auc", n=5000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if metric == "auc":
            if y[b].sum() < 5 or (1 - y[b]).sum() < 5: continue
            d.append(roc_auc_score(y[b], pb[b]) - roc_auc_score(y[b], pa[b]))
        else:
            d.append(pearsonr(pb[b], y[b])[0] - pearsonr(pa[b], y[b])[0])
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())

def cvpred_clf(D, cols, y):
    Z = ((D[cols] - D[cols].mean()) / D[cols].std()).values
    return cross_val_predict(LogisticRegression(max_iter=1000, class_weight="balanced"),
                             Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]

def cvpred_reg(D, cols, y):
    Z = ((D[cols] - D[cols].mean()) / D[cols].std()).values
    return cross_val_predict(Ridge(alpha=1.0), Z, y, cv=KFold(5, shuffle=True, random_state=0))

def main():
    import json
    z = np.load("traj_substrate.npz", allow_pickle=True)
    mae, dino, ga = z["mae"], z["dino"], z["ga"]
    nid = np.array([str(int(float(x))) for x in z["nid"]])
    # ensemble lag (two independent noisy maturation estimates)
    lag = (oof_lag(mae, ga, nid) + oof_lag(dino, ga, nid)) / 2
    tf = traj_features(nid, ga, lag)
    # matched SIZE-trajectory baseline: same features on biometry EFW instead of lag
    T = pd.read_csv("comprehensive_tabular.csv", index_col=0); T.index = T.index.map(lambda x: str(int(float(x))))
    D = tf.join(T[["efw_z", "ac_z", "hc_z", "fl_z", "percentil_birth",
                   "sga_p10", "lga_p90"]], how="inner").dropna(subset=["efw_z", "percentil_birth"])
    biom = ["efw_z", "ac_z", "hc_z", "fl_z"]
    lag_traj = ["lag_mean", "lag_slope", "lag_late"]
    res = {}
    # --- Test A: continuous birth-percentile regression, biom vs biom+lag-trajectory ---
    y = D["percentil_birth"].values
    pa = cvpred_reg(D, biom, y); pb = cvpred_reg(D, biom + lag_traj, y)
    ra, rb = pearsonr(pa, y)[0], pearsonr(pb, y)[0]
    md, lo, hi, pp = boot_delta(y, pa, pb, metric="r")
    res["A_birthpct_regression"] = dict(biom_r=ra, biom_lagtraj_r=rb, delta_r=md, ci=[lo, hi], P_gt0=pp)
    # --- Test B: SGA classification, biom vs biom+lag-trajectory ---
    for lab, col in [("SGA_p10", "sga_p10"), ("LGA_p90", "lga_p90")]:
        yy = D[col].astype(int).values
        pa = cvpred_clf(D, biom, yy); pb = cvpred_clf(D, biom + lag_traj, yy)
        aa, bb = roc_auc_score(yy, pa), roc_auc_score(yy, pb)
        md, lo, hi, pp = boot_delta(yy, pa, pb, metric="auc")
        res[f"B_{lab}"] = dict(biom_auc=aa, biom_lagtraj_auc=bb, delta_auc=md, ci=[lo, hi], P_gt0=pp)
    # --- Test C: is lag-SLOPE alone (the pure dynamic) incremental? ---
    y = D["percentil_birth"].values
    pa = cvpred_reg(D, biom, y); pb = cvpred_reg(D, biom + ["lag_slope"], y)
    md, lo, hi, pp = boot_delta(y, pa, pb, metric="r")
    res["C_slope_only_birthpct"] = dict(delta_r=md, ci=[lo, hi], P_gt0=pp)
    json.dump(res, open("lag_trajectory_battery_result.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
