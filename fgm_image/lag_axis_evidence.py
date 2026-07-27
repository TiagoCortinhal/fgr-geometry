"""
Evidence battery: is appearance-lag a REAL, independent, useful latent axis?

Runs on the clean config-B latent (_lagB_traj_clean.npy — corrupt biometry masked).
Five tests, each a distinct claim a reviewer would ask for:

  E1 dedicated axis      — a latent dim the lag scalar dominates, separable from size
  E2 non-redundant       — that axis carries variance the 5 biometry z-scores cannot explain
  E3 stable across seeds — the axis reappears in independent retrains (see lag_axis_stability.py)
  E4 incremental signal  — the axis adds outcome info beyond the size axis
  E5 stable per-fetus     — the axis separates fetuses far more than it wobbles within one,
                            and predicts the held-out lag scalar

Writes lag_axis_evidence.json + lag_axis_evidence.png (4-panel figure).
E3 seed values are produced by fgm_image/lag_axis_stability.py (4 retrains) and pasted in.
"""
import json, numpy as np, pandas as pd
from numpy.linalg import lstsq
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

IMG = "/Users/tiago/dev/fgr-geometry/results/img_align"
LAG_DIM, SIZE_DIM = 2, 4          # clean-latent roles (z2 = near-pure lag, z4 = size)
SEED_STABILITY = [0.919, 0.845, 0.908, 0.504]   # from lag_axis_stability.py


def load():
    tC = np.load(f"{IMG}/_lagB_traj_clean.npy")
    z = np.load(f"{IMG}/_merged_seq.npz", allow_pickle=True)
    lab = np.load(f"{IMG}/_merged_labels.npz", allow_pickle=True)
    X, L, fids, F = z["X"], z["L"], z["fids"], int(z["F"]); N = len(fids)
    E = np.array([tC[i, L[i] - 1, :] for i in range(N)])
    birth = pd.Series(lab["birth"], index=lab["fids"]).reindex(fids).values
    cg = pd.read_csv(f"{IMG}/_citus_groups.csv").set_index("Cod").grp_citus.reindex([int(f) for f in fids]).values
    lz = np.load(f"{IMG}/_lag_seq.npz", allow_pickle=True); ls, lm = lz["lag_seq"], lz["lag_mask"]
    lag = np.array([(ls[i][:L[i]][lm[i][:L[i]] > 0].mean() if (lm[i][:L[i]] > 0).any() else np.nan) for i in range(N)])
    # _merged_seq interleaves biometry and image visits; the last SLOT is often an image-only
    # visit with missing biometry (all-zero, mask-bit 0). Select the last slot whose biometry
    # mask bit is set — the true last biometry scan. (Effect on the R2-purity test is negligible,
    # 0.011->0.010, but keep the selection correct for consistency with the rest of the repo.)
    def _lbv(i):
        for t in range(L[i] - 1, -1, -1):
            if X[i, t, F:2 * F].sum() > 0: return t
        return L[i] - 1
    Bz = np.array([X[i, _lbv(i), :F] for i in range(N)]).copy(); Bz[Bz < -10] = np.nan
    return tC, E, L, N, birth, cg, lag, Bz, F


def run():
    tC, E, L, N, birth, cg, lag, Bz, F = load()
    ml = np.isfinite(lag); ok = np.isfinite(birth)
    # E1
    z_lag = E[:, LAG_DIM]
    e1_lag = float(np.corrcoef(z_lag[ml], lag[ml])[0, 1])
    e1_biom = float(np.nanmax([abs(np.corrcoef(E[np.isfinite(Bz[:, j]), LAG_DIM], Bz[np.isfinite(Bz[:, j]), j])[0, 1]) for j in range(F)]))
    # E2
    mB = np.all(np.isfinite(Bz), 1) & ml
    Bg = np.column_stack([Bz[mB], np.ones(mB.sum())])
    resid = E[mB, LAG_DIM] - Bg @ lstsq(Bg, E[mB, LAG_DIM], rcond=None)[0]
    r2_biom = float(1 - resid.var() / E[mB, LAG_DIM].var())
    r_resid = float(np.corrcoef(resid, lag[mB] - Bg @ lstsq(Bg, lag[mB], rcond=None)[0])[0, 1])
    # E4
    size = E[:, SIZE_DIM]
    def fit_r(cols, y, m): return abs(np.corrcoef((cols[m] @ lstsq(np.column_stack([cols[m], np.ones(m.sum())]), y[m], rcond=None)[0][:cols.shape[1]]), y[m])[0, 1])
    bp_s = float(abs(pearsonr(size[ok], birth[ok])[0]))
    bp_sl = float(fit_r(np.column_stack([size, E[:, LAG_DIM]]), birth, ok))
    aucs = {}
    for grp in ["SGA", "LGA"]:
        y = (cg == grp).astype(int); m = np.isfinite(size)
        a_s = roc_auc_score(y[m], LogisticRegression(max_iter=500).fit(size[m, None], y[m]).predict_proba(size[m, None])[:, 1])
        a_sl = roc_auc_score(y[m], LogisticRegression(max_iter=500).fit(np.column_stack([size[m], E[m, LAG_DIM]]), y[m]).predict_proba(np.column_stack([size[m], E[m, LAG_DIM]]))[:, 1])
        aucs[grp] = (round(float(a_s), 3), round(float(a_sl), 3))
    # E5
    within = np.array([np.std(tC[i, :int(L[i]), LAG_DIM]) for i in range(N) if L[i] >= 2])
    between = float(np.std([np.mean(tC[i, :int(L[i]), LAG_DIM]) for i in range(N) if L[i] >= 2]))
    pred = cross_val_predict(Ridge(1.0), E[ml][:, [LAG_DIM]], lag[ml], cv=5)
    r_heldout = float(pearsonr(pred, lag[ml])[0])
    ev = {
        "E1_dedicated_axis": {"z_lag_r": round(e1_lag, 3), "z_max_biom_r": round(e1_biom, 3)},
        "E2_non_redundant": {"R2_from_biometry": round(r2_biom, 3), "resid_vs_lag_r": round(r_resid, 3)},
        "E3_stable_across_retrains": {"seed_best_r": SEED_STABILITY, "mean": round(float(np.mean(SEED_STABILITY)), 3)},
        "E4_incremental_signal": {"birthpct_size": round(bp_s, 3), "birthpct_size_plus_lag": round(bp_sl, 3),
                                  "SGA_auc": aucs["SGA"], "LGA_auc": aucs["LGA"]},
        "E5_stable_per_fetus": {"within_SD": round(float(within.mean()), 3), "between_SD": round(between, 3),
                                "ratio": round(between / float(within.mean()), 2), "heldout_lag_r": round(r_heldout, 3)},
    }
    json.dump(ev, open(f"{IMG}/lag_axis_evidence.json", "w"), indent=2)
    return ev


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
