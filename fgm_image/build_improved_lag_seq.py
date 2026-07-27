"""
Build the IMPROVED per-visit appearance-lag sequence for config-B (longitudinal).

Winner of the pooling battery (2026-07-09): multi-layer fusion (block1+6+12 patch-MEANS,
2304-d) for the GA clock + MEDIAN aggregation of per-image lags within each GA-week visit bin.

Pipeline:
  1. fused per-image feature = concat(emb_l0, emb_l5, emb_l11) (2304-d), z-scored
  2. GA clock = Ridge(fused -> datesGA), group-CV 5-fold (no fetus leakage), 6<=GA<=42 filter
  3. per-image lag = predGA - datesGA
  4. per (fid, GA-week bin) -> MEDIAN lag
  5. map onto the existing _merged_seq.npz 12-slot grid via each slot's GA (X[:,:,-1]*36+6)

Output: results/img_align/_lag_seq_improved.npz  (lag_seq, lag_mask, fids, L)  -- same shape as _lag_seq.npz
so config-B can consume it by swapping the input path.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold

IMG = "/Users/tiago/dev/fgr-geometry/results/img_align"


def perimage_improved_lag():
    ml = np.load(f"{IMG}/emb_usfm_multilayer.npz", allow_pickle=True)
    fused = np.concatenate([ml["emb_l0"], ml["emb_l5"], ml["emb_l11"]], axis=1)  # (61804, 2304)
    fid = np.array([int(x) for x in ml["fetus_id"]])
    ga = np.array([float(x) for x in ml["ga_weeks_recovered"]])
    gm = np.isfinite(ga) & (ga >= 6) & (ga <= 42)
    fused, fid, ga = fused[gm], fid[gm], ga[gm]
    Ez = (fused - fused.mean(0)) / (fused.std(0) + 1e-8)
    pga = cross_val_predict(Ridge(200.0), Ez, ga, cv=GroupKFold(5), groups=fid)
    clock_r = float(np.corrcoef(pga, ga)[0, 1])
    return pd.DataFrame({"fid": fid, "gab": np.round(ga).astype(int), "lag": pga - ga}), clock_r


def build(MAXT=12):
    df, clock_r = perimage_improved_lag()
    # median lag per (fid, GA-week bin)
    med = df.groupby(["fid", "gab"]).lag.median()
    # align onto existing sequence grid
    base = np.load(f"{IMG}/_merged_seq.npz", allow_pickle=True)
    X, L, fids = base["X"], base["L"], base["fids"]
    N = len(fids); lag_seq = np.zeros((N, MAXT), np.float32); lag_mask = np.zeros((N, MAXT), np.float32)
    for i, f in enumerate(fids):
        for t in range(int(L[i])):
            gaw = int(round(X[i, t, -1] * 36.0 + 6.0))
            for cand in (gaw, gaw - 1, gaw + 1):        # tolerate +-1 wk binning slack
                if (int(f), cand) in med.index:
                    lag_seq[i, t] = med.loc[(int(f), cand)]; lag_mask[i, t] = 1.0; break
    cov = lag_mask.sum() / L.sum()
    np.savez(f"{IMG}/_lag_seq_improved.npz", lag_seq=lag_seq, lag_mask=lag_mask, fids=fids, L=L)
    return dict(clock_r=round(clock_r, 3), coverage=round(float(cov), 3),
                n_fetus=int(N), mean_fill=round(float(lag_mask.sum() / (N * MAXT)), 3))


if __name__ == "__main__":
    import json
    print(json.dumps(build(), indent=2))
