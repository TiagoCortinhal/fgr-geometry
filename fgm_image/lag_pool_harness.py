"""
Shared harness for the lag-pooling ablation battery (2026-07-09).

Two pooling levels:
  Level A  patches (196) -> image vector   (currently mean over patch tokens)
  Level B  images        -> per-fetus lag  (currently mean over a fetus's images)

Every variant is scored on THREE metrics, not clock R2 alone:
  clock_GA_r     : image->GA Ridge, group-CV (5-fold, no fetus leakage)
  lag_SGA_r      : per-fetus lag <-> SGA (point-biserial)  [the metric that matters]
  lag_birthpct_r : per-fetus lag <-> birth percentile
  lag_LGA_r      : per-fetus lag <-> LGA

CRITICAL data hygiene (learned the hard way):
  - the `ga` field in _patchgrid_pooled.npz contains wrong-pregnancy contamination
    (range -181..+204 wk). ALWAYS filter to 6 <= GA <= 42 before training the clock,
    or the clock collapses to r=0.19 instead of 0.85.

Inputs (results/img_align/):
  _patchgrid_pooled.npz  : pool_mean/max/std/center/periph (61804,768) + fid,ga,fn
  _citus_groups.csv      : Cod -> grp_citus (SGA/AGA/LGA)
  _merged_labels.npz     : birth (fids-indexed), fids
Raw grids (USB, block6=b5): /Users/tiago/usb/patchgrids/b5_grid_*.npz  (2048,196,768) fp16
  loader: from fgm_image.patchgrid_io import iter_grids, load_grids
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold
from scipy import stats

IMG = "/Users/tiago/dev/fgr-geometry/results/img_align"


def load_pooled():
    P = np.load(f"{IMG}/_patchgrid_pooled.npz", allow_pickle=True)
    ga = P["ga"]; gm = np.isfinite(ga) & (ga >= 6) & (ga <= 42)   # wrong-pregnancy filter
    return {k: (P[k][gm] if P[k].ndim and P[k].shape[0] == len(ga) else P[k])
            for k in ["pool_mean", "pool_max", "pool_std", "pool_center", "pool_periph"]}, \
           P["fid"][gm].astype(int), ga[gm]


def outcomes():
    cg = pd.read_csv(f"{IMG}/_citus_groups.csv").set_index("Cod").grp_citus
    lab = np.load(f"{IMG}/_merged_labels.npz", allow_pickle=True)
    return cg, pd.Series(lab["birth"], index=lab["fids"])


def clock_perimg(feat, fid, ga, alpha=100.0):
    """group-CV image->GA; returns (clock_r, per-image lag=predGA-GA)."""
    Ez = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    pga = cross_val_predict(Ridge(alpha), Ez, ga, cv=GroupKFold(5), groups=fid)
    return float(np.corrcoef(pga, ga)[0, 1]), pga - ga


def fetus_agg(perimg_lag, fid, how="mean"):
    g = pd.DataFrame({"fid": fid, "lag": perimg_lag}).groupby("fid").lag
    if how == "mean": return g.mean()
    if how == "median": return g.median()
    if how == "trim": return g.apply(lambda x: stats.trim_mean(x, 0.2) if len(x) >= 3 else x.mean())
    raise ValueError(how)


def metrics(fl, clock_r, cg, birthf, levelA="", levelB=""):
    idx = fl.index
    sga = (cg.reindex(idx) == "SGA").astype(float).values
    lga = (cg.reindex(idx) == "LGA").astype(float).values
    bp = birthf.reindex(idx).values; lv = fl.values
    m = np.isfinite(lv) & np.isfinite(bp)
    return dict(levelA=levelA, levelB=levelB, n=int(m.sum()), clock_GA_r=round(clock_r, 3),
                lag_SGA_r=round(stats.pointbiserialr(sga[m], lv[m])[0], 3),
                lag_LGA_r=round(stats.pointbiserialr(lga[m], lv[m])[0], 3),
                lag_birthpct_r=round(stats.pearsonr(lv[m], bp[m])[0], 3))
