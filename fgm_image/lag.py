#!/usr/bin/env python3
"""
Appearance-age LAG — the maturation-clock discordance marker.  [MAIN IDEA]

The USFM image manifold is the sole readable maturation clock for this cohort
(image -> GA r=0.84; biometry IG-21 z-scores have GA normalized out, r=0.05).
The appearance-age lag = (image-predicted GA) - (dates GA), per image, averaged
per fetus. A negative lag means the fetus "looks younger than its dates."

WHY A LAG IS VALID (the general principle):
  A lag is meaningful ONLY when the two differenced quantities are measured
  INDEPENDENTLY. GA qualifies -- dates GA (LMP/dating scan) and appearance GA
  (image) are separate instruments, so their disagreement is real latent
  information. Maternal BMI does NOT: there is only one measurement (the scale),
  so an image-BMI "lag" is just prediction error (r=-0.965 with recorded BMI).
  Before building a lag on any covariate, check it has two independent measurements.

KEY RESULTS (block 6; see results/img_align/appearance_age_results.json):
  - SGA fetuses look younger: lag_mean -0.41 (SGA) vs +0.27 wk (non-SGA);
    lag<->SGA r=-0.166 (p<0.01), correctly signed for growth restriction.
  - Size-independent: biometry explains only 5.4% of the lag; survives biometry
    removal (residual r=-0.087, p=0.04).
  - Static, not drifting: longitudinal drift (lag_slope) is NULL (p=0.16-0.71).
  - Layer-invariant: ~-0.16 across USFM blocks 1/6/12.
  - Redundant for screening: biometry SGA AUC 0.781 -> +lag 0.789 (Delta +0.008).

VALUE: representational/biological, not predictive. "SGA fetuses look younger" is
an appearance-maturation statement biometry cannot make. Candidate uses beyond
FGR: brain/organ maturation phenotyping, neurodevelopment proxies,
standardization/QC (flag scans whose appearance-age disagrees with dates),
re-dating support where LMP is unreliable.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from scipy import stats


def maturation_clock(E, ga, fid, alpha=5.0, cv=5):
    """Group-CV image->GA regressor. Returns per-image predicted GA (no leakage:
    a fetus's own images never train its prediction)."""
    Ez = StandardScaler().fit_transform(E)
    return cross_val_predict(Ridge(alpha), Ez, ga, cv=GroupKFold(cv), groups=fid)


def appearance_age_lag(E, ga, fid, min_imgs=3, min_span=1.0):
    """Per-fetus appearance-age lag_mean and lag_slope (drift over GA).
    E: (N,D) image embeddings; ga: dates GA per image; fid: fetus id per image.
    Returns DataFrame indexed by fid with columns lag_mean, lag_slope, n."""
    pga = maturation_clock(E, ga, fid)
    M = pd.DataFrame({"fid": fid, "ga": ga, "resid": pga - ga})
    rows = []
    for f, g in M.groupby("fid"):
        if len(g) >= min_imgs and g.ga.std() > min_span:
            slope = np.polyfit(g.ga, g.resid, 1)[0]
            rows.append((f, g.resid.mean(), slope, len(g)))
    return pd.DataFrame(rows, columns=["fid", "lag_mean", "lag_slope", "n"]).set_index("fid")


def lag_association(lag, y, binary=True):
    """Association of a per-fetus lag with an outcome y (aligned by index)."""
    m = np.isfinite(lag) & np.isfinite(y)
    if binary:
        return stats.pointbiserialr(y[m], lag[m])
    return stats.pearsonr(lag[m], y[m])


def lag_size_independence(lag, biometry):
    """Fraction of lag explained by biometry (R^2) and the biometry-residualized
    lag. If R^2 is small and the residual keeps its outcome association, the lag
    is NOT just small-size read back."""
    m = np.isfinite(lag) & np.isfinite(biometry).all(1)
    lr = LinearRegression().fit(biometry[m], lag[m])
    r2 = lr.score(biometry[m], lag[m])
    resid = np.full_like(lag, np.nan, dtype=float)
    resid[m] = lag[m] - lr.predict(biometry[m])
    return r2, resid
