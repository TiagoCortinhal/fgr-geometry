"""
Per-unit clocks for the appearance-age lag battery (Option 3).

Two levers on the USFM->GA image clock, compared to the mean-patches / mean-images
baseline (clock_GA_r=0.848, lag_SGA_r=-0.155):

(a) PER-PATCH clock.  Each of the 196 patch tokens of each image is its own
    training row (patch768 -> that image's GA).  A regularized Ridge is fit with
    fetus-grouped 5-fold CV (no fetus leakage), the 196 held-out per-patch GA
    predictions are pooled (mean) back to an image predGA, and image lag = predGA
    - GA is aggregated per fetus.  We report the in-sample vs held-out gap as an
    explicit overfit check.

(b) PER-IMAGE clock with INVERSE-VARIANCE fetus pooling.  Using the per-patch
    predictions, each image gets a predGA and a within-image variance across its
    196 patch predictions (disagreement).  Per fetus, image lags are pooled with
    weights 1/var_image so images whose patches agree count more.

Scale: 52,637 images x 196 patches ~= 10.3M rows x 768 feats.  We stream the raw
grids from USB once, accumulating per-fold raw Gram matrices (X^T X, X^T y, sums),
then reconstruct the globally-standardized Ridge system analytically per fold.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy import stats

REPO = "/Users/tiago/dev/fgr-geometry"
IMG = f"{REPO}/results/img_align"
GRID_DIR = os.environ.get("PATCHGRID_DIR", "/Users/tiago/usb/patchgrids")
D = 768
NPATCH = 196


def _shards(block=5):
    return sorted(glob.glob(f"{GRID_DIR}/b{block}_grid_*.npz"))


def stream_accumulate(meta_f, block=5, n_folds=5):
    """Single streaming pass. Returns per-fold raw accumulators for patch rows.

    meta_f: DataFrame with columns shard,row,fid,ga,fold (GA-filtered images).
    For each fold f, accumulates over that fold's patches (196 per image):
        n[f]            row count
        Sx[f]  (D,)     sum_x  x
        Sxx[f] (D,D)    sum_x  x x^T
        Sxy[f] (D,)     sum_x  x*y      (y = image GA broadcast to 196)
        Sy[f]           sum_x  y
    """
    n = np.zeros(n_folds)
    Sx = np.zeros((n_folds, D))
    Sxx = np.zeros((n_folds, D, D))
    Sxy = np.zeros((n_folds, D))
    Sy = np.zeros(n_folds)
    by_shard = {s: g for s, g in meta_f.groupby("shard")}
    shards = _shards(block)
    for si, path in enumerate(shards):
        if si not in by_shard:
            continue
        g = by_shard[si]
        z = np.load(path, allow_pickle=True)
        grid = z["grid"]  # (n_img_shard, 196, 768) fp16
        rows = g["row"].values
        ga = g["ga"].values.astype(np.float64)
        folds = g["fold"].values
        # expand to patches: (n_kept, 196, 768) -> (n_kept*196, 768)
        sub = grid[rows].astype(np.float64)            # (k,196,768)
        k = sub.shape[0]
        Xp = sub.reshape(k * NPATCH, D)
        yp = np.repeat(ga, NPATCH)
        fp = np.repeat(folds, NPATCH)
        for f in range(n_folds):
            m = fp == f
            if not m.any():
                continue
            Xf = Xp[m]; yf = yp[m]
            n[f] += Xf.shape[0]
            Sx[f] += Xf.sum(0)
            Sxx[f] += Xf.T @ Xf
            Sxy[f] += Xf.T @ yf
            Sy[f] += yf.sum()
        del z, grid, sub, Xp
    return dict(n=n, Sx=Sx, Sxx=Sxx, Sxy=Sxy, Sy=Sy)


def _global_standardizer(acc):
    N = acc["n"].sum()
    tot_x = acc["Sx"].sum(0)
    mu = tot_x / N
    diag_xx = np.array([acc["Sxx"][f].diagonal() for f in range(len(acc["n"]))]).sum(0)
    var = diag_xx / N - mu ** 2
    sigma = np.sqrt(np.maximum(var, 1e-12))
    return mu, sigma


def _fold_std_system(acc, folds, mu, sigma):
    """Reconstruct standardized (centered) Ridge system X_s^T X_s, X_s^T y for a
    UNION of raw folds, where X_s = (X - mu)/sigma. Returns Gxx, Gxy, n, ybar,
    and xbar_s (mean of standardized features) for intercept handling."""
    n = sum(acc["n"][f] for f in folds)
    Sx = sum(acc["Sx"][f] for f in folds)
    Sxx = sum(acc["Sxx"][f] for f in folds)
    Sxy = sum(acc["Sxy"][f] for f in folds)
    Sy = sum(acc["Sy"][f] for f in folds)
    invs = 1.0 / sigma
    # standardized second moments (not yet centered on the training subset)
    # X_s^T X_s = D (Sxx - mu Sx^T - Sx mu^T + n mu mu^T) D
    M = Sxx - np.outer(mu, Sx) - np.outer(Sx, mu) + n * np.outer(mu, mu)
    XsXs = (invs[:, None] * M) * invs[None, :]
    # X_s^T y = D (Sxy - mu Sy)
    Xsy = invs * (Sxy - mu * Sy)
    # sum of standardized features = D (Sx - n mu)
    Sxs = invs * (Sx - n * mu)
    return XsXs, Xsy, n, Sy / n, Sxs / n


def _solve(acc, train, mu, sigma, alpha):
    XsXs, Xsy, n, ybar, xbar = _fold_std_system(acc, train, mu, sigma)
    # center: Xc^T Xc = XsXs - n xbar xbar^T ; Xc^T yc = Xsy - n xbar ybar
    Gxx = XsXs - n * np.outer(xbar, xbar)
    Gxy = Xsy - n * xbar * ybar
    w = np.linalg.solve(Gxx + alpha * np.eye(D), Gxy)
    b = ybar - xbar @ w
    return w, b


def fit_fold_weights(acc, mu, sigma, alpha, n_folds=5):
    """Returns (cv_weights, full_weights). cv_weights[k] = Ridge fit on all folds
    except k (held-out). full_weights = Ridge fit on ALL folds (for in-sample
    overfit check). Both in standardized feature space."""
    cv = {k: _solve(acc, [f for f in range(n_folds) if f != k], mu, sigma, alpha)
          for k in range(n_folds)}
    full = _solve(acc, list(range(n_folds)), mu, sigma, alpha)
    return cv, full


def predict_perpatch(meta_f, cv_weights, full_weights, mu, sigma, block=5):
    """Stream again; for each image produce, from its 196 per-patch GA predictions:
      predGA_cv    mean per-patch predGA using the image's HELD-OUT fold weights
      patch_var    within-image variance of those 196 held-out predictions
      predGA_full  mean per-patch predGA using FULL-DATA weights (in-sample)
    Returns DataFrame indexed like meta_f."""
    invs = 1.0 / sigma
    wf, bf = full_weights
    by_shard = {s: g for s, g in meta_f.groupby("shard")}
    shards = _shards(block)
    n = len(meta_f)
    p_cv = np.full(n, np.nan); p_var = np.full(n, np.nan); p_full = np.full(n, np.nan)
    pos = {idx: i for i, idx in enumerate(meta_f.index)}
    for si, path in enumerate(shards):
        if si not in by_shard:
            continue
        g = by_shard[si]
        z = np.load(path, allow_pickle=True)
        grid = z["grid"]
        rows = g["row"].values
        folds = g["fold"].values
        sub = grid[rows].astype(np.float64)           # (k,196,768)
        Xs = (sub - mu) * invs                          # standardize
        # in-sample (full weights), vectorized over all patches of the shard
        pp_full = Xs @ wf + bf                          # (k,196)
        for j in range(sub.shape[0]):
            w, b = cv_weights[folds[j]]
            pp = Xs[j] @ w + b                          # (196,) held-out predGA
            gi = pos[g.index[j]]
            p_cv[gi] = pp.mean(); p_var[gi] = pp.var(); p_full[gi] = pp_full[j].mean()
        del z, grid, sub, Xs
    res = meta_f.copy()
    res["predGA_cv"] = p_cv
    res["patch_var"] = p_var
    res["predGA_full"] = p_full
    return res


def invvar_fetus_agg(perimg_lag, fid, weight):
    """Inverse-variance weighted per-fetus mean of image lags. weight = 1/var."""
    df = pd.DataFrame({"fid": np.asarray(fid), "lag": np.asarray(perimg_lag),
                       "w": np.asarray(weight)})
    df = df[np.isfinite(df.lag) & np.isfinite(df.w) & (df.w > 0)]
    g = df.groupby("fid")
    return (g.apply(lambda x: np.average(x.lag, weights=x.w)))



def build_meta(block=5, n_folds=5):
    """Scan raw grids -> GA-filtered image manifest with fetus-grouped CV folds."""
    from sklearn.model_selection import GroupKFold
    rows = []
    for si, path in enumerate(_shards(block)):
        z = np.load(path, allow_pickle=True)
        rows.append(pd.DataFrame({
            "shard": si, "row": np.arange(len(z["fetus_id"])),
            "fid": z["fetus_id"].astype(float), "ga": z["ga_weeks_recovered"].astype(float),
            "fn": z["new_filename"]}))
    meta = pd.concat(rows, ignore_index=True)
    gm = np.isfinite(meta.ga) & (meta.ga >= 6) & (meta.ga <= 42)
    meta_f = meta[gm].reset_index(drop=True).copy()
    fold = np.empty(len(meta_f), int)
    for k, (_, te) in enumerate(GroupKFold(n_folds).split(meta_f, groups=meta_f.fid.values)):
        fold[te] = k
    meta_f["fold"] = fold
    return meta_f


def run(block=5, alphas=(10.0, 100.0, 1000.0, 10000.0, 100000.0), n_folds=5):
    """Full Option-3 pipeline. Two streaming passes over the raw grids:
       1. accumulate per-fold Gram matrices (X^T X, X^T y) over all patch rows,
       2. predict per-image mean per-patch GA (held-out + in-sample) & patch var.
    Returns (meta_f, acc, cv/full weights, predictions). ~40 min on the USB.
    Metrics are scored with fgm_image.lag_pool_harness so they match the battery.
    """
    meta_f = build_meta(block, n_folds)
    acc = stream_accumulate(meta_f, block, n_folds)
    mu, sigma = _global_standardizer(acc)
    cvW, fullW = fit_fold_weights(acc, mu, sigma, alphas[1], n_folds)  # single-alpha entry
    preds = predict_perpatch(meta_f, cvW, fullW, mu, sigma, block)
    return meta_f, acc, (cvW, fullW), preds


if __name__ == "__main__":
    run()
