import json
import os
import time
import numpy as np
from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import featuresets as FS
from fgrgeom.latent import FactorAnalysisMissing

OUT = "results/nl/gplvm.json"
Q = 6
NUM_INDUCING = 10
MAX_ITERS = 20
HELDOUT_FRAC = 0.15
FA_MAX_ITER = 80  # FactorAnalysisMissing EM is a per-sample Python loop; cap it.
SUBSAMPLE = 250  # GPy missing_data mode is O(n) per output dim per opt-iter and
# slow; both models fit on the SAME seeded row subsample for a fair head-to-head.


def _standardize(X, M):
    mu = np.array([X[M[:, j], j].mean() if M[:, j].any() else 0.0
                   for j in range(X.shape[1])])
    sd = np.array([X[M[:, j], j].std() if M[:, j].sum() > 1 else 1.0
                   for j in range(X.shape[1])])
    sd[sd == 0] = 1.0
    return mu, sd


def _fit_gplvm(Ystd, q=Q, num_inducing=NUM_INDUCING, max_iters=MAX_ITERS, seed=C.SEED):
    from GPy.models.bayesian_gplvm_minibatch import BayesianGPLVMMiniBatch
    from GPy.kern import RBF
    np.random.seed(seed)
    kern = RBF(q, ARD=True)
    m = BayesianGPLVMMiniBatch(Ystd, q, kernel=kern, num_inducing=num_inducing,
                               missing_data=True)
    m.optimize(max_iters=max_iters, messages=False)
    return m


def _ard_relevance(m):
    ls = np.asarray(m.kern.lengthscale.values, dtype=float)
    var = float(m.kern.variance)
    rel = var / (ls ** 2)
    rel = rel / rel.max()
    return rel


def fit_both(X, M, seed=C.SEED, return_models=False):
    """Hold out a fraction of OBSERVED entries, fit GP-LVM and FA on the rest,
    reconstruct held-out entries, compare in original feature units. Mask-aware:
    missing AND held-out entries are NaN/unobserved to both models. No imputation.
    Returns recon metrics, ARD relevance, and the two latent matrices (from the
    same train mask, so the two latents are directly comparable)."""
    rng = np.random.default_rng(seed)
    obs_idx = np.argwhere(M)
    ntest = int(HELDOUT_FRAC * len(obs_idx))
    sel = rng.choice(len(obs_idx), size=ntest, replace=False)
    test = obs_idx[sel]
    Mtr = M.copy()
    Mtr[test[:, 0], test[:, 1]] = False

    mu, sd = _standardize(X, Mtr)
    Ystd = ((X - mu) / sd)
    Ystd[~Mtr] = np.nan

    t0 = time.time()
    m = _fit_gplvm(Ystd, seed=seed)
    print("    gplvm fit done", round(time.time() - t0, 1), "s", flush=True)
    Zgp = np.asarray(m.X.mean)
    pred_std, _ = m.predict(Zgp)  # standardized space (Ystd was passed in)

    t1 = time.time()
    fa = FactorAnalysisMissing(k=Q, max_iter=FA_MAX_ITER, seed=seed).fit(X, Mtr)
    print("    fa fit done", round(time.time() - t1, 1), "s", flush=True)
    Zfa, _ = fa.transform(X, Mtr)
    fa_recon_orig = fa.center_[None, :] + fa.scale_[None, :] * (
        fa.mu_[None, :] + Zfa @ fa.W_.T)
    fa_pred_std = (fa_recon_orig - mu) / sd  # into the same standardized space

    # Score held-out reconstruction in STANDARDIZED space so mixed-scale
    # features (biom z ~1 vs doppler pctl ~1e2) contribute comparably.
    ti, tj = test[:, 0], test[:, 1]
    true = (X[ti, tj] - mu[tj]) / sd[tj]
    denom = (true ** 2).sum()
    gp_err = ((true - pred_std[ti, tj]) ** 2).sum()
    fa_err = ((true - fa_pred_std[ti, tj]) ** 2).sum()

    rel = _ard_relevance(m)
    recon = {
        "scoring": "standardized (per-feature z, train moments)",
        "n_test_entries": int(ntest),
        "gp_mse": float(gp_err / ntest),
        "fa_mse": float(fa_err / ntest),
        "gp_r2": float(1 - gp_err / denom),
        "fa_r2": float(1 - fa_err / denom),
        "gp_minus_fa_mse": float((gp_err - fa_err) / ntest),
        "ard_relevance": [round(float(x), 4) for x in np.sort(rel)[::-1]],
        "gplvm_effective_dim": int((rel > 0.05).sum()),
        "kern_variance": float(m.kern.variance),
        "noise_var": float(m.likelihood.variance),
    }
    return recon, rel, Zgp, Zfa


def _cv_auc(Zfeat, y, seed=C.SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(Zfeat, y):
        sc = StandardScaler().fit(Zfeat[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.transform(Zfeat[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(Zfeat[te]))[:, 1]
    return float(roc_auc_score(y, p))


def _cv_r2(Zfeat, y, seed=C.SEED):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in kf.split(Zfeat):
        sc = StandardScaler().fit(Zfeat[tr])
        rg = Ridge(alpha=1.0).fit(sc.transform(Zfeat[tr]), y[tr])
        p[te] = rg.predict(sc.transform(Zfeat[te]))
    return float(r2_score(y, p))


def outcome_pred(Zgp, Zfa, panel, rows):
    out = panel.outcomes.iloc[rows].reset_index(drop=True)
    res = {}
    for tgt in ["sga", "lga"]:  # severe_sga too rare for a meaningful CV-AUC here
        y = out[tgt].values.astype(float)
        ok = ~np.isnan(y)
        yb = y[ok].astype(int)
        if yb.sum() < 10:
            continue
        res[tgt] = {"n": int(ok.sum()), "pos": int(yb.sum()),
                    "gplvm_auc": _cv_auc(Zgp[ok], yb),
                    "fa_auc": _cv_auc(Zfa[ok], yb)}
    y = out["percentile_birth_pop"].values.astype(float)
    ok = ~np.isnan(y)
    res["percentile_birth_pop"] = {"n": int(ok.sum()),
                                   "gplvm_r2": _cv_r2(Zgp[ok], y[ok]),
                                   "fa_r2": _cv_r2(Zfa[ok], y[ok])}
    return res


def linearity(Zgp, Zfa, seed=C.SEED):
    """How linearly related are the GP-LVM and FA latents. If a linear map
    reconstructs one from the other with R2 ~ 1, the manifolds coincide up to
    a linear reparametrisation (flat). Lower R2 -> GP-LVM bends away from FA."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    def cvmap(A, B):
        kf = KFold(5, shuffle=True, random_state=seed)
        Bp = np.zeros_like(B)
        for tr, te in kf.split(A):
            lr = LinearRegression().fit(A[tr], B[tr])
            Bp[te] = lr.predict(A[te])
        return float(r2_score(B, Bp, multioutput="variance_weighted"))

    return {"fa_to_gp_r2": cvmap(Zfa, Zgp),
            "gp_to_fa_r2": cvmap(Zgp, Zfa)}


def negative_control(shape, M, kdim=2, seed=C.SEED):
    """Linear-Gaussian single manifold of matching shape and mask. Confirms the
    pipeline does not fabricate curvature: GP-LVM should not beat FA on held-out
    reconstruction and ARD should recover ~kdim directions."""
    rng = np.random.default_rng(seed + 1)
    n, d = shape
    Ztrue = rng.normal(size=(n, kdim))
    W = rng.normal(size=(d, kdim))
    X = Ztrue @ W.T + rng.normal(scale=0.5, size=(n, d))
    recon, rel, _, _ = fit_both(X, M, seed=seed)
    recon["true_dim"] = kdim
    recon["ard_relevance_full"] = [round(float(x), 4) for x in np.sort(rel)[::-1]]
    return recon


def run_set(panel, name, seed=C.SEED):
    X, M, names = FS.build(panel, name)
    cc_full = int(M.all(1).sum())
    n_full = X.shape[0]
    rng = np.random.default_rng(seed)
    if SUBSAMPLE and n_full > SUBSAMPLE:
        rows = np.sort(rng.choice(n_full, size=SUBSAMPLE, replace=False))
    else:
        rows = np.arange(n_full)
    Xs, Ms = X[rows], M[rows]
    t0 = time.time()
    recon, rel, Zgp, Zfa = fit_both(Xs, Ms, seed=seed)
    res = {
        "n_full": n_full, "n_used": int(len(rows)),
        "subsampled": bool(len(rows) < n_full),
        "n_features": int(X.shape[1]),
        "observed_frac": float(Ms.mean()),
        "complete_case_rows_full": cc_full,
        "complete_case_usable": cc_full >= 50,
        "method": "GPy BayesianGPLVMMiniBatch missing_data=True (mask-aware, no imputation)",
        "latent_dim_requested": Q,
        "num_inducing": NUM_INDUCING, "max_iters": MAX_ITERS,
        "reconstruction_heldout": recon,
        "ard_relevance_full_fit": [round(float(x), 4) for x in np.sort(rel)[::-1]],
        "gplvm_effective_dim_full_fit": int((rel > 0.05).sum()),
        "outcome_prediction": outcome_pred(Zgp, Zfa, panel, rows),
        "latent_linearity": linearity(Zgp, Zfa),
        "seconds": round(time.time() - t0, 1),
    }
    return res, (Xs.shape, Ms)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    panel = P.load_panel()
    out = {"sets": {}}
    shapes = {}
    for name in ["minimal", "full"]:
        res, (shp, M) = run_set(panel, name)
        out["sets"][name] = res
        shapes[name] = (shp, M)
    shp, M = shapes["minimal"]
    out["negative_control_linear_manifold"] = negative_control(shp, M, kdim=2)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
