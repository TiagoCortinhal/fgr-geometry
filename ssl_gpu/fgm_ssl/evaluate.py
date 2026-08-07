"""Evaluation — IDENTICAL protocol to the frozen-feature analyses.

Every arm goes through this file and nothing else, so a difference between arms
is a difference in the encoder and not in how it was scored. The adjustment
(GA + maternal BMI residualised from BOTH sides), the 5-fold out-of-fold ridge,
and the permutation null are the same ones used for frozen USFM and radiomics.

TRAP THIS FILE EXISTS TO AVOID: residualising only one side, or adjusting after
the fact, both leave the acquisition confound in and turn maternal habitus into
an apparent physiological signal.
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


def residualise(X, covariates):
    """Project covariates out of X. Pass the SAME covariates for image and target."""
    if not covariates:
        return X
    A = np.column_stack([np.ones(len(X))] +
                        [np.asarray(c).reshape(len(X), -1) for c in covariates])
    return X - A @ np.linalg.lstsq(A, X, rcond=None)[0]


def heldout_cc(X, Y, n_pc=10, folds=5, seed=0):
    """Out-of-fold canonical correlation. Block-level primary endpoint."""
    out = []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        k = min(n_pc, X.shape[1], len(tr) - 1)
        p = PCA(k, random_state=0).fit(X[tr])
        c = CCA(n_components=1, max_iter=2000).fit(p.transform(X[tr]), Y[tr])
        a, b = c.transform(p.transform(X[te]), Y[te])
        out.append(np.corrcoef(a[:, 0], b[:, 0])[0, 1])
    return float(np.mean(out))


def heldout_r(y, X, n_pc=8, folds=5, seed=0):
    """Out-of-fold ridge correlation. Single-variable endpoint."""
    p = np.zeros_like(y)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        pc = PCA(min(n_pc, X.shape[1], len(tr) - 1), random_state=0).fit(X[tr])
        p[te] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(
            pc.transform(X[tr]), y[tr]).predict(pc.transform(X[te]))
    return float(np.corrcoef(p, y)[0, 1])


def permutation_p(stat_fn, X, Y, n_perm=1000, seed=0):
    """Permute the TARGET rows only. Permuting features would also destroy the
    covariate adjustment already applied to them."""
    obs = stat_fn(X, Y)
    rng = np.random.default_rng(seed)
    null = np.array([stat_fn(X, Y[rng.permutation(len(Y))]) for _ in range(n_perm)])
    return dict(observed=float(obs),
                p=float((1 + (null >= obs).sum()) / (1 + len(null))),
                null_p95=float(np.percentile(null, 95)),
                null_mean=float(null.mean()))


def split_spread_delta(X_new, X_ref, Y, stat_fn, n_splits=200, seed=0):
    """Does the new encoder beat the incumbent? Spread over INDEPENDENT CV splits.

    The stop rule rests on this interval, so it must not be a bootstrap.
    `stat_fn` runs its own KFold internally; resampling rows with replacement
    and feeding them in would duplicate fetuses across train and test folds of
    the same split, leaking train into test and producing intervals that can
    exclude their own point estimate. That failure has already occurred twice in
    this project and is recorded in the project's tooling notes.

    Instead: re-run BOTH arms on the same fresh CV seed and take the paired
    difference. Every fetus appears exactly once per split, so there is no
    leakage; the spread reflects genuine split-to-split variability.
    """
    d = np.array([stat_fn(X_new, Y, seed=s) - stat_fn(X_ref, Y, seed=s)
                  for s in range(seed, seed + n_splits)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(d.mean()), interval=[float(lo), float(hi)],
                frac_le_0=float((d <= 0).mean()), n_splits=int(n_splits),
                method="paired difference over independent CV seeds (NOT a bootstrap)")


def benjamini_hochberg(pvals, q=0.10):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    n = len(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = o[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return adj <= q, adj


def evaluate_arm(IMG, panel, ga, bmi, endpoints, n_perm=1000, seed=0, min_n=120):
    """Run every prespecified endpoint for one arm.

    IMG: (n_fetuses, d) representation, rows aligned to panel rows.
    endpoints: {name: column-index array into panel} for blocks, or a 1-d array
    for a single variable.
    """
    n = len(IMG)
    cov = [np.where(np.isfinite(ga), ga, np.nanmean(ga)).reshape(n, 1),
           np.where(np.isfinite(bmi), bmi, 0.0).reshape(n, 1)]
    res = {}
    for name, tgt in endpoints.items():
        Y = panel[:, tgt] if np.ndim(tgt) and np.asarray(tgt).dtype.kind == "i" \
            else np.asarray(tgt).reshape(n, -1)
        keep = np.isfinite(IMG).all(1) & (np.isfinite(Y).sum(1) >= Y.shape[1] - 1)
        if keep.sum() < min_n:
            res[name] = dict(skipped=True, n=int(keep.sum()),
                             reason=f"n={int(keep.sum())} below min_n={min_n}")
            continue
        Ys = np.where(np.isfinite(Y[keep]), Y[keep], 0.0)
        Xs = IMG[keep]
        cv = [c[keep] for c in cov]
        # A covariate must never be the target: residualising BMI on BMI (or GA
        # on GA) drives the residual to ~0 and produces a meaningless number.
        # This is the exact defect that made a growth-block test degenerate
        # earlier in the project, caught in review. Drop the offending covariate.
        drop = {"C1_maternal_BMI": 1, "C2_GA": 0}.get(name)
        use_cv = [c for i, c in enumerate(cv) if i != drop]
        ladder = {}
        rungs = [("raw", [])]
        if drop != 0:
            rungs.append(("GA", use_cv[:1]))
        rungs.append(("GA+BMI" if drop is None else "adjusted", use_cv))
        for lab, use in rungs:
            ladder[lab] = heldout_cc(residualise(Xs, use), residualise(Ys, use))
        cv = use_cv
        Xa, Ya = residualise(Xs, cv), residualise(Ys, cv)
        perm = permutation_p(lambda a, b: heldout_cc(a, b), Xa, Ya,
                             n_perm=n_perm, seed=seed)
        res[name] = dict(n=int(keep.sum()), ladder=ladder, **perm)
    return res
