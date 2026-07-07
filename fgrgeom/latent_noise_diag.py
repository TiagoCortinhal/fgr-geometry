import json
import numpy as np
from fgrgeom import config as C
from fgrgeom import panel as P

RESULTS = C.DATA.parent.parent / "fgr-geometry" / "results"

MINIMAL = ("biom", "doppler")
FULL = ("biom", "ratios", "doppler", "cardiac", "maternal", "bp")
ADDED = ("ratios", "cardiac", "maternal", "bp")


def _block(name):
    if name.startswith("dop:"):
        return "doppler"
    if name.startswith("card:"):
        return "cardiac"
    if name.startswith("mat:"):
        return "maternal"
    if name.startswith("bp:"):
        return "bp"
    if ":hc_ac" in name or ":fl_ac" in name:
        return "ratios"
    return "biom"


class ARDFactorAnalysisMissing:
    """FA with EM + exact missing handling and a per-factor ARD prior on the
    loading columns: W[:,k] ~ N(0, alpha_k^{-1} I). alpha_k -> large prunes a
    factor (automatic relevance / effective rank). Standardised observed-only."""

    def __init__(self, k, ard=True, max_iter=120, tol=3e-4, alpha_cap=1e8,
                 seed=C.SEED):
        self.k = k
        self.ard = ard
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_cap = alpha_cap
        self.seed = seed

    def _standardize(self, X, M):
        mu = np.array([X[M[:, j], j].mean() if M[:, j].any() else 0.0
                       for j in range(X.shape[1])])
        sd = np.array([X[M[:, j], j].std() if M[:, j].sum() > 1 else 1.0
                       for j in range(X.shape[1])])
        sd[sd == 0] = 1.0
        self.center_, self.scale_ = mu, sd
        Xs = (X - mu) / sd
        Xs[~M] = 0.0
        return Xs

    def fit(self, X, M):
        rng = np.random.default_rng(self.seed)
        Xs = self._standardize(X, M)
        n, d = Xs.shape
        k = self.k
        W = rng.normal(scale=0.1, size=(d, k))
        psi = np.ones(d)
        mu = np.zeros(d)
        alpha = np.ones(k)
        ll_prev = -np.inf
        self.ll_trace_ = []
        # group samples by identical missingness pattern (exact same EM math,
        # cov/logdet computed once per pattern, applied batched).
        pat, inv = np.unique(M, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        groups = [(pat[p], np.where(inv == p)[0]) for p in range(len(pat))]
        Ik = np.eye(k)
        log2pi = np.log(2 * np.pi)
        for it in range(self.max_iter):
            Ez = np.zeros((n, k))
            Ezz = np.zeros((n, k, k))
            ll = 0.0
            for o, idx in groups:
                if not o.any():
                    Ezz[idx] = Ik
                    continue
                Wo = W[o]
                psio = psi[o]
                Xog = Xs[np.ix_(idx, o)] - mu[o]        # (g, |o|)
                WtP = Wo.T / psio                        # (k, |o|)
                cov = np.linalg.inv(Ik + WtP @ Wo)       # (k, k)
                EzG = (Xog @ WtP.T) @ cov                # (g, k)
                Ez[idx] = EzG
                Ezz[idx] = cov[None] + np.einsum("gi,gj->gij", EzG, EzG)
                Sig = Wo @ Wo.T + np.diag(psio)
                _, logdet = np.linalg.slogdet(Sig)
                sol = np.linalg.solve(Sig, Xog.T)        # (|o|, g)
                quad = np.einsum("gi,ig->g", Xog, sol)
                ll += -0.5 * (len(idx) * (logdet + o.sum() * log2pi)
                              + quad.sum())
            self.ll_trace_.append(ll)
            Adiag = np.diag(alpha) if self.ard else np.zeros((k, k))
            for j in range(d):
                obs = M[:, j]
                if not obs.any():
                    continue
                xj = Xs[obs, j]
                Ez_j = Ez[obs]
                Ezz_j = Ezz[obs].sum(axis=0)
                rhs = (xj[:, None] * Ez_j).sum(axis=0)
                W[j] = np.linalg.solve(Ezz_j + Adiag + 1e-6 * np.eye(k), rhs)
                mu[j] = (xj - Ez_j @ W[j]).mean()
                covmean = (Ezz[obs] - np.einsum("ik,il->ikl",
                                                Ez[obs], Ez[obs])).mean(0)
                r = (xj - mu[j]) - Ez_j @ W[j]
                psi[j] = max((r ** 2).mean() + W[j] @ covmean @ W[j], 1e-4)
            if self.ard:
                alpha = np.minimum(d / (np.sum(W ** 2, axis=0) + 1e-12),
                                   self.alpha_cap)
            if it > 0 and abs(ll - ll_prev) < self.tol * max(1.0, abs(ll_prev)):
                break
            ll_prev = ll
        self.W_, self.psi_, self.mu_, self.alpha_ = W, psi, mu, alpha
        self.Ez_ = Ez
        self.n_iter_ = it + 1
        return self

    def transform(self, X, M):
        Xs = (X - self.center_) / self.scale_
        Xs[~M] = 0.0
        n, k = X.shape[0], self.k
        Ez = np.zeros((n, k))
        for i in range(n):
            o = M[i]
            if not o.any():
                continue
            Wo = self.W_[o]
            WtP = Wo.T / self.psi_[o]
            cov = np.linalg.inv(np.eye(k) + WtP @ Wo)
            Ez[i] = cov @ (WtP @ (Xs[i, o] - self.mu_[o]))
        return Ez


def _communality(W, psi):
    sig = (W ** 2).sum(axis=1)  # ||W_j||^2 shared
    return sig / (sig + psi)     # fraction of model-implied var that is shared


def _principal_angles(A, B):
    """Principal angles (deg) between column spaces of A and B (same n rows)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1, 1)
    return np.degrees(np.arccos(s))


def _cv_r2(Z, y, folds=5, seed=C.SEED):
    """Linear CV R^2 of latent Z predicting y (complete-case on y)."""
    ok = np.isfinite(y)
    Z, y = Z[ok], y[ok]
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fold = np.array_split(idx, folds)
    pred = np.full(n, np.nan)
    for f in range(folds):
        te = fold[f]
        tr = np.concatenate([fold[g] for g in range(folds) if g != f])
        A = np.column_stack([np.ones(len(tr)), Z[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(len(te)), Z[te]]) @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def _fit_plain(X, M, k, seed=C.SEED, max_iter=120):
    fa = ARDFactorAnalysisMissing(k=k, ard=False, seed=seed,
                                  max_iter=max_iter).fit(X, M)
    return fa


def positive_control(seed=0):
    """Planted: rank-2 structured block + pure-noise blocks matching the added
    columns. ARD-FA must (a) prune to ~2 factors, (b) zero noise loadings."""
    rng = np.random.default_rng(seed)
    n = 977
    d_sig, d_noise = 26, 32
    Zt = rng.normal(size=(n, 2))
    Wt = rng.normal(size=(d_sig, 2))
    Xsig = Zt @ Wt.T + 0.3 * rng.normal(size=(n, d_sig))
    Xnoise = rng.normal(size=(n, d_noise))
    X = np.concatenate([Xsig, Xnoise], axis=1)
    M = np.ones_like(X, bool)
    fa = ARDFactorAnalysisMissing(k=6, ard=True, seed=seed).fit(X, M)
    relev = 1.0 / fa.alpha_
    relev_n = relev / relev.max()
    eff = int((relev_n > 0.5).sum())
    comm = _communality(fa.W_, fa.psi_)
    return {
        "init_k": 6,
        "effective_factors_relev_gt_0.5": eff,
        "relevance_normalized": [round(float(x), 4) for x in np.sort(relev_n)[::-1]],
        "mean_communality_signal_block": round(float(comm[:d_sig].mean()), 4),
        "mean_communality_noise_block": round(float(comm[d_sig:].mean()), 4),
        "passes": bool(eff <= 3 and comm[:d_sig].mean() > 0.4
                       and comm[d_sig:].mean() < 0.15),
    }


def run():
    pan = P.load_panel()
    Xm, Mm, nm = P.flatten(pan, include=MINIMAL)
    Xf, Mf, nf = P.flatten(pan, include=FULL)
    blocks_f = np.array([_block(x) for x in nf])
    shared = np.array([b in ("biom", "doppler") for b in blocks_f])
    y = pan.outcomes["percentile_birth_pop"].to_numpy(float)

    out = {"n": int(Xm.shape[0]), "d_minimal": int(Xm.shape[1]),
           "d_full": int(Xf.shape[1]),
           "deps": {"ripser": False, "hdbscan": False, "torch": True}}

    # --- 1. positive control for ARD (gates everything) ---
    out["ard_positive_control"] = positive_control()

    # --- 2. plain FA k=2 minimal vs full: capacity / communality by block ---
    K = 2
    fa_m = _fit_plain(Xm, Mm, K)
    fa_f = _fit_plain(Xf, Mf, K)
    comm_m = _communality(fa_m.W_, fa_m.psi_)
    comm_f = _communality(fa_f.W_, fa_f.psi_)
    cap_f = (fa_f.W_ ** 2).sum(axis=1)  # per-column shared energy

    by_block = {}
    for b in FULL:
        sel = blocks_f == b
        by_block[b] = {
            "n_cols": int(sel.sum()),
            "mean_communality_full": round(float(comm_f[sel].mean()), 4),
            "mean_psi_full": round(float(fa_f.psi_[sel].mean()), 4),
            "capacity_total_sumWsq": round(float(cap_f[sel].sum()), 4),
            "capacity_per_col": round(float(cap_f[sel].mean()), 4),
        }
    tot_cap = cap_f.sum()
    shared_cap = cap_f[shared].sum()
    out["plain_fa_k2"] = {
        "by_block": by_block,
        "capacity_fraction_shared_biom_doppler": round(float(shared_cap / tot_cap), 4),
        "capacity_fraction_added": round(float((tot_cap - shared_cap) / tot_cap), 4),
        "mean_communality_minimal_bdshared": round(float(comm_m.mean()), 4),
        "mean_communality_full_bdshared": round(float(comm_f[shared].mean()), 4),
        "mean_communality_full_added": round(
            float(comm_f[~shared].mean()), 4),
    }

    # --- 3. centile-R2 at fixed k=2: does adding noise cols degrade the latent? ---
    Zm = fa_m.transform(Xm, Mm)
    Zf = fa_f.transform(Xf, Mf)
    out["centile_r2_k2"] = {
        "minimal": round(_cv_r2(Zm, y), 4),
        "full": round(_cv_r2(Zf, y), 4),
    }

    # --- 3b. SAME diagnostics at the pipeline default k=6, where the documented
    # 0.42->0.29 centile-R2 degradation lives (excess-capacity effect). ---
    K6 = 6
    fa_m6 = _fit_plain(Xm, Mm, K6)
    fa_f6 = _fit_plain(Xf, Mf, K6)
    comm_m6 = _communality(fa_m6.W_, fa_m6.psi_)
    comm_f6 = _communality(fa_f6.W_, fa_f6.psi_)
    by_block6 = {}
    for b in FULL:
        sel = blocks_f == b
        by_block6[b] = round(float(comm_f6[sel].mean()), 4)
    Zm6 = fa_m6.transform(Xm, Mm)
    Zf6 = fa_f6.transform(Xf, Mf)
    # how much does the full k=6 subspace (restricted to shared rows) rotate from
    # the minimal k=6 subspace, and from the core 2-D plane?
    ang6_full = _principal_angles(fa_m6.W_, fa_f6.W_[shared])
    ang6_core = _principal_angles(fa_m.W_, fa_f6.W_[shared])  # 2-D core vs 6-D full
    out["k6_default"] = {
        "centile_r2_minimal": round(_cv_r2(Zm6, y), 4),
        "centile_r2_full": round(_cv_r2(Zf6, y), 4),
        "mean_communality_minimal_bd": round(float(comm_m6.mean()), 4),
        "mean_communality_full_bdshared": round(float(comm_f6[shared].mean()), 4),
        "mean_communality_full_added": round(float(comm_f6[~shared].mean()), 4),
        "by_block_communality": by_block6,
        "max_angle_full6_vs_minimal6_shared": round(float(ang6_full.max()), 3),
        "max_angle_core2D_vs_full6_shared": round(float(ang6_core.max()), 3),
    }

    # --- 4. principal angles: minimal plane vs full plane restricted to shared cols.
    Wm = fa_m.W_                       # (26, 2)
    Wf_shared = fa_f.W_[shared]        # (26, 2) same row order (biom+doppler)
    ang = _principal_angles(Wm, Wf_shared)
    # bootstrap minimal-vs-minimal reference spread
    rng = np.random.default_rng(C.SEED)
    n = Xm.shape[0]
    boot = []
    for _ in range(24):
        bi = rng.integers(0, n, n)
        fb = _fit_plain(Xm[bi], Mm[bi], K, seed=int(rng.integers(1e6)),
                        max_iter=80)
        boot.append(_principal_angles(Wm, fb.W_).max())
    boot = np.array(boot)
    # random 2-D subspace floor in 26-D
    randang = []
    for _ in range(200):
        A = rng.normal(size=(Wm.shape[0], K))
        B = rng.normal(size=(Wm.shape[0], K))
        randang.append(_principal_angles(A, B).max())
    randang = np.array(randang)
    out["principal_angles_deg"] = {
        "minimal_vs_full_restricted": [round(float(a), 3) for a in ang],
        "max_angle": round(float(ang.max()), 3),
        "bootstrap_minimal_self_max_angle_mean": round(float(boot.mean()), 3),
        "bootstrap_minimal_self_max_angle_p95": round(float(np.percentile(boot, 95)), 3),
        "random_floor_max_angle_mean": round(float(randang.mean()), 3),
        "plane_within_bootstrap_spread": bool(
            ang.max() <= np.percentile(boot, 95)),
    }

    # --- 5. ARD-FA on full set: does it gate the noise dims? ---
    fa_ard = ARDFactorAnalysisMissing(k=6, ard=True, seed=C.SEED).fit(Xf, Mf)
    relev = 1.0 / fa_ard.alpha_
    relev_n = relev / relev.max()
    eff = int((relev_n > 0.5).sum())
    comm_ard = _communality(fa_ard.W_, fa_ard.psi_)
    # subspace of the surviving ARD factors vs minimal plane (restrict to shared rows)
    surv = np.argsort(relev_n)[::-1][:max(eff, 2)]
    Ward_shared = fa_ard.W_[np.ix_(shared, surv)]
    # take top-2 surviving for plane comparison
    ang_ard = _principal_angles(Wm, fa_ard.W_[np.ix_(shared, surv[:2])])
    Zard = fa_ard.transform(Xf, Mf)
    out["ard_fa_full"] = {
        "init_k": 6,
        "effective_factors_relev_gt_0.5": eff,
        "relevance_normalized": [round(float(x), 4) for x in np.sort(relev_n)[::-1]],
        "mean_communality_shared": round(float(comm_ard[shared].mean()), 4),
        "mean_communality_added": round(float(comm_ard[~shared].mean()), 4),
        "max_angle_surviving_plane_vs_minimal": round(float(ang_ard.max()), 3),
        "centile_r2_surviving": round(_cv_r2(Zard[:, surv], y), 4),
    }

    with open(RESULTS / "latent_noise_diag.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
