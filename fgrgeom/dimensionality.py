import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L

RESULTS = C.DATA.parent.parent / "fgr-geometry" / "results"


def participation_ratio(eig):
    eig = np.asarray(eig, float)
    eig = eig[eig > 0]
    if eig.size == 0:
        return 0.0
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def pairwise_cov(Xs, M):
    """Pairwise-complete correlation/covariance on standardized observed data.
    May be non-PSD; caller checks negative eigenvalue magnitude."""
    d = Xs.shape[1]
    Cm = np.full((d, d), np.nan)
    for j in range(d):
        for l in range(j, d):
            both = M[:, j] & M[:, l]
            if both.sum() > 2:
                a, b = Xs[both, j], Xs[both, l]
                Cm[j, l] = Cm[l, j] = np.mean(a * b)
    return Cm


def standardize_obs(X, M):
    mu = np.array([X[M[:, j], j].mean() if M[:, j].any() else 0.0
                   for j in range(X.shape[1])])
    sd = np.array([X[M[:, j], j].std() if M[:, j].sum() > 1 else 1.0
                   for j in range(X.shape[1])])
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xs[~M] = 0.0
    return Xs


def raw_spectrum(X, M):
    Xs = standardize_obs(X, M)
    Cm = pairwise_cov(Xs, M)
    Cm = np.nan_to_num(Cm)
    ev = np.linalg.eigvalsh(Cm)[::-1]
    return ev, Cm


def permutation_null(X, M, n_perm=200, seed=C.SEED):
    """Parallel-analysis null: independently shuffle each column (within observed
    rows) to destroy cross-column correlation, recompute the raw eigenspectrum."""
    rng = np.random.default_rng(seed)
    Xs = standardize_obs(X, M)
    d = X.shape[1]
    null_ev = np.zeros((n_perm, d))
    null_pr = np.zeros(n_perm)
    for b in range(n_perm):
        Xp = Xs.copy()
        Mp = M.copy()
        for j in range(d):
            idx = np.where(M[:, j])[0]
            perm = rng.permutation(idx)
            Xp[idx, j] = Xs[perm, j]
        Cm = np.nan_to_num(pairwise_cov(Xp, Mp))
        ev = np.linalg.eigvalsh(Cm)[::-1]
        null_ev[b] = ev
        null_pr[b] = participation_ratio(ev[ev > 0])
    return null_ev, null_pr


def boot_raw(X, M, n_boot=200, seed=C.SEED):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    ev2 = np.zeros(n_boot)
    pr = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ev, _ = raw_spectrum(X[idx], M[idx])
        ev2[b] = ev[1] if ev.size > 1 else np.nan
        pr[b] = participation_ratio(ev[ev > 0])
    return ev2, pr


def latent_spectrum(W):
    """Rotation-invariant: eigenvalues of W^T W = variance carried by each latent
    direction in data space. Independent of FA rotation gauge."""
    ev = np.linalg.eigvalsh(W.T @ W)[::-1]
    return ev


def principal_angles(A, B):
    """Principal angles between column spaces of A and B (d x q). Returns cos(angles)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, -1, 1)


def axis_stability(panel, include, k, q=2, n_sub=20, frac=0.8, seed=C.SEED):
    """Refit FA on fetus subsamples. Compare to the full-data fit two ways:
    (1) per-axis cosine after greedy sign/permutation match (secondary, rotation-fragile);
    (2) subspace overlap = mean cos(principal angle) between top-q directions of W W^T
    (rotation-invariant, the rung-1 vs rung-2 discriminator)."""
    X, M, _ = P.flatten(panel, include=include)
    full = L.FactorAnalysisMissing(k=k, seed=seed).fit(X, M)
    Wf = full.W_
    # top-q principal directions of the full common covariance
    Uf = np.linalg.svd(Wf, full_matrices=False)[0][:, :q]
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    sub_overlap = []
    axis_cos = []
    for s in range(n_sub):
        idx = rng.choice(n, int(frac * n), replace=False)
        try:
            m = L.FactorAnalysisMissing(k=k, seed=seed + 1 + s).fit(X[idx], M[idx])
        except np.linalg.LinAlgError:
            continue
        Ws = m.W_
        Us = np.linalg.svd(Ws, full_matrices=False)[0][:, :q]
        cosang = principal_angles(Uf, Us)
        sub_overlap.append(cosang)
        # per-axis greedy match on full W columns (sign-invariant cosine)
        nf = Wf / (np.linalg.norm(Wf, axis=0) + 1e-12)
        ns = Ws / (np.linalg.norm(Ws, axis=0) + 1e-12)
        Ccos = np.abs(nf.T @ ns)
        taken = set()
        row = []
        for jj in range(min(q, k)):
            order = np.argsort(-Ccos[jj])
            pick = next((o for o in order if o not in taken), order[0])
            taken.add(pick)
            row.append(Ccos[jj, pick])
        axis_cos.append(row)
    return (np.array(sub_overlap), np.array(axis_cos), Wf)


def run(include=("biom", "doppler"), k=6, n_boot=200, n_perm=200,
        n_sub=20, drop_efw=False, tag=None):
    panel = P.load_panel()
    X, M, names = P.flatten(panel, include=include)
    if drop_efw:
        keep = [i for i, nm in enumerate(names) if "efw" not in nm]
        X, M = X[:, keep], M[:, keep]
        names = [names[i] for i in keep]

    ev_raw, Cm = raw_spectrum(X, M)
    neg = ev_raw[ev_raw < 0]
    pr_raw = participation_ratio(ev_raw[ev_raw > 0])

    fit = L.fit_latent(panel, k=k, include=include)
    ev_lat = latent_spectrum(fit["W"] if not drop_efw else
                             L.FactorAnalysisMissing(k=k).fit(X, M).W_)
    pr_lat = participation_ratio(ev_lat)

    null_ev, null_pr = permutation_null(X, M, n_perm=n_perm)
    null_ev2 = null_ev[:, 1]
    bev2, bpr = boot_raw(X, M, n_boot=n_boot)

    sub_overlap, axis_cos, _ = axis_stability(panel, include, k, q=2, n_sub=n_sub)

    def ci(a):
        a = a[np.isfinite(a)]
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    out = {
        "include": list(include), "k": k, "drop_efw": drop_efw,
        "n_features": X.shape[1], "n_fetus": X.shape[0],
        "obs_fraction": float(M.mean()),
        "raw_eigs_top6": ev_raw[:6].tolist(),
        "raw_neg_eig_min": float(neg.min()) if neg.size else 0.0,
        "raw_neg_eig_count": int(neg.size),
        "raw_pr": pr_raw,
        "raw_eig2": float(ev_raw[1]),
        "raw_eig2_boot_ci": ci(bev2),
        "raw_pr_boot_ci": ci(bpr),
        "latent_WtW_eigs": ev_lat.tolist(),
        "latent_pr": pr_lat,
        "perm_null_eig2_mean": float(null_ev2.mean()),
        "perm_null_eig2_p95": float(np.percentile(null_ev2, 95)),
        "perm_null_eig2_max": float(null_ev2.max()),
        "eig2_above_null": bool(ev_raw[1] > np.percentile(null_ev2, 95)),
        "perm_null_pr_mean": float(null_pr.mean()),
        "obs_pr_vs_null_pr": [pr_raw, float(null_pr.mean())],
        "subspace_overlap_top2_mean_cosangle": float(np.nanmean(sub_overlap))
        if sub_overlap.size else None,
        "subspace_overlap_min_cosangle_mean": float(np.nanmean(sub_overlap[:, -1]))
        if sub_overlap.size else None,
        "per_axis_cos_mean": np.nanmean(axis_cos, axis=0).tolist()
        if axis_cos.size else None,
        "per_axis_cos_std": np.nanstd(axis_cos, axis=0).tolist()
        if axis_cos.size else None,
    }

    tag = tag or "_".join(include) + ("_noefw" if drop_efw else "")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"dimensionality_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(1, min(8, ev_raw.size) + 1)
    ax[0].plot(x, ev_raw[:len(x)], "o-", label="observed")
    ax[0].fill_between(x, np.percentile(null_ev[:, :len(x)], 2.5, axis=0),
                       np.percentile(null_ev[:, :len(x)], 97.5, axis=0),
                       alpha=0.3, label="perm null 95%")
    ax[0].axhline(1.0, ls=":", c="k", lw=0.7)
    ax[0].set_xlabel("component"); ax[0].set_ylabel("eigenvalue")
    ax[0].set_title(f"raw spectrum  PR={pr_raw:.2f}"); ax[0].legend(fontsize=8)
    el = np.arange(1, ev_lat.size + 1)
    ax[1].bar(el, ev_lat)
    ax[1].set_xlabel("latent direction"); ax[1].set_ylabel("eig(W'W)")
    ax[1].set_title(f"latent spectrum  PR={pr_lat:.2f}")
    fig.suptitle(f"FGR latent dimensionality [{tag}]")
    fig.tight_layout()
    fig.savefig(RESULTS / f"dimensionality_{tag}.png", dpi=110)
    plt.close(fig)
    return out


def main():
    summ = {}
    for inc, de in [(("biom", "doppler"), False),
                    (("biom", "doppler"), True),
                    (("biom", "doppler", "cardiac"), False)]:
        o = run(include=inc, drop_efw=de)
        tag = "_".join(inc) + ("_noefw" if de else "")
        summ[tag] = {
            "raw_pr": o["raw_pr"], "latent_pr": o["latent_pr"],
            "raw_eig2": o["raw_eig2"], "eig2_above_null": o["eig2_above_null"],
            "perm_null_eig2_p95": o["perm_null_eig2_p95"],
            "subspace_overlap_top2": o["subspace_overlap_top2_mean_cosangle"],
            "per_axis_cos_mean": o["per_axis_cos_mean"],
        }
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
