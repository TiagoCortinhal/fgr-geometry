import json
import numpy as np

from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L

RESULTS = C.DATA.parent.parent / "fgr-geometry" / "results" / "nl"


def _maskaware_sqdist(Xs, M, min_overlap=5):
    """Pairwise-complete squared Euclidean distance, scaled D/|S_ij| to a common
    dimensionality. Xs already standardized with missing entries set to 0; M is the
    boolean observed mask. Pairs with co-observed dims < min_overlap -> inf."""
    Pm = M.astype(float)
    Xz = Xs * Pm
    overlap = Pm @ Pm.T
    sq2 = (Xz ** 2) @ Pm.T
    cross = Xz @ Xz.T
    sq = sq2 + sq2.T - 2.0 * cross
    sq = np.maximum(sq, 0.0)
    D = Xs.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        d2 = np.where(overlap > 0, D * sq / overlap, np.inf)
    d2[overlap < min_overlap] = np.inf
    np.fill_diagonal(d2, np.inf)
    return d2, overlap


def twonn(d2, discard_frac=0.1):
    """TwoNN intrinsic-dimension estimator (Facco et al. 2017) from a squared
    distance matrix with inf for invalid/self pairs. Returns (id, n_used, n_dropped).
    A point is dropped if it lacks two finite neighbor distances."""
    n = d2.shape[0]
    mus = []
    dropped = 0
    for i in range(n):
        row = d2[i]
        finite = row[np.isfinite(row)]
        if finite.size < 2:
            dropped += 1
            continue
        r = np.sqrt(np.sort(finite)[:2])
        r1, r2 = r[0], r[1]
        if r1 <= 0 or r2 <= 0:
            dropped += 1
            continue
        mus.append(r2 / r1)
    mu = np.sort(np.array(mus))
    m = mu.size
    if m < 10:
        return float("nan"), m, dropped
    k = int((1.0 - discard_frac) * m)
    mu = mu[:k]
    F = np.arange(1, mu.size + 1) / m
    x = np.log(mu)
    y = -np.log(1.0 - F)
    ok = np.isfinite(x) & np.isfinite(y)
    d_id = float(np.sum(x[ok] * y[ok]) / np.sum(x[ok] * x[ok]))
    return d_id, int(m), int(dropped)


def participation_ratio(eig):
    eig = np.asarray(eig, float)
    eig = eig[eig > 0]
    if eig.size == 0:
        return 0.0
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def fa_sim_panel_free(X, M, k, seed):
    """Sample a synthetic dataset from the FA generative model fitted (mask-aware)
    on X,M: x_std = W z + sqrt(psi) eps, z ~ N(0,I). Re-applies the real mask M.
    This is the NEGATIVE single-(linear-Gaussian)-manifold control matched to this
    exact feature set / missingness. Returns standardized Xs_sim, M."""
    fa = L.FactorAnalysisMissing(k=k, seed=seed).fit(X, M)
    n, d = X.shape
    rng = np.random.default_rng(seed + 7)
    z = rng.standard_normal((n, k))
    eps = rng.standard_normal((n, d)) * np.sqrt(fa.psi_)
    Xs_sim = z @ fa.W_.T + eps  # already in standardized space (mu ~ 0)
    Xs_sim = Xs_sim + fa.mu_
    Xs_sim[~M] = 0.0
    return Xs_sim, fa


def run_set(name, k=6, n_perm=100, n_sub=20, n_null=5):
    from fgrgeom import dimensionality as DM
    from fgrgeom import featuresets as FS

    panel = P.load_panel()
    X, M, names = FS.build(panel, name)
    Xs = DM.standardize_obs(X, M)

    # linear battery
    ev_raw, _ = DM.raw_spectrum(X, M)
    pr_raw = DM.participation_ratio(ev_raw[ev_raw > 0])
    neg = ev_raw[ev_raw < 0]

    fit = L.fit_latent(panel, k=k, include=FS.SETS[name])
    ev_lat = DM.latent_spectrum(fit["W"])
    pr_lat = DM.participation_ratio(ev_lat)

    null_ev, null_pr = DM.permutation_null(X, M, n_perm=n_perm)
    null_ev2 = null_ev[:, 1]

    sub_overlap, axis_cos, _ = DM.axis_stability(panel, FS.SETS[name], k,
                                                 q=2, n_sub=n_sub)

    # nonlinear intrinsic dim on the real data (mask-aware distances)
    d2, overlap = _maskaware_sqdist(Xs, M)
    id_real, n_used, n_drop = twonn(d2)
    min_ov = float(overlap[~np.eye(overlap.shape[0], dtype=bool)].min())
    med_ov = float(np.median(overlap[~np.eye(overlap.shape[0], dtype=bool)]))

    # FA-sim null intrinsic dim (linear Gaussian manifold, same mask)
    id_null = []
    for s in range(n_null):
        Xs_sim, _ = fa_sim_panel_free(X, M, k=k, seed=C.SEED + s)
        d2s, _ = _maskaware_sqdist(Xs_sim, M)
        idn, _, _ = twonn(d2s)
        id_null.append(idn)
    id_null = np.array(id_null, float)

    out = {
        "set": name,
        "include": list(FS.SETS[name]),
        "n_features": int(X.shape[1]),
        "n_fetus": int(X.shape[0]),
        "obs_fraction": float(M.mean()),
        "raw_eigs_top6": ev_raw[:6].tolist(),
        "raw_neg_eig_count": int(neg.size),
        "raw_neg_eig_min": float(neg.min()) if neg.size else 0.0,
        "raw_pr": pr_raw,
        "latent_pr": pr_lat,
        "raw_eig2": float(ev_raw[1]),
        "perm_null_eig2_p95": float(np.percentile(null_ev2, 95)),
        "perm_null_eig2_mean": float(null_ev2.mean()),
        "eig2_above_null": bool(ev_raw[1] > np.percentile(null_ev2, 95)),
        "perm_null_pr_mean": float(null_pr.mean()),
        "subspace_overlap_top2_mean_cosangle": float(np.nanmean(sub_overlap))
        if sub_overlap.size else None,
        "subspace_overlap_min_cosangle_mean": float(np.nanmean(sub_overlap[:, -1]))
        if sub_overlap.size else None,
        "per_axis_cos_mean": np.nanmean(axis_cos, axis=0).tolist()
        if axis_cos.size else None,
        "nonlinear_id_twonn": id_real,
        "nonlinear_id_n_points_used": n_used,
        "nonlinear_id_n_points_dropped": n_drop,
        "min_pair_overlap_dims": min_ov,
        "median_pair_overlap_dims": med_ov,
        "fa_sim_null_id_mean": float(np.nanmean(id_null)),
        "fa_sim_null_id_std": float(np.nanstd(id_null)),
        "fa_sim_null_id_all": id_null.tolist(),
        "id_below_linear_null": bool(id_real < np.nanmean(id_null) - 2 * np.nanstd(id_null)),
    }
    return out


def main():
    from fgrgeom import featuresets as FS
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = {}
    for name in FS.SETS:
        if name == "full":
            o = run_set(name, n_perm=100)
        else:
            o = run_set(name, n_perm=200)
        table[name] = o
        print(f"{name:>13} feat={o['n_features']:>2} "
              f"raw_pr={o['raw_pr']:.2f} lat_pr={o['latent_pr']:.2f} "
              f"eig2={o['raw_eig2']:.2f}>null95={o['perm_null_eig2_p95']:.2f}"
              f"({o['eig2_above_null']}) "
              f"sub2={o['subspace_overlap_top2_mean_cosangle']:.3f} "
              f"ID={o['nonlinear_id_twonn']:.2f} "
              f"null_ID={o['fa_sim_null_id_mean']:.2f}+-{o['fa_sim_null_id_std']:.2f} "
              f"drop={o['nonlinear_id_n_points_dropped']}")
    with open(RESULTS / "featureset_ablation.json", "w") as f:
        json.dump(table, f, indent=2)
    return table


if __name__ == "__main__":
    main()
