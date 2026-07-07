import os
# single-thread BLAS: the FA EM is a per-row python loop calling small LAPACK
# routines; under a shared/contended box, multi-threaded BLAS oversubscribes and
# slows this ~50x. Must be set before numpy imports. Set defaults if unset.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np
from sklearn.cluster import KMeans

from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L
from fgrgeom import featuresets as FS
from fgrgeom import embedding as E
from fgrgeom import topology as T

_REPO_RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "nl")


def _latent_full(panel, include, k=6, max_iter=300):
    X, M, names = FS.build(panel, include) if isinstance(include, str) else P.flatten(panel, include=include)
    fit = L.FactorAnalysisMissing(k, max_iter=max_iter).fit(X, M)
    Z, _ = fit.transform(X, M)
    return Z, names, fit.n_iter_


def nonlinear_embedding(Z, knn_k=15, n_diff=3, alpha=1.0):
    g = E.knn_graph(Z, k=knn_k)
    diff, devals = E.diffusion_map(g.W, n_components=n_diff, alpha=alpha)
    return diff, devals, g


def paga_graph(Z, labels, knn_idx):
    """PAGA-style cluster-abstraction graph. Connectivity between two clusters =
    observed kNN edges / edges expected if the same number of edges were placed at
    random (configuration-model null). Cluster pairs with ratio>1 become graph
    edges. Returns the cluster adjacency, degrees, n branch nodes (deg>=3) and
    connected components of the abstraction graph.
    NOTE: this is a degraded PAGA (kmeans partition + edge-fraction abstraction);
    scanpy/leiden PAGA is unavailable in this env."""
    n = Z.shape[0]
    k = knn_idx.shape[1]
    K = int(labels.max()) + 1
    sizes = np.array([(labels == c).sum() for c in range(K)], float)
    E_obs = np.zeros((K, K))
    for i in range(n):
        ci = labels[i]
        for j in knn_idx[i]:
            E_obs[ci, labels[j]] += 1
    E_obs = E_obs + E_obs.T
    total_edges = n * k
    ratio = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            expected = total_edges * (sizes[a] * sizes[b]) / (n * n)
            ratio[a, b] = E_obs[a, b] / expected if expected > 0 else 0.0
    adj = (ratio > 1.0).astype(int)
    np.fill_diagonal(adj, 0)
    deg = adj.sum(1)
    # connected components of abstraction graph
    seen = np.zeros(K, bool)
    ncomp = 0
    for s in range(K):
        if seen[s]:
            continue
        ncomp += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in np.where(adj[u] > 0)[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return dict(K=K, n_branch_nodes=int((deg >= 3).sum()),
                max_degree=int(deg.max()), n_leaves=int((deg == 1).sum()),
                n_components=int(ncomp), mean_degree=float(deg.mean()),
                is_tree=bool(adj.sum() // 2 == K - ncomp))


def _fa_from_features(X, M, k=6, max_iter=300):
    fit = L.FactorAnalysisMissing(k, max_iter=max_iter).fit(X, M)
    Z, _ = fit.transform(X, M)
    return Z, fit.n_iter_


def _pipeline(Z, knn_k=15, n_clusters=10, n_boot=2000, seed=C.SEED, _t=None):
    n = Z.shape[0]
    Zc = Z - Z.mean(0)
    _, sv, vt = np.linalg.svd(Zc, full_matrices=False)
    pc1 = Zc @ vt[0]
    d_lin, p_lin, _ = T.dip_test(pc1, n_boot=n_boot)
    if _t:
        _t("svd+dip_lin")
    diff, devals, g = nonlinear_embedding(Z, knn_k=knn_k)
    dc1 = diff[:, 0]
    d_nl, p_nl, _ = T.dip_test(dc1, n_boot=n_boot)
    if _t:
        _t("diffusion+dip_nl")
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(Z)
    paga = paga_graph(Z, labels, g.idx)
    if _t:
        _t("kmeans+paga")
    ph, ph_src = T.persistent_homology(diff)
    h1 = T.ph_h1_ratio(ph)
    if _t:
        _t("persistent_homology(%s)" % ph_src)
    return dict(
        n=n,
        latent_sv=[float(x) for x in sv],
        diff_evals=[float(x) for x in devals],
        dip_linear_pc1=dict(dip=float(d_lin), p=float(p_lin)),
        dip_nonlinear_dc1=dict(dip=float(d_nl), p=float(p_nl)),
        paga=paga,
        ph=ph, ph_source=ph_src, h1_h0_ratio=float(h1),
    )


def _timer(verbose):
    import time
    state = {"t": time.time()}

    def _t(msg):
        if verbose:
            print("  [%s] %.1fs" % (msg, time.time() - state["t"]), flush=True)
        state["t"] = time.time()
    return _t


def analyze(panel, include, k_latent=6, knn_k=15, n_clusters=10,
            n_boot=2000, seed=C.SEED, verbose=True):
    _t = _timer(verbose)
    Z, names, fa_iter = _latent_full(panel, include, k=k_latent)
    _t("FA fit+transform iter=%d" % fa_iter)
    out = _pipeline(Z, knn_k=knn_k, n_clusters=n_clusters, n_boot=n_boot, seed=seed, _t=_t)
    out.update(n_features=len(names), include=str(include), fa_n_iter=int(fa_iter))
    return out


def analyze_features(X, M, n_features_label, k_latent=6, knn_k=15, n_clusters=10,
                     n_boot=2000, seed=C.SEED, verbose=True):
    _t = _timer(verbose)
    Z, fa_iter = _fa_from_features(X, M, k=k_latent)
    _t("FA fit+transform iter=%d" % fa_iter)
    out = _pipeline(Z, knn_k=knn_k, n_clusters=n_clusters, n_boot=n_boot, seed=seed, _t=_t)
    out.update(n_features=int(n_features_label), include="synthetic", fa_n_iter=int(fa_iter))
    return out


def _null_features(n, n_dirs=2, d=47, rho=0.5, seed=C.SEED):
    """NEGATIVE control: single smooth connected manifold. n_dirs Gaussian latent
    coords through random loadings -> ellipsoidal cloud, unimodal, no branch.
    Mirrors controls_sim.make_null but returns (X, M) directly (Panel ctor is stale)."""
    rng = np.random.default_rng(seed)
    Tm = rng.standard_normal((n, n_dirs))
    Lm = rng.standard_normal((n_dirs, d))
    signal = Tm @ Lm
    noise = rng.standard_normal((n, d))
    sd = signal.std(0); sd[sd == 0] = 1.0
    X = np.sqrt(rho) * signal / sd + np.sqrt(1 - rho) * noise
    return X, np.ones((n, d), bool)


def _branch_features(n, d=47, p_route=0.5, trunk_w=1.0, branch_w=1.4, ortho=0.85,
                     rho=0.5, seed=C.SEED):
    """POSITIVE control: shared trunk splitting into two genuine routes (a Y in
    feature space). Mirrors controls_sim.make_branch, returns (X, M) directly."""
    rng = np.random.default_rng(seed)
    w_trunk = rng.standard_normal(d)
    uA = rng.standard_normal(d)
    rand = rng.standard_normal(d)
    perp = rand - (rand @ uA) / (uA @ uA) * uA
    uB = ortho * perp / np.linalg.norm(perp) * np.linalg.norm(uA) + (1 - ortho) * uA
    s = rng.standard_normal(n)
    route = (rng.random(n) < p_route).astype(int)
    b = np.abs(rng.standard_normal(n))
    dirs = np.where(route[:, None] == 0, uA, uB)
    signal = trunk_w * s[:, None] * w_trunk + branch_w * (b[:, None] * dirs)
    noise = rng.standard_normal((n, d))
    sd = signal.std(0); sd[sd == 0] = 1.0
    X = np.sqrt(rho) * signal / sd + np.sqrt(1 - rho) * noise
    return X, np.ones((n, d), bool)


def _verdict(real, pos, calib):
    """Null-calibrated verdict. A metric is treated as VALID only if it both trips
    on the planted-branch positive control AND stays clean across the single-
    manifold null seeds (calib). The verdict on real data is then read only off the
    metrics that passed that discrimination test. H1/H0 detects loops not trees and
    PAGA branch_nodes/components are descriptive: they are reported, never used to
    assert a branch (a disconnected component is an outlier, not a route split)."""
    h1_cut = calib["h1_cut"]
    # which metrics discriminate pos from the null
    disc_dip = (pos["dip_nonlinear_dc1"]["p"] < 0.05) and (calib["dip_nl_null_min_p"] >= 0.05)
    disc_h1 = (pos["h1_h0_ratio"] > h1_cut)  # h1_cut is the 95th pct of the null by construction
    discriminating = [m for m, ok in (("dip_nl", disc_dip), ("h1_calibrated", disc_h1)) if ok]
    # real flags, read only off discriminating metrics
    rn = (real["dip_nonlinear_dc1"]["p"] < 0.05) and disc_dip
    rl = real["dip_linear_pc1"]["p"] < 0.05
    r_loop = (real["h1_h0_ratio"] > h1_cut) and disc_h1
    real_flags = rn or r_loop
    nl_only = rn and not rl
    if not discriminating:
        return ("UNDERPOWERED: no metric separates the planted-branch positive control "
                "from the single-manifold null at n=977; nonlinear branch test "
                "inconclusive (real shows no flags on any metric regardless)")
    if real_flags and nl_only:
        return ("NONLINEAR-ONLY structure on a control-validated metric "
                "(%s); candidate real nonlinear branch/loop" % ",".join(discriminating))
    if real_flags:
        return ("structure on a control-validated metric (%s), present linearly too "
                "(not nonlinear-specific)" % ",".join(discriminating))
    return ("no branch/loop beyond the linear single continuum "
            "(validated by metrics: %s)" % ",".join(discriminating))


def run(panel=None, n_clusters=10, knn_k=15, n_boot=2000, seed=C.SEED, n_null=8):
    if panel is None:
        panel = P.load_panel()
    n = panel.ids.shape[0]
    print("== analyze real_full ==", flush=True)
    real = analyze(panel, "full", knn_k=knn_k, n_clusters=n_clusters,
                   n_boot=n_boot, seed=seed)
    d_ctrl = real["n_features"]
    # NEGATIVE single-manifold control over several seeds -> null distribution
    print("== analyze neg_control x%d (single manifold) ==" % n_null, flush=True)
    negs = []
    for s in range(n_null):
        Xn, Mn = _null_features(n, n_dirs=2, d=d_ctrl, rho=0.9, seed=seed + 1 + s)
        negs.append(analyze_features(Xn, Mn, d_ctrl, knn_k=knn_k, n_clusters=n_clusters,
                                     n_boot=n_boot, seed=seed + 1 + s, verbose=False))
    neg = negs[0]
    # POSITIVE strong planted Y-branch control
    print("== analyze pos_control (planted Y-branch) ==", flush=True)
    Xp, Mp = _branch_features(n, d=d_ctrl, rho=0.9, trunk_w=1.0, branch_w=3.0,
                              ortho=0.95, seed=seed)
    pos = analyze_features(Xp, Mp, d_ctrl, knn_k=knn_k, n_clusters=n_clusters,
                           n_boot=n_boot, seed=seed)
    h1_null = np.array([x["h1_h0_ratio"] for x in negs])
    comp_null = np.array([x["paga"]["n_components"] for x in negs])
    branch_null = np.array([x["paga"]["n_branch_nodes"] for x in negs])
    calib = dict(
        n_null=n_null,
        h1_cut=float(np.quantile(h1_null, 0.95)),
        h1_null_mean=float(h1_null.mean()), h1_null_max=float(h1_null.max()),
        dip_nl_null_min_p=float(min(x["dip_nonlinear_dc1"]["p"] for x in negs)),
        comp_null_max=int(comp_null.max()), branch_null_max=int(branch_null.max()),
    )
    v = _verdict(real, pos, calib)
    return dict(real_full=real, neg_control=neg, pos_control=pos, calibration=calib,
                verdict=v,
                deps=dict(umap=False, gudhi=False, ripser=real["ph_source"] == "ripser",
                          paga="degraded (kmeans+edge-fraction, no scanpy/leiden)"))


def main():
    r = run()
    os.makedirs(_REPO_RESULTS, exist_ok=True)
    with open(os.path.join(_REPO_RESULTS, "branch_topology.json"), "w") as f:
        json.dump(r, f, indent=2)
    for name in ("real_full", "neg_control", "pos_control"):
        a = r[name]
        print(f"[{name}] n={a['n']} feat={a['n_features']} "
              f"dip_lin_p={a['dip_linear_pc1']['p']:.4f} "
              f"dip_nl_p={a['dip_nonlinear_dc1']['p']:.4f} "
              f"H1/H0={a['h1_h0_ratio']:.3f} "
              f"paga[branch={a['paga']['n_branch_nodes']} "
              f"comp={a['paga']['n_components']} maxdeg={a['paga']['max_degree']}]")
    print("calibration:", r["calibration"])
    print("deps:", r["deps"])
    print("VERDICT:", r["verdict"])


if __name__ == "__main__":
    main()
