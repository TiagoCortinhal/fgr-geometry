"""TASK 1: a CONTROLLED full-set branch/topology test.

The full feature set (58 cols, 4-block + ratios + bp, 87% observed, 5 complete-case
rows) was the one set whose branch battery was UNCONTROLLED: the planted-branch
positive control was never scored with a load-bearing branch discriminator, so a
null verdict there could not be trusted (underpowered). This module fixes that.

What "controlled" means here, concretely:
  POWER GATE (positive control). Plant a Y-branch directly in the 58-d feature
  space, paste the REAL full-set missingness mask on top (no imputation; the FA EM
  marginalises the missing dims), fit the same FA latent, and check that the planted
  route label is RECOVERABLE from the latent/diffusion coords by cross-validated AUC
  (route_auc). route_auc is the only branch metric validated against ground truth;
  if it does not clear the random-label baseline the battery is underpowered and we
  escalate power (branch separation / orthogonality / latent k / kNN) until it does,
  and report what it took. cluster/dip/PH/PAGA are unsupervised descriptors only.

  NEGATIVE control (single smooth manifold, several seeds) -> the null distribution
  of every unsupervised descriptor; thresholds are the null p95.

  REAL data has no route labels, so route_auc is undefined there; the real verdict
  is read ONLY off the unsupervised descriptors that (a) trip on the planted branch
  AND (b) stay clean across the single-manifold null (the discriminating set). A
  descriptor that cannot separate the planted branch from the null carries no weight.

NO imputation anywhere: the real mask is reused for the controls, the FA is the
observed-only missing-data EM, the diffusion/PH/PAGA run on the resulting latent.

deps: ripser (PH, present), diptest (dip null, present), torch/umap unused here,
hdbscan ABSENT (not needed; kmeans partitions + configuration-model PAGA only).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
import numpy as np

from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L
from fgrgeom import featuresets as FS
from fgrgeom import embedding as E
from fgrgeom import topology as T

_REPO_RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


# --- synthetic full-dim panels carrying the REAL missingness ----------------
def _colstd(S):
    m = S.mean(0)
    s = S.std(0)
    s[s == 0] = 1.0
    return (S - m) / s


def _branch_X(M, route, branch_w, ortho, trunk_w=1.0, rho=0.6, seed=C.SEED):
    """POSITIVE control geometry in the full d-dim feature space. Shared trunk s,
    each row pushed along route A or B (near-orthogonal dirs). Real mask M pasted
    on top -> identical missingness to real data, no imputation. route is given so
    the same labels can score recovery."""
    rng = np.random.default_rng(seed)
    n, d = M.shape
    w_trunk = rng.standard_normal(d)
    uA = rng.standard_normal(d)
    rand = rng.standard_normal(d)
    perp = rand - (rand @ uA) / (uA @ uA) * uA
    uB = ortho * perp / np.linalg.norm(perp) * np.linalg.norm(uA) + (1 - ortho) * uA
    s = rng.standard_normal(n)
    b = np.abs(rng.standard_normal(n))
    dirs = np.where(route[:, None] == 0, uA, uB)
    signal = trunk_w * s[:, None] * w_trunk + branch_w * (b[:, None] * dirs)
    Zsig = _colstd(signal)
    noise = rng.standard_normal((n, d))
    X = np.sqrt(rho) * Zsig + np.sqrt(1 - rho) * noise
    X = X.copy()
    X[~M] = np.nan
    return X


def _null_X(M, n_dirs=2, rho=0.6, seed=C.SEED):
    """NEGATIVE control: a single smooth connected ellipsoidal manifold (no branch),
    real mask pasted on. Matched to the branch control on dim/scale/missingness."""
    rng = np.random.default_rng(seed)
    n, d = M.shape
    Tm = rng.standard_normal((n, n_dirs))
    Lm = rng.standard_normal((n_dirs, d))
    signal = _colstd(Tm @ Lm)
    noise = rng.standard_normal((n, d))
    X = np.sqrt(rho) * signal + np.sqrt(1 - rho) * noise
    X = X.copy()
    X[~M] = np.nan
    return X


# --- latent + branch battery -------------------------------------------------
def _fa_latent(X, M, k, max_iter):
    fa = L.FactorAnalysisMissing(k=k, seed=C.SEED, max_iter=max_iter).fit(X, M)
    Z, _ = fa.transform(X, M)
    return Z, fa.n_iter_


def _cv_auc(Xc, y, seed=C.SEED):
    """5-fold stratified CV AUC of label y from coords Xc (logistic)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return float("nan")
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(Xc, y):
        clf = LogisticRegression(max_iter=500).fit(Xc[tr], y[tr])
        p[te] = clf.predict_proba(Xc[te])[:, 1]
    return float(roc_auc_score(y, p))


def _paga_sweep(Z, knn_idx, resolutions, seed=C.SEED):
    """PAGA-style abstraction graph over MANY resolutions (n_clusters) and MANY
    kmeans startpoints (n_init random restarts inside KMeans). Reports, across the
    sweep, the max branch-node count and the min component count (a branch shows as
    a deg>=3 abstraction node in a single connected component)."""
    from sklearn.cluster import KMeans
    n = Z.shape[0]
    k = knn_idx.shape[1]
    rows = []
    for K in resolutions:
        labels = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Z)
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
            for bb in range(K):
                if a == bb:
                    continue
                expected = total_edges * (sizes[a] * sizes[bb]) / (n * n)
                ratio[a, bb] = E_obs[a, bb] / expected if expected > 0 else 0.0
        adj = (ratio > 1.0).astype(int)
        np.fill_diagonal(adj, 0)
        deg = adj.sum(1)
        seen = np.zeros(K, bool)
        ncomp = 0
        for ss in range(K):
            if seen[ss]:
                continue
            ncomp += 1
            stack = [ss]
            seen[ss] = True
            while stack:
                u = stack.pop()
                for v in np.where(adj[u] > 0)[0]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
        rows.append(dict(K=int(K), branch_nodes=int((deg >= 3).sum()),
                         max_degree=int(deg.max()), n_components=int(ncomp)))
    return dict(per_resolution=rows,
                max_branch_nodes=int(max(r["branch_nodes"] for r in rows)),
                min_components=int(min(r["n_components"] for r in rows)),
                max_degree=int(max(r["max_degree"] for r in rows)))


def branch_stats(Z, route=None, knn_k=15, resolutions=(6, 8, 10, 12, 15),
                 n_boot=1000, seed=C.SEED):
    """Full branch battery on a latent cloud Z.
    Supervised (positive control only): route_auc_cv from diffusion coords, plus a
    random-label baseline (null AUC floor). Unsupervised (applied everywhere):
    kmeans(2) silhouette on the diffusion plane, dip on pc1 and on diffusion-1, the
    ripser H1/H0 lifetime ratio, and a multi-resolution PAGA sweep."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    Zc = Z - Z.mean(0)
    vt = np.linalg.svd(Zc, full_matrices=False)[2]
    pc1 = Zc @ vt[0]
    g = E.knn_graph(Z, k=knn_k)
    diff, devals = E.diffusion_map(g.W, n_components=5)
    devals = np.asarray(devals, float)
    eigengap = float(devals[0] / devals[1]) if devals[1] > 0 else float("inf")
    d_lin, p_lin, _ = T.dip_test(pc1, n_boot=n_boot)
    d_d1, p_d1, _ = T.dip_test(diff[:, 0], n_boot=n_boot)
    km = KMeans(2, n_init=10, random_state=seed).fit(diff[:, :2])
    sil = float(silhouette_score(diff[:, :2], km.labels_))
    bal = float(min(np.bincount(km.labels_)) / n)
    ph, ph_src = T.persistent_homology(diff)
    h1 = float(T.ph_h1_ratio(ph))
    paga = _paga_sweep(Z, g.idx, resolutions, seed=seed)
    out = dict(
        n=int(n), diff_evals=devals.tolist(), diff_eigengap=eigengap,
        dip_pc1=float(d_lin), dip_p_pc1=float(p_lin),
        dip_diff1=float(d_d1), dip_p_diff1=float(p_d1),
        kmeans2_silhouette=sil, kmeans2_balance=bal,
        h1_h0_ratio=h1, ph_source=ph_src, paga=paga,
        randlabel_auc_cv=_cv_auc(diff, (rng.random(n) < 0.5).astype(int), seed=seed),
    )
    if route is not None:
        out["route_auc_cv"] = _cv_auc(diff, route, seed=seed)
    return out


# --- positive-control power escalation ---------------------------------------
def _escalate_positive(M, k, knn_k, resolutions, n_boot, seed):
    """Plant a Y-branch and confirm the battery RECOVERS it (route_auc clears the
    random-label baseline by a margin). If not, escalate power along a fixed ladder
    (cleaner branch -> more orthogonal -> larger latent k / kNN) and report the first
    rung that works and what it took. The point: a powered positive, not a lucky one."""
    route = (np.random.default_rng(seed).random(M.shape[0]) < 0.5).astype(int)
    ladder = [
        dict(branch_w=2.0, ortho=0.90, rho=0.6, k=k, knn_k=knn_k),
        dict(branch_w=3.0, ortho=0.95, rho=0.6, k=k, knn_k=knn_k),
        dict(branch_w=4.0, ortho=0.97, rho=0.7, k=k, knn_k=knn_k),
        dict(branch_w=5.0, ortho=0.99, rho=0.8, k=max(k, 8), knn_k=knn_k),
        dict(branch_w=6.0, ortho=0.99, rho=0.85, k=max(k, 10), knn_k=max(knn_k, 20)),
    ]
    trace = []
    for rung, cfg in enumerate(ladder):
        Xp = _branch_X(M, route, branch_w=cfg["branch_w"], ortho=cfg["ortho"],
                       rho=cfg["rho"], seed=seed)
        Zp, it = _fa_latent(Xp, M, k=cfg["k"], max_iter=200)
        st = branch_stats(Zp, route=route, knn_k=cfg["knn_k"],
                          resolutions=resolutions, n_boot=n_boot, seed=seed)
        rec = (np.isfinite(st["route_auc_cv"]) and st["route_auc_cv"] > 0.75
               and st["route_auc_cv"] - st["randlabel_auc_cv"] > 0.15)
        trace.append(dict(rung=rung, **cfg, fa_iter=int(it),
                          route_auc=st["route_auc_cv"],
                          randlabel_auc=st["randlabel_auc_cv"],
                          silhouette=st["kmeans2_silhouette"], h1=st["h1_h0_ratio"],
                          recovered=bool(rec)))
        if rec:
            return st, route, dict(recovered=True, rung=rung, config=cfg,
                                   ladder_trace=trace)
    # no rung recovered -> report the strongest attempt
    return st, route, dict(recovered=False, rung=None,
                           config=ladder[-1], ladder_trace=trace)


def run(panel=None, k=6, knn_k=15, resolutions=(6, 8, 10, 12, 15),
        n_null=10, n_boot=1000, seed=C.SEED, verbose=True):
    t0 = time.time()
    if panel is None:
        panel = P.load_panel()
    X, M, names = FS.build(panel, "full")
    n, d = X.shape

    if verbose:
        print(f"== full set: n={n} d={d} obs={M.mean():.3f} "
              f"complete={int(M.all(1).sum())} ==", flush=True)

    # POSITIVE CONTROL FIRST (power gate, with escalation)
    if verbose:
        print("== positive control: planted Y-branch + power escalation ==", flush=True)
    pos, route, esc = _escalate_positive(M, k, knn_k, resolutions, n_boot, seed)
    if verbose:
        for r in esc["ladder_trace"]:
            print(f"  rung{r['rung']} bw={r['branch_w']} ortho={r['ortho']} "
                  f"k={r['k']} knn={r['knn_k']} -> route_auc={r['route_auc']:.3f} "
                  f"rand={r['randlabel_auc']:.3f} sil={r['silhouette']:.3f} "
                  f"recovered={r['recovered']}", flush=True)

    # NEGATIVE CONTROL: single-manifold null over seeds -> threshold distribution
    if verbose:
        print(f"== negative control: single-manifold null x{n_null} ==", flush=True)
    negs = []
    for s in range(n_null):
        Xn = _null_X(M, n_dirs=2, rho=0.6, seed=seed + 1000 + s)
        Zn, _ = _fa_latent(Xn, M, k=k, max_iter=200)
        negs.append(branch_stats(Zn, route=None, knn_k=knn_k,
                                 resolutions=resolutions, n_boot=n_boot,
                                 seed=seed + 1000 + s))

    def col(rows, key):
        return np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
    sil_null = col(negs, "kmeans2_silhouette")
    h1_null = col(negs, "h1_h0_ratio")
    rand_null = col(negs, "randlabel_auc_cv")
    eg_null = col(negs, "diff_eigengap")
    branch_null = np.array([r["paga"]["max_branch_nodes"] for r in negs])
    comp_null = np.array([r["paga"]["min_components"] for r in negs])
    thresholds = dict(
        n_null=n_null,
        silhouette_p95=float(np.percentile(sil_null, 95)),
        silhouette_max=float(sil_null.max()),
        h1_p95=float(np.percentile(h1_null, 95)),
        h1_max=float(h1_null.max()),
        randlabel_auc_p95=float(np.percentile(rand_null, 95)),
        eigengap_p95=float(np.percentile(eg_null, 95)),
        dip_fpr_pc1=float(np.mean(col(negs, "dip_p_pc1") < 0.05)),
        dip_fpr_diff1=float(np.mean(col(negs, "dip_p_diff1") < 0.05)),
        paga_branch_nodes_max=int(branch_null.max()),
        paga_min_components_min=int(comp_null.min()),
    )

    # REAL data: same FA latent, same battery, NO route labels
    if verbose:
        print("== real full-set latent + battery ==", flush=True)
    Zr, it_r = _fa_latent(X, M, k=k, max_iter=300)
    real = branch_stats(Zr, route=None, knn_k=knn_k, resolutions=resolutions,
                        n_boot=n_boot, seed=seed)
    real["fa_n_iter"] = int(it_r)

    verdict = _verdict(pos, esc, real, thresholds)
    out = dict(
        n=int(n), n_features=int(d), include="full",
        observed_fraction=float(M.mean()), n_complete_case=int(M.all(1).sum()),
        k_latent=int(k), knn_k=int(knn_k), resolutions=list(resolutions),
        positive_control=dict(recovered=esc["recovered"], rung=esc["rung"],
                              config=esc["config"], route_auc_cv=pos.get("route_auc_cv"),
                              randlabel_auc_cv=pos["randlabel_auc_cv"],
                              silhouette=pos["kmeans2_silhouette"],
                              h1_h0_ratio=pos["h1_h0_ratio"],
                              paga_max_branch_nodes=pos["paga"]["max_branch_nodes"],
                              ladder_trace=esc["ladder_trace"]),
        thresholds=thresholds,
        real=real,
        verdict=verdict,
        deps=dict(ripser=real["ph_source"] == "ripser", diptest=_have("diptest"),
                  hdbscan=_have("hdbscan"), torch=_have("torch"),
                  paga="degraded (kmeans partition + configuration-model abstraction)"),
        runtime_s=round(time.time() - t0, 1),
    )
    return out


def _verdict(pos, esc, real, thr):
    """Read the real verdict only off descriptors that BOTH trip on the planted
    branch AND stay clean on the single-manifold null (the discriminating set)."""
    if not esc["recovered"]:
        return ("UNDERPOWERED: the planted Y-branch is not recoverable from the "
                "full-set latent even after power escalation "
                f"(best route_auc={pos.get('route_auc_cv'):.3f} vs "
                f"randlabel={pos['randlabel_auc_cv']:.3f}); no full-set branch "
                "threshold is meaningful, the negative cannot be trusted.")
    # discriminating unsupervised descriptors: trip on positive AND clean on null
    disc = []
    if (pos["kmeans2_silhouette"] > thr["silhouette_p95"]):
        disc.append("silhouette")
    if (pos["h1_h0_ratio"] > thr["h1_p95"]) and thr["h1_max"] < pos["h1_h0_ratio"]:
        disc.append("h1")
    if thr["dip_fpr_diff1"] <= 0.05:
        disc.append("dip_diff1")
    # real flags on the discriminating descriptors
    flags = []
    if "silhouette" in disc and real["kmeans2_silhouette"] > thr["silhouette_p95"]:
        flags.append("silhouette")
    if "h1" in disc and real["h1_h0_ratio"] > thr["h1_p95"]:
        flags.append("h1")
    if "dip_diff1" in disc and real["dip_p_diff1"] < 0.05:
        flags.append("dip_diff1")
    head = (f"POWERED: planted branch recovered at route_auc="
            f"{pos['route_auc_cv']:.2f} (rung {esc['rung']}, randlabel "
            f"{pos['randlabel_auc_cv']:.2f}); discriminating descriptors=[%s]"
            % ",".join(disc) if disc else
            f"POWERED on route_auc={pos['route_auc_cv']:.2f} but NO unsupervised "
            "descriptor separates the planted branch from the null (route_auc is the "
            "only branch detector with power here)")
    if not disc:
        return head + "; real full-set verdict: no usable unsupervised branch test."
    if flags:
        return (head + f"; REAL trips [{','.join(flags)}] above the null p95 -> "
                "candidate full-set branch, inspect.")
    return (head + "; REAL trips none of the discriminating descriptors above the "
            "null p95 -> CONTROLLED NEGATIVE: no branch beyond the single continuum "
            "on the full set.")


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def main():
    out = run()
    os.makedirs(_REPO_RESULTS, exist_ok=True)
    with open(os.path.join(_REPO_RESULTS, "branch_fullset.json"), "w") as f:
        json.dump(out, f, indent=2)
    pc = out["positive_control"]
    print(f"\npositive_control recovered={pc['recovered']} rung={pc['rung']} "
          f"route_auc={pc['route_auc_cv']:.3f} rand={pc['randlabel_auc_cv']:.3f}")
    print("thresholds:", {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in out["thresholds"].items()})
    r = out["real"]
    print(f"real: sil={r['kmeans2_silhouette']:.3f} h1={r['h1_h0_ratio']:.3f} "
          f"dip_p_pc1={r['dip_p_pc1']:.3f} dip_p_diff1={r['dip_p_diff1']:.3f} "
          f"eigengap={r['diff_eigengap']:.2f} "
          f"paga_branch={r['paga']['max_branch_nodes']} "
          f"paga_mincomp={r['paga']['min_components']}")
    print("deps:", out["deps"])
    print("runtime_s:", out["runtime_s"])
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
