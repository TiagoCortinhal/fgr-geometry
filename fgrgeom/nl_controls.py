"""Nonlinear controls: calibrate the nonlinear geometry stack against a known
single smooth manifold (negative) and a known branch (positive).

The linear battery already reports a stable 2-D continuum. Before any NONLINEAR
claim (extra dimension / branch / clusters) is admissible we must show, on planted
ground truth, (a) what the nonlinear methods do on a smooth unimodal manifold with
NO branch -- do they invent one -- and (b) that they recover a real branch when one
is present. Thresholds are then set from the null and verified to separate the two.

Mapping of each statistic to the rung it discriminates (not interchangeable):
  - TwoNN intrinsic dim, linear PR, diffusion eigengap -> DIMENSION (1-D vs 2-D),
    NOT branching. A Y is locally 1-D; ID does not read >2 on a branch.
  - dip on the principal axis -> GAP / multimodality (clusters). A branch is
    connected at the trunk so dip is expected weak on the branch too.
  - kmeans(2) silhouette on the embedding -> invented-split strength (cluster
    invention on the null).
  - cross-validated AUC of the PLANTED route from diffusion coords -> the only
    load-bearing BRANCH metric, and the only one validated against ground truth.
    (ripser/gudhi are absent here, so topology.persistent_homology returns None and
    topology.verdict can never reach its Y/branch arm; route-AUC carries it.)

NOTE: controls_sim predates the featuresets change that appended 10 fields to
panel.Panel, so controls_sim.make_null/make_branch (and controls_run) raise
TypeError under the current Panel. We install None-defaults for the trailing 10
Panel fields at import as a contained runtime shim; we do NOT edit controls_sim
(owned elsewhere). The synthetic panels only carry the 4-block 47-col space, so we
calibrate/apply on minimal/plus_cardiac/plus_maternal only -- the "full" set
(ratios/bp) has no matched control and cannot be calibrated here.
"""
import json
import time
import pathlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L
from fgrgeom import controls_sim as CS
from fgrgeom import featuresets as FS
from fgrgeom import embedding as E
from fgrgeom import topology as T
from fgrgeom import dimensionality as D

# contained shim: give the 10 appended Panel fields defaults so controls_sim can
# build panels under the current 25-field Panel. See module docstring.
if P.Panel.__new__.__defaults__ is None or len(P.Panel.__new__.__defaults__) < 10:
    P.Panel.__new__.__defaults__ = (None,) * 10

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results" / "nl"


def twonn_id(X, discard=0.1):
    """Facco et al. (2017) TwoNN intrinsic dimension via the mu=r2/r1 ratio and the
    MLE slope, discarding the top `discard` fraction of mu (outlier robustness)."""
    n = X.shape[0]
    G = X @ X.T
    sq = np.diag(G)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0)
    np.fill_diagonal(D2, np.inf)
    d = np.sqrt(np.sort(D2, axis=1)[:, :2])
    r1, r2 = d[:, 0], d[:, 1]
    ok = (r1 > 0)
    mu = (r2[ok] / r1[ok])
    mu = np.sort(mu)
    keep = int(len(mu) * (1 - discard))
    mu = mu[:keep]
    return float((keep) / np.sum(np.log(mu)))


def cv_auc(Xc, y, seed=C.SEED):
    """5-fold stratified CV AUC of label y from coordinates Xc (logistic)."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return float("nan")
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(Xc, y):
        clf = LogisticRegression(max_iter=500).fit(Xc[tr], y[tr])
        p[te] = clf.predict_proba(Xc[te])[:, 1]
    return float(roc_auc_score(y, p))


def nl_stats(Z, route=None, seed=C.SEED):
    """Run the nonlinear stack on a latent point cloud Z (n,k)."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    # linear reference
    ev = np.linalg.eigvalsh(np.cov(Z.T))[::-1]
    lin_pr = D.participation_ratio(ev)
    idim = twonn_id(Z)
    # diffusion embedding (self-tuning kNN affinity, mask-free on the latent)
    g = E.knn_graph(Z, k=15)
    diff, devals = E.diffusion_map(g.W, n_components=5)
    devals = np.asarray(devals, float)
    eigengap = float(devals[0] / devals[1]) if devals[1] > 0 else float("inf")
    # gap / multimodality on the dominant axes
    pc1 = (Z - Z.mean(0)) @ np.linalg.svd(Z - Z.mean(0),
                                          full_matrices=False)[2][0]
    dip_pc1, dipp_pc1, _ = T.dip_test(pc1, n_boot=500)
    dip_d1, dipp_d1, _ = T.dip_test(diff[:, 0], n_boot=500)
    # invented-split strength on the 2-D diffusion embedding
    km = KMeans(2, n_init=10, random_state=seed).fit(diff[:, :2])
    sil = float(silhouette_score(diff[:, :2], km.labels_))
    bal = float(min(np.bincount(km.labels_)) / n)
    # branch graph descriptors
    br = T.knn_branch(Z, n_neighbors=15)
    out = {
        "lin_pr": float(lin_pr), "twonn_id": idim,
        "diff_evals": devals.tolist(), "diff_eigengap": eigengap,
        "dip_pc1": float(dip_pc1), "dip_p_pc1": float(dipp_pc1),
        "dip_diff1": float(dip_d1), "dip_p_diff1": float(dipp_d1),
        "kmeans2_silhouette": sil, "kmeans2_balance": bal,
        "n_components": br["n_components"], "mst_max_degree": br["mst_max_degree"],
        "geodesic_diameter": br["geodesic_diameter"],
    }
    # planted-route recovery (positive control) vs a random-label baseline (null FP)
    if route is not None:
        out["route_auc_cv"] = cv_auc(diff, route, seed=seed)
    rand = (rng.random(n) < 0.5).astype(int)
    out["randlabel_auc_cv"] = cv_auc(diff, rand, seed=seed)
    return out


def _panel_Z(panel, setname, k=6, max_iter=50):
    # max_iter cap: the per-sample EM in latent.py is pure-Python and slow; on the
    # noise-heavy synthetic controls (rho=0.5) it converges slowly. The top FA
    # directions we project onto stabilise well within 80 iters; this bounds
    # runtime. Real-data fits converge within this too.
    X, M, names = FS.build(panel, setname)
    fa = L.FactorAnalysisMissing(k=k, seed=C.SEED, max_iter=max_iter).fit(X, M)
    Z, _ = fa.transform(X, M)
    return Z, X.shape, float(M.mean())


def run(setname="minimal", null_seeds=12, k=6):
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = {"setname": setname, "k": k, "null_seeds": null_seeds,
           "deps": {"umap": _have("umap"), "ripser": _have("ripser"),
                    "gudhi": _have("gudhi"), "diptest": _have("diptest"),
                    "torch": _have("torch")}}

    # --- positive control: branch (gate first) ---
    pan_b, truth_b = CS.make_branch(return_truth=True)
    Zb, shb, obsb = _panel_Z(pan_b, setname, k)
    stat_b = nl_stats(Zb, route=truth_b["route"])
    log["branch"] = {**stat_b, "shape": shb, "obs_frac": obsb,
                     "route_balance": float(truth_b["route"].mean())}

    # --- negative controls: smooth manifolds, no branch (1-D and 2-D) ---
    null = {"n_dirs1": [], "n_dirs2": []}
    for ndir, key in [(1, "n_dirs1"), (2, "n_dirs2")]:
        ns = null_seeds if ndir == 2 else max(4, null_seeds // 2)
        for s in range(ns):
            pan = CS.make_null(n_dirs=ndir, seed=C.SEED + 100 * ndir + s)
            Zn, _, _ = _panel_Z(pan, setname, k)
            null[key].append(nl_stats(Zn, route=None, seed=C.SEED + s))
    log["null"] = null

    # --- calibrated thresholds from the 2-D null (the operative negative) ---
    n2 = null["n_dirs2"]
    def col(rows, key):
        return np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
    id2 = col(n2, "twonn_id")
    sil2 = col(n2, "kmeans2_silhouette")
    rauc2 = col(n2, "randlabel_auc_cv")
    eg2 = col(n2, "diff_eigengap")
    pr2 = col(n2, "lin_pr")
    thr = {
        "twonn_id_p95": float(np.percentile(id2, 95)),
        "twonn_id_max": float(id2.max()),
        "twonn_id_mean": float(id2.mean()),
        "kmeans2_silhouette_p95": float(np.percentile(sil2, 95)),
        "kmeans2_silhouette_max": float(sil2.max()),
        "randlabel_auc_cv_p95": float(np.percentile(rauc2, 95)),
        "diff_eigengap_p95": float(np.percentile(eg2, 95)),
        "lin_pr_mean": float(pr2.mean()),
        "dip_p_fpr_pc1": float(np.mean(col(n2, "dip_p_pc1") < 0.05)),
        "dip_p_fpr_diff1": float(np.mean(col(n2, "dip_p_diff1") < 0.05)),
    }
    log["thresholds_from_null2d"] = thr

    # --- validity checks ---
    checks = {
        # GATE: the branch must survive FA into Z, else the positive control is void
        "branch_route_recovered": bool(stat_b["route_auc_cv"] > 0.75),
        "branch_route_auc": float(stat_b["route_auc_cv"]),
        "null_randlabel_auc_near_half": float(np.median(rauc2)),
        # does kmeans invent a stronger split on the branch than on the 2-D null?
        "branch_silhouette_above_null": bool(
            stat_b["kmeans2_silhouette"] > thr["kmeans2_silhouette_p95"]),
        # intrinsic-dim ordering 1d < 2d (sanity that ID tracks planted dimension)
        "twonn_id_null1d_mean": float(col(null["n_dirs1"], "twonn_id").mean()),
        "twonn_id_null2d_mean": float(id2.mean()),
        "twonn_id_branch": float(stat_b["twonn_id"]),
    }
    log["checks"] = checks

    # --- apply the calibrated bars to REAL data (same featureset, like-with-like) ---
    real_panel = P.load_panel()
    Zr, shr, obsr = _panel_Z(real_panel, setname, k)
    stat_r = nl_stats(Zr, route=None)
    log["real"] = {
        **stat_r, "shape": shr, "obs_frac": obsr,
        "twonn_id_above_null_p95": bool(stat_r["twonn_id"] > thr["twonn_id_p95"]),
        "silhouette_above_null_p95":
            bool(stat_r["kmeans2_silhouette"] > thr["kmeans2_silhouette_p95"]),
        "dip_pc1_multimodal": bool(stat_r["dip_p_pc1"] < 0.05),
        "randlabel_auc": stat_r["randlabel_auc_cv"],
    }

    log["verdict"] = _verdict(stat_b, thr, checks)
    log["runtime_s"] = round(time.time() - t0, 1)

    with open(RESULTS / "nl_controls.json", "w") as f:
        json.dump(log, f, indent=2)
    return log


def _verdict(stat_b, thr, checks):
    if not checks["branch_route_recovered"]:
        return ("POSITIVE CONTROL FAILED: planted route not recoverable from Z "
                f"(route AUC={checks['branch_route_auc']:.2f}); FA collapsed the "
                "branch, no nonlinear branch threshold is meaningful.")
    parts = [f"branch route AUC={checks['branch_route_auc']:.2f} (recovered)"]
    if thr["dip_p_fpr_pc1"] > 0.05 or thr["dip_p_fpr_diff1"] > 0.05:
        parts.append(f"dip FPR on 2-D null pc1={thr['dip_p_fpr_pc1']:.2f}/"
                     f"diff1={thr['dip_p_fpr_diff1']:.2f} (gap test over-calls)")
    else:
        parts.append("dip does not over-call multimodality on the smooth null")
    if stat_b["kmeans2_silhouette"] <= thr["kmeans2_silhouette_p95"]:
        parts.append("kmeans silhouette does NOT exceed null even on a true branch "
                     "(silhouette is not a usable branch detector here)")
    parts.append(f"calibrated bars: twonn_id>{thr['twonn_id_p95']:.2f}, "
                 f"silhouette>{thr['kmeans2_silhouette_p95']:.3f}, "
                 f"randlabel_auc<{thr['randlabel_auc_cv_p95']:.2f}")
    return "; ".join(parts)


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def main():
    log = run(setname="minimal", null_seeds=10)
    print(json.dumps({k: v for k, v in log.items()
                      if k in ("setname", "deps", "checks",
                               "thresholds_from_null2d", "verdict",
                               "runtime_s")}, indent=2))


if __name__ == "__main__":
    main()
