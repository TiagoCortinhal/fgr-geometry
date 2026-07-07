import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # Accelerate multithreaded eigh is pathological here

import json
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis, KernelPCA
from sklearn.manifold import trustworthiness
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist, squareform

from fgrgeom import panel as P
from fgrgeom import config as C
from fgrgeom import featuresets as F

SEED = getattr(C, "SEED", 0)
_REPO_RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _continuity(X, X_emb, n_neighbors):
    # trustworthiness with the roles of original/embedding swapped:
    # penalises true neighbours that are pushed apart in the embedding.
    return trustworthiness(X_emb, X, n_neighbors=n_neighbors)


def _prune_and_completecase(X, M, names, obs_thr):
    col_obs = M.mean(0)
    keep = col_obs >= obs_thr
    Xk, Mk = X[:, keep], M[:, keep]
    rows = Mk.all(1)
    kept_names = [n for n, k in zip(names, keep) if k]
    dropped_cols = [(n, round(float(o), 3)) for n, k, o in zip(names, keep, col_obs) if not k]
    return Xk[rows], rows, kept_names, dropped_cols


def _standardize(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def _diffusion_embed(Xs, n_components, knn_k, gamma):
    # affinity from rbf over knn graph, then Coifman-Lafon diffusion coords.
    from fgrgeom import embedding as E
    g = E.knn_graph(Xs, k=knn_k)
    coords, evals = E.diffusion_map(g.W, n_components=n_components, alpha=1.0, t=1.0)
    return coords


def _recon_cv(model_ctor, Xs, n_splits=5):
    # held-out reconstruction MSE; fit (scaler implicit: Xs already standardized,
    # but refit components per fold) on train, reconstruct test, MSE over all entries.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    errs = []
    for tr, te in kf.split(Xs):
        try:
            m = model_ctor()
            m.fit(Xs[tr])
            Zt = m.transform(Xs[te])
            Xr = m.inverse_transform(Zt)
            errs.append(float(np.mean((Xs[te] - Xr) ** 2)))
        except Exception as e:  # FA has no inverse_transform path here
            return None
    return float(np.mean(errs))


def _fa_recon_cv(Xs, d, n_splits=5):
    # FA reconstruction: x_hat = mu + W z, z = E[z|x]; closed form via components_.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    errs = []
    for tr, te in kf.split(Xs):
        fa = FactorAnalysis(n_components=d, random_state=SEED, max_iter=2000)
        fa.fit(Xs[tr])
        Z = fa.transform(Xs[te])
        Xr = Z @ fa.components_ + fa.mean_
        errs.append(float(np.mean((Xs[te] - Xr) ** 2)))
    return float(np.mean(errs))


def _embed_methods(Xs, d, knn_k, gammas):
    out = {}
    out["pca"] = PCA(n_components=d, random_state=SEED).fit_transform(Xs)
    fa = FactorAnalysis(n_components=d, random_state=SEED, max_iter=2000)
    out["fa"] = fa.fit_transform(Xs)
    for g in gammas:
        kp = KernelPCA(n_components=d, kernel="rbf", gamma=g, random_state=SEED)
        out["kpca_g%.4g" % g] = kp.fit_transform(Xs)
    out["diffusion"] = _diffusion_embed(Xs, d, knn_k, None)
    return out


def _gamma_grid(Xs):
    D = pdist(Xs)
    med = float(np.median(D))
    base = 1.0 / (2 * med ** 2) if med > 0 else 1.0
    factors = [0.1, 0.3, 1.0, 3.0, 10.0]
    return base, [base * f for f in factors], med


def _eval_matrix(Xs, d, knn_k, nn, recon=True):
    base, gammas, med = _gamma_grid(Xs)
    embs = _embed_methods(Xs, d, knn_k, gammas)
    rows = {}
    for name, emb in embs.items():
        rows[name] = {
            "trustworthiness": float(trustworthiness(Xs, emb, n_neighbors=nn)),
            "continuity": float(_continuity(Xs, emb, nn)),
        }
    if recon:
        # reconstruction (linear-comparable methods only; diffusion has no pre-image)
        rows["pca"]["recon_mse"] = _recon_cv(
            lambda: PCA(n_components=d, random_state=SEED), Xs)
        rows["fa"]["recon_mse"] = _fa_recon_cv(Xs, d)
        # recon only at the base (median-heuristic) gamma to bound runtime; the
        # full gamma sweep is still evaluated above for trust/continuity.
        gkey = "kpca_g%.4g" % base
        rows[gkey]["recon_mse"] = _recon_cv(
            lambda: KernelPCA(n_components=d, kernel="rbf", gamma=base,
                              fit_inverse_transform=True, alpha=1e-3,
                              random_state=SEED), Xs)
    return rows, {"gamma_base_median_heuristic": base,
                  "median_pdist": med, "gammas": gammas}


def _null_matched(Xs, d, knn_k, nn, n_rep=5):
    # negative single-manifold control: MVN with the SAME mean/cov as Xs.
    # report the kernel-vs-linear trustworthiness gain the null produces.
    rng = np.random.default_rng(SEED)
    mu = Xs.mean(0)
    cov = np.cov(Xs, rowvar=False)
    n = Xs.shape[0]
    gains = []
    best_kpca_t = []
    pca_t = []
    for _ in range(n_rep):
        Xn = rng.multivariate_normal(mu, cov, size=n)
        Xn, _, _ = _standardize(Xn)
        rows, _ = _eval_matrix(Xn, d, knn_k, nn, recon=False)
        pt = rows["pca"]["trustworthiness"]
        kt = max(v["trustworthiness"] for k, v in rows.items() if k.startswith("kpca"))
        pca_t.append(pt)
        best_kpca_t.append(kt)
        gains.append(kt - pt)
    return {"null_pca_trust_mean": float(np.mean(pca_t)),
            "null_best_kpca_trust_mean": float(np.mean(best_kpca_t)),
            "null_kpca_minus_pca_gain_mean": float(np.mean(gains)),
            "null_kpca_minus_pca_gain_sd": float(np.std(gains)),
            "n_rep": n_rep}


def run_set(panel, name, d=2, knn_k=15, nn=10, thresholds=(0.5, 0.9)):
    X, M, names = F.build(panel, name)
    res = {"set": name, "include": list(F.SETS[name]),
           "n_total": int(X.shape[0]), "n_features_raw": int(X.shape[1]),
           "d": d, "nn": nn, "knn_k": knn_k, "thresholds": {}}
    for thr in thresholds:
        Xc, rows_mask, kept, dropped = _prune_and_completecase(X, M, names, thr)
        n_cc = int(Xc.shape[0])
        entry = {"obs_threshold": thr, "n_features_kept": len(kept),
                 "n_complete_case": n_cc, "n_dropped_fetuses": int(X.shape[0] - n_cc),
                 "dropped_columns": dropped, "kept_columns": kept}
        if n_cc < 4 * len(kept) or n_cc < 50:
            entry["skipped"] = "complete-case too small for n>p neighborhood eval"
            res["thresholds"][str(thr)] = entry
            continue
        print(f"  [{name} thr={thr}] ncc={n_cc} feat={len(kept)} evaluating...",
              flush=True)
        Xs, _, _ = _standardize(Xc)
        rows, ginfo = _eval_matrix(Xs, d, knn_k, nn)
        entry["methods"] = rows
        entry["gamma_info"] = ginfo
        best_kpca = max((v["trustworthiness"] for k, v in rows.items()
                         if k.startswith("kpca")))
        entry["real_kpca_minus_pca_trust"] = float(best_kpca - rows["pca"]["trustworthiness"])
        entry["real_diffusion_minus_pca_trust"] = float(
            rows["diffusion"]["trustworthiness"] - rows["pca"]["trustworthiness"])
        print(f"  [{name} thr={thr}] real done, running null...", flush=True)
        entry["null"] = _null_matched(Xs, d, knn_k, nn)
        entry["verdict_gain_over_null"] = float(
            entry["real_kpca_minus_pca_trust"] - entry["null"]["null_kpca_minus_pca_gain_mean"])
        res["thresholds"][str(thr)] = entry
    return res


def main():
    panel = P.load_panel()
    out = {"n": int(len(panel.ids)), "seed": SEED,
           "deps": "sklearn-only (KernelPCA, FactorAnalysis, PCA, trustworthiness); "
                   "no torch/umap/ripser/gudhi", "sets": {}}
    os.makedirs(os.path.join(_REPO_RESULTS, "nl"), exist_ok=True)
    path = os.path.join(_REPO_RESULTS, "nl", "kpca_diffusion.json")
    for name in ("minimal", "full"):
        out["sets"][name] = run_set(panel, name)
        with open(path, "w") as f:  # checkpoint after each set
            json.dump(out, f, indent=2)
        print(f"== {name} set written ==", flush=True)
    for sname, s in out["sets"].items():
        for thr, e in s["thresholds"].items():
            if "methods" not in e:
                print(f"{sname:>7} thr={thr}: SKIP ({e.get('skipped')})")
                continue
            print(f"{sname:>7} thr={thr}: ncc={e['n_complete_case']} "
                  f"feat={e['n_features_kept']} "
                  f"real_kpca-pca={e['real_kpca_minus_pca_trust']:+.4f} "
                  f"null_gain={e['null']['null_kpca_minus_pca_gain_mean']:+.4f} "
                  f"net={e['verdict_gain_over_null']:+.4f}")
    print("saved:", path)
    return out


if __name__ == "__main__":
    main()
