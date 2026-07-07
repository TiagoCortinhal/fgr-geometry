import json
import pathlib
import warnings
import numpy as np
from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import featuresets as F

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results" / "nl"
RNG = np.random.default_rng(0)

# WARNING: UMAP/t-SNE/Isomap/LLE preserve LOCAL neighbourhoods at the cost of
# GLOBAL geometry. Inter-cluster distances, branch lengths and the apparent
# number of clusters in these 2-D scatters are NOT trustworthy and routinely
# manufacture structure from noise. Every structural claim below is checked by a
# permutation null, never by eyeballing the embedding.


def coobserved_distance(X, M):
    """Pairwise standardized Euclidean distance over co-observed dims only.
    Returns (D, bad_pairs). Columns standardized on their observed values.
    d_ij = sqrt(p / |obs_ij| * sum_obs (xs_i - xs_j)^2), p = n_features."""
    n, p = X.shape
    Xs = X.copy()
    for j in range(p):
        col = X[:, j]
        m = M[:, j]
        if m.sum() < 2:
            Xs[:, j] = 0.0
            continue
        mu = col[m].mean()
        sd = col[m].std()
        sd = sd if sd > 1e-12 else 1.0
        Xs[:, j] = (col - mu) / sd
    Z = np.where(M, Xs, 0.0).astype(float)
    Mf = M.astype(float)
    Z2 = Z ** 2
    cross = Z @ Z.T                 # sum_co z_i z_j (missing dims are zero)
    szi = Z2 @ Mf.T                 # sum over j-observed of z_i^2
    szj = Mf @ Z2.T                 # sum over i-observed of z_j^2
    K = Mf @ Mf.T                   # co-observed counts
    sse = szi + szj - 2 * cross
    sse = np.clip(sse, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.sqrt(p / K * sse)
    bad = int((K == 0).sum() - n * (K.diagonal() == 0).any())  # off-diag zeros
    bad = int(np.triu(K == 0, 1).sum())
    D[K == 0] = np.nan
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    return D, bad


def classical_mds(D, ncomp=2):
    """Classical (Torgerson) MDS on a distance matrix. Linear baseline; equals
    PCA scores when D is Euclidean. Returns 2-D scores."""
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    B = (B + B.T) / 2
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w = w[order][:ncomp]
    V = V[:, order][:, :ncomp]
    w = np.clip(w, 0, None)
    return V * np.sqrt(w)


def _silhouette_perm(emb, labels, nperm=500):
    """Label-permutation null on the silhouette of a binary label in the
    embedding. Answers: does the labelled group form a separated region beyond
    chance? Returns (observed, perm_mean, perm_p)."""
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import pdist, squareform
    Demb = squareform(pdist(emb))
    lab = labels.astype(int)
    if len(np.unique(lab)) < 2:
        return None, None, None
    obs = float(silhouette_score(Demb, lab, metric="precomputed"))
    perm = np.empty(nperm)
    for b in range(nperm):
        perm[b] = silhouette_score(Demb, RNG.permutation(lab), metric="precomputed")
    p = float((np.sum(perm >= obs) + 1) / (nperm + 1))
    return obs, float(perm.mean()), p


def embed_set(panel, name):
    from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding, trustworthiness
    X, M, names = F.build(panel, name)
    n = X.shape[0]
    cc_mask = M.all(axis=1)
    n_cc = int(cc_mask.sum())
    log = {"name": name, "n": n, "n_features": int(X.shape[1]),
           "observed_fraction": float(M.mean()),
           "n_complete_case": n_cc,
           "complete_case_dropout": int(n - n_cc),
           "complete_case_usable_for_embedding": bool(n_cc >= 50)}

    D, bad = coobserved_distance(X, M)
    # rows with any undefined distance get dropped (no co-observed overlap)
    finite_row = ~np.isnan(D).any(axis=1)
    n_drop = int((~finite_row).sum())
    log["empty_overlap_pairs"] = int(bad)
    log["rows_dropped_no_overlap"] = n_drop
    idx = np.where(finite_row)[0]
    Dd = D[np.ix_(idx, idx)]
    n_used = len(idx)
    log["n_used_distance"] = n_used

    out_y = panel.outcomes
    centile = out_y["percentile_birth_pop"].to_numpy()[idx]
    ssga = out_y["severe_sga"].to_numpy()[idx]

    embeddings = {}
    tw = {}

    mds = classical_mds(Dd, 2)
    embeddings["mds"] = mds
    tw["mds_pca_linear"] = float(trustworthiness(Dd, mds, n_neighbors=10, metric="precomputed"))

    iso = Isomap(n_neighbors=10, n_components=2, metric="precomputed")
    Yiso = iso.fit_transform(Dd)
    embeddings["isomap"] = Yiso
    tw["isomap"] = float(trustworthiness(Dd, Yiso, n_neighbors=10, metric="precomputed"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=2, metric="precomputed", init="random",
                    perplexity=30, random_state=0)
        Ytsne = tsne.fit_transform(Dd)
    embeddings["tsne"] = Ytsne
    tw["tsne"] = float(trustworthiness(Dd, Ytsne, n_neighbors=10, metric="precomputed"))

    try:
        import umap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            um = umap.UMAP(n_components=2, metric="precomputed",
                           n_neighbors=15, random_state=0)
            Yumap = um.fit_transform(Dd)
        embeddings["umap"] = Yumap
        tw["umap"] = float(trustworthiness(Dd, Yumap, n_neighbors=10, metric="precomputed"))
        log["umap_available"] = True
    except Exception as e:
        log["umap_available"] = False
        log["umap_error"] = str(e)

    # LLE needs raw X (no precomputed metric); run on complete-case only.
    lle_log = {"requires_raw_X": True, "n_complete_case": n_cc}
    if n_cc >= 50:
        Xc = X[cc_mask]
        Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=0)
            Ylle = lle.fit_transform(Xc)
        from sklearn.metrics import pairwise_distances
        Dc = pairwise_distances(Xc)
        lle_log["trustworthiness"] = float(
            trustworthiness(Dc, Ylle, n_neighbors=10))
        lle_log["status"] = "ran_on_complete_case"
    else:
        lle_log["status"] = "skipped_underpowered_complete_case"
    log["lle"] = lle_log

    # Cluster/branch survival check: does severe_sga separate in each embedding
    # beyond a label-permutation null? (Linear MDS included as reference.)
    survival = {}
    for k, emb in embeddings.items():
        obs, pm, pv = _silhouette_perm(emb, ssga, nperm=500)
        survival[k] = {"silhouette_obs": obs, "silhouette_perm_mean": pm,
                       "perm_p": pv}
    log["severe_sga_separation"] = survival

    log["trustworthiness"] = tw
    log["trustworthiness_gain_vs_pca"] = {
        k: float(v - tw["mds_pca_linear"]) for k, v in tw.items() if k != "mds_pca_linear"}

    _scatter(name, embeddings, centile, ssga)
    return log, embeddings


def _scatter(name, embeddings, centile, ssga):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in ("mds", "isomap", "tsne", "umap") if k in embeddings]
    fig, axes = plt.subplots(2, len(keys), figsize=(4 * len(keys), 8))
    if len(keys) == 1:
        axes = axes.reshape(2, 1)
    for j, k in enumerate(keys):
        E = embeddings[k]
        sc0 = axes[0, j].scatter(E[:, 0], E[:, 1], c=centile, s=8,
                                 cmap="viridis", vmin=0, vmax=100)
        axes[0, j].set_title(f"{k}  (centile)")
        m = ssga.astype(bool)
        axes[1, j].scatter(E[~m, 0], E[~m, 1], s=6, c="lightgray", label="not severe")
        axes[1, j].scatter(E[m, 0], E[m, 1], s=14, c="red", label="severe_sga")
        axes[1, j].set_title(f"{k}  (severe_sga)")
        if j == 0:
            axes[1, j].legend(fontsize=7)
    fig.colorbar(sc0, ax=axes[0, :].tolist(), shrink=0.6, label="birth centile")
    fig.suptitle(f"{name}: nonlinear embeddings (GLOBAL distances NOT trustworthy)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / f"embed_{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    panel = P.load_panel()
    sets = ["minimal", "full"]
    report = {
        "warning": ("UMAP/t-SNE/Isomap/LLE preserve local neighbourhoods only; "
                    "global distances, branch lengths and apparent cluster count "
                    "are unreliable. Structural claims checked by permutation null, "
                    "single-manifold negative control deferred to controls_run."),
        "method": ("mask-aware co-observed standardized-Euclidean distances "
                   "(n~977 kept); complete-case raw featuresets too small to embed "
                   "(logged); classical MDS on the same distances is the linear/PCA "
                   "baseline for trustworthiness."),
        "sets": {},
    }
    for s in sets:
        log, _ = embed_set(panel, s)
        report["sets"][s] = log
        tw = log["trustworthiness"]
        print(f"[{s}] n_used={log['n_used_distance']} cc={log['n_complete_case']} "
              f"tw_pca={tw['mds_pca_linear']:.3f} "
              + " ".join(f"{k}={tw[k]:.3f}" for k in tw if k != "mds_pca_linear"))
        for k, v in log["severe_sga_separation"].items():
            if v["silhouette_obs"] is not None:
                print(f"    sep[{k}] sil={v['silhouette_obs']:.3f} "
                      f"null={v['silhouette_perm_mean']:.3f} p={v['perm_p']:.3f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "manifold_embed.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main()
