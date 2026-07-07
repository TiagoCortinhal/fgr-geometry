import pickle
from collections import namedtuple
import numpy as np
from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L

KNNGraph = namedtuple("KNNGraph", ["idx", "dist", "W", "k"])
Embedding = namedtuple("Embedding", ["Z", "pca", "pca_evr", "diff", "graph", "ids"])

RESULTS = C.DATA.parent / "results"  # fetal_growth_mechanism/results
# repo-local results dir (gitignored); prefer it if it exists
import os
_REPO_RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _resolve_Z(panel_or_Z, k=6, include=("biom", "doppler")):
    """Accept a Panel or a precomputed (n,k) latent matrix. Returns (Z, ids)."""
    if isinstance(panel_or_Z, np.ndarray):
        return panel_or_Z, np.arange(panel_or_Z.shape[0])
    if isinstance(panel_or_Z, P.Panel):
        d = L.fit_latent(panel_or_Z, k=k, include=include)
        return d["Z"], panel_or_Z.ids
    raise TypeError("expected Panel or ndarray, got %r" % type(panel_or_Z))


def pca(Z, n_components=None):
    """Plain PCA on the latent means. Returns (scores, components, evr, mean)."""
    mu = Z.mean(0)
    Zc = Z - mu
    U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    if n_components is None:
        n_components = Vt.shape[0]
    var = (s ** 2) / (Z.shape[0] - 1)
    evr = var / var.sum()
    scores = U[:, :n_components] * s[:n_components]
    return scores, Vt[:n_components], evr[:n_components], mu


def knn_graph(Z, k=15):
    """Symmetric kNN graph with Gaussian (self-tuning) affinity.
    Returns KNNGraph(idx, dist, W, k). W is dense (n,n) symmetric affinity."""
    n = Z.shape[0]
    # pairwise sq distances
    G = Z @ Z.T
    sq = np.diag(G)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0)
    np.fill_diagonal(D2, np.inf)
    idx = np.argsort(D2, axis=1)[:, :k]
    dist = np.sqrt(np.take_along_axis(D2, idx, axis=1))
    # self-tuning scale: distance to k-th neighbour per point (Zelnik-Manor & Perona)
    sigma = dist[:, -1].copy()
    sigma[sigma == 0] = np.median(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0
    A = np.zeros((n, n))
    for i in range(n):
        A[i, idx[i]] = np.exp(-dist[i] ** 2 / (sigma[i] * sigma[idx[i]]))
    W = np.maximum(A, A.T)  # symmetrise
    return KNNGraph(idx=idx, dist=dist, W=W, k=k)


def diffusion_map(W, n_components=3, alpha=1.0, t=1.0):
    """Diffusion map from a precomputed affinity W (Coifman & Lafon).
    alpha=1 removes the sampling density. Returns (coords (n,n_components), evals)."""
    d = W.sum(1)
    d[d == 0] = 1e-12
    # density normalisation
    Wa = W / np.outer(d ** alpha, d ** alpha)
    da = Wa.sum(1)
    da[da == 0] = 1e-12
    # symmetric normalised matrix M_s = D^-1/2 Wa D^-1/2, same spectrum as P=D^-1 Wa
    dinv = 1.0 / np.sqrt(da)
    Ms = Wa * np.outer(dinv, dinv)
    Ms = (Ms + Ms.T) / 2
    evals, evecs = np.linalg.eigh(Ms)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    # back to right eigenvectors of P; drop trivial first component
    psi = evecs * dinv[:, None]
    lam = np.clip(evals, 0, None)
    coords = psi[:, 1:n_components + 1] * (lam[1:n_components + 1] ** t)
    return coords, evals[1:n_components + 1]


def embed(panel_or_Z, k_latent=6, include=("biom", "doppler"),
          knn_k=15, n_pca=None, n_diff=3, alpha=1.0):
    Z, ids = _resolve_Z(panel_or_Z, k=k_latent, include=include)
    scores, comps, evr, _ = pca(Z, n_pca)
    g = knn_graph(Z, k=knn_k)
    diff, devals = diffusion_map(g.W, n_components=n_diff, alpha=alpha)
    return Embedding(Z=Z, pca=scores, pca_evr=evr, diff=diff, graph=g, ids=ids), comps, devals


def save(emb, path):
    with open(path, "wb") as f:
        pickle.dump(emb._asdict(), f)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel = P.load_panel()
    emb, comps, devals = embed(panel, knn_k=15, n_diff=3)
    os.makedirs(_REPO_RESULTS, exist_ok=True)

    save(emb, os.path.join(_REPO_RESULTS, "embedding.pkl"))
    np.savez(os.path.join(_REPO_RESULTS, "embedding_arrays.npz"),
             Z=emb.Z, pca=emb.pca, pca_evr=emb.pca_evr, diff=emb.diff,
             knn_idx=emb.graph.idx, knn_dist=emb.graph.dist, ids=emb.ids)

    bc = panel.outcomes["percentile_birth_pop"].reindex(emb.ids).to_numpy(float)
    col = bc.copy()
    fin = np.isfinite(col)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, (X, name) in zip(ax, [(emb.pca, "PCA"), (emb.diff, "diffusion")]):
        a.scatter(X[~fin, 0], X[~fin, 1], c="0.8", s=8, label="birth centile NA")
        sc = a.scatter(X[fin, 0], X[fin, 1], c=col[fin], s=10, cmap="viridis")
        a.set_title("%s of FGR latent" % name)
        a.set_xlabel("%s-1" % name)
        a.set_ylabel("%s-2" % name)
        fig.colorbar(sc, ax=a, label="birth centile")
    fig.tight_layout()
    figpath = os.path.join(_REPO_RESULTS, "embedding_scatter.png")
    fig.savefig(figpath, dpi=130)
    plt.close(fig)

    print("n=%d  latent_k=%d" % (emb.Z.shape[0], emb.Z.shape[1]))
    print("PCA explained-variance-ratio:",
          np.array2string(emb.pca_evr, precision=3))
    print("PCA cumulative:",
          np.array2string(np.cumsum(emb.pca_evr), precision=3))
    print("diffusion top eigenvalues (non-trivial):",
          np.array2string(devals, precision=4))
    print("kNN graph: k=%d, affinity nnz=%d, mean degree=%.2f"
          % (emb.graph.k, int((emb.graph.W > 0).sum()),
             (emb.graph.W > 0).sum(1).mean()))
    print("birth centile colour: %d finite / %d total" % (fin.sum(), len(col)))
    print("saved:", figpath)


if __name__ == "__main__":
    main()
