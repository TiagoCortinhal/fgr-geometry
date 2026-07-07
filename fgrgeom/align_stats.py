import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from fgrgeom.image_panel import load_image_panel, get_aligned
from fgrgeom.confound_residualize import residualize, three_versions

N_PERM = 1000
KNN = 15


def _spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (np.sqrt(ra @ ra) * np.sqrt(rb @ rb)))


def rsa(img, Z, rng):
    """Distance-matrix Spearman (RSA) + Mantel permutation p."""
    Di = squareform(pdist(img))
    Dz = squareform(pdist(Z))
    iu = np.triu_indices(Di.shape[0], 1)
    vi, vz = Di[iu], Dz[iu]
    obs = _spearman(vi, vz)
    n = Di.shape[0]
    ge = 1
    for _ in range(N_PERM):
        p = rng.permutation(n)
        vp = Dz[p][:, p][iu]
        if _spearman(vi, vp) >= obs:
            ge += 1
    return obs, ge / (N_PERM + 1)


def _cca_corrs(X, Y):
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    Qx, _ = np.linalg.qr(Xc)
    Qy, _ = np.linalg.qr(Yc)
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return np.clip(s, 0, 1)


def svcca(img, Z, rng):
    """Canonical correlations on PCA-reduced spaces; value = mean, plus top corrs and perm p on the mean."""
    corrs = _cca_corrs(img, Z)
    obs = float(corrs.mean())
    n = img.shape[0]
    ge = 1
    for _ in range(N_PERM):
        p = rng.permutation(n)
        if _cca_corrs(img, Z[p]).mean() >= obs:
            ge += 1
    return obs, ge / (N_PERM + 1), [float(c) for c in corrs]


def _knn(D, k):
    n = D.shape[0]
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    S = np.zeros((n, n), bool)
    rows = np.repeat(np.arange(n), k)
    S[rows, idx.ravel()] = True
    return S


def knn_overlap(img, Z, rng, k=KNN):
    """Mutual k-NN overlap: mean fraction of shared neighbors; perm p."""
    Di = squareform(pdist(img))
    Dz = squareform(pdist(Z))
    Si = _knn(Di, k)
    Sz = _knn(Dz, k)

    def ov(perm):
        Szp = Sz[perm][:, perm]
        return float((Si & Szp).sum(1).mean() / k)

    base = np.arange(img.shape[0])
    obs = ov(base)
    ge = 1
    for _ in range(N_PERM):
        if ov(rng.permutation(img.shape[0])) >= obs:
            ge += 1
    return obs, ge / (N_PERM + 1)


def _concat_aligned(ip, pan, lat):
    from fgrgeom import latent as L
    Z = lat["Z"]
    tab_pos = {int(f): i for i, f in enumerate(pan.ids)}
    from fgrgeom.image_panel import _tabular_size
    size, _ = _tabular_size(pan)
    m = ip.concat_mask
    ids = ip.ids[m]
    coords = ip.concat[m]
    ga = ip.ga[m]
    rows = [tab_pos[int(f)] for f in ids]
    return coords, Z[rows], ga, size[rows], ids


def run():
    ip = load_image_panel()
    from fgrgeom import panel as P
    from fgrgeom import latent as L
    pan = P.load_panel()
    lat = L.fit_latent(pan, k=6, include=("biom", "doppler"))

    planes = ["cerebral", "abdominal", "femur", "concat"]
    out = {}
    for plane in planes:
        if plane == "concat":
            img, Z, ga, size, ids = _concat_aligned(ip, pan, lat)
        else:
            img, Z, ga, size, ids = get_aligned(plane, ip=ip, pan=pan, lat=lat)
        versions, kept = three_versions(img, Z, ga, size)
        out[plane] = {"n_in": int(len(ids)), "n_kept": int(kept.sum()), "versions": {}}
        for vname, (im, zr) in versions.items():
            rng = np.random.default_rng(0)
            r_val, r_p = rsa(im, zr, rng)
            s_val, s_p, s_corrs = svcca(im, zr, rng)
            k_val, k_p = knn_overlap(im, zr, rng)
            out[plane]["versions"][vname] = {
                "rsa": {"value": r_val, "p": r_p},
                "svcca": {"value": s_val, "p": s_p, "canonical_corrs": s_corrs},
                "knn_overlap": {"value": k_val, "p": k_p, "k": KNN},
            }
    import os
    os.makedirs("results/img_align", exist_ok=True)
    path = "results/img_align/align_stats.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out, path


if __name__ == "__main__":
    o, p = run()
    print(p)
    print(json.dumps(o, indent=2))
