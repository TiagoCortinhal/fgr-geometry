from collections import namedtuple
import numpy as np
import pandas as pd
from fgrgeom import config as C

EMB_COLS = [f"emb_{i}" for i in range(768)]
PLANES = ("abdominal", "cerebral", "femur")

ImagePanel = namedtuple("ImagePanel", [
    "ids",            # (n,) shared fetus-id order = panel.ids intersect image, sorted
    "planes",         # tuple of plane names
    "mask",           # dict plane -> (n,) bool, fetus has >=1 scan in that plane
    "pooled",         # dict plane -> (n, 768) mean-pooled embedding, nan where absent
    "pooled_w",       # dict plane -> (n, 768) confidence-weighted pooled, nan where absent
    "pca",            # dict plane -> (n, d) PCA coords on available rows, nan where absent
    "pca_w",          # dict plane -> (n, d) PCA coords from weighted pooling
    "pca_var",        # dict plane -> float variance retained
    "concat",         # (n, d) PCA on planes-concat over complete-case fetuses, nan elsewhere
    "concat_mask",    # (n,) bool complete-case across all planes used in concat
    "concat_var",
    "ga",             # (n,) mean ga_weeks per fetus across its scans, nan if no scans
    "n_comp",
])


def _pool(df):
    """Mean and confidence-weighted-mean pooling of the 768-d embedding per (fetus, plane).
    Weighted variant uses label_confidence; missing confidence filled with the plane median
    so no scan is silently dropped. Returns two dicts plane -> {fetus_id: vec}."""
    mean_p, wmean_p = {}, {}
    for plane, sub in df.groupby("plane"):
        med = sub["label_confidence"].median()
        w = sub["label_confidence"].fillna(med).to_numpy(float)
        E = sub[EMB_COLS].to_numpy(float)
        fids = sub["fetus_id"].to_numpy()
        m_, wm_ = {}, {}
        for f in np.unique(fids):
            sel = fids == f
            m_[f] = E[sel].mean(0)
            wi = w[sel]
            wm_[f] = (E[sel] * wi[:, None]).sum(0) / wi.sum() if wi.sum() > 0 else E[sel].mean(0)
        mean_p[plane] = m_
        wmean_p[plane] = wm_
    return mean_p, wmean_p


def _pca(X, mask, n_comp):
    """PCA on rows where mask is True; return (coords (n,d) nan off-mask, var_retained)."""
    from numpy.linalg import svd
    Xo = X[mask]
    mu = Xo.mean(0)
    Xc = Xo - mu
    U, S, Vt = svd(Xc, full_matrices=False)
    d = min(n_comp, Vt.shape[0])
    comps = Vt[:d]
    var = float((S[:d] ** 2).sum() / (S ** 2).sum())
    coords = np.full((X.shape[0], d), np.nan)
    coords[mask] = Xc @ comps.T
    return coords, var


def load_image_panel(n_comp=18, planes=PLANES, csv=None):
    """Build the per-fetus image panel pooled to one vector per plane, aligned to the
    fetus ids shared with panel.load_panel(). NO imputation across missing planes;
    per-plane availability carried in `mask`. PCA-reduces each plane to n_comp comps."""
    from fgrgeom import panel as P
    path = csv if csv is not None else (C.DATA / "scans_long_usfm.csv")
    df = pd.read_csv(path)
    df = df[df["plane"].isin(planes)].copy()

    pan = P.load_panel()
    tab_ids = set(int(x) for x in pan.ids)
    df = df[df["fetus_id"].isin(tab_ids)].copy()

    # shared id order: tabular order restricted to fetuses with any image scan
    img_ids = set(int(x) for x in df["fetus_id"].unique())
    ids = np.array([int(x) for x in pan.ids if int(x) in img_ids])
    n = len(ids)
    pos = {f: i for i, f in enumerate(ids)}

    mean_p, wmean_p = _pool(df)

    mask, pooled, pooled_w = {}, {}, {}
    pca, pca_w, pca_var = {}, {}, {}
    for plane in planes:
        P0 = np.full((n, 768), np.nan)
        Pw = np.full((n, 768), np.nan)
        m = np.zeros(n, bool)
        for f, v in mean_p.get(plane, {}).items():
            P0[pos[f]] = v
            m[pos[f]] = True
        for f, v in wmean_p.get(plane, {}).items():
            Pw[pos[f]] = v
        pooled[plane], pooled_w[plane], mask[plane] = P0, Pw, m
        c, var = _pca(P0, m, n_comp)
        cw, _ = _pca(Pw, m, n_comp)
        pca[plane], pca_w[plane], pca_var[plane] = c, cw, var

    # concat: complete-case fetuses present in ALL planes; PCA on stacked pooled means
    cmask = np.all([mask[p] for p in planes], axis=0)
    Xcat = np.concatenate([pooled[p] for p in planes], axis=1)
    concat, cvar = _pca(Xcat, cmask, n_comp)

    ga = np.full(n, np.nan)
    gm = df.groupby("fetus_id")["ga_weeks"].mean()
    for f, v in gm.items():
        if int(f) in pos:
            ga[pos[int(f)]] = v

    return ImagePanel(ids=ids, planes=tuple(planes), mask=mask, pooled=pooled,
                      pooled_w=pooled_w, pca=pca, pca_w=pca_w, pca_var=pca_var,
                      concat=concat, concat_mask=cmask, concat_var=cvar,
                      ga=ga, n_comp=n_comp)


def _tabular_size(pan):
    """Per-fetus size proxy: mean of observed biometry-z across all visit cells.
    Returns (size (N,), valid (N,) bool) aligned to pan.ids order."""
    bz = pan.biom_z.reshape(pan.biom_z.shape[0], -1)
    bm = pan.biom_mask.reshape(bz.shape)
    cnt = bm.sum(1)
    s = np.where(cnt > 0, np.nansum(np.where(bm, bz, 0.0), 1) / np.maximum(cnt, 1), np.nan)
    return s, cnt > 0


def get_aligned(plane, ip=None, pan=None, lat=None, weighted=False, k=6,
                include=("biom", "doppler")):
    """Row-aligned bundle for one plane, complete-case on the image side.
    Returns (img_coords (n_p,d), tabular_Z (n_p,k), ga (n_p,), size (n_p,), ids (n_p,)).
    tabular_Z is fit_latent(panel, k, include).Z subset+reordered to the SAME fetuses.
    size = mean observed biometry-z (tabular size proxy).
    Pass ip/pan/lat to avoid re-pooling and refitting the EM on every call (the
    latent fit is a 300-iter EM over n=977; cache it across planes/bootstraps)."""
    from fgrgeom import panel as P
    from fgrgeom import latent as L
    if ip is None:
        ip = load_image_panel()
    if pan is None:
        pan = P.load_panel()
    if lat is None:
        lat = L.fit_latent(pan, k=k, include=include)
    Z = lat["Z"]
    tab_pos = {int(f): i for i, f in enumerate(pan.ids)}
    size, _ = _tabular_size(pan)

    m = ip.mask[plane]
    ids = ip.ids[m]
    coords = (ip.pca_w[plane] if weighted else ip.pca[plane])[m]
    ga = ip.ga[m]
    rows = [tab_pos[int(f)] for f in ids]
    Zsub = Z[rows]
    size_sub = size[rows]
    return coords, Zsub, ga, size_sub, ids
