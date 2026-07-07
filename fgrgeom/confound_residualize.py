import numpy as np


def residualize(X, confounds):
    """OLS partial-out: return residuals of each column of X after regressing on
    confounds (intercept added automatically). Rows must be complete-case (no nan).
    X (n,p), confounds (n,c) -> residuals (n,p)."""
    X = np.asarray(X, float)
    C = np.asarray(confounds, float)
    if C.ndim == 1:
        C = C[:, None]
    A = np.concatenate([np.ones((C.shape[0], 1)), C], axis=1)
    beta, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ beta


def three_versions(img_coords, tabular_Z, ga, size):
    """Produce {raw, ga_resid, ga_size_resid} for an (image_coords, tabular_Z) pair.
    Each value is (img_resid, Z_resid) with the named confounds partialled out of BOTH
    sides. Rows with nan in coords/Z/ga/size are dropped jointly first.
    Returns (versions dict, kept_mask)."""
    img = np.asarray(img_coords, float)
    Z = np.asarray(tabular_Z, float)
    ga = np.asarray(ga, float).reshape(-1, 1)
    size = np.asarray(size, float).reshape(-1, 1)
    ok = (np.isfinite(img).all(1) & np.isfinite(Z).all(1)
          & np.isfinite(ga[:, 0]) & np.isfinite(size[:, 0]))
    img, Z, ga, size = img[ok], Z[ok], ga[ok], size[ok]
    out = {"raw": (img, Z)}
    out["ga_resid"] = (residualize(img, ga), residualize(Z, ga))
    gs = np.concatenate([ga, size], axis=1)
    out["ga_size_resid"] = (residualize(img, gs), residualize(Z, gs))
    return out, ok
