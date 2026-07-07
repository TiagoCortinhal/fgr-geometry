import json
import numpy as np
from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L
from fgrgeom.image_panel import load_image_panel, get_aligned, _tabular_size
from fgrgeom.confound_residualize import residualize, three_versions

PLANES = ("abdominal", "cerebral", "femur")
OUT = "results/img_align/direction_attrib.json"


def _multi_r(y, X):
    """Multiple correlation of 1-D y with the d-dim block X (= max corr of y with any
    image linear combination = the asked 'best image canonical variate')."""
    y = y - y.mean()
    A = np.concatenate([np.ones((X.shape[0], 1)), X - X.mean(0)], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    ss = (y ** 2).sum()
    return float(np.sqrt(max((yhat ** 2).sum() / ss, 0.0))) if ss > 0 else 0.0


def _perm_p(y, X, obs, rng, n_perm=5000):
    null = np.array([_multi_r(rng.permutation(y), X) for _ in range(n_perm)])
    p = float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))
    return p, float(null.mean()), float(np.quantile(null, 0.95))


def _cca(Xa, Xb):
    """Canonical correlations between blocks Xa, Xb via whitened cross-cov SVD."""
    Xa = Xa - Xa.mean(0)
    Xb = Xb - Xb.mean(0)
    n = Xa.shape[0]
    Sa = Xa.T @ Xa / n + 1e-6 * np.eye(Xa.shape[1])
    Sb = Xb.T @ Xb / n + 1e-6 * np.eye(Xb.shape[1])
    Sab = Xa.T @ Xb / n
    Wa = np.linalg.inv(_sqrtm(Sa))
    Wb = np.linalg.inv(_sqrtm(Sb))
    M = Wa @ Sab @ Wb
    s = np.linalg.svd(M, compute_uv=False)
    return np.clip(s, 0, 1)


def _sqrtm(S):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return (V * np.sqrt(w)) @ V.T


def _redist_axis(ip, pan, lat):
    """Fixed unit redistribution axis u_redist in R^k: top PC of the tabular latent
    after partialling ga+size out, computed once over all image fetuses. Also returns
    its loadings on the tabular factors and correlation with the size proxy / Doppler
    span so the axis label can be checked rather than assumed."""
    Z = lat["Z"]
    size, valid = _tabular_size(pan)
    tab_pos = {int(f): i for i, f in enumerate(pan.ids)}
    rows = np.array([tab_pos[int(f)] for f in ip.ids])
    ga = ip.ga
    Zr, sr, gr = Z[rows], size[rows], ga
    ok = np.isfinite(Zr).all(1) & np.isfinite(sr) & np.isfinite(gr)
    conf = np.column_stack([gr[ok], sr[ok]])
    Zres = residualize(Zr[ok], conf)
    # top PC of residual latent
    Cz = Zres.T @ Zres / Zres.shape[0]
    w, V = np.linalg.eigh(Cz)
    u = V[:, -1]
    u = u / np.linalg.norm(u)
    # angle of u to the size axis in Z (size-axis = OLS direction of size on Z)
    bsize, *_ = np.linalg.lstsq(Zr[ok] - Zr[ok].mean(0), sr[ok] - sr[ok].mean(), rcond=None)
    bsize = bsize / np.linalg.norm(bsize)
    ang = float(np.degrees(np.arccos(np.clip(abs(u @ bsize), 0, 1))))
    # Doppler vs biometry loading mass of u via the factor loading matrix W (d x k)
    W = lat["W"]
    load = W @ u  # per tabular feature contribution along u
    names = list(lat["colnames"])
    dopp = np.array([n.startswith("dop:") for n in names])
    mass = (load ** 2)
    dopp_frac = float(mass[dopp].sum() / mass.sum()) if mass.sum() > 0 else float("nan")
    return u, ang, dopp_frac, names, dopp.tolist()


def run(n_perm=5000):
    rng = np.random.default_rng(C.SEED)
    ip = load_image_panel()
    pan = P.load_panel()
    lat = L.fit_latent(pan, k=6, include=("biom", "doppler"))
    u_redist, redist_angle, redist_dopp_frac, names, dopp = _redist_axis(ip, pan, lat)

    res = {"meta": {"n_perm": n_perm, "k": 6, "n_comp": ip.n_comp,
                    "redist_axis_angle_to_size_deg": redist_angle,
                    "redist_axis_doppler_loading_frac": redist_dopp_frac,
                    "tabular_colnames": names,
                    "note": "size test on ga_resid (size present); redist on ga_size_resid "
                            "(size removed both sides). Judge effect size vs perm-null, not p. "
                            "3 planes x 2 axes = 6 tests."},
           "planes": {}}

    for plane in PLANES:
        coords, Z, ga, size, ids = get_aligned(plane, ip=ip, pan=pan, lat=lat)
        versions, kept = three_versions(coords, Z, ga, size)
        ga_k = ga[kept]
        size_k = size[kept]
        Z_k = Z[kept]
        n_k = int(kept.sum())

        # SIZE axis (positive control / trivial for abdominal): ga_resid version.
        img_g, Z_g = versions["ga_resid"]
        size_axis = residualize(size_k.reshape(-1, 1), ga_k.reshape(-1, 1))[:, 0]
        r_size = _multi_r(size_axis, img_g)
        p_size, null_size, q95_size = _perm_p(size_axis, img_g, kept, rng, n_perm)

        # REDISTRIBUTION axis: ga_size_resid version, projected on fixed u_redist.
        img_gs, Z_gs = versions["ga_size_resid"]
        redist_axis = Z_gs @ u_redist
        r_redist = _multi_r(redist_axis, img_gs)
        p_redist, null_redist, q95_redist = _perm_p(redist_axis, img_gs, kept, rng, n_perm)

        # NOVEL structure: full CCA img vs full tabular Z (ga+size residualized), plus
        # image-variance fraction NOT explained by [ga,size,Z], plus split-half stability
        # of the leading image-residual PC.
        cc = _cca(img_gs, Z_gs)
        # perm null for top canonical corr
        cca_null = np.array([_cca(img_gs, rng.permutation(Z_gs))[0] for _ in range(500)])
        cca_p = float((np.sum(cca_null >= cc[0] - 1e-12) + 1) / (500 + 1))
        # residual image var after partialling full tabular Z (already ga+size resid)
        img_res = residualize(img_gs, Z_gs)
        frac_unexpl = float((img_res ** 2).sum() / ((img_gs - img_gs.mean(0)) ** 2).sum())
        stab = _splithalf_stability(img_res, rng)

        res["planes"][plane] = {
            "n_kept": n_k,
            "size_axis": {"r": r_size, "perm_p": p_size, "perm_null_mean": null_size,
                          "perm_null_q95": q95_size},
            "redist_axis": {"r": r_redist, "perm_p": p_redist, "perm_null_mean": null_redist,
                            "perm_null_q95": q95_redist},
            "novel": {"cca_corrs": cc.tolist(), "top_cca_perm_p": cca_p,
                      "img_var_frac_unexplained_by_tabular": frac_unexpl,
                      "leading_residual_pc_splithalf_loading_corr": stab},
            "size_trivial_flag": plane == "abdominal",
        }
    import os
    os.makedirs("results/img_align", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    return res


def _splithalf_stability(X, rng, reps=50):
    """Mean |corr| of the leading-PC loading vector between random half splits.
    High residual variance is meaningless unless its leading axis is reproducible."""
    n = X.shape[0]
    vals = []
    for _ in range(reps):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        va = _top_pc(X[a])
        vb = _top_pc(X[b])
        vals.append(abs(float(va @ vb)))
    return float(np.mean(vals))


def _top_pc(X):
    Xc = X - X.mean(0)
    Cx = Xc.T @ Xc / Xc.shape[0]
    w, V = np.linalg.eigh(Cx)
    v = V[:, -1]
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    out = run()
    pl = out["planes"]
    print("redist axis: angle_to_size=%.1fdeg doppler_frac=%.3f"
          % (out["meta"]["redist_axis_angle_to_size_deg"],
             out["meta"]["redist_axis_doppler_loading_frac"]))
    for p in PLANES:
        d = pl[p]
        print("%-10s n=%d | SIZE r=%.3f p=%.4f null=%.3f | REDIST r=%.3f p=%.4f null=%.3f | "
              "novel top_cca=%.3f p=%.3f unexpl=%.2f stab=%.2f"
              % (p, d["n_kept"], d["size_axis"]["r"], d["size_axis"]["perm_p"],
                 d["size_axis"]["perm_null_mean"], d["redist_axis"]["r"],
                 d["redist_axis"]["perm_p"], d["redist_axis"]["perm_null_mean"],
                 d["novel"]["cca_corrs"][0], d["novel"]["top_cca_perm_p"],
                 d["novel"]["img_var_frac_unexplained_by_tabular"],
                 d["novel"]["leading_residual_pc_splithalf_loading_corr"]))
