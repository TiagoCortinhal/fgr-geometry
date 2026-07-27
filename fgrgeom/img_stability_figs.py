import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from scipy.stats import spearmanr

from fgrgeom.image_panel import load_image_panel, get_aligned
from fgrgeom.confound_residualize import three_versions

PLANES = ("abdominal", "cerebral", "femur")
VERSIONS = ("raw", "ga_resid", "ga_size_resid")
OUT = "results/img_align"
FIGS = os.path.join(OUT, "figs")
DATA = "/Users/tiago/dev/fgr-geometry/data"


def top_cancorr(X, Y):
    """First canonical correlation via QR-whitening (stable, no overfit-prone inverse)."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    Qx, _ = np.linalg.qr(Xc)
    Qy, _ = np.linalg.qr(Yc)
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return float(s[0])


def rsa(X, Y):
    """Spearman correlation between the two pairwise-distance RDMs (upper triangles)."""
    dx = pdist(X)
    dy = pdist(Y)
    return float(spearmanr(dx, dy).correlation)


def nn_overlap(X, Y, k=10):
    """Mean fraction of shared k-NN between the two spaces (self excluded)."""
    Dx = cdist(X, X)
    Dy = cdist(Y, Y)
    np.fill_diagonal(Dx, np.inf)
    np.fill_diagonal(Dy, np.inf)
    nx = np.argsort(Dx, 1)[:, :k]
    ny = np.argsort(Dy, 1)[:, :k]
    ov = [len(set(nx[i]) & set(ny[i])) / k for i in range(X.shape[0])]
    return float(np.mean(ov))


def _perm_null(stat, X, Y, n_perm, rng, k=10):
    out = np.empty(n_perm)
    n = X.shape[0]
    for i in range(n_perm):
        p = rng.permutation(n)
        out[i] = stat(X, Y[p]) if stat is not nn_overlap else nn_overlap(X, Y[p], k)
    return out


def _boot(stat, X, Y, n_boot, rng, k=10):
    out = np.empty(n_boot)
    n = X.shape[0]
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb, Yb = X[idx], Y[idx]
        out[i] = stat(Xb, Yb) if stat is not nn_overlap else nn_overlap(Xb, Yb, k)
    return out


def _stat_block(stat, img, Z, n_boot, n_perm, rng, k=10):
    point = stat(img, Z) if stat is not nn_overlap else nn_overlap(img, Z, k)
    bt = _boot(stat, img, Z, n_boot, rng, k)
    nl = _perm_null(stat, img, Z, n_perm, rng, k)
    return {
        "point": point,
        "boot_lo": float(np.percentile(bt, 2.5)),
        "boot_hi": float(np.percentile(bt, 97.5)),
        "null_mean": float(nl.mean()),
        "null_lo": float(np.percentile(nl, 2.5)),
        "null_hi": float(np.percentile(nl, 97.5)),
        "p_perm": float((np.sum(nl >= point) + 1) / (n_perm + 1)),
    }


def _subsample(stat, img, Z, fracs, rng, reps=40, k=10):
    n = img.shape[0]
    out = {}
    for f in fracs:
        m = max(20, int(round(f * n)))
        vals = []
        for _ in range(reps):
            idx = rng.choice(n, m, replace=False)
            v = stat(img[idx], Z[idx]) if stat is not nn_overlap else nn_overlap(img[idx], Z[idx], k)
            vals.append(v)
        out[f"{f:.2f}"] = {"n": m, "mean": float(np.mean(vals)),
                           "lo": float(np.percentile(vals, 2.5)),
                           "hi": float(np.percentile(vals, 97.5))}
    return out


def run(n_boot=500, n_perm=500, k_nn=10, seed=0):
    os.makedirs(FIGS, exist_ok=True)
    rng = np.random.default_rng(seed)
    ip = load_image_panel()
    from fgrgeom import panel as P
    from fgrgeom import latent as L
    pan = P.load_panel()
    lat = L.fit_latent(pan, k=6, include=("biom", "doppler"))

    res = {"meta": {"n_boot": n_boot, "n_perm": n_perm, "k_nn": k_nn, "seed": seed,
                    "n_comp": ip.n_comp}, "planes": {}}

    for plane in PLANES:
        coords, Z, ga, size, ids = get_aligned(plane, ip=ip, pan=pan, lat=lat)
        versions, kept = three_versions(coords, Z, ga, size)
        ga_k = ga[kept].reshape(-1, 1)
        size_k = size[kept].reshape(-1, 1)
        gs = np.concatenate([ga_k, size_k], axis=1)
        Z_raw = versions["raw"][1]
        n_p = versions["raw"][0].shape[0]
        pres = {"n": n_p, "versions": {}}
        # GA+size-only baseline against RAW Z (the line to beat in raw)
        pres["baseline_gasize_vs_rawZ"] = {
            "rsa": rsa(gs, Z_raw),
            "cancorr": top_cancorr(gs, Z_raw),
            "nn_overlap": nn_overlap(gs, Z_raw, k_nn),
        }
        for v in VERSIONS:
            img, Zr = versions[v]
            vb = {
                "rsa": _stat_block(rsa, img, Zr, n_boot, n_perm, rng),
                "cancorr": _stat_block(top_cancorr, img, Zr, n_boot, n_perm, rng),
                "nn_overlap": _stat_block(nn_overlap, img, Zr, n_boot, n_perm, rng, k_nn),
                "subsample_rsa": _subsample(rsa, img, Zr, (0.5, 0.75), rng),
            }
            pres["versions"][v] = vb
        res["planes"][plane] = pres
        print(plane, "done", n_p)

    # FetalCLIP robustness: headline only = cerebral ga_size_resid RSA
    fc_csv = os.path.join(DATA, "scans_long_fetalclip.csv")
    ip_fc = load_image_panel(csv=fc_csv)
    coords, Z, ga, size, ids = get_aligned("cerebral", ip=ip_fc, pan=pan, lat=lat)
    versions, kept = three_versions(coords, Z, ga, size)
    img, Zr = versions["ga_size_resid"]
    res["fetalclip_headline"] = {
        "stat": "cerebral_ga_size_resid_RSA",
        "n": int(img.shape[0]),
        "rsa": _stat_block(rsa, img, Zr, n_boot, n_perm, rng),
    }
    print("fetalclip done", img.shape[0])

    with open(os.path.join(OUT, "stability.json"), "w") as f:
        json.dump(res, f, indent=2)

    _figs(res)
    return res


def _figs(res):
    colors = {"raw": "#1f77b4", "ga_resid": "#ff7f0e", "ga_size_resid": "#2ca02c"}
    for stat, ylab, fname in [("rsa", "RSA (Spearman of RDMs)", "rsa_per_plane.png"),
                              ("cancorr", "Top canonical correlation", "cancorr_vs_null.png"),
                              ("nn_overlap", "k-NN overlap fraction", "nn_overlap.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = np.arange(len(PLANES))
        w = 0.25
        for j, v in enumerate(VERSIONS):
            x = xs + (j - 1) * w
            pts, los, his, nlo, nhi = [], [], [], [], []
            for plane in PLANES:
                b = res["planes"][plane]["versions"][v][stat]
                pts.append(b["point"]); los.append(b["point"] - b["boot_lo"])
                his.append(b["boot_hi"] - b["point"])
                nlo.append(b["null_lo"]); nhi.append(b["null_hi"])
            ax.bar(x, pts, w, color=colors[v], alpha=0.8, label=v)
            ax.errorbar(x, pts, yerr=[los, his], fmt="none", ecolor="k", capsize=3, lw=1)
            for xi, lo, hi in zip(x, nlo, nhi):
                ax.add_patch(plt.Rectangle((xi - w / 2, lo), w, hi - lo,
                                           color="gray", alpha=0.35, zorder=5))
        # GA+size baseline line per plane
        for i, plane in enumerate(PLANES):
            bl = res["planes"][plane]["baseline_gasize_vs_rawZ"][stat]
            ax.plot([xs[i] - 1.5 * w, xs[i] + 1.5 * w], [bl, bl], "r--", lw=1.5,
                    label="GA+size baseline" if i == 0 else None)
        ax.set_xticks(xs); ax.set_xticklabels(PLANES)
        ax.set_ylabel(ylab)
        ax.set_title(ylab + " per plane (boot 95% CI, gray=perm null, red=GA+size baseline)",
                     fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, fname), dpi=130)
        plt.close(fig)

    # 4th figure: FetalCLIP vs USFM headline (RSA cerebral ga_size_resid)
    fig, ax = plt.subplots(figsize=(5, 5))
    usfm = res["planes"]["cerebral"]["versions"]["ga_size_resid"]["rsa"]
    fc = res["fetalclip_headline"]["rsa"]
    for i, (lab, b) in enumerate([("USFM", usfm), ("FetalCLIP", fc)]):
        ax.errorbar([i], [b["point"]], yerr=[[b["point"] - b["boot_lo"]],
                    [b["boot_hi"] - b["point"]]], fmt="o", color="#2ca02c", capsize=4)
        ax.add_patch(plt.Rectangle((i - 0.18, b["null_lo"]), 0.36,
                     b["null_hi"] - b["null_lo"], color="gray", alpha=0.35))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["USFM", "FetalCLIP"])
    ax.set_ylabel("RSA cerebral ga_size_resid")
    ax.set_title("Headline robustness (gray=perm null)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "headline_robustness.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
