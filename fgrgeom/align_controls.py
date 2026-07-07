import json
import numpy as np
from fgrgeom import config as C
from fgrgeom.image_panel import load_image_panel, get_aligned
from fgrgeom.confound_residualize import three_versions, residualize


def _metrics():
    """Scalar (no-permutation) RSA / SVCCA wrappers built on the sibling
    align_stats PRIMITIVES so the math is byte-identical to the real-result
    pipeline, but cheap enough to call inside an outer permutation null.
    align_stats.rsa/svcca take (img,Z,rng) and run their OWN 1000-perm loop
    returning tuples, so they cannot be used directly here. We reuse
    align_stats._spearman (RSA distance-corr) and align_stats._cca_corrs
    (mean canonical correlation, Raghu et al. SVCCA). Returns (rsa_fn, svcca_fn,
    source) with rsa_fn(X,Y)->float, svcca_fn(X,Y)->float."""
    from scipy.spatial.distance import pdist
    try:
        from fgrgeom import align_stats as A

        def _rsa(X, Y):
            return A._spearman(pdist(np.asarray(X, float)),
                               pdist(np.asarray(Y, float)))

        def _svcca(X, Y):
            return float(A._cca_corrs(np.asarray(X, float),
                                      np.asarray(Y, float)).mean())

        return _rsa, _svcca, "align_stats_primitives"
    except Exception:
        from scipy.stats import spearmanr

        def _rsa(X, Y):
            return float(spearmanr(pdist(np.asarray(X, float)),
                                   pdist(np.asarray(Y, float))).correlation)

        def _cancorr(A_, B_):
            Qa, _ = np.linalg.qr(A_ - A_.mean(0))
            Qb, _ = np.linalg.qr(B_ - B_.mean(0))
            s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
            return np.clip(s, 0.0, 1.0)

        def _svcca(X, Y):
            return float(_cancorr(np.asarray(X, float), np.asarray(Y, float)).mean())

        return _rsa, _svcca, "fallback_canonical"


def _perm_null(X, Y, rsa_fn, svcca_fn, n_perm, seed):
    """Shuffle the X<->Y pairing; recompute BOTH metrics end-to-end each shuffle
    (CCA refit captures the n>>d overfit floor). Returns dict of null arrays."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    rsa_n, sv_n = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        rsa_n[i] = rsa_fn(X, Y[p])
        sv_n[i] = svcca_fn(X, Y[p])
    return {"rsa": rsa_n, "svcca": sv_n}


def _scored(real, null):
    """p (one-sided, real>=null) + empirical 95th-percentile bar + null mean."""
    null = np.asarray(null, float)
    n = null.size
    return {
        "real": float(real),
        "null_mean": float(null.mean()),
        "bar95": float(np.percentile(null, 95)),
        "p": float((1 + (null >= real).sum()) / (1 + n)),
        "pass": bool(real > np.percentile(null, 95)),
    }


def run(n_perm=500, n_comp=18, k=6, seed=C.SEED):
    """Load-bearing controls for the image<->tabular alignment.
    (a) permutation null, (b) GA+size-only baseline, (c) noisy positive control,
    (d) matched-Gaussian negative. Empirical thresholds only (the SVCCA overfit
    floor is nonzero at n~900,d~18, so every bar is derived from the null/Gaussian,
    never a hardcoded constant)."""
    from fgrgeom import panel as P
    from fgrgeom import latent as L

    rsa_fn, svcca_fn, source = _metrics()
    pan = P.load_panel()
    lat = L.fit_latent(pan, k=k, include=("biom", "doppler"))
    ip = load_image_panel(n_comp=n_comp)
    rng = np.random.default_rng(seed)

    out = {
        "metric_source": source,
        "assumed_dep": "uses sibling primitives align_stats._spearman (RSA) and "
                       "align_stats._cca_corrs (SVCCA mean canonical corr) as scalar "
                       "metrics; identical math to the real-result pipeline, no nested "
                       "perm. If source==fallback_canonical, align_stats was unimportable "
                       "and bars hold only under the local canonical defs.",
        "n_perm": n_perm, "n_comp": n_comp, "k": k, "seed": seed,
        "planes": {},
    }

    for plane in ("abdominal", "cerebral", "femur"):
        coords, Z, ga, size, ids = get_aligned(plane, ip=ip, pan=pan, lat=lat, k=k)
        versions, kept = three_versions(coords, Z, ga, size)
        n_p = int(kept.sum())
        rec = {"n": n_p, "versions": {}}

        for vname, (img_r, Z_r) in versions.items():
            real_rsa = rsa_fn(img_r, Z_r)
            real_sv = svcca_fn(img_r, Z_r)
            null = _perm_null(img_r, Z_r, rsa_fn, svcca_fn, n_perm, seed + 1)
            rec["versions"][vname] = {
                "rsa": _scored(real_rsa, null["rsa"]),
                "svcca": _scored(real_sv, null["svcca"]),
            }

        # (b) GA+size-only baseline: 2-D driver space aligned to each manifold.
        gs = np.concatenate([ga[kept].reshape(-1, 1), size[kept].reshape(-1, 1)], axis=1)
        img_raw, Z_raw = versions["raw"]
        rec["baseline_gasize"] = {
            "gs_vs_img_rsa": rsa_fn(gs, img_raw),
            "gs_vs_img_svcca": svcca_fn(gs, img_raw),
            "gs_vs_Z_rsa": rsa_fn(gs, Z_raw),
            "gs_vs_Z_svcca": svcca_fn(gs, Z_raw),
            "note": "trivial driver alignment; the raw img<->Z number is non-trivial "
                    "only insofar as ga_size_resid still beats its own perm null",
        }
        out["planes"][plane] = rec

    # (c) positive control: tabular latent vs a noisy copy of itself.
    coords, Z, ga, size, ids = get_aligned("cerebral", ip=ip, pan=pan, lat=lat, k=k)
    ok = np.isfinite(Z).all(1)
    Zc = Z[ok]
    sigma = 0.5 * Zc.std(0, keepdims=True)
    Znoisy = Zc + rng.normal(size=Zc.shape) * sigma
    out["positive_control"] = {
        "n": int(Zc.shape[0]), "noise_sigma_frac": 0.5,
        "rsa": rsa_fn(Zc, Znoisy), "svcca": svcca_fn(Zc, Znoisy),
        "pass": bool(svcca_fn(Zc, Znoisy) > 0.9),
        "note": "0.5*sd Gaussian noise per column; SVCCA must stay near 1",
    }

    # (d) negative control: tabular latent vs matched Gaussian (same n, per-col sd).
    G = rng.normal(size=Zc.shape) * Zc.std(0, keepdims=True)
    neg_rsa, neg_sv = rsa_fn(Zc, G), svcca_fn(Zc, G)
    nnull = _perm_null(Zc, G, rsa_fn, svcca_fn, n_perm, seed + 7)
    out["negative_control"] = {
        "n": int(Zc.shape[0]), "d": int(Zc.shape[1]),
        "rsa": _scored(neg_rsa, nnull["rsa"]),
        "svcca": _scored(neg_sv, nnull["svcca"]),
        "note": "matched Gaussian should NOT beat its own permutation null; "
                "neg svcca>0 is the overfit floor, not signal",
    }

    return out


def main():
    from pathlib import Path
    out = run()
    outdir = Path(__file__).resolve().parents[1] / "results" / "img_align"
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "align_controls.json"
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    return out, fp


if __name__ == "__main__":
    o, fp = main()
    print(json.dumps(o, indent=2))
    print("WROTE", fp)
