import json
import pathlib
import numpy as np
from scipy.spatial import cKDTree

from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import featuresets as F

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results" / "nl"

# Complete-case is unusable on the full sets (5 fetuses). We restrict each
# feature set to its dense columns (observed fraction > DENSE_THRESH) and then
# take complete cases over those columns only. No imputation: rows still missing
# any dense column are dropped and the count is logged.
DENSE_THRESH = 0.9


def dense_complete(panel, name):
    X, M, names = F.build(panel, name)
    obs = M.mean(0)
    dense = obs > DENSE_THRESH
    keep_cols = np.where(dense)[0]
    Xd, Md = X[:, keep_cols], M[:, keep_cols]
    rows = Md.all(1)
    Xc = Xd[rows]
    dropped_cols = [names[i] for i in range(len(names)) if not dense[i]]
    info = {
        "n_features_total": int(X.shape[1]),
        "n_features_dense": int(dense.sum()),
        "dropped_sparse_cols": dropped_cols,
        "n_fetus_total": int(X.shape[0]),
        "n_complete_dense": int(rows.sum()),
        "n_dropped_incomplete": int((~rows).sum()),
        "dense_cols": [names[i] for i in keep_cols],
    }
    return Xc, info


def standardize(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


# ---- linear effective rank ----

def participation_ratio(eig):
    eig = np.asarray(eig, float)
    eig = eig[eig > 0]
    if eig.size == 0:
        return 0.0
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def linear_dim(Xs):
    cov = np.cov(Xs, rowvar=False)
    ev = np.linalg.eigvalsh(cov)[::-1]
    ev = ev[ev > 1e-10]
    pr = participation_ratio(ev)
    tot = ev.sum()
    cum = np.cumsum(ev) / tot
    rank90 = int(np.searchsorted(cum, 0.90) + 1)
    rank95 = int(np.searchsorted(cum, 0.95) + 1)
    return {"pr": pr, "rank90": rank90, "rank95": rank95, "n_pos_eig": int(ev.size)}


# ---- nonlinear intrinsic dimension estimators ----

def twonn(Xs, discard=0.1):
    """Facco et al. 2017. mu = r2/r1; d from slope of log(mu) vs -log(1-F)."""
    tree = cKDTree(Xs)
    dist, _ = tree.query(Xs, k=3)
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = r1 > 0
    mu = r2[ok] / r1[ok]
    mu = mu[mu > 1.0]
    mu = np.sort(mu)
    n = mu.size
    Femp = np.arange(1, n + 1) / (n + 1)
    keep = int(n * (1 - discard))
    x = np.log(mu[:keep])
    y = -np.log(1 - Femp[:keep])
    d = float(np.sum(x * y) / np.sum(x * x))
    return d


def mle_levina_bickel(Xs, k=20):
    """Levina-Bickel 2004 MLE with MacKay-Ghahramani averaging of inverse."""
    n = Xs.shape[0]
    k = min(k, n - 1)
    tree = cKDTree(Xs)
    dist, _ = tree.query(Xs, k=k + 1)
    T = dist[:, 1:]
    inv = []
    for i in range(n):
        ti = T[i]
        if ti[-1] <= 0 or np.any(ti <= 0):
            continue
        s = np.sum(np.log(ti[-1] / ti[:-1]))
        if s <= 0:
            continue
        m = (k - 1) / s
        inv.append(1.0 / m)
    if not inv:
        return float("nan")
    return float(1.0 / np.mean(inv))


def correlation_dim(Xs, lo=0.1, hi=0.3):
    """Grassberger-Procaccia: slope of log C(r) vs log r in mid-range."""
    n = Xs.shape[0]
    iu = np.triu_indices(n, 1)
    diff = Xs[iu[0]] - Xs[iu[1]]
    d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    d = d[d > 0]
    rlo, rhi = np.quantile(d, lo), np.quantile(d, hi)
    rs = np.geomspace(rlo, rhi, 15)
    npair = d.size
    Cr = np.array([(d < r).sum() / npair for r in rs])
    m = Cr > 0
    lr, lc = np.log(rs[m]), np.log(Cr[m])
    if lr.size < 3:
        return float("nan")
    A = np.vstack([lr, np.ones_like(lr)]).T
    slope = float(np.linalg.lstsq(A, lc, rcond=None)[0][0])
    return slope


def estimate_all(Xs):
    return {
        "twonn": twonn(Xs),
        "mle_k10": mle_levina_bickel(Xs, k=10),
        "mle_k20": mle_levina_bickel(Xs, k=20),
        "corr_dim": correlation_dim(Xs),
    }


def bootstrap_ci(Xs, fn, n_boot=100, seed=C.SEED):
    rng = np.random.default_rng(seed)
    n = Xs.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        idx = np.unique(idx)  # avoid duplicate rows -> zero distances
        try:
            vals.append(fn(Xs[idx]))
        except Exception:
            continue
    a = np.array([v for v in vals if np.isfinite(v)])
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def run_set(panel, name, n_boot=100):
    Xc, info = dense_complete(panel, name)
    Xs = standardize(Xc)
    lin = linear_dim(Xs)
    nl = estimate_all(Xs)
    ci = {
        "twonn": bootstrap_ci(Xs, twonn, n_boot),
        "mle_k20": bootstrap_ci(Xs, lambda Z: mle_levina_bickel(Z, k=20), n_boot),
        "corr_dim": bootstrap_ci(Xs, correlation_dim, n_boot),
        "linear_pr": bootstrap_ci(Xs, lambda Z: linear_dim(standardize(Z))["pr"], n_boot),
    }
    nl_mean = np.nanmean([nl["twonn"], nl["mle_k20"], nl["corr_dim"]])
    gap = float(nl_mean - lin["pr"])
    # "materially above": nonlinear ID exceeds linear PR by > 1 dim and the
    # twonn bootstrap lower bound sits above the linear PR point estimate.
    curved = bool(gap > 1.0 and ci["twonn"][0] > lin["pr"])
    return {
        "info": info,
        "linear": lin,
        "nonlinear": nl,
        "boot_ci_95": ci,
        "nl_mean": float(nl_mean),
        "linear_pr": lin["pr"],
        "nl_minus_linear": gap,
        "curved": curved,
    }


def main(n_boot=100):
    panel = P.load_panel()
    out = {"dense_thresh": DENSE_THRESH, "n_boot": n_boot, "sets": {}}
    for name in F.SETS:
        r = run_set(panel, name, n_boot=n_boot)
        out["sets"][name] = r
        i = r["info"]
        print(f"{name:14s} n={i['n_complete_dense']:4d}/{i['n_fetus_total']} "
              f"(drop {i['n_dropped_incomplete']}, cols {i['n_features_dense']}/{i['n_features_total']})  "
              f"PR={r['linear_pr']:.2f} TwoNN={r['nonlinear']['twonn']:.2f} "
              f"MLE20={r['nonlinear']['mle_k20']:.2f} corr={r['nonlinear']['corr_dim']:.2f} "
              f"curved={r['curved']}")
    n_curved = sum(v["curved"] for v in out["sets"].values())
    out["verdict"] = (
        "nonlinear intrinsic dimension materially above linear effective rank "
        "(curved structure)" if n_curved >= 3 else
        "nonlinear intrinsic dimension consistent with linear effective rank "
        "(no curvature beyond the linear continuum)")
    out["n_sets_curved"] = int(n_curved)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "intrinsic_dim.json", "w") as f:
        json.dump(out, f, indent=2)
    print("VERDICT:", out["verdict"])
    return out


if __name__ == "__main__":
    main()
