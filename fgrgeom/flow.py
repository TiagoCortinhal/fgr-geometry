import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import features  # noqa: F401  (kept for sibling parity; used in main)
from fgrgeom import latent  # noqa: F401

# Flow axes: drop efw_z (Hadlock-derived from ac/hc/bpd/fl -> mechanically collinear,
# and ~mostly missing at early visits). The flow field is built on the 4 primaries.
FLOW_AXES = ["ac_z_ig21", "hc_z_ig21", "bpd_z_ig21", "fl_z_ig21"]
OUTDIR = Path("results/flow")


def _axis_idx(panel):
    return [panel.biom_cols.index(a) for a in FLOW_AXES]


def velocity_samples(panel):
    """Per-fetus, per consecutive-visit-transition velocity vectors in biometry-z space.
    Complete-case per transition: needs both endpoints fully observed on the 4 FLOW_AXES
    and a positive ga gap. Returns dict keyed by transition label, each a record dict
    with arrays: p (starting position, n_t x 4), v (velocity z/week, n_t x 4),
    ga_mid (weeks), fid, and a dropout count. NO imputation."""
    idx = _axis_idx(panel)
    Bz = panel.biom_z[:, :, idx]            # (n, V, 4)
    Bm = panel.biom_mask[:, :, idx]
    ga = panel.ga_days                       # (n, V)
    V = len(C.VISITS)
    out = {}
    for j in range(V - 1):
        lab = f"{C.VISITS[j]}->{C.VISITS[j+1]}"
        p, v, gm, fid = [], [], [], []
        dropped = 0
        for i in range(len(panel.ids)):
            a, b = Bz[i, j], Bz[i, j + 1]
            ok = Bm[i, j].all() and Bm[i, j + 1].all()
            g0, g1 = ga[i, j], ga[i, j + 1]
            if not (ok and np.isfinite(g0) and np.isfinite(g1) and g1 > g0):
                dropped += 1
                continue
            dt = (g1 - g0) / 7.0
            p.append(a); v.append((b - a) / dt)
            gm.append((g0 + g1) / 2.0 / 7.0); fid.append(panel.ids[i])
        out[lab] = {
            "p": np.array(p), "v": np.array(v), "ga_mid": np.array(gm),
            "fid": np.array(fid), "n": len(p), "dropped": dropped,
        }
    return out


def fit_jacobian(p, v, ridge=1e-2):
    """Linear flow field v = A p + b by ridge least squares. Returns A (4x4), b,
    eigenvalues of A, spectral abscissa (max real part), trace, and the stationary
    point p* = -A^-1 b (the field's fixed point / attractor location)."""
    n, d = p.shape
    Xa = np.hstack([p, np.ones((n, 1))])
    G = Xa.T @ Xa + ridge * np.eye(d + 1)
    coef = np.linalg.solve(G, Xa.T @ v)     # (d+1, d)
    A = coef[:d].T                           # rows = dv_i/dp_j
    b = coef[d]
    w = np.linalg.eigvals(A)
    try:
        pstar = np.linalg.solve(A, -b)
    except np.linalg.LinAlgError:
        pstar = np.full(d, np.nan)
    return {"A": A, "b": b, "eig": w, "abscissa": float(np.max(w.real)),
            "trace": float(np.trace(A)), "pstar": pstar}


def expanding_axis(A):
    """Eigenvector of A with the largest real-part eigenvalue (the direction along which
    the flow stretches / branches). Returned as a real unit vector."""
    w, Vv = np.linalg.eig(A)
    k = int(np.argmax(w.real))
    e = Vv[:, k].real
    nrm = np.linalg.norm(e)
    return e / nrm if nrm > 0 else e, float(w[k].real)


def bimodality_coef(x):
    """Sarle's bimodality coefficient. > 5/9 ~ 0.555 suggests bimodality/branching;
    Gaussian -> 1/3. Soft signal only."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan
    g = stats.skew(x)
    k = stats.kurtosis(x, fisher=True)
    return float((g ** 2 + 1) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def dispersion_by_visit(panel):
    """Spread of the population in biometry-z at each visit (observed-only, complete
    rows on the 4 axes). Generalised variance (det of cov) and total variance (trace).
    Contracting flow -> spread shrinks across GA; diverging -> grows."""
    idx = _axis_idx(panel)
    Bz = panel.biom_z[:, :, idx]
    Bm = panel.biom_mask[:, :, idx]
    res = {}
    for j, vis in enumerate(C.VISITS):
        rows = Bm[:, j].all(axis=1)
        Xr = Bz[rows, j]
        if len(Xr) < 5:
            res[vis] = {"n": int(len(Xr)), "total_var": None, "gen_var": None}
            continue
        cov = np.cov(Xr.T)
        res[vis] = {"n": int(len(Xr)), "total_var": float(np.trace(cov)),
                    "gen_var": float(np.linalg.det(cov)),
                    "ga_weeks": float(np.nanmedian(panel.ga_days[rows, j] / 7.0))}
    return res


def bootstrap_jacobian(p, v, n_boot=500, seed=C.SEED):
    """Resample fetuses; CI for trace(A) and spectral abscissa. A positive-abscissa
    fraction near 1 is evidence of a genuine expanding (divergent/branching) direction
    overriding mean reversion."""
    rng = np.random.default_rng(seed)
    n = len(p)
    tr, ab = [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        f = fit_jacobian(p[s], v[s])
        tr.append(f["trace"]); ab.append(f["abscissa"])
    tr, ab = np.array(tr), np.array(ab)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"trace_ci": q(tr), "abscissa_ci": q(ab),
            "p_abscissa_pos": float((ab > 0).mean())}


def analyse(panel, n_boot=500):
    samples = velocity_samples(panel)
    disp = dispersion_by_visit(panel)
    windows = {}
    for lab, s in samples.items():
        if s["n"] < 20:
            windows[lab] = {"n": s["n"], "dropped": s["dropped"], "note": "too few"}
            continue
        fj = fit_jacobian(s["p"], s["v"])
        e_axis, e_val = expanding_axis(fj["A"])
        proj_start = s["p"] @ e_axis
        bc = bimodality_coef(proj_start)
        bt = bootstrap_jacobian(s["p"], s["v"], n_boot=n_boot)
        windows[lab] = {
            "n": s["n"], "dropped": s["dropped"],
            "ga_mid_weeks": float(np.median(s["ga_mid"])),
            "trace": fj["trace"], "trace_ci": bt["trace_ci"],
            "spectral_abscissa": fj["abscissa"], "abscissa_ci": bt["abscissa_ci"],
            "p_abscissa_pos": bt["p_abscissa_pos"],
            "eigvals_real": [float(x) for x in np.sort(fj["eig"].real)[::-1]],
            "eigvals_imag_max": float(np.max(np.abs(fj["eig"].imag))),
            "pstar": [float(x) for x in fj["pstar"]],
            "expanding_axis": dict(zip(FLOW_AXES, [float(x) for x in e_axis])),
            "expanding_eig": e_val,
            "bimodality_coef_along_axis": bc,
        }

    # Where does divergence first appear? earliest window with abscissa CI lower bound > 0.
    diverge_at = None
    for lab in samples:
        w = windows.get(lab, {})
        ci = w.get("abscissa_ci")
        if ci and ci[0] > 0:
            diverge_at = lab
            break

    verdict = _verdict(windows, disp, diverge_at)
    return {"axes": FLOW_AXES, "windows": windows, "dispersion_by_visit": disp,
            "divergence_first_window": diverge_at, "verdict": verdict}


def _verdict(windows, disp, diverge_at):
    """Map the spectra to the rung question. Contraction everywhere -> rung1 single
    attractor; a stable expanding direction -> rung2 multi-direction; expanding axis
    that also turns bimodal late -> rung3 branching. Honest, exploratory wording."""
    abscissas = [w.get("spectral_abscissa") for w in windows.values()
                 if isinstance(w, dict) and "spectral_abscissa" in w]
    any_pos = [w for w in windows.values()
               if isinstance(w, dict) and w.get("abscissa_ci", [0, 0])[0] > 0]
    bim = [w for w in any_pos if (w.get("bimodality_coef_along_axis") or 0) > 0.555]
    if not abscissas:
        return "insufficient data"
    if not any_pos:
        return ("rung1: all flow eigen-directions contract (CI<0) -> single attractor / "
                "mean reversion; no divergent axis detected")
    if bim:
        return ("rung3 candidate: a divergent axis exists AND its projection is bimodal "
                f"late (first at {diverge_at}) -> possible branching")
    return ("rung2 candidate: at least one flow direction expands (CI>0) while others "
            f"contract -> multi-direction continuum; divergence onset {diverge_at}")


def figure(panel, result, path):
    """PCA(2) quiver of the velocity field per window + dispersion curve."""
    samples = velocity_samples(panel)
    allp = np.vstack([s["p"] for s in samples.values() if s["n"] > 0])
    mu = allp.mean(0)
    U, S, Vt = np.linalg.svd(allp - mu, full_matrices=False)
    pcs = Vt[:2]                              # (2,4)
    labs = list(samples.keys())
    fig, axes = plt.subplots(1, len(labs) + 1, figsize=(4.2 * (len(labs) + 1), 4))
    for ax, lab in zip(axes[:-1], labs):
        s = samples[lab]
        if s["n"] < 20:
            ax.set_title(f"{lab}\n(n={s['n']})"); continue
        pp = (s["p"] - mu) @ pcs.T
        vv = s["v"] @ pcs.T
        # bin to a grid and average
        gx = np.linspace(pp[:, 0].min(), pp[:, 0].max(), 9)
        gy = np.linspace(pp[:, 1].min(), pp[:, 1].max(), 9)
        ix = np.clip(np.digitize(pp[:, 0], gx), 1, len(gx) - 1) - 1
        iy = np.clip(np.digitize(pp[:, 1], gy), 1, len(gy) - 1) - 1
        cx, cy, ux, uy = [], [], [], []
        for a in range(len(gx) - 1):
            for c in range(len(gy) - 1):
                m = (ix == a) & (iy == c)
                if m.sum() >= 3:
                    cx.append((gx[a] + gx[a + 1]) / 2); cy.append((gy[c] + gy[c + 1]) / 2)
                    ux.append(vv[m, 0].mean()); uy.append(vv[m, 1].mean())
        ax.scatter(pp[:, 0], pp[:, 1], s=4, c="0.8", zorder=1)
        ax.quiver(cx, cy, ux, uy, angles="xy", color="C3", zorder=2)
        w = result["windows"][lab]
        e = np.array(list(w["expanding_axis"].values())) @ pcs.T
        ax.annotate("", xy=mu[:2] * 0 + e * 2, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="C0", lw=2))
        ax.set_title(f"{lab}  abs={w['spectral_abscissa']:.2f}\n"
                     f"tr={w['trace']:.2f} n={s['n']}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax = axes[-1]
    d = result["dispersion_by_visit"]
    vis = [v for v in C.VISITS if d[v]["total_var"] is not None]
    ax.plot([d[v]["ga_weeks"] for v in vis], [d[v]["total_var"] for v in vis], "o-")
    ax.set_title("population spread vs GA\n(trace cov, 4 axes)")
    ax.set_xlabel("GA weeks"); ax.set_ylabel("total variance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(n_boot=500):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = P.load_panel()
    result = analyse(panel, n_boot=n_boot)
    with open(OUTDIR / "flow_stats.json", "w") as f:
        json.dump(result, f, indent=2)
    figure(panel, result, OUTDIR / "flow_field.png")
    print(json.dumps({"divergence_first_window": result["divergence_first_window"],
                      "verdict": result["verdict"],
                      "windows": {k: {kk: v[kk] for kk in
                                      ("n", "trace", "spectral_abscissa", "abscissa_ci",
                                       "p_abscissa_pos", "bimodality_coef_along_axis")
                                      if kk in v}
                                  for k, v in result["windows"].items()}}, indent=2))
    return result


if __name__ == "__main__":
    main()
