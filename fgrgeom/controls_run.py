import json
from pathlib import Path
import numpy as np
from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L

# Phase 5b: run the geometry battery on SIMULATED controls and calibrate the
# decision thresholds the real-data verdict is read against. The null is a single
# continuum (rung 1): it must NOT trip a second stable direction or a branch.
# The branch control must recover both.
#
# The three metrics target DIFFERENT rungs, so the verdict is a combination rule,
# not one cutoff:
#   participation ratio (PR) of the latent W'W spectrum  -> rung1 (~1) vs rung2/3 (>1)
#   2nd raw eigenvalue above permutation null            -> a 2nd stable direction (rung2)
#   Hartigan dip p on the principal axis                 -> multimodality (rung3)
#   persistent H1 lifetime ratio (branch score)          -> a loop/bifurcation (rung3)
#
# Sibling interfaces actually used (imported inside functions):
#   controls_sim.make_null(n, n_dirs, rho, seed) -> Panel
#   controls_sim.make_branch(n, p_route, ..., seed) -> Panel
#   dimensionality.{raw_spectrum, participation_ratio, permutation_null,
#                   latent_spectrum}     (run() loads the REAL panel, so we use
#                                         these primitives on the sim panels)
#   topology.run(panel, n_boot) -> dict(dip_p=, ph=, ...); topology.ph_h1_ratio(ph)
#   flow.analyse(panel) -> dict(verdict=, ...)   [reported, not thresholded]

N_SEEDS = 12
K_FIT = 4
INCLUDE = ("biom", "doppler")
NULL_QUANTILE = 0.95
DIP_ALPHA = 0.05
TOPO_NBOOT = 500
PERM_NPERM = 100


def _metrics(panel):
    from fgrgeom import dimensionality as dim
    from fgrgeom import topology as topo

    X, M, _ = P.flatten(panel, include=INCLUDE)
    ev_raw, _ = dim.raw_spectrum(X, M)
    fit = L.fit_latent(panel, k=K_FIT, include=INCLUDE)
    pr_lat = float(dim.participation_ratio(dim.latent_spectrum(fit["W"])))
    null_ev, _ = dim.permutation_null(X, M, n_perm=PERM_NPERM)
    eig2_null_p95 = float(np.percentile(null_ev[:, 1], 95))
    eig2_above = bool(ev_raw[1] > eig2_null_p95)

    tr = topo.run(panel=panel, n_boot=TOPO_NBOOT)
    dip_p = float(tr["dip_p"])
    branch_score = float(topo.ph_h1_ratio(tr["ph"]))

    return {"pr": pr_lat, "raw_eig2": float(ev_raw[1]),
            "eig2_null_p95": eig2_null_p95, "eig2_above_null": eig2_above,
            "dip_p": dip_p, "branch_score": branch_score,
            "topo_verdict": tr["verdict"]}


def run_controls(n_seeds=N_SEEDS, n=977):
    from fgrgeom import controls_sim as sim
    null, branch = [], []
    for s in range(n_seeds):
        null.append(_metrics(sim.make_null(n=n, seed=s)))
        branch.append(_metrics(sim.make_branch(n=n, seed=s)))
    return null, branch


def _col(rows, key):
    return np.array([r[key] for r in rows], float)


def calibrate(null, branch):
    pr_null, pr_branch = _col(null, "pr"), _col(branch, "pr")
    bs_null, bs_branch = _col(null, "branch_score"), _col(branch, "branch_score")
    dip_null, dip_branch = _col(null, "dip_p"), _col(branch, "dip_p")
    e2_null, e2_branch = _col(null, "eig2_above_null"), _col(branch, "eig2_above_null")

    pr_cut = float(np.quantile(pr_null, NULL_QUANTILE))
    branch_cut = float(np.quantile(bs_null, NULL_QUANTILE))

    return {
        "pr_cut": pr_cut,
        "branch_cut": branch_cut,
        "dip_alpha": DIP_ALPHA,
        "n_seeds": len(null),
        "k_fit": K_FIT,
        "include": list(INCLUDE),
        "null_quantile": NULL_QUANTILE,
        "calibration": {
            "pr_null_median": float(np.median(pr_null)),
            "pr_branch_median": float(np.median(pr_branch)),
            "branch_score_null_median": float(np.median(bs_null)),
            "branch_score_branch_median": float(np.median(bs_branch)),
            "dip_p_null_median": float(np.median(dip_null)),
            "dip_p_branch_median": float(np.median(dip_branch)),
            "eig2_above_null_rate_null": float(e2_null.mean()),
            "eig2_above_null_rate_branch": float(e2_branch.mean()),
        },
        "power": {  # fraction of BRANCH sims each signal flags
            "pr": float((pr_branch >= pr_cut).mean()),
            "branch_score": float((bs_branch >= branch_cut).mean()),
            "dip": float((dip_branch < DIP_ALPHA).mean()),
        },
        "specificity": {  # fraction of NULL sims that stay quiet (want ~1)
            "pr_below": float((pr_null < pr_cut).mean()),
            "branch_below": float((bs_null < branch_cut).mean()),
            "dip_ns": float((dip_null >= DIP_ALPHA).mean()),
        },
    }


def decide(m, thr):
    """Combination rule on one record's metrics dict + calibrated thresholds.
    rung3 (branch): a persistent loop AND a multimodal axis (mirrors topology's
    Y/branch). rung2: more than one stable direction (PR up or 2nd eig real) but
    no branch. rung1: single continuum."""
    branchy = m["branch_score"] >= thr["branch_cut"] and m["dip_p"] < thr["dip_alpha"]
    if branchy:
        return "rung3_branch"
    if m["pr"] >= thr["pr_cut"] or m.get("eig2_above_null"):
        return "rung2_multidirection"
    return "rung1_continuum"


def main(n_seeds=N_SEEDS):
    results = Path(__file__).resolve().parents[1] / "results"
    results.mkdir(exist_ok=True)
    try:
        null, branch = run_controls(n_seeds=n_seeds)
    except ImportError as e:
        print(f"PENDING sibling module: cannot import '{getattr(e, 'name', e)}'. "
              f"controls_run is runnable; calibrates once siblings land.")
        return None

    thr = calibrate(null, branch)
    null_calls = [decide(r, thr) for r in null]
    branch_calls = [decide(r, thr) for r in branch]
    thr["null_called_continuum_frac"] = float(
        np.mean([c == "rung1_continuum" for c in null_calls]))
    thr["branch_called_branch_frac"] = float(
        np.mean([c == "rung3_branch" for c in branch_calls]))
    thr["null_calls"] = null_calls
    thr["branch_calls"] = branch_calls

    path = results / "geometry_thresholds.json"
    path.write_text(json.dumps(thr, indent=2))
    print(f"wrote {path}")
    print(json.dumps({k: thr[k] for k in
                      ("pr_cut", "branch_cut", "dip_alpha", "power",
                       "specificity", "null_called_continuum_frac",
                       "branch_called_branch_frac")}, indent=2))
    return thr


if __name__ == "__main__":
    main()
