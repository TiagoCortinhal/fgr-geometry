"""Orchestrator: load foundation panel, run phases 1..5, write results/summary.json.

Each phase is an adapter (below) around the real sibling module, mapping its rich
return dict down to the few scalars decide_rung needs. The real modules own their
own latent include choice (the maternal block pollutes the default FA, so a shared
pre-fit would force the wrong include on a phase). A failing/absent phase is
recorded, never crashes the run. Note: the full run invokes permutation nulls and
persistent homology and is minutes-scale, not interactive.
"""
import json
import traceback
from pathlib import Path

from fgrgeom import config
from fgrgeom import panel as panelmod

RESULTS = Path(__file__).resolve().parent / "results"


def _phase1(panel):
    # Dimensionality: a second direction counts as stable iff raw eig2 clears the
    # within-column permutation null. Subspace overlap is reported, not thresholded.
    from fgrgeom import dimensionality as dim
    out = dim.run(include=("biom", "doppler"))
    n_stable = 2 if out.get("eig2_above_null") else 1
    return {"n_stable_axes": n_stable, "raw_eig2": out.get("raw_eig2"),
            "eig2_above_null": out.get("eig2_above_null"), "latent_pr": out.get("latent_pr"),
            "subspace_overlap_top2": out.get("subspace_overlap_top2_mean_cosangle")}


def _phase2(panel):
    # Axis meaning / clinical anchoring (descriptive; not used by decide_rung).
    from fgrgeom import clinical_anchor as ca
    res = ca.anchor(panel, k=3, include=("biom", "doppler"))
    return {"axis_labels": res.get("labels") if isinstance(res, dict) else None}


def _phase3(panel):
    # Continuum vs discrete modes: multimodality of the principal axis (dip).
    from fgrgeom import topology as topo
    out = topo.run(panel=panel)
    return {"clusters_present": bool(out.get("dip_p", 1.0) < 0.05),
            "dip_p": out.get("dip_p"), "topology_verdict": out.get("verdict"),
            "_topo": out}


def _phase4(panel, p3=None):
    # Branching: a persistent 1-cycle on top of a multimodal axis (topology), or a
    # divergent flow field. Reuse phase3 topology output if available.
    from fgrgeom import topology as topo
    from fgrgeom import flow
    out = p3.get("_topo") if p3 else None
    if out is None:
        out = topo.run(panel=panel)
    branch_loop = topo.ph_h1_ratio(out.get("ph")) > 0.25 and out.get("dip_p", 1.0) < 0.05
    fl = flow.analyse(panel)
    flow_diverge = fl.get("divergence_first_window") is not None
    return {"branching": bool(branch_loop or flow_diverge),
            "branch_loop": bool(branch_loop), "flow_diverge": bool(flow_diverge),
            "flow_verdict": fl.get("verdict")}


def _phase5(panel):
    from fgrgeom import clinical_anchor as ca
    res = ca.pole_test(panel)
    return {"pole_test": res if isinstance(res, dict) else str(type(res).__name__)}


PHASES = {1: _phase1, 2: _phase2, 3: _phase3, 4: _phase4, 5: _phase5}

# Calibration-pending thresholds. NOT fitted, NOT authoritative. Each phase that
# computes its own permutation/bootstrap null should expose a boolean pass flag;
# the verdict reads those flags rather than re-deriving significance here.
THRESH = {
    "stable_axis_floor": 1,        # phase1: min axes surviving the null to leave rung 1
    "second_axis_min": 2,          # phase1: axes needed to consider rung 2/3
    "_calibration": "PENDING - thresholds are placeholders, not calibrated on data",
}


def _run_phase(num, panel, **kw):
    try:
        out = PHASES[num](panel, **kw)
        out.setdefault("status", "ok")
        return out
    except ImportError as e:
        return {"status": "missing", "phase": num, "detail": str(e)}
    except Exception:
        return {"status": "error", "phase": num, "detail": traceback.format_exc(limit=4)}


def decide_rung(phases):
    """Map collected phase metrics to a rung verdict. Returns (verdict, reason).

    verdict in {1, 2, 3, 'inconclusive'}. The default when anything required is
    absent or underpowered is 'inconclusive' - never fall through to the rung-2
    prior, or the study could not honestly find a 1-D continuum if that is the truth.
    Required metric keys (to be produced by the phases):
      phase1.n_stable_axes (int)            - axes surviving the permutation null
      phase3.clusters_present (bool)        - discrete modes vs continuum
      phase4.branching (bool)               - trunk splits into routes
    """
    p1 = phases.get(1, {})
    p3 = phases.get(3, {})
    p4 = phases.get(4, {})

    n = p1.get("n_stable_axes")
    if p1.get("status") != "ok" or n is None:
        return "inconclusive", "phase1 dimensionality unavailable"

    if n < THRESH["stable_axis_floor"]:
        return "inconclusive", "no axis survived the null (underpowered or no signal)"
    if n < THRESH["second_axis_min"]:
        return 1, "single stable axis -> 1-D severity continuum"

    # n >= 2: distinguish multi-direction continuum (rung 2) from branching (rung 3).
    clusters = p3.get("clusters_present")
    branching = p4.get("branching")
    if p3.get("status") != "ok" or clusters is None:
        return "inconclusive", "phase3 cluster/continuum test unavailable for rung 2 vs 3"
    if p4.get("status") != "ok" or branching is None:
        return "inconclusive", "phase4 branching test unavailable for rung 2 vs 3"

    if clusters and branching:
        return 3, "multi-axis with separable modes and branching routes"
    if not clusters and not branching:
        return 2, "multi-direction continuum, single field, no discrete routes"
    return "inconclusive", "phase3/phase4 disagree (clusters vs branching mismatch)"


def main():
    RESULTS.mkdir(exist_ok=True)
    pnl = panelmod.load_panel()

    foundation = {
        "n_fetuses": len(pnl.ids),
        "biom_z_shape": list(pnl.biom_z.shape),
        "doppler_shape": list(pnl.doppler.shape),
        "cardiac_shape": list(pnl.cardiac.shape),
        "maternal_shape": list(pnl.maternal.shape),
        "outcomes": list(config.OUTCOMES),
        "visits": list(config.VISITS),
    }

    phases = {}
    for num in sorted(PHASES):
        kw = {"p3": phases.get(3)} if num == 4 else {}
        phases[num] = _run_phase(num, pnl, **kw)

    verdict, reason = decide_rung(phases)

    summary = {
        "foundation": foundation,
        "thresholds": THRESH,
        "phases": {str(k): v for k, v in phases.items()},
        "rung_verdict": verdict,
        "rung_reason": reason,
        "phase_status": {str(k): v.get("status") for k, v in phases.items()},
    }

    out = RESULTS / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {out}")
    print(f"verdict={verdict} ({reason})")
    print("phase status:", summary["phase_status"])
    return summary


if __name__ == "__main__":
    main()
