"""Synthetic controls for the geometry battery.

Two generators return the same Panel structure load_panel() produces, so every
downstream geometry experiment (flatten / fit_latent / velocity_features) runs on
them unchanged:

  make_null   -> NEGATIVE control. One smooth, connected, NON-branching latent
                 manifold (a low-d Gaussian ellipsoid). No clusters, no routes.
                 If the battery reports >1 stable direction or a branch here, the
                 battery is over-calling structure.
  make_branch -> POSITIVE control. A shared trunk that splits into two genuine
                 routes (a Y/V in feature space). The battery SHOULD recover a
                 branch / two directions here. If it cannot, it is under-powered.

Both are matched to the real panel on: feature count and per-(block) layout,
per-column observed mean/sd (covariance SCALE), and the missingness pattern
(real per-fetus masks are bootstrap-resampled, preserving the Doppler-near-complete
/ early-biometry-sparse structure and any cross-feature missingness coupling).

What is NOT matched on purpose: the real cross-feature correlation structure. That
is exactly the thing under test, so the controls impose a KNOWN geometry (ellipsoid
vs branch) and let scale/missingness be realistic. Outcomes are synthesised from the
trunk/severity coordinate so outcome-conditioned tooling has signal; they are not
calibrated to real outcome base rates beyond rough order.
"""
import numpy as np
import pandas as pd
from math import erf
from fgrgeom import config as C
from fgrgeom import panel as P

# Block layout, matching panel / flatten column order exactly.
_BLOCKS = [
    ("biom", len(C.VISITS) * len(C.BIOM_Z)),   # 20, visit-major
    ("doppler", len(C.DOPPLER_PCTL)),          # 6
    ("cardiac", len(C.CARDIAC_PCTL)),          # 13
    ("maternal", len(C.MATERNAL) + len(C.MATERNAL_DISEASE)),  # 8
]
_D = sum(d for _, d in _BLOCKS)


def _template(panel=None):
    """Per-column observed mean/sd over the full 47-col layout, plus the stacked
    real per-fetus mask and ga_days, for scale + missingness matching."""
    if panel is None:
        panel = P.load_panel()
    order = ("biom", "doppler", "cardiac", "maternal")
    X, M, names = P.flatten(panel, include=order)
    mean = np.array([X[M[:, j], j].mean() if M[:, j].any() else 0.0
                     for j in range(X.shape[1])])
    sd = np.array([X[M[:, j], j].std() if M[:, j].sum() > 1 else 1.0
                   for j in range(X.shape[1])])
    sd[sd == 0] = 1.0
    return dict(mean=mean, sd=sd, mask=M, names=names, ga=panel.ga_days,
                src_ids=panel.ids)


def _colstd(S):
    """Standardise each column to zero mean / unit variance (empirically), so the
    chosen signal fraction holds regardless of how the raw signal was built."""
    m = S.mean(0)
    s = S.std(0)
    s[s == 0] = 1.0
    return (S - m) / s


def _assemble(signal, severity, rng, tmpl, n, rho, truth, return_truth):
    """Common back end: combine standardised signal with noise at signal fraction
    rho, rescale to real per-column moments, paste real (bootstrap) missingness,
    split into panel blocks, synthesise outcomes from `severity`.

    Missingness is drawn from a SEPARATE rng seeded only by n, so null and branch of
    the same size get the identical mask -> geometry is the only A/B difference."""
    Zsig = _colstd(signal)
    noise = rng.standard_normal((n, _D))
    Z = np.sqrt(rho) * Zsig + np.sqrt(1.0 - rho) * noise
    X = tmpl["mean"] + tmpl["sd"] * Z

    # Missingness: resample whole real per-fetus mask rows (preserves marginals and
    # all cross-feature / cross-block missingness coupling, incl. Doppler ~complete).
    src = tmpl["mask"]
    mask_rng = np.random.default_rng(10_000 + n)
    pick = mask_rng.integers(0, src.shape[0], size=n)
    M = src[pick].copy()
    X[~M] = np.nan

    ids = np.arange(n)
    panel = _to_panel(X, M, ids, tmpl, pick)
    panel = panel._replace(outcomes=_outcomes(severity, ids, rng))
    if return_truth:
        return panel, truth
    return panel


def _to_panel(X, M, ids, tmpl, pick):
    n = len(ids)
    off, blocks = 0, {}
    for name, d in _BLOCKS:
        blocks[name] = (X[:, off:off + d], M[:, off:off + d])
        off += d

    bx, bm = blocks["biom"]
    V, B = len(C.VISITS), len(C.BIOM_Z)
    biom_z = bx.reshape(n, V, B)
    biom_mask = bm.reshape(n, V, B)
    ga = tmpl["ga"][pick].copy()  # carry the donor fetus' ga grid

    dop, dop_m = blocks["doppler"]
    car, car_m = blocks["cardiac"]
    mat, mat_m = blocks["maternal"]

    nr = len(C.RATIOS)
    ratios = np.zeros((n, V, nr))
    ratios_mask = np.zeros((n, V, nr), bool)
    nbp = len(C.BP)
    bp = np.zeros((n, nbp))
    bp_mask = np.zeros((n, nbp), bool)
    nrv, nrd = len(C.RAW_DOPPLER_VISITS), len(C.RAW_DOPPLER)
    raw_doppler = np.zeros((n, nrv, nrd))
    raw_doppler_mask = np.zeros((n, nrv, nrd), bool)

    return P.Panel(
        ids=ids, biom_z=biom_z, biom_mask=biom_mask, ga_days=ga,
        biom_cols=list(C.BIOM_Z),
        doppler=dop, doppler_mask=dop_m, doppler_cols=list(C.DOPPLER_PCTL),
        cardiac=car, cardiac_mask=car_m, cardiac_cols=list(C.CARDIAC_PCTL),
        maternal=mat, maternal_mask=mat_m,
        maternal_cols=C.MATERNAL + C.MATERNAL_DISEASE,
        outcomes=None,
        ratios=ratios, ratios_mask=ratios_mask, ratios_cols=list(C.RATIOS),
        bp=bp, bp_mask=bp_mask, bp_cols=list(C.BP),
        raw_doppler=raw_doppler, raw_doppler_mask=raw_doppler_mask,
        raw_doppler_cols=list(C.RAW_DOPPLER),
        raw_doppler_visits=list(C.RAW_DOPPLER_VISITS))


def _outcomes(severity, ids, rng):
    """Map a standardised severity coordinate (+ = more growth-restricted) to a
    plausible outcome table. Monotone, noisy; not base-rate calibrated."""
    s = (severity - severity.mean()) / (severity.std() or 1.0)
    z = -s + 0.3 * rng.standard_normal(len(s))
    cdf = 0.5 * (1.0 + np.array([erf(v / np.sqrt(2)) for v in z]))
    centile = np.clip(cdf * 100, 0, 100)
    p = 1.0 / (1.0 + np.exp(-(s - 1.0)))  # rises with severity
    df = pd.DataFrame({
        "percentile_birth_pop": centile,
        "sga": (centile < 10).astype(float),
        "severe_sga": (centile < 3).astype(float),
        "lga": (centile > 90).astype(float),
        "PEwithSGA": (rng.random(len(s)) < 0.15 * p).astype(float),
        "PartoPret": (rng.random(len(s)) < 0.3 * p).astype(float),
        "NICU": (rng.random(len(s)) < 0.3 * p).astype(float),
    }, index=ids)[C.OUTCOMES]
    return df


def make_null(n=977, n_dirs=2, rho=0.5, seed=C.SEED, tmpl=None, return_truth=False):
    """NEGATIVE control: a single smooth, connected, NON-branching manifold.

    n_dirs latent Gaussian coordinates are mapped through fixed random loadings to
    the 47 features -> an ellipsoidal point cloud, unimodal, no clusters/branch.
    n_dirs=1 gives a 1-D severity line; n_dirs=2 a 2-D continuum (still rung<=2,
    NO branch). rho in (0,1) is the per-column signal variance fraction (the rest is
    independent feature noise -> sets the cov SCALE split signal/noise).

    return_truth=True -> (panel, truth) with truth={"kind":"null","T":(n,n_dirs),
    "L":(n_dirs,_D)} so a battery's recovered geometry can be scored vs ground truth.
    """
    rng = np.random.default_rng(seed)
    if tmpl is None:
        tmpl = _template()
    T = rng.standard_normal((n, n_dirs))
    L = rng.standard_normal((n_dirs, _D))
    signal = T @ L
    severity = T[:, 0]  # first latent dir doubles as severity
    truth = {"kind": "null", "T": T, "L": L}
    return _assemble(signal, severity, rng, tmpl, n, rho, truth, return_truth)


def make_branch(n=977, p_route=0.5, trunk_w=1.0, branch_w=1.4, ortho=0.85,
                rho=0.5, seed=C.SEED, tmpl=None, return_truth=False):
    """POSITIVE control: shared trunk that splits into two genuine routes.

    Every fetus shares a trunk coordinate s ~ N(0,1) (loaded broadly, e.g. overall
    size/severity). Each is then assigned route A or B (Bernoulli p_route) and pushed
    a non-negative distance b ~ |N(0,1)| along that route's direction. With two
    near-orthogonal route directions this draws a Y/V in feature space: a shared
    stem that branches.

      trunk_w   weight on the shared trunk direction
      branch_w  weight on the route displacement (controls branch separation;
                set 0 to collapse back to a no-branch trunk-only line)
      ortho     1 -> routes exactly orthogonal; lower -> routes share more direction
                (less separated branch). In [0,1].
      p_route   P(route B)
      rho       per-column signal variance fraction (cov scale split).

    Severity (for outcomes) = trunk + signed branch distance, so the two routes are
    two ways to reach high severity (the branching-continuum, rung-3, hypothesis).

    return_truth=True -> (panel, truth) with truth={"kind":"branch","route":(n,) in
    {0,1},"s":trunk,"b":branch distance,"uA","uB","w_trunk"} so branch recovery is
    scorable against the planted route labels and directions.
    """
    rng = np.random.default_rng(seed)
    if tmpl is None:
        tmpl = _template()
    # Loadings.
    w_trunk = rng.standard_normal(_D)
    uA = rng.standard_normal(_D)
    # Build uB near-orthogonal to uA: uB = ortho*perp + (1-ortho)*uA.
    rand = rng.standard_normal(_D)
    perp = rand - (rand @ uA) / (uA @ uA) * uA
    uB = ortho * perp / np.linalg.norm(perp) * np.linalg.norm(uA) + (1 - ortho) * uA

    s = rng.standard_normal(n)
    route = (rng.random(n) < p_route).astype(int)   # 0=A, 1=B
    b = np.abs(rng.standard_normal(n))               # distance along the route
    dirs = np.where(route[:, None] == 0, uA, uB)
    signal = trunk_w * s[:, None] * w_trunk + branch_w * (b[:, None] * dirs)
    sign = np.where(route == 0, -1.0, 1.0)
    severity = s + sign * b
    truth = {"kind": "branch", "route": route, "s": s, "b": b,
             "uA": uA, "uB": uB, "w_trunk": w_trunk}
    return _assemble(signal, severity, rng, tmpl, n, rho, truth, return_truth)


def _pca_pr(X, M):
    """Cheap geometry probe WITHOUT EM: participation ratio of the complete-case
    correlation spectrum over columns that are jointly observed in >=50% of rows.
    Diagnostic only (the real geometry battery uses fit_latent); kept fast so main()
    runs in seconds."""
    keep = M.mean(0) > 0.5
    rows = M[:, keep].all(1)
    Xc = X[np.ix_(rows, keep)]
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
    ev = np.linalg.eigvalsh(np.corrcoef(Xc.T))[::-1]
    ev = ev[ev > 1e-9]
    return ev.sum() ** 2 / (ev ** 2).sum(), rows.sum(), keep.sum()


def main():
    from fgrgeom import features as F
    tmpl = _template()
    print(f"template: {_D} cols, n_src={tmpl['mask'].shape[0]}, "
          f"miss={1 - tmpl['mask'].mean():.3f}")
    runs = [("null", make_null(tmpl=tmpl, return_truth=True)),
            ("branch", make_branch(tmpl=tmpl, return_truth=True))]
    for name, (pan, truth) in runs:
        X, M, _ = P.flatten(pan, include=("biom", "doppler", "maternal"))
        vel = F.velocity_features(pan, log=False)
        pr, nr, nc = _pca_pr(X, M)
        print(f"{name:7s} X{X.shape} miss={1 - M.mean():.3f} truth={truth['kind']} "
              f"vel{vel.shape} cc_pr={pr:.2f} (n={nr},d={nc}) "
              f"sga={pan.outcomes['sga'].mean():.3f}")
    print("note: full geometry uses latent.fit_latent (k=6 EM ~45s/panel); not run "
          "here. Panels are runnable by any battery module unchanged.")


if __name__ == "__main__":
    main()
