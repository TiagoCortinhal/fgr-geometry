# FGR latent geometry: hypotheses and phase map

Exploratory first look. The question is which "rung" the FGR latent sits on, fit
on a pure fetal-growth phenotype latent (biometry + late Doppler; the maternal
block is left out of the geometry latent because it can manufacture non-fetal
axes). This is a structure question, not an AUC question.

## Competing hypotheses

- **H1 - 1-D severity continuum.** One stable axis. FGR severity is a single
  monotone gradient (size deficit) and everything else (redistribution, outcome)
  is a readout of position on that one axis. Null/most parsimonious.
- **H2 - multi-direction continuum.** Two (or more) stable, roughly orthogonal
  directions, e.g. an overall-size axis and a redistribution/placental axis, with
  no discrete groups: fetuses fill the plane continuously. Prior evidence from the
  parent project leans here (~2 stable directions, no clean clusters).
- **H3 - branching paths.** A shared trunk that splits into separable routes
  (e.g. early placental vs late constitutional), i.e. discrete modes / a tree, not
  a filled continuum.

The design must be honestly powered to land on H1 if H1 is the truth: the verdict
defaults to **inconclusive**, never to the H2 prior, whenever a discriminating
phase is missing or underpowered.

## What each phase tests (and which rung it discriminates)

| Phase | Module (registry in run_all.py) | Tests | Discriminates |
|-------|----------------------------------|-------|---------------|
| 1 | phase1_dimension | # latent axes surviving a permutation/parallel-analysis null on the pure fetal latent | H1 (1 axis) vs H2/H3 (>=2) |
| 2 | phase2_axes | bootstrap stability + loadings/meaning of the leading axes (is axis 2 real, and is it size vs redistribution) | supports/undermines the >=2 claim |
| 3 | phase3_clusters | discrete clusters vs a filled continuum (dip / gap / silhouette vs null) | continuum (H2) vs modes (H3) |
| 4 | phase4_branching | single monotone trunk vs branching routes (principal curve/tree, local intrinsic dim, partic. ratio along the risk field) | H2 (single field) vs H3 (branching) |
| 5 | phase5_outcomes | do the axes anchor to outcomes (sga, severe_sga, PEwithSGA, NICU, PartoPret) | external validity that axes are phenotype, not nuisance; does not by itself pick a rung |

Verdict logic lives in `run_all.decide_rung`:
- phase1 < floor -> inconclusive (no signal / underpowered)
- phase1 == 1 stable axis -> rung 1
- phase1 >= 2 and no clusters and no branching -> rung 2
- phase1 >= 2 and clusters and branching -> rung 3
- any required phase missing, or phase3/phase4 disagree -> inconclusive

Thresholds in `run_all.THRESH` are placeholders marked calibration-pending; each
phase should expose its own permutation/bootstrap pass flag and the orchestrator
reads those rather than re-deriving significance.

## Data limitations (these cap what any rung claim can mean)

- **Doppler is ONE late snapshot** (eco visit ~26-39 wk, per-fetus percentiles),
  not a trajectory. Early Doppler (20s/28s/32s) is ~0-20% populated. We cannot see
  the temporal redistribution cascade, only its late state.
- **No AEDF / absent-or-reversed end-diastolic flow, no oligohydramnios, no sFlt-1/
  PlGF.** The canonical early-placental-FGR markers are simply not in this data, so
  a "branching into a placental route" claim is structurally under-instrumented.
- **Biometry is longitudinal (4 visits) but efw_z is Hadlock-derived** from ac/hc/
  bpd/fl, so near-collinear by construction; it will pin a factor to overall size.
  Factor-1 "size" is partly mechanical, not a discovered axis.
- **Early visits are sparse** (biometry ~19-28% missing at 20s/28s/32s, ~1% at eco).
  No imputation anywhere: missingness is marginalised, complete-case only where a
  method demands it and the dropout is logged.
- **n = 977**, single cohort, no external replication. A 2nd or 3rd axis that is
  real here may not generalise.
- Known data-quality sentinels exist in the parent project (e.g. a handful of
  impossible efw/hc values); phases should screen rather than trust raw extremes.

## Confidence / calibration tracker (update after each run)

Verdict fields in results/summary.json are left empty until an actual run fills
them. Do not pre-write a rung here.

| Date | Phase 1 n_stable_axes | Phase 3 clusters | Phase 4 branching | Verdict | Confidence | Notes |
|------|----------------------|------------------|-------------------|---------|------------|-------|
| TBD  | -                    | -                | -                 | pending | -          | phases not yet implemented; orchestrator skips missing modules |
| 2026-06-29 | ~2 (linear) | none (dip p=0.99) | none detected | H2 multi-direction continuum (linear) | moderate-high | nonlinear/full-featureset battery, see below |

## 2026-06-29 update: nonlinear + full-featureset battery (results/nl/)

Ran 6 methods x {minimal curated, full 58-var} plus a calibrated control suite.
Verdict UNCHANGED from the prior linear result: stable ~2-D continuum, no curvature,
no clusters, no detectable branch. Full vs minimal does not change geometry (full
adds noise: lower trustworthiness, FA centile R2 0.42 -> 0.29).

CONFIRMED (passed a control):
- No curvature: nonlinear ID <= linear effective rank in all 5 sets; Isomap
  trustworthiness = MDS (geodesic = Euclidean). [intrinsic_dim, manifold_embed]
- No nonlinear reconstruction advantage at d=2 / k=6: KPCA matched-MVN null shows the
  ~0.01 trust edge is negligible and buys no recon; AE/VAE lose to FA at n=977 5-fold;
  GP-LVM's own linear-Gaussian negative control favors FA by 0.058. [kpca, ae, gplvm]
- No discrete clusters on minimal: dip p=0.99 unimodal; the branch route discriminator
  (route_auc_cv 0.96 on planted positive, 0.48 on null) validates the negative.
  forced-k=2 silhouette exceeds null p95 but fires identically for a planted branch
  (~0.52), so it is NOT a cluster discriminator -> elongated continuum, not modes.
- Full vs minimal does not change the geometry verdict.

PROVISIONAL / underpowered:
- Branch ABSENCE on the FULL set is uncontrolled: the dip+PH+PAGA battery failed to
  recover its own planted Y-branch (dip p=0.87). Calibrated controls exist for MINIMAL
  only. "No branch on minimal" is controlled; "no branch on full" is not.
- GP-LVM full-set recon hint (+0.018 R2, n=250, single split) is contradicted by the
  AE on the same set at n=977 5-fold (FA wins). Do not treat as a nonlinear signal.
- Downstream centile-R2 robustness edge on full (AE/VAE/GP > FA) is corroborated by
  two methods but is supervised robustness to a noisy 58-d soup, NOT geometry.

Calibration notes (keep honest, update as evidence accrues):
- Prior (pre-data, from parent project): H2 ~ moderate, H1 ~ plausible, H3 ~ low.
- After phase 1: ...
- After phases 3-4: ...
- Standing caveat: with no early Doppler / no AEDF, H3 cannot be strongly
  confirmed even if branching is hinted; report as "suggestive, under-instrumented".
