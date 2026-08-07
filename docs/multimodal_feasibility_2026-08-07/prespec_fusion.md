# Prespecification — precision fusion, conditional dependence, decision fusion
Written BEFORE any test statistic was computed. Endpoints and directions fixed here.

## Cohort
IMPACT n=977. Image side: per-fetus pooled USFM emb_l5 -> PCs. Tabular: canonical 25 in
4 blocks (growth 5 / maternal 4 / Doppler 5 / cardiac 11).

## EXPERIMENT 1 — PRECISION FUSION
Claim: images predict per-fetus MEASUREMENT RELIABILITY, so Psi_j becomes Psi_j(image).

1a PRIMARY (longitudinal deviation). Fit a smooth growth curve per fetus over its 2-4
   biometry visits; deviation of each visit from its OWN curve approximates measurement
   error (a fetus cannot leave and rejoin its own trajectory). Target = |deviation|.
   Test: do images from that visit predict it, beyond GA?
1b ROBUSTNESS (cross-sectional residual). |residual| per cell from the factor model.
   Weaker: confounds "poorly measured" with "unusual fetus".
1c PAYOFF. Split Psi into per-fetus Psi_j(image), refit, score held-out coverage against
   the current 0.974 @95% / 0.927 @90%. SUCCESS = moves TOWARD nominal without hurting means.

CONTROLS: (i) maternal BMI alone as competitor -- images must beat plain BMI to add anything;
(ii) permutation null; (iii) DIRECTION CHECK -- predicted noise must INCREASE with BMI.
DECLARED IN ADVANCE: I expect this to work WEAKLY. Echo-quality grading correlates only 0.26
with images. A null here is a real answer.

## EXPERIMENT 2 — CONDITIONAL DEPENDENCE
Claim: blocks independent marginally, coupled within a subgroup.
Stratify on a variable OUTSIDE the panel (birthweight centile) -- NOT on EFW-z, which is in
the panel and would induce collider dependence among the remaining blocks.
Recompute Omega and cross-block R2 within strata.
CONTROLS: random-subgroup null at matched n; report n per stratum (complete cases ~240 total,
so ~80 per tertile -- underpowered, and a null must be reported as such).

## EXPERIMENT 3 — DECISION-LEVEL FUSION
Claim: near-orthogonal blocks combine MULTIPLICATIVELY; orthogonality is an asset here.
Per-block score -> combine -> compare against (a) each single block, (b) a joint model on all
25 variables at once.

ENDPOINTS FIXED NOW (5, no others will be added):
  E1 SGA_birth      (<p10)   expected n~169
  E2 severeSGA      (<p3)    expected n~61
  E3 AGA                     (the complement; reference group)
  E4 LGA_birth      (>p90)   expected n~78
  E5 NICU admission          expected n~46

DIRECTIONS FIXED NOW: for SGA / severeSGA the growth block predicts POSITIVE (smaller fetus ->
higher risk); Doppler predicts POSITIVE via redistribution. For LGA the growth block predicts
POSITIVE in the opposite tail. For NICU no block direction is asserted.

PRIMARY COMPARISON: fused AUC vs BEST SINGLE BLOCK, not vs chance. Beating chance is trivial;
the claim is that fusion beats its own best component.
MULTIPLICITY: 5 endpoints x 5 models. BH at q=0.10 across all tests.
NULL: label permutation, 400 draws, per endpoint.
STOP RULE: if fused does not beat the best single block on ANY endpoint, the decision-fusion
claim fails and will be reported as failed.
