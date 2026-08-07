# Multimodal feasibility on the IMPACT cohort — 2026-08-07

Exhaustive assessment of whether the ultrasound images carry any signal aligned
with the tabular registry, plus a target-free information decomposition of the
tabular blocks themselves.

## Headline results

**1. No clinically meaningful multimodal signal.** 477 individual registry
variables tested against pooled-USFM image PCs, each adjusted for gestational
age + estimated weight + biometry. Median adjusted r = **-0.005**; only 7.3%
exceed 0.15. The top ten hits are near-duplicate encodings of maternal body
habitus (BMI 0.493, weight 0.468, waist 0.443). Residualising the images on BMI
collapses the maternal block from +0.384 to **-0.014**. The best non-habitus
hit, third-trimester blood pressure (0.215), collapses to 0.037 (p=0.12) once
BMI enters.

**What the frozen encoder encodes is acquisition condition** — maternal habitus
sets ultrasound penetration and the embedding reads it off the pixels.

**2. The blocks are neither redundant nor synergistic.** O-information
(target-free, no outcome required) puts the curated 4-block clinical panel at
Omega = -0.014, inside its block-shuffle null, and the 5-block panel with images
at -0.003. Bracketed by two real-data positive controls: repeated visits of one
measure read redundancy-dominated (+0.240), as does the raw registry (+0.29 to
+0.53, but clerically — half the top-40 are dates and identifiers).

Not redundant => an absent panel cannot be reconstructed => marginalisation is
the honest operation. Not synergistic => no fusion architecture is leaving
information on the table.

**3. The archived image<->cardiac 0.248 does not reproduce.** It is recovered
only as an UNADJUSTED value and collapses five-fold on GA adjustment (0.248 ->
0.018 -> 0.046, p=0.20). Raw echo parameters are unscaled cm measurements that
grow with gestation (mean |r| with GA = 0.248) and image appearance also encodes
GA (PC1 = -0.276) — a shared maturation channel. Not proof the original was
wrong; the original image representation could not be reconstructed.

## Files

| file | contents |
|---|---|
| `multimodal_survey.{json,png}` | the 477-variable screen, block ladders, BMI decomposition |
| `cardiac_reproduction.{json,png}` | both cardiac representations, full ladders, GA mechanism |
| `o_information_blocks.json` | 5-block target-free decomposition + controls |
| `o_information_all_tabular.json` | registry-wide arms, rank-deficiency artefact, corrections |
| `lag_reliability_corrected.json` | split-half reliability; supersedes the invalid pooled ICC |
| `lag_variance_decomposition.json` | the superseded ICC=0 result, kept with its correction |
| `kalman_{lag_channel,image_pcs}.json` | image channels on the longitudinal size state |
| `image_pc_block.json` | 5-PC image block; PCs cannot have within-block structure |
| `vae_{architecture_sweep,imputation_competitor}.json` | masked VAE vs marginalisation, tuned both arms |
| `gplvm_nonlinear_latent.json` | nonlinear unsupervised latent, null after decircularising |
| `scripts/` | every analysis script, as run |

## Tooling

Published as the skill `impact-fetal-panel` (30 helpers). SKILL.md documents
every estimator trap that produced a wrong number during this work: bootstrap
inside cross-validation, the AUC complement bug, column-wise permutation nulls,
circular controls, informative initialisation, rank-deficient log-determinants,
the pooled-ICC mixture violation, and European-decimal parsing.

## Caveats

Six agent claims required correction during this work and are documented inline
in the JSONs. Everything reported here is what survived recomputation. The
screen is univariate per variable; the pooled per-fetus image representation is
one of several possible; 477 of 1013 numeric columns clear the coverage filter.
