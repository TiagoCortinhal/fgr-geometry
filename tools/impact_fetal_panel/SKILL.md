---
name: impact-fetal-panel
description: Load and analyse the IMPACT fetal-growth multiblock panel (977 fetuses; growth/maternal/Doppler/cardiac blocks plus USFM image features). Use for cross-block information budgets, factor/latent analysis, missing-data marginalisation, the residual data-quality screen, image maturation-lag or image-PC work, and the Kalman size-state model. Ships helpers that avoid the specific estimator traps this cohort punishes.
---

# IMPACT fetal-growth panel

Helpers for the 977-fetus IMPACT cohort. `kernel.py` auto-loads, so the
functions below already exist in your python kernel.

Run cells with `environment="fgrgeom"`. Call `fgm_setup()` first (it puts the
`fgm` package on `sys.path` and chdirs to the repo).

## Loading

```python
fgm_setup()
P = fgm_panel()                       # X, Z, cols, blocks, fids, mu, sd
lag, k = fgm_image_lag(P["fids"])     # per-fetus maturation lag + image count
IMG, pca = fgm_image_pcs(P["fids"], n_pc=5)
```

`fgm_panel()` masks `|z| > SENTINEL_Z` on the biometry columns to NaN — the
correction that makes a separate head factor form. Omitting it silently
changes the factor inventory.

## Cohort facts that decide experiment design

- **Blocks are near-orthogonal.** Cross-block out-of-fold R² ≈ 0.023–0.04;
  within-block ≈ 0.51. A random partition of the same variables scores ~0.375,
  putting the clinical partition at the bottom of the null. This is the
  paper's central number — quote the ridge value, not OLS, and never the
  in-sample one.
- **Complete cases are 240 of 977** across all 28 variables. Any complete-case
  method discards three-quarters of the cohort, systematically the sick end.
  Use the missing-data likelihood.
- **Imaging is single-session** — 98.5% of fetuses have one image study-date.
  No image representation can be longitudinal here. Biometry has 2–4 visits;
  Doppler and cardiac are one row per fetus.
- **Nonlinear detection floor is R² ≈ 0.10 at n ≈ 883.** Below that a null
  means underpowered, not absent.

## Estimator traps this cohort punishes

Each of these produced a wrong published-looking number before being caught.

**Bootstrap inside cross-validation.** Resampling rows with replacement
duplicates fetuses across folds and leaks train into test; intervals come out
excluding their own point estimate. Use the spread over independent CV splits
(`fgm_cv_r2` with several seeds).

**`1 - U/(n1*n0)` for AUC.** `U/(n1*n0)` is already P(a>b). Use `fgm_auc`.
The tell is every AUC below 0.5, including size for smallness.

**Column-wise permutation nulls.** Destroys within-block structure as well as
cross-block, so the test is trivially easy. Use `fgm_block_shuffle_null`,
which permutes rows within each block.

**Circular controls.** If a variable was among the model's outputs, do not
then evaluate whether the model finds it — refit without it first. This
inverted both a GPLVM ARD result (0.46 vs 0.19 nulls became 0.205 vs
0.175–0.241) and its predictive arm.

**Informative initialisation.** A latent seeded from a linear fit inherits the
linear signal. Always refit from a random start as the gate; a signal that
collapses (e.g. +0.040 → −0.002) was inherited, not learned.

**Pooled one-way ICC on the image lag.** Mixes single-image fetuses (lag
variance 0.98) with repeated-measure ones (6.54) and returns exactly 0. Use
the ≥4-image restricted estimate (~0.06) or assumption-free split-half
(0.27–0.53 rising with image count).

**Reading numbers off truncated output.** Print counts explicitly rather than
summing displayed rows.

## Controls

```python
fgm_positive_control(fit_predict)          # plant known signal, check recovery
fgm_block_shuffle_null(Z, groups, stat)    # within-block row permutation
tc, dtc, omega = fgm_o_information(Z, groups)   # target-free redundancy/synergy
fgm_omega_null(Z, groups)                  # the above + its null band + verdict
```

## Beyond the canonical panel

`fgm_panel()` returns only the paper's 25 variables. For the whole registry:

```python
A = fgm_all_tabular()                 # every numeric column above 50% coverage
V = fgm_visit_matrix("efw_z_ig21")    # wide per-visit matrix, longitudinal arm
fgm_classify_registry_columns(A["cols"])   # administrative vs clinical split
fgm_canonical_block_vars()            # the paper's 25 by block
fgm_derived_variables()               # algebraic dependencies to drop first
```

Registry scale: 1,431 columns, 1,013 numeric, **564 above 50% coverage**, but
only **5 fetuses complete across all 564** — the full multiplet is not
estimable. `Percentil_Sapse` (0.478) is the panel's coverage bottleneck; one
variable halves the complete-case count.

`fgm_all_tabular` routes every column through `fgm_eurofloat`, because this
registry stores many numerics as European-decimal strings (`'25,97'`) that
`to_numeric` silently turns into NaN — a defect that once discarded most
values of a key column and produced a sample size reported as real coverage.
Only swap the separator when the comma is acting as a decimal point; blanket
stripping corrupts genuine thousands separators.

**Rank-deficiency is the killer.** Always route registry variables through
`fgm_decorrelate` (or use `fgm_omega_report`, which does it and returns the
conditioning diagnostics) before computing Omega on more than ~15 of them.
A near-zero covariance eigenvalue sends the log-determinant to -inf and Omega
to a large arbitrary negative number reading as spectacular synergy — this
registry produced Omega = -31 that way, from coded duplicate columns.

**Established O-information regimes on this cohort** (all target-free):

| multiplet | TC | Omega | verdict |
|---|---|---|---|
| 4 repeated visits of one biometry measure | +1.09 | +0.240 | redundancy |
| 4 different biometry variables, one visit | +0.47 | +0.056 | redundancy |
| raw registry, 10–28 independent vars | +2.4 to +6.6 | +0.29 to +0.53 | redundancy (clerical — see below) |
| curated 4 clinical blocks | +0.77 | −0.014 | **inside null** |
| 5 blocks incl. image | +1.08 | −0.003 | **inside null** |

Repeated measures and the raw registry are both redundancy-dominated (the
latter from clerical duplication and derived variables, not biology). The
curated clinical panel sits inside its null band — neither redundant nor
synergistic — which is the paper's result, bracketed by two real-data
positive controls.

**The raw-registry redundancy is clerical, not biological.** Coverage-ranking
selects mostly administrative fields — 50% of the top 40 were dates,
identifiers or one-hot ethnicity dummies (which sum to 1 by construction;
visit dates are collinear with gestational age). Route column names through
`fgm_classify_registry_columns` and keep the clinical set before interpreting
any registry-wide verdict. The result still demonstrates the estimator detects
redundancy where redundancy exists — it says nothing about physiology.

**Within-block synergy is algebraic, not physiological.** Removing the derived
variable collapses it: maternal −1.596 → −0.0001 without BMI, growth −0.133 →
+0.056 without EFW, cardiac −0.229 → +0.192 without MPI.

**Reading Omega with singleton multiplets:** the block-shuffle null band is
very tight (~1e-4) when the multiplet is individual variables rather than
blocks, so almost any real data reads as significant. Report TC alongside
Omega for scale, and expect within-block synergy to track known algebraic
dependencies (EFW from Hadlock inputs, BMI from weight and height, MPI from
ICT/IRT/ET) rather than physiology — test by refitting without the derived
variable.

Gate every null on a positive control at the settings you will actually
report — not at settings you tuned elsewhere.

## The image↔cardiac cross-modal direction — DID NOT REPRODUCE

An archived summary reports held-out canonical correlation **0.248** as the
FULLY ADJUSTED value, strengthening along the ladder (0.218@GA → 0.225@GA+size
→ 0.248@GA+size+biometry), perm p=3e-4. **A direct re-run does not reproduce
this.**

| cardiac representation | raw | GA | GA+size | +biometry | perm p |
|---|---|---|---|---|---|
| raw echo morphology (7), n=751 | **0.248** | 0.018 | 0.042 | 0.046 | 0.20 |
| canonical Percentil_* (11), n=660 | 0.107 | 0.104 | 0.105 | 0.093 | 0.062 |

The 0.248 is recovered only as the **UNADJUSTED** value, and it collapses
five-fold on GA adjustment. Neither arm clears its permutation null.

**Mechanism:** raw echo params are unscaled measurements in cm and grow with
gestation (mean |r| with GA = 0.248; Circunf_cardiaca 0.40), while image
appearance also encodes GA (pooled-USFM PC1 r = −0.276). The unadjusted
correlation is a shared maturation channel. Percentile-scored columns are
age-normalised (mean |r| = 0.040) and show no inflation.

This is a failure to reproduce with current tooling, not proof the original
was wrong — the original image representation and CCA variant could not be
reconstructed. **Do not build on the 0.248 without resolving it.**

```python
GA = fgm_ga_at_echo(P["fids"])
fgm_ga_leakage(Y, GA, names)        # confound audit BEFORE any cross-modal test
fgm_crossmodal_ladder(IMG, Y, GA, EFW, BIO)   # read the LADDER, not the raw value
```

```python
E, enames = fgm_echo_raw(P["fids"])        # raw morphology params
Xa = fgm_residualise(IMG, [GA, EFW, BIO])  # fill NaNs FIRST
cc = fgm_heldout_cca(Xa, Ya)               # PCA fitted inside the fold
```

Cohort: 918 fetuses have a pooled image embedding; 664 also have ≥10/11
canonical cardiac; 283 have all 11.

## Screening images against the whole registry

```python
GA  = fgm_ga_at_echo(P["fids"])
V   = fgm_registry_variables(P["fids"])        # name -> vector, admin fields dropped
COV = np.column_stack([np.ones(n), GA, EFW, BIO])
rows = fgm_image_screen(IMG, V, COV)           # per-variable, adjusted
rej, q = fgm_bh([r["p"] for r in rows])        # BH is mandatory at this width
```

Screening hundreds of variables without BH is p-hacking; the user has named
this risk explicitly. Always report the GA leakage of any hit alongside it —
`fgm_ga_leakage` — because a maturation channel produces large unadjusted
correlations on any block containing unscaled measurements.

**Maternal is an acquisition confound, not a finding.** The maternal block
aligns with images at 0.384 fully adjusted (p=0.002, n=913), but it decomposes
entirely into body habitus: BMI alone 0.491, weight 0.468, versus age 0.038 and
height 0.062. Image PC3 carries it (r=0.38 with BMI). Residualising the images
on BMI collapses the whole block to **−0.014**. Higher BMI degrades ultrasound
penetration — the encoder is reading acquisition conditions off the pixels.
Treat any image↔tabular hit as suspect until BMI-residualised.

## Exhaustive multimodal survey — the answer is no

477 individual registry variables tested against pooled-USFM image PCs, each
adjusted for GA + size + biometry (held-out ridge, 5-fold). **Median adjusted
r = −0.005; only 7.3% exceed 0.15.**

Every hit that clears a null is an **acquisition property**:

| rank | variable | adjusted r |
|---|---|---|
| 1 | Vis1TR_BMI | 0.493 |
| 2–10 | nine more BMI / weight / waist encodings | 0.44–0.49 |
| best non-habitus | Vis3TR_MAP (blood pressure) | 0.215 → **0.037** after BMI |

Block level: maternal 0.384 (p=0.002) — the only block clearing its null, and
it decomposes to BMI 0.491 / weight 0.468 versus age 0.038 / height 0.062.
Growth −0.179, Doppler −0.040, cardiac 0.093. Residualising the images on BMI
takes maternal to **−0.014**.

```python
BASE = fgm_nuisance_design(GA, EFW, BIO)
fgm_confound_ladder(y, IMG, BASE, [("BMI", bmi), ("presentation", pres), ("quality", cal)])
```

**What the frozen encoder actually encodes is acquisition condition** —
maternal habitus sets ultrasound penetration, and the embedding reads it off
the pixels. That is the useful finding here, and a caution for anyone applying
foundation-model embeddings to clinical ultrasound.

## Settled null results — do not re-run without a new angle

The image channel has been tested as: maturation lag, lag magnitude,
pre-specified interactions, boosted trees, Gaussian process, GPLVM (MAP, ARD),
masked VAE, 5-PC block, Kalman measurement channel, and per-visit GA-aligned
PCs. All inert in the paper's own units (posterior narrowing ≤ 0.28%, coverage
unchanged, screen a trade not a gain). The lag loads +0.117 on the size state;
image PC1 is 64% of variance and loads +0.008. Image PC3 correlates 0.34 with
maternal BMI — an acquisition condition, not physiology.

Evaluate image contributions in the paper's units (posterior width, coverage,
screen), not AUC.
