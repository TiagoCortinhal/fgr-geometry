# WP2 — Latent geometry of fetal growth restriction: results

*Standalone results narrative for the WP2 latent-geometry paper. Consolidated
2026-07-10 from the full project arc. Companion to the running log
`fgr_geometry_status.md`; this document is the ordered result set, that one is
the chronological handoff.*

Cohort: IMPACT trial, n≈977 fetuses (908 in the longitudinal modelling cohort,
927 with images). Biometry is genuinely longitudinal (~4 visits); Doppler and
echocardio are late single snapshots. GA dated by FUReco (eco), LMP fallback;
GA clock retrained on CiTUS dating (image→GA r=0.847).

---

## 1. The FGR latent is a stable, approximately-linear, ~2-D continuum

- **Dimensionality:** a second direction is real and stable (λ₂=2.47, bootstrap
  CI [2.31,2.87], clears parallel-analysis null 1.39; top-2 subspace stability
  0.993). A third direction is borderline, not established.
- **Topology:** unimodal (dip test ns); persistent homology shows only
  sampling-noise loops → a single arc, no branch.
- **Flow:** velocity field contractive (spectral abscissa −0.049, CI negative,
  p(divergent)=0) → trajectories converge, FGR sits at one END of the continuum,
  not on a divergent branch.
- **Clinical anchoring (deflationary):** the second axis carries a CPR/UtA
  redistribution signature but is entangled with size (principal angle 57.6°,
  low variance ~0.08 vs size R²=0.88) — one severity continuum with a weak,
  size-coupled redistribution component, NOT two clean mechanisms.
- **FGR vs PE:** the geometry separates a FETAL growth-restriction axis from a
  MATERNAL pre-eclampsia axis (PE-without-SGA is neutral on the SGA axis).

## 2. Temporal modelling recovers a real third dimension (growth dynamics)

Model progression static → trajectory-params → temporal GRU-VAE. The temporal
GRU-VAE recovers a generalizing 3rd dimension (held-out eff dim 2.80 vs static
~2.0); a GA-shuffle control collapses it (2.80→2.10) and flattens the r-rise
slope (0.023→0.008), proving the gain is time, not GRU capacity.

> **Provenance note (h=32 vs h=64):** the eff-dim 2.80 and shuffle-control
> numbers in this section were produced with GRU **h=64** (the width used during
> the temporal-progression battery). A later ablation showed h=32 reproduces the
> same ~2.8-D result and anchor AUCs with half the parameters and the tightest CI,
> so h=32 was adopted as the canonical width and ALL config-B lag results (§4, §6,
> §7 and the canonical model below) are at h=32. The two families of numbers are
> therefore not from a single run; the h=32/h=64 equivalence rests on that
> ablation, not on a re-run of the full temporal battery at h=32 — a clean re-run
> of the eff-dim/shuffle control at h=32 is the one outstanding confirmation.

## 3. Images are near-orthogonal to growth — but are the sole maturation clock

- Image (USFM) and biometry occupy near-orthogonal manifolds (all canonical
  r<0.3; 99% of image variance orthogonal to the size axis). Every attempt to
  make image content improve the growth model failed (soft-tissue, patch texture,
  plane-aware features, learned compression) — the size axis is not a direction
  in USFM image space.
- USFM image space IS the sole readable maturation clock (image→GA r≈0.85;
  biometry IG-21 z-scores have GA normalized out, r≈0.05). This motivates the
  appearance-age lag.

## 4. Appearance-age lag: how images legitimately enter the model

lag = image-predicted GA − dates GA, per image, aggregated per fetus. Valid
because the two differenced quantities are measured independently (a control:
an image-BMI "lag" is just prediction error, r=−0.965 with recorded BMI).

- **Group signal:** SGA fetuses look younger — lag SGA −0.39, AGA +0.21, LGA
  +1.10 (Kruskal p=6.6e-8); survives machine/date confounder adjustment.
- **Config B (lag REPLACES image embeddings in the GRU-VAE):** works
  (birth-pct r +0.39, LGA AUC 0.79); config C (lag as an EXTRA channel beside
  images) collapses to baseline. Lag is the right way for appearance to enter.
- **But raw lag carries no independent growth signal beyond biometry**
  (OOF biom+GA no-lag +0.338 vs +lag +0.341). Its representational value is
  (a) a second, independent maturation-timing axis (corr with size ≈+0.01), and
  (b) the size-orthogonalised **discordance residual**, which is the one place
  the appearance channel adds SGA signal beyond biometry (SGA_prenCONFIRMED
  +0.36, severeSGA +0.28, earlySGA +0.52; growth-trajectory r strengthens
  −0.15→−0.33 across gestation → mid-gestation growth-restriction etiology).

## 5. Negative results that bound the scope (all controls-backed)

- **Contrastive VAE does NOT expand dimensionality** (held-out eff dim 3.53→1.98)
  → the ~2-D latent is a property of the cohort, not the model.
- **Doppler brain-sparing / NICU / preterm:** all null after FDR — the
  clinically-hoped Doppler cascade is absent in this late-snapshot cohort.
- **Echocardio fusion (corrected z-scores, static side-channel):** does NOT
  improve the growth representation at any latent width (Z=6/8/16/32); cardiac
  is barely encoded (recon r≤0.15). Cardiac belongs as an OUTCOME variable.
- **Social/parental variables (12 tested):** 0/12 cohere with the 8D divisions.

## 6. Lag-pooling battery (44 variants) — the patch mean is not the bottleneck

- **Level A (patches→image):** no pooling beats the plain patch mean — fixed
  (max/top-k), LEARNED attention (clock 0.79, diffuse ~68/196 patches), and
  per-patch clock all lose. 9th consecutive image-null on spatial structure.
- **Level B (images→fetus) + multi-layer features:** the real gains.
  median/trim > mean; learned image-attention lifts lag↔SGA 0.155→0.181;
  multi-layer fusion (block1+6+12) lifts both clock (0.868) and lag↔SGA.
- **Best pooled lag = fuse(b1+6+12) + median aggregation.** Swapped into the
  longitudinal config-B GRU-VAE, it **sharpens the continuous growth axis**:
  birth-pct r 0.323→0.369, LGA AUC 0.773→0.795, at a small SGA cost
  (0.720→0.706). The better lag does its intended job (size/birthweight axis);
  the SGA tail is unchanged-to-slightly-worse.

## 7. 8D-division characterization (corrected echo)

KMeans on the frozen config-B 8D latent → 3 size-ordered divisions (C0 small
n=20 / C1 mid n=648 / C2 large n=240, silhouette 0.34 — a soft continuum). With
CORRECTED echo z-scores: **5/13 cardiac variables cohere across divisions**
(cardiac_area, RV_longitudinal, LV_basal, RV_basal, septum; q<0.1) plus
umbilical-artery Doppler. Chamber/wall size tracks division size; the small
division shows a notably thick septum (z+1.23) despite small chambers — a
growth-restriction remodeling signature. Divisions cohere on fetal size+cardiac,
weakly on placental Doppler, and not at all on social/parental variables.

---

## CANONICAL MODEL (decision, on the record 2026-07-10)

**Frozen config-B GRU-VAE** — per-visit sequence (biometry 5 IG-21 z + masks,
appearance-lag scalar+mask, GA), GRU h=32, latent Z=8, β=0.1, unsupervised
(reconstructs its own seq-mean inputs). birth-pct and SGA/LGA are evaluation-only,
never inputs.

- **Lag is kept** — it earns a dedicated latent axis (z0, +0.76; also dominant on
  z5 +0.58 and z6 −0.66) and supplies the maturation degree-of-freedom; removing
  it flattens the latent to size-only.
- **Echo is NOT a latent input** — it underperformed at every width; use it as an
  outcome/characterization variable.
- **The improved (fuse+median) lag is the recommended lag** when the target is the
  continuous birth-pct/LGA axis (birth-pct r +0.046, LGA +0.022); keep the original
  lag if SGA screening is the primary target (SGA −0.014). Both latents are saved.

Latents: `_lagB_traj.npy` (original lag), `_lagB_improved_traj.npy` (improved lag).
