# Region-quality sweep: can images add distinctive regions to the GBTM/trajectory latent? (2026-07-20)

Multi-agent adversarial sweep. Goal: NOT classification AUC, but a **distinctive latent with cohesive
analyzable regions**, and whether introducing ultrasound images into the GBTM/trajectory framework
creates new distinctive regions.

## Headline
**No.** Across 6 model families, images add no new distinctive regions to the biometry-trajectory latent.
The most distinctive analyzable region model is the **biometry trajectory** (univariate EFW-z GBTM: 3 cohesive
phenotypes — faltering / tracking / accelerating — subsample ARI 0.86–0.88). Images are a maturation axis
orthogonal to the growth-trajectory regions and to all clinical tabular data.

## What GBTM is
A trajectory model. Latent = per-fetus EFW-z growth-curve coefficients (level@GA28, slope, curvature);
the 3 classes are ordered regions in that space. It used **EFW-z only** (univariate). The multivariate
GBMTM (HC/AC/FL/EFW) is a valid stronger baseline but adds no new regions (faltering is symmetric, no
head-sparing split); its coefficient space is a continuum, not new clusters.

## Model families (all NO CHANGE)
1. Joint biometry+image trajectory VAE — no new region; image-shuffle null; Procrustes shows image doesn't reshape latent.
2. Trajectory-coeff + image-lag fusion clustering — partition identical (ARI 0.99), fails image-shuffle.
3. Stacked image-PCA channel — no region added.
4. Single-component PCA sweep (only PC1..PC8) — all no change (PC8 apparent effect = 5 outliers, artifact).
5. Cumulative image-dimensionality sweep (images reduced to dim=1..8, up to 87% var) — no change at every dim.
6. Gold-standard multivariate multlcmm (biometry-only vs +image-lag) — image channel inert, ARI=1.000 vs shuffled.

## Files
- `region_quality_prereg.md`, `image_dl_prereg.md` — frozen protocols
- `tables/image_dl_results.json` — full per-model region metrics + verdicts
- `tables/*.csv` — per-fetus latents + region assignments (908 fetuses)
- `plots/*.png` — per-model region figures + recovery comparison
- `models/*.rds` — fitted multivariate GBMTM models (R lcmm)
- `latents_3d/*.html` — interactive 3D latents (store-only, see MANIFEST.md for artifact version_ids)

See MANIFEST.md for the artifact-store version_id of every deliverable (the large `.npz`/`.rds`/`.html`
are git-ignored per repo hygiene and live durably in the artifact store).
