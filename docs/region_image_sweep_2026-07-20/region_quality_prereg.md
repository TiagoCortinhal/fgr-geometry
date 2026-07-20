# REGION-QUALITY Pre-Registration — joint biometry+image trajectory latents (frozen before region metrics)

## Goal (user, explicit): NOT AUC. A DISTINCTIVE latent space with COHESIVE, ANALYZABLE REGIONS. Introduce images
into the GBTM/trajectory framework and ask whether they create NEW distinctive regions or refine existing ones.

## What GBTM currently is (answer to user): UNIVARIATE, EFW-z ONLY (hlme: efw_z ~ ga_c + ga_c^2). 3 ordered classes
in (level@28, slope, curv) space: faltering(-0.59,-0.034) / tracking(+0.57,+0.027) / accelerating(+1.96,+0.074).
Data shows HC/AC/FL slopes are only weakly correlated with EFW (r 0.21-0.30) so multivariate adds real info; and the
faltering class has head-sparing heterogeneity (32/65 HC-spared, 20 AC-spared, 13 symmetric) a multivariate model can split.

## Substrate: joint_traj_substrate.csv (900 fetuses): per-fetus trajectory coeffs for efw/hc/ac/fl/bpd z (level@28,
slope,curv) + image-maturation-lag (level n=875, slope n=196 ONLY — image trajectory is mostly a LEVEL, treat slope cautiously).
Plus image_substrate.npz (pooled + sequence USFM) for DL models. Eval-only labels: gbtm_class, birth, sga_p10, severe, lga, pe_any.

## REGION-QUALITY METRICS (the deliverable axis — NOT classification AUC)
For every candidate latent, quantify REGION STRUCTURE, label-free where possible:
 1. distinctiveness: silhouette / Calinski-Harabasz of the discovered regions; dip-test for multimodality (are regions
    real or is it one continuum?); gap statistic for k.
 2. cohesion: within-region homogeneity on HELD-OUT variables not used to build the latent (Doppler CPR/UtA/UA, birth-pct)
    vs a GA+size-gradient null — a cohesive region differs on things beyond what defined it.
 3. reproducibility: subsample ARI of region assignment (must be >0.6 to be a real partition, cf GBTM EFW ARI 0.86).
 4. NEGATIVE CONTROLS: GA-shuffle (within fetus, destroys trajectory), IMAGE-shuffle (permute image channel across fetuses
    -> if regions unchanged, images add nothing), label-shuffle, matched-random-capacity. PE-any = maternal negative control.

## COMPARISON (the core question): region structure of biometry-only vs each image-augmented model.
 Verdict per model: ADDS DISTINCTIVE REGIONS (new cohesive region absent from biometry-only, survives image-shuffle) /
 REFINES (splits an existing region into cohesive subregions) / NO CHANGE (image-shuffle reproduces it) / DEGRADES.

## Model portfolio (waves, kept independent per CDC protocol)
 W1a: multivariate GBMTM (multlcmm HC/AC/FL/EFW z) — biometry-only stronger baseline; does it split faltering into sym/asym?
 W1b: multivariate GBMTM + image-maturation-lag as an added trajectory channel — does the image channel add/refine regions?
 W2a: joint biometry+image trajectory VAE (GRU over per-visit [biometry z | image features]) — latent region analysis.
 W2b: trajectory-coefficient + image fusion clustering (GMM/HDBSCAN on joint_traj_substrate) — region discovery.
 W3: red-team the 2-3 most promising latents (image-shuffle, GA-shuffle, cohesion, reproducibility) + 3D interactive latents.

## Outputs: per-model region-distinctiveness + cohesion metrics with GA-shuffle/image-shuffle controls; comparison
biometry-only vs image-augmented; 3D interactive latents of the 2-3 most promising; results JSON + memory doc verdict.
SGA-CONFIRMED BANNED. Canonical config-B UNCHANGED. Labels eval-only.
