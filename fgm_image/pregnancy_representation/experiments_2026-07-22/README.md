# Pregnancy-representation experiments (2026-07-22)

Adversarial validation of what frozen fetal-ultrasound foundation encoders (USF-MAE ViT-B/16,
DINOv2-S) encode about pregnancy, on per-visit image embeddings (540 fetuses, ~2600 visits, GA 12-42wk,
clinical longitudinal cohort). Fetus-grouped CV throughout (no fetus in train+test). `results/` is
git-ignored — code + this README only; result JSONs/figures live in the artifact store.

## Headline finding
A **cross-encoder-validated umbilical-artery (placental) direction** exists in the image-embedding
space, orthogonal to gestational age and fetal size:
- held-out CCA image<->Doppler, GA-residualized: USF-MAE cc=0.366, DINOv2 cc=0.315 (image-shuffle
  p=0.002 both; only cross-encoder survivor besides the trivial biometry/size channel).
- dominated by umbilical-artery Zscore_AU (|loading|~0.95); NOT size (birthpct r~0.03); holds under
  biometry adjustment; PE maternal control at chance (AUC~0.48).
- stable across gestation (early cc 0.243 / mid 0.226 / late 0.193); monotonic per-fetus gradient
  along UA status (tertile means -0.23 / -0.02 / +0.41, high-vs-low p=1.5e-22; severe-UA AUC 0.73).

Nulls (reportable): outcome magnitude (SGA/LGA/birthpct), cardiac, maternal are cross-encoder nulls;
USF-MAE reconstruction-error atypicality is null for growth; decile trajectory geometry does not fan.

## Files
- `characterize_ua_direction.py` — reproduces the UA-direction characterization (overall CCA,
  GA-window stability, UA-status gradient). `python characterize_ua_direction.py` in env fgrgeom.
- `contrastive/` — the contrastive-trajectory experiment (does a supervised/contrastive loss build a
  better-organized latent?):
  - `traj_model.py` — GRU trajectory VAE, 3 variants: (a) unsupervised recon+KL; (b) +**SupCon**
    (Khosla 2020) on birth-percentile deciles applied to a projection head of the latent mean; (c)
    +regression head to birth percentile. SupCon: L2-normalize projected latents, scaled cosine
    similarity /temp=0.1, for each anchor pull same-decile fetuses together vs all others
    (log-softmax over positives). Labels enter ONLY the loss, never the encoder input.
  - `run_eval.py` — fetus-grouped 5-fold; PCA(48) fit on TRAIN visits only; deciles from TRAIN
    birthpct bins; headline metric = pooled held-out birthpct-r via a Ridge train->test readout;
    per-fold held-out silhouette by decile. Modes: real / label_shuffle / ga_shuffle.
  - `battery.py` — 5 seeds x {real, ga_shuffle} + N label-shuffle nulls; bootstrap CI + perm p.
  - `analyze.py` — aggregates battery_progress.json into the reported numbers.
  - RESULT: contrastive/regression give a small GENERALIZING readout axis (held-out r~0.17-0.18 vs
    label-shuffle null ~0.08, p=0.010) but NO decile geometry (silhouette negative), and the effect
    is temporal (GA-shuffle collapses it 0.18->0.06). Contrastive carves a faint linear axis; it does
    not manufacture fanned-out organization.

## Method notes
- Encoders are FROZEN; we never fine-tune. Embeddings extracted once (see
  fgm_image/pregnancy_representation/).
- "GA-residualized" = regress out GA + GA^2 from both blocks before CCA, so alignment is not the
  maturation clock re-expressed.
- Cross-encoder replication (survive in BOTH USF-MAE and DINOv2 at p<0.05) is the multiplicity filter.
- Latent-architecture sweep (separate): beta-TCVAE beats FA/PCA on GA by only +0.019 (GRU temporal
  pooling, not the loss; TC/KL weights inert); non-GA monotonicity at shuffle floor for every method.
  Transparent FA+varimax is ~98% as GA-organized — nonlinear/contrastive complexity is not worth it.
