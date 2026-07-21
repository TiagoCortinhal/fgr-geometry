# A Longitudinal Representation of Pregnancy — Research Plan (design, no training yet)

## 1. The thesis
An organized latent manifold of pregnancy spanning the full gestational range we have, in which
**movement along a direction corresponds to monotone increase/decrease of interpretable clinical
variables** (a gradient field, not clusters). Built from:
- a **minimal subset** of tabular variables (found, not assumed), and
- **new image-derived features** from encoders we have not used, in the spirit of the GA-clock and
  its appearance-age "lag" — derivations, not raw embeddings.

The paper's contribution is *understanding*: which variables and which image-derived axes organize
pregnancy, and what the interpretable directions mean.

## 2. What we've already exhausted (do not repeat)
- **USFM** (ultrasound foundation features) — primary features to date; growth is near-orthogonal to
  its image manifold; it IS a maturation clock (GA) and gives the confirmed image↔cardiac direction.
- **Radiomics (raw texture)** — just run on 21k IMPACT frames: recovers the same *directions* as USFM
  (cardiac cross-modal, abdominal-LGA, continuum-by-plane) but captures **less** of every signal.
- **ResNet50-ImageNet** — weak static baseline.
- Raw-pixel appearance→growth is a settled **null**; the live signals are maturation (GA) and the
  cross-modal cardiac direction.

## 3. New encoders to bring in (grounded in current literature, Oct–Nov 2025)
| Encoder | Family | Domain | Weights | Why it's new signal |
|---|---|---|---|---|
| **USF-MAE** | Masked-autoencoder ViT-B/16 | ultrasound (author README: ~370k img / 46 datasets "OpenUS-46") | checkpoints supplied locally by user (~/Downloads, 100ep + 500ep .pt files, ~448MB each, verified loadable as MAE ViT-B/16) | reconstruction SSL → gives per-patch **reconstruction-error** (atypicality) for free; different objective from USFM |
| **FetalCLIP** | Vision-language (CLIP) | **fetal** ultrasound | repo public (biomedia-mbzuai/fetalclip); weights were HF-CDN-blocked before — retry | fetal-domain-specific semantics; text-anchored axes |
| **DINOv2** | Self-distillation ViT | natural images | downloads freely (torch.hub/HF) | different inductive bias; strong medical feature extractor even frozen; independent "opinion" for cross-encoder agreement |

Cross-encoder **agreement** is itself the robustness test (as radiomics-vs-USFM already served):
a direction is only trusted if ≥2 encoders recover it.

## 4. The image-derivation menu (the creative core — scalars/axes, not raw embeddings)
Each is a *derived* quantity designed to organize the latent, extending the GA-clock/lag idea:
- **A. Multi-encoder GA clocks + lags** — per-encoder appearance→GA regressor; lag = residual
  (appearance older/younger than dates). Cross-encoder lag agreement.
- **B. Organ-specific maturation clocks + differential** — separate GA clocks for cerebral / abdominal
  / femur; **brain-age − body-age** differential (an appearance head-sparing axis).
- **C. Appearance velocity / acceleration** — derivative of the appearance representation along GA
  (longitudinal; needs the clinical repeated-visit set).
- **D. Developmental typicality** — Mahalanobis distance from the GA-conditional appearance
  distribution; **USF-MAE reconstruction error** as an independent atypicality scalar.
- **E. Trajectory curvature / deviation** — how a fetus's appearance path bends from the mean
  progression over gestation.
- **F. Prediction uncertainty** — image-quality-derived GA uncertainty (validated idea, npj Dig Med
  2025) as a reliability/derived feature.
- **G. Cross-modal projection** — projection onto the confirmed image↔cardiac direction as a scalar.

## 5. Minimal tabular subset (found, not assumed)
- Candidate pool: biometry z (HC/AC/FL/BPD/EFW × visit), Doppler (UA/UtA/MCA/CPR), a few maternal.
- Selection = smallest set that best **organizes** the manifold: stability-selection / elastic-net
  toward the latent, or greedy forward selection under the organization criterion (§7), with a
  fetus-grouped hold-out so the subset generalizes.

## 6. The latent model — options to decide together (no VAE-by-default)
Three candidates, compared on the same organization criterion:
- **(a) Factor analysis / PCA + varimax rotation** — linear, fully transparent axes; the honest baseline.
- **(b) Diffusion maps / PHATE** — nonlinear manifold built for continua/trajectories; natural for a
  smooth GA progression with branch directions.
- **(c) GRU-VAE trajectory latent** — the project's established sequence model (config-B lineage).
The choice is a design discussion, not a default.

## 7. "Organized" — the metric
- **Monotonicity**: each retained tabular variable increases/decreases monotonically along a latent
  axis (Spearman gradient magnitude, sign-consistent).
- **GA ordering**: gestational age runs smoothly along the principal trajectory (low crossing/energy).
- **Direction interpretability**: each axis carries a nameable gradient (size, maturation, head-body,
  placental, cardiac). Reported as gradients with strength — within-noise stated as within-noise.

## 8. Adversarial validation (the spine)
- GroupKFold **by fetus** throughout.
- Every image-derivation must show **incremental organizing value beyond GA + size** (the recurring
  trap: most image signal is just maturation or size).
- Negative controls: **image-shuffle**, **GA-shuffle**, **PE as maternal negative control**.
- **Cross-encoder replication**: a direction must survive in ≥2 encoders to be reported.
- Multiplicity-aware across the derivation × encoder × axis grid.

## 9. Compute reality (RESOLVED — CPU is feasible, benchmarked 2026-07-21)
No GPU connected; machine is CPU-only, 24 GB RAM. **Benchmarked, not assumed:** DINOv2-S runs at
18 ms/img on 10 CPU threads → ~14 min for 47k clinical frames, ~20 min for all 68k. USF-MAE ViT-B is
~2–3× slower per frame but still well under an hour for the full set. So extraction is **minutes-to-
an-hour per encoder on CPU — GPU is NOT required.** The real constraint was weight-host allowlisting
(DINOv2 fbaipublicfiles + Google Drive for USF-MAE now granted; USF-MAE checkpoints supplied locally).
Practice: shard-to-disk (proven daemonized pattern), one encoder at a time, gated-fetal frames only.

## 10. Phasing
- **Phase 0 — feasibility probe (cheap, first):** confirm each model's weights download in-sandbox and
  extract on a 200-image subset; verify the embeddings separate plane/GA sanely. De-risks before hours.
- **Phase 1 — extraction:** full per-frame embeddings for the chosen encoders, sharded to disk.
- **Phase 2 — derivations:** build the A–G battery → per-fetus / per-visit derived feature table.
- **Phase 3 — minimal tabular subset** selection.
- **Phase 4 — assemble the pregnancy latent;** compare (a)/(b)/(c) on the organization metric.
- **Phase 5 — adversarial validation;** which directions survive across encoders.
- **Deliverable:** the organized pregnancy manifold, its interpretable directions, the minimal variable
  set, and which image-derivations earned their place — as a paper-ready results set.

## 11. Decisions needed before Phase 0
1. **Encoders**: all three (USF-MAE + FetalCLIP + DINOv2) for cross-encoder agreement, or a subset?
2. **Cohort / GA span**: clinical longitudinal (widest span, within-fetus) as primary; IMPACT for
   cross-sectional density; or both pooled?
3. **Compute**: CPU extraction is benchmarked-feasible (§9), so no GPU needed — confirm we proceed on CPU.


---

## Progress log

### 2026-07-21 — Phase 0 complete, full extraction launched
**Encoders obtained & feasibility proven (CPU):**
- DINOv2-S (384-d): weights via fbaipublicfiles (net-access granted); 18 ms/img.
- USF-MAE ViT-B/16 (1536-d embed, decoder intact): checkpoints user-supplied at ~/Downloads
  (100ep + 500ep .pt, ~448 MB); loads exactly (0 missing/unexpected keys).

**Phase-0 probe (180 frames, balanced plane, GA 12–39 wk, both cohorts):**
- DINOv2 GA r=0.61, plane AUC 0.90.
- USF-MAE embeddings GA r=0.79, plane AUC 0.95 (strongest maturation reader yet).
- USF-MAE per-patch recon error is non-degenerate & structured, BUT Phase-0 exposed a **confound**:
  MAE reconstructs fetal tissue well (low error) and dark background/cone edges poorly (high error),
  so raw global error ≈ framing/zoom + plane, not tissue atypicality. Error-map fig: usfmae_error_maps_probe.png.

**Atypicality derivation ladder (decided):** (1) raw error [neg control], (2) ROI-restricted
(cone & ~caliper), (3) **GA+plane-conditioned tissue error** [candidate signal]. ROI alone insufficient
(raw-vs-ROI r=0.88; plane still dominates) → conditioning required. Per-patch error + tissue mask saved
per frame so (3) is built post-hoc.

**Masks:** clinical cone_mask (52,172 by name) + inpaint_mask (caliper/annotation, 2.4% on) →
ROI = cone & ~inpaint (validated clinical_curated_mask_check.png). IMPACT cone by name + caliper by
DICOM-UID (12,758/20,413 matched; rest = no-caliper frames).

**Full extraction RUNNING (cohort decision: Clinical + IMPACT):** extract_image_derivations.py
daemonized, 50,670 frames (clinical 30,257 + impact 20,413). Per frame: dino(384) + mae(1536) +
err_raw + err_roi + err_patch(196) + tissue(196) → handoff/imgderiv/ sharded 2000. ~2.2 frames/s,
ETA ~6 h, 0 errors, smoke-tested. Full-battery timing is hours on CPU (not minutes; GPU not needed).

**Next after extraction:** derivation battery (A–G; USF-MAE atypicality is the priority/novel piece) →
minimal tabular subset → pregnancy-latent assembly (FA / PHATE / GRU-VAE compared) → adversarial
validation (GroupKFold-by-fetus, incremental-beyond-GA+size, image/GA-shuffle, PE neg control,
cross-encoder ≥2 agreement).
