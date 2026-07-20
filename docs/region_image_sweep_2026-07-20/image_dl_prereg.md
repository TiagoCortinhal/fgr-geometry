# IMAGE-DL TRAJECTORY-RECOVERY Pre-Registration (frozen before any class-recovery metric)

## Question
GBTM on longitudinal biometry (EFW-z) found 3 reproducible, cohesive growth-trajectory phenotypes:
  Class 1 FALTERING (n=73, mid-gestation deceleration), Class 2 TRACKING (n=733), Class 3 ACCELERATING (n=102).
Can IMAGE-based deep-learning models recover the SAME partition from ultrasound appearance alone?

## Why this is a valid, non-circular test
The GBTM classes were derived from BIOMETRY (EFW-z curves), NOT from images. Recovering them from image features is
therefore a genuine CROSS-MODAL test, not circular. The class labels are EVAL-ONLY: never inputs to any image model.

## Strong prior (from the project's prior arc — sets expectations, not the verdict)
Images behave as a MATURATION CLOCK, near-orthogonal to the growth/FGR space (image->FGR null was encoder-general:
ResNet<USFM<FetalCLIP all fail on size/growth). The image maturation-lag, after GA-residualization, correlates with
UA-Doppler (r=-0.27) and birth-pct (r=+0.19) and DOES track the GBTM classes (lag: falter -0.94, track -0.33, accel
+0.23 wk; Kruskal p=2.5e-4). So a MODEST recovery of the faltering class via the maturation/UA bridge is plausible;
a strong full-partition recovery is NOT expected. A well-characterized NULL is a first-class result here.

## Data / substrate (image_substrate.npz, artifact fe15dc7c)
- pooled multi-layer USFM per fetus: pool_l0/l5/l11 (908x768 each)
- temporal image sequences: seq_l5 (908x12x768), seq_mask, seq_ga (mean 2.57 real slots/fetus)
- eval-only: gbtm_class (1/2/3), mean_lag, birth, sga_p10, severe, lga, pe_any
- ResNet50-ImageNet + FetalCLIP features extracted per-agent as needed (USB/raw pixels NOT mounted this session)

## FROZEN protocol (applies to every model in every wave)
1. SPLIT: GroupKFold(5) by fetus — no fetus in train & test. Every metric is held-out OOF.
2. TARGET: the 3-class GBTM partition (primary) + per-class one-vs-rest. Labels EVAL-ONLY.
3. PRIMARY METRICS: 3-class balanced accuracy + macro-AUC (OOF); ARI(predicted, GBTM) vs random; per-class OvR AUC.
4. BASELINES (the honest bar): 
   - SIZE-ONLY baseline: last-visit biometry size scalar -> classes (what non-image data trivially gives; the CEILING
     for "did images add anything").
   - random/capacity control: same model on shuffled features / random features of matched dim.
5. NEGATIVE-CONTROL BATTERY (every candidate): label-shuffle (>=20), GA-shuffle within fetus, PE-any as maternal
   negative control (should NOT be recovered by a fetal-growth-phenotype model), matched-capacity random-feature.
6. ADVERSARIAL SELF-AUDIT per model: circular-axis check (is any feature a biometry proxy?), leakage (fetus grouping,
   GA reintroduction), multiplicity (# architectures x targets tried -> adjust bar), matched-baseline strength.
7. SURVIVOR criterion: OOF class-recovery that (a) EXCEEDS the size-only baseline by a bootstrap-CI margin excluding 0,
   (b) COLLAPSES under label-shuffle AND GA-shuffle, (c) does NOT recover PE, (d) reproduces on independent relaunch.
   Anything short of all four = QUALIFIED or NULL, reported honestly.

## Waves
W1 static (multi-layer USFM; ResNet/FetalCLIP swap). W2 temporal (image-GRU, image-trajectory-AE, cross-modal fusion,
lag-bridge). W3 red-team survivors + cohesion on independent variables + comparison figure + results JSON + memory doc.

## Outputs
per-model {recovery AUC/ARI, baseline delta+CI, which controls passed, verdict SURVIVOR/QUALIFIED/NULL}; comparison
figure vs biometry-GBTM ceiling and size-only floor; results JSON; memory-doc section. SGA-CONFIRMED BANNED throughout.
