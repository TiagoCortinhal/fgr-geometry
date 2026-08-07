# SSL / fine-tuned encoder vs frozen USFM — GPU package
#
# QUICK START (you are on the node, in a terminal): see RUN.md


Tests whether an encoder trained **on our own frames** recovers cross-modal
signal that frozen pooled features miss. Read `PRESPEC.md` first — endpoints,
comparators and the stop rule are fixed there, before any training.

## Why this exists

Every image null in this project used frozen, frame-level pooled USFM features
(with PyRadiomics texture as a check; the two agree at held-out cc = 0.963).
Both are pooled summaries of the same pixels and neither was tuned on this
cohort. This package removes that objection.

## Four arms

| arm | what it answers |
|---|---|
| `frozen` | the incumbent — reproduces the existing numbers, no training |
| `mae` | does a better general representation help? (masked autoencoding) |
| `contrast` | same, via same-fetus positives |
| `supervised` | **the decisive arm** — end-to-end pixels → tabular target |

If `supervised`, trained directly on the target, finds nothing, no unsupervised
proxy will.

## Layout

```
ssl_gpu/                    <- in the repo
  RUN.md              terminal commands, start here
  PRESPEC.md          endpoints and stop rule, fixed before training
  run_ssl.py          training, one arm per invocation
  score_arms.py       evaluation (CPU, after training)
  build_inputs.py     regenerates the two npz inputs from the cohort
  smoke_test.py       synthetic end-to-end check
  fgm_ssl/            data.py models.py evaluate.py
  data/               NOT in the repo (gitignored -- cohort data)
    panel.npz           from build_inputs.py  (156 KB)
    frozen_usfm.npz     from build_inputs.py  (3 MB)
    image_clusters.csv  the manifest
```

Frames are referenced by `--image-root`, never copied.

Expect ~2–4 h/arm for 100 epochs on 21k frames at 224² on one A100; the
supervised arm is ~5x that because it trains one model per fold.

## The three things that would invalidate the result

**1. Frame-level splitting.** A fetus has ~22 near-duplicate frames; splitting
by frame leaks train into test and manufactures a positive.
`fetus_level_folds` is the only splitter in the package, and `run_ssl.py`
asserts no fetus appears on both sides.

**2. A silently-broken run.** A null from a model that never trained is
meaningless. `run_ssl.py` refuses to write embeddings if the loss did not move
or the embedding variance collapsed; `score_arms.py` refuses to issue a verdict
unless a trained arm passed the positive-control gate (image → maternal BMI,
which must be strong — it is the confound we already measured at 0.512 raw).

**3. Adjusting one side only.** `evaluate.py` residualises GA and BMI from
**both** the image and the target, and drops a covariate when it *is* the
target — the degenerate-self-adjustment defect caught in review earlier.

**3b. Scoring a trained encoder on its own training fetuses.** The supervised
arm sees the target during training, so scoring it on those fetuses measures
memorisation. It runs OUT-OF-FOLD (K models, each embedding only its held-out
fold) and `score_arms.py` honours `heldout_fids` for every trained arm. A smoke
test with the leak present reported +0.851 on the trained block and ~0 on all
others.

**4. Bootstrap inside cross-validation.** The stop-rule interval is the spread
over *independent CV seeds* (`split_spread_delta`), not a bootstrap. Resampling
rows with replacement into a statistic that runs its own KFold duplicates
fetuses across train and test folds and yields intervals that can exclude their
own point estimate. This package contains no bootstrap-inside-CV.

## Reading the output

`results/scores.json` carries, per arm and endpoint: the raw → GA → GA+BMI
ladder, a permutation p, BH-adjusted q, and for the trained arms a paired
bootstrap delta against the frozen incumbent.

**The verdict line is the answer.** It reports `INCONCLUSIVE` rather than a null
whenever no trained arm passed its positive control — a broken run must never
read as evidence about the data.

Expected under the project's current conclusion: C1/C2 strong on every arm,
E1 positive (biometry is measured from these images — measurement recovery),
E2 null, and E3/E4 the genuine unknowns.

A `BEATS INCUMBENT` on E2 or E4 with a CI excluding zero overturns the
multimodal null and becomes the paper.

## Smoke test

The package was validated end-to-end on synthetic frames with a planted
per-fetus factor: all arms train, the fetus-level splitter never leaks, the
out-of-fold path scores every fetus from a model that never saw it, and the gate
correctly refuses a verdict when the positive control fails.

An earlier version of this smoke test reported +0.851 for the supervised arm on
its own trained block. That was the training-fetus leak, not a positive control,
and it is what motivated the out-of-fold path. The honest positive control on
real data is C1 (image → maternal BMI, expected strong from the frozen-feature
result of 0.512).
