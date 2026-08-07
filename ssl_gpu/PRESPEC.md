# Pre-specification — SSL / fine-tuned encoder vs frozen USFM
Written BEFORE any pretraining. Fixes endpoints, comparators and stop rule.

## The question
Every image null in this project used FROZEN, FRAME-LEVEL POOLED features
(USFM emb_l5, and PyRadiomics texture as a check). Those two agree at held-out
cc = 0.963, which argues the nulls are not an artefact of one encoder — but both
are pooled summaries of the same pixels, and neither was ever tuned on this
cohort or this task.

Q: does an encoder trained ON OUR OWN FRAMES recover cross-modal signal that
frozen pooled features miss?

## Arms (all evaluated identically)
A0  FROZEN USFM emb_l5, 12 PCs                 -- the incumbent, reproduces prior numbers
A1  SSL-MAE          masked autoencoding on our frames
A2  SSL-CONTRASTIVE  same-fetus positives, different-fetus negatives
A3  SUPERVISED       end-to-end regression from pixels to the tabular target
                     (the STRONGEST test: if a supervised net trained directly
                     on the target finds nothing, no unsupervised proxy will)

A3 is the decisive arm. A1/A2 answer "does a better general representation
help"; A3 answers "is the information in the pixels AT ALL".

## Endpoints — FIXED NOW, identical to the frozen-feature analyses
Primary (block-level held-out canonical correlation, GA + maternal BMI adjusted):
  E1 growth block    (5 biometry z)      incumbent USFM +0.159
  E2 Doppler block   (5 percentiles)     incumbent +0.077 (ns)
  E3 cardiac block   (11 percentiles)    incumbent +0.111
Secondary (single variable, split-sample selected):
  E4 Percentil_LV_basal                  incumbent +0.165 full / +0.109 split-mean
Confound / positive controls:
  C1 maternal BMI    -- MUST be strongly predicted (positive control: the encoder
                        works). Incumbent raw 0.512.
  C2 gestational age -- MUST be predicted (positive control).
  C3 growth block    -- measurement recovery; expected positive, NOT novel signal.

## Adjustment and validation (unchanged from the frozen analyses)
GA + maternal BMI residualised from BOTH sides. 5-fold out-of-fold. Permutation
null 1000 draws. BH q=0.10 across arms x endpoints.
FETUS-LEVEL SPLITS EVERYWHERE -- a fetus's frames must never straddle train/test.
This is the single most important implementation detail: frame-level splitting
would leak and manufacture a positive.

## Stop rule
SSL/fine-tuning SUCCEEDS only if some arm beats the frozen incumbent on E2 or E4
by a margin whose bootstrap CI excludes zero, while C1/C2 confirm the encoder
trained at all. Beating the incumbent on E1/C3 alone does NOT count -- biometry
is measured from these images and is measurement recovery.

## Declared expectations
- C1, C2 will be strongly positive on every arm (or the run is broken).
- E1/C3 will be positive on every arm (measurement recovery).
- E2 will stay null. E3/E4 are the genuine unknowns.
- If A3 (supervised, end-to-end, direct on target) is null, that is close to
  decisive for this cohort and should be reported as such.

## What would falsify the project's negative conclusion
A3 or A1/A2 clearing E2 or E4 above the frozen incumbent with a CI excluding
zero. That result would overturn the multimodal null and become the paper.
