# Literature positioning — what our dataset's problem is called, and who else has it

Search: OpenAlex, title-restricted (free-text Boolean returned citation-sorted
generic reviews, not the methods literature). Every DOI below verified against
Crossref by fetching the record and comparing titles.

## The honest situation

Six independent analyses on this cohort say the same thing: images and the
tabular registry do not share information beyond gestational age and maternal
habitus. That is not a failed project — it is a **negative result with unusually
strong controls**, and there is an established literature it belongs to.

## The four papers our findings map onto

### 1. Wang et al., "What Makes Training Multi-Modal Classification Networks Hard?" CVPR 2020
DOI 10.1109/cvpr42600.2020.01271

The multi-modal network receives strictly more information, so it should match
or beat its uni-modal counterpart. They observe the opposite, consistently
across modality combinations and benchmarks, and attribute it to
increased-capacity overfitting plus modalities that overfit and generalise at
different rates.

**Maps to:** our decision-fusion arm. Fused beat the best single block by
+0.000 to +0.027 across five endpoints; nothing survived BH at q=0.10. The
growth block was the best single predictor on all five.

### 2. Geirhos et al., "Shortcut learning in deep neural networks," Nat Mach Intell 2020
DOI 10.1038/s42256-020-00257-z

Networks solve tasks through unintended cues that do not transfer.

**Maps to:** our strongest image signal by far. Image to maternal block was
+0.512 raw and **+0.003 after BMI adjustment**; the effect replicates in every
anatomical plane (cerebral 0.43 to 0.02, abdominal 0.34 to 0.06, femur 0.23 to
0.04) and on an encoder-independent texture representation (0.298 to 0.023).
Maternal habitus sets ultrasound penetration and the encoder reads it off the
pixels. This is a textbook shortcut, measured four ways.

### 3. "Vision-Language Foundation Models Do Not Transfer to Medical Imaging Classification," 2025
DOI 10.64898/2025.12.06.25341759

Both web-pretrained and biomedically-pretrained VLMs underperform ImageNet CNNs
on ChestX-ray14.

**Maps to:** USFM versus PyRadiomics. The two representations correlate at
**cc = 0.963** held-out — the ultrasound foundation model sees essentially what
219 hand-crafted texture features see. Whatever USFM adds is not what our
tabular blocks need.

### 4. Shwartz-Ziv & Armon, "Tabular data: Deep learning is not all you need," Inf. Fusion 2022
DOI 10.1016/j.inffus.2021.11.011

**Maps to:** the whole representation-learning line. Our linear factor model
beat or tied every deep alternative tried — GPLVM, masked autoencoder,
variational autoencoder — on held-out likelihood, the criterion chosen before
comparing.

## What this suggests for a paper

The multimodal angle is not the thread. **The missing-data mechanism is**, and
it is what already survived review: analytic marginalisation over unacquired
blocks, calibrated intervals, a screen that flags implausible records, and a
synthetic benchmark. The image work belongs in it as a *bounded negative
control* — one paragraph and one figure panel showing that a fifth block was
tested and has no home in the latent, with the shortcut quantified.

That framing is stronger than a multimodal paper we cannot support, and it is
honest about six analyses that pointed the same way.

## The one live thread

Percentil_LV_basal: selected in 15 of 20 random splits out of 11 candidates,
mean held-out r = +0.109, significant in 13 of those 15, basal >> longitudinal
on both representations. Small (r ~ 0.11), encoder-dependent (weak on
radiomics), and not enough for a paper on its own.
