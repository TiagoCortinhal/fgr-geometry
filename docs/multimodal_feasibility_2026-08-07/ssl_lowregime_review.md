# What the literature says: ultrasound SSL, low-regime learning, and orthogonal data

Search: OpenAlex, title-restricted (free-text Boolean returns citation-sorted
generic reviews). Every DOI verified against Crossref. One candidate
(arXiv 2203.02053, "Mind the Gap") did NOT resolve in Crossref and is excluded
rather than cited from memory.

## 1. Ultrasound SSL — the field exists, and it targets a different problem

| work | DOI | what it does |
|---|---|---|
| Jiao et al., SSL for Ultrasound Video, **ISBI 2020** | 10.1109/isbi45749.2020.9098666 | learns transferable representations from unlabelled US **video**, no annotation |
| USCL, MICCAI 2021 | 10.1007/978-3-030-87237-3_60 | video contrastive pretraining for US diagnosis |
| Anatomy-Aware Contrastive, **fetal US** 2023 | 10.1007/978-3-031-25066-8_23 | contrastive learning with anatomical structure as the signal |
| FUSC, Ultrasound Med Biol 2024 | 10.1016/j.ultrasmedbio.2024.01.010 | unsupervised clustering of **88,063** second-trimester images into fetal views |
| USFM, Med Image Anal 2024 | 10.1016/j.media.2024.103202 | the ultrasound foundation model we already use |

**The pattern:** every one of these learns image representations for
**image-domain tasks** — view classification, segmentation, diagnosis from the
image itself. None of them is trying to predict a *separate tabular modality*
from images. The problem they solve is label scarcity, not cross-modal transfer.

**What this means for us.** SSL or fine-tuning would give a better image
representation. Our evidence says the bottleneck is not representation quality:
USFM and 219 hand-crafted texture features agree at held-out cc = 0.963 and give
the same block-level answers. Two very different feature extractors converging
on the same nulls is evidence about the *information*, not the *encoder*.

**The honest caveat:** both are frame-level summaries of the same pixels. A
fine-tuned or SSL encoder trained on *our* cohort with a task-relevant proxy
could in principle find a subspace both miss. Nothing we ran rules that out.

## 2. Low-regime — the field says use the unlabelled majority, not more capacity

- **TabPFN** (Nature 2025, 10.1038/s41586-024-08328-6) — transformer pretrained
  on synthetic tabular tasks, strong exactly at n < 10,000. Our n = 977.
- **Data-scarcity survey** (J Big Data 2023, 10.1186/s40537-023-00727-2) —
  taxonomy of scarcity strategies.
- Few-shot/meta-learning for medical images (e.g. MetaMed,
  10.1016/j.patcog.2021.108111) is an active line, all **image-label** tasks.

Note what is *absent*: title-restricted searches for "SSL does not help small
datasets", "pretraining benefits diminish with dataset size", and
"fine-tuning versus frozen features" returned essentially nothing. **The
negative-result literature for SSL at small n is not there** — which is itself
informative about what gets published, and about where an honest contribution
could sit.

## 3. Orthogonal / weakly-aligned data — the thinnest area, and the closest to us

Searches for "unpaired multimodal representation learning", "partially paired",
"weakly aligned multimodal" returned almost nothing at title level. What exists
is either image-to-image translation (unpaired MR-CT) or report generation —
not two heterogeneous modalities with no shared latent.

The nearest applied work is **ICH-PRNet** (Neural Networks 2025,
10.1016/j.neunet.2024.107096), cross-modal imaging + tabular prognosis — but it
assumes the modalities are complementary and does not test whether they are.

**This is the gap.** The field has methods for *fusing* imaging and tabular data
and essentially no established protocol for *deciding whether fusion is
warranted on a given cohort*. We built one by accident: information
decomposition, block ladders with acquisition-confound controls, split-sample
selection, and encoder-agreement checks.

## 4. What would actually survive — three options, ranked

**A. SSL / fine-tuning on our own cohort (what you asked about).**
The literature supports the method. Our evidence says representation quality is
not the binding constraint — but the strongest counter-argument is real: both
representations we compared are frame-level pooled summaries, and USFM was never
tuned on this cohort or this task. A masked-autoencoder or anatomy-aware
contrastive encoder trained on our 21,192 IMPACT + 46,956 clinical fetal frames,
evaluated on the SAME prespecified endpoints, is the one experiment that could
overturn the null rather than confirm it.
Precedent for the recipe: 10.1007/978-3-031-25066-8_23 (fetal, anatomy-aware),
10.1109/isbi45749.2020.9098666 (ISBI, video).

**B. Positive results we already hold, reframed.**
Not everything was null. LV_basal survives split-sample selection (15/20 splits,
held-out r = 0.109, significant in 13/15). Images recover fetal biometry at
+0.159 after GA and BMI. The maternal-habitus channel is strong and
reproducible (0.512 to 0.003 on BMI adjustment) — that is a *finding about
ultrasound acquisition*, not a failure.

**C. The methodological contribution — the decision protocol itself.**
Section 3 says this does not exist. A paper that says "here is how to decide,
before committing to a fusion architecture, whether your two modalities share
information — and here is the cohort where the answer was no" is publishable,
useful, and something we can fully support today.

## Recommendation

**A and C together.** Run the SSL/fine-tune experiment properly — it is the one
thing that could produce a positive, the fetal-US precedent exists, and if it
also comes back null the protocol paper becomes much stronger for having tried
the strongest available method. Prespecify the endpoints before training so the
result is interpretable either way.

If GPU access is the constraint, C alone is defensible now.
