# Prespecification — targeted ventricular-geometry test
Written BEFORE any test statistic. Amends prespec_fusion.md.

## The selection problem, stated first
Percentil_LV_basal and Percentil_RV_basal were chosen BECAUSE they topped an
11-variable screen on this same cohort. Re-testing them on the same rows is
circular and would inflate any result. Two guards:

  SPLIT-SAMPLE (primary). Split fetuses 50/50 by a fixed seed. Rank the 11
  cardiac variables on HALF A only. Test the top-2 from that ranking on HALF B
  only. If LV_basal/RV_basal are real they will be selected in A and survive in
  B; if the 11-variable maximum was noise, the A-ranking will not transfer.

  FULL-COHORT (secondary, reported as CONFIRMATORY-ONLY). The same two
  variables on all rows, explicitly labelled as not independent of selection.

## Hypotheses and directions, fixed now
H1 basal geometry. The image representation predicts the BASAL ventricular
   percentiles (LV_basal, RV_basal) beyond GA + maternal BMI.
   DIRECTION: positive (higher predicted score -> higher measured percentile).
H2 specificity. If H1 is real, the LONGITUDINAL ventricular percentiles
   (LV_longitudinal, RV_longitudinal) should show a WEAKER effect --
   they measure a different axis of the same chambers. A signal equally
   present in both is non-specific and argues for a global confound.

## Protocol
Representations: radiomics (219 texture features -> 12 PCs) and USFM (12 PCs),
reported separately. Adjustment: GA + maternal BMI, both sides residualised.
Held-out ridge, 5-fold. Permutation null 1000 draws on the fully adjusted arm.
BH across the 4 variables x 2 representations = 8 tests, q=0.10.

## Stop rule
The claim SUCCEEDS only if the split-sample arm shows a basal variable
selected in A and clearing its permutation null in B, on at least one
representation. Full-cohort significance alone does NOT count.

## Declared expectation
I expect this to FAIL the split-sample arm. Eleven variables at n~755 with a
block-level effect of 0.11-0.17 is the regime where a maximum is often noise.
A null here is a real answer and will be reported as one.
