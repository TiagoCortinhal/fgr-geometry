# impact-fetal-panel — analysis tools for the IMPACT cohort

30 helpers for the 977-fetus IMPACT multiblock panel. Exported from the
published Claude Science skill of the same name; `SKILL.md` is the full
reference and documents every estimator trap these encode against.

## Use

```python
import sys; sys.path.insert(0, "tools/impact_fetal_panel")
from fgm_tools import *

fgm_setup()
P   = fgm_panel()                        # X, Z, cols, blocks, fids
GA  = fgm_ga_at_echo(P["fids"])
IMG, pca = fgm_image_pcs(P["fids"], n_pc=5)
```

(Inside a Claude Science session the same functions load automatically via
`skill({skill: "impact-fetal-panel"})` — no import needed.)

## What's here

**Loading** — `fgm_panel`, `fgm_all_tabular`, `fgm_registry_variables`,
`fgm_eurofloat`, `fgm_visit_matrix`, `fgm_echo_raw`, `fgm_image_lag`,
`fgm_image_pcs`, `fgm_ga_at_echo`, `fgm_setup`, `fgm_run`

**Statistics** — `fgm_cv_r2`, `fgm_auc`, `fgm_o_information`, `fgm_omega_null`,
`fgm_omega_report`, `fgm_decorrelate`, `fgm_heldout_cca`, `fgm_residualise`,
`fgm_nuisance_design`, `fgm_bh`

**Controls** — `fgm_positive_control`, `fgm_block_shuffle_null`,
`fgm_ga_leakage`, `fgm_confound_ladder`, `fgm_crossmodal_ladder`,
`fgm_image_screen`

**Registry knowledge** — `fgm_classify_registry_columns`,
`fgm_canonical_block_vars`, `fgm_derived_variables`

## Why these exist

Each encodes against a specific defect that produced a wrong, publishable-looking
number during this project:

- `fgm_auc` — `1 - U/(n1*n0)` takes the complement twice; every AUC lands below 0.5
- `fgm_cv_r2` — bootstrap inside CV duplicates rows across folds and leaks
- `fgm_block_shuffle_null` — column-wise permutation destroys within-block structure too
- `fgm_decorrelate` — a near-zero covariance eigenvalue sends O-information to -31
- `fgm_eurofloat` — European-decimal strings silently become NaN
- `fgm_ga_leakage` / `fgm_confound_ladder` — unscaled measurements carry gestational age,
  and image appearance does too; the unadjusted correlation is a maturation channel

Results produced with these: `docs/multimodal_feasibility_2026-08-07/`.
