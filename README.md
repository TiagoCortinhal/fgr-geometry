# fgr-geometry

A first look at the geometry of the fetal growth restriction (FGR) latent. The
question this repo tries to answer, honestly and with the power to find the null:
does the FGR phenotype sit on

1. a single 1-D severity continuum,
2. a multi-direction continuum (e.g. a size axis plus a redistribution/placental
   axis), or
3. a branching continuum (a shared trunk that splits into distinct routes)?

This is exploratory. The analysis is built to be able to return "1-D continuum"
if that is the truth, not biased toward finding extra structure.

## Data

The clinical data is private and is NOT included here. The scripts read it in
place from a separate project directory (see `fgrgeom/config.py`, `DATA`). For
convenience a gitignored `data` symlink can point at it:

    data -> ../fetal_growth_mechanism/data

Source files (per-fetus, n=977, joined 1:1 with zero unmatched rows):
visits_long.csv, visits_long_z.csv (longitudinal biometry over 4 visits
20s/28s/32s/eco), impact_features.csv, impact_outcomes.csv.

Key data limitation: biometry is longitudinal (4 visits), but Doppler, cardiac
and maternal blocks are a SINGLE late snapshot per fetus (the eco visit,
~26-39 wk). Doppler is therefore treated as one late reading, not a trajectory.
There is NO imputation anywhere; missing modalities are marginalised and masks
are carried through.

## Layout

`fgrgeom/`
- `config.py` - paths, column groups, visit/GA windows.
- `panel.py` - loads and joins the four CSVs into the per-fetus Panel; flatten().
- `features.py` - longitudinal velocity/slope features (complete-case per fetus).
- `latent.py` - missing-data factor analysis (exact EM E/M), the shared latent.
- `dimensionality.py` - how many stable directions (spectrum, permutation null, bootstrap, subspace stability).
- `embedding.py` - low-dimensional PCA / kNN graph / diffusion-map embedding of the latent.
- `topology.py` - branch vs single-arc read (Hartigan dip, kNN/MST, persistent homology).
- `flow.py` - biometry growth as a dynamical flow field (per-window Jacobian, expansion/contraction).
- `clinical_anchor.py` - varimax-labelled axes projected onto held-out clinical outcomes.
- `controls_sim.py` - synthetic null (ellipsoid) and branch (Y/V) panels matched to real scale/missingness.
- `controls_run.py` - runs the geometry battery on the controls to calibrate decision thresholds.

`run_all.py` - orchestrator: loads the panel, runs the phases, writes
`results/summary.json` with a rung verdict. `results/` is gitignored.

`docs/hypotheses.md` - the three competing hypotheses and the per-phase test map.

## Running

    pip install -r requirements.txt
    python -m fgrgeom.dimensionality   # or any module's main()
    python run_all.py

Each module also runs standalone via its `main()`. Some phases (permutation
nulls, persistent homology on the controls) are minutes-scale, not interactive.
