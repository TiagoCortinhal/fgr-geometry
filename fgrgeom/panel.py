from collections import namedtuple
import numpy as np
import pandas as pd
from fgrgeom import config as C

Panel = namedtuple("Panel", [
    "ids",            # (n,) int fetus ids
    "biom_z",         # (n, V, B) longitudinal biometry z, np.nan where missing
    "biom_mask",      # (n, V, B) bool observed
    "ga_days",        # (n, V) ga at each visit cell, np.nan if visit absent
    "biom_cols",      # list[str] length B
    "doppler",        # (n, Dd) late Doppler percentiles
    "doppler_mask",   # (n, Dd) bool observed
    "doppler_cols",
    "cardiac",        # (n, Dc) late cardiac percentiles
    "cardiac_mask",
    "cardiac_cols",
    "maternal",       # (n, M) maternal covariates + disease
    "maternal_mask",
    "maternal_cols",
    "outcomes",       # DataFrame indexed by fetus id, config.OUTCOMES
    # --- full variable set (appended; existing signatures unchanged) ---
    "ratios",         # (n, V, R) longitudinal asymmetry ratios (raw), nan where missing
    "ratios_mask",    # (n, V, R) bool observed
    "ratios_cols",    # list[str] length R
    "bp",             # (n, Bbp) visit blood pressure
    "bp_mask",
    "bp_cols",
    "raw_doppler",        # (n, RV, RD) SPARSE raw longitudinal Doppler PI/CPR
    "raw_doppler_mask",
    "raw_doppler_cols",   # list[str] length RD
    "raw_doppler_visits", # list[str] length RV
])


def _load_raw():
    vz = pd.read_csv(C.DATA / "visits_long_z.csv")
    vz = vz[vz["visit"].isin(C.VISITS)].copy()
    feat = pd.read_csv(C.DATA / "impact_features.csv")
    out = pd.read_csv(C.DATA / "impact_outcomes.csv")
    return vz, feat, out


def load_panel():
    """Build the per-fetus structured panel. NO imputation; missingness carried as masks."""
    vz, feat, out = _load_raw()

    ids = np.sort(vz[C.KEY_LONG].unique())
    n = len(ids)
    id_pos = {f: i for i, f in enumerate(ids)}
    vpos = {v: j for j, v in enumerate(C.VISITS)}
    V, B = len(C.VISITS), len(C.BIOM_Z)

    biom_z = np.full((n, V, B), np.nan)
    ga = np.full((n, V), np.nan)
    for r in vz.itertuples(index=False):
        i = id_pos[getattr(r, C.KEY_LONG)]
        j = vpos[r.visit]
        ga[i, j] = r.ga_days
        for b, col in enumerate(C.BIOM_Z):
            val = getattr(r, col)
            if pd.notna(val):
                biom_z[i, j, b] = val
    biom_mask = ~np.isnan(biom_z)

    # Per-fetus late snapshot tables, reindexed onto the longitudinal id order.
    feat = feat.set_index(C.KEY_IMPACT).reindex(ids)
    out = out.set_index(C.KEY_IMPACT).reindex(ids)

    def matrix(df, cols):
        m = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return m, ~np.isnan(m)

    doppler, doppler_mask = matrix(feat, C.DOPPLER_PCTL)
    cardiac, cardiac_mask = matrix(feat, C.CARDIAC_PCTL)

    # Maternal: per-fetus constant from visits_long (first non-null), disease from impact.
    vl = pd.read_csv(C.DATA / "visits_long.csv")
    vl = vl[vl["visit"].isin(C.VISITS)]

    # Longitudinal asymmetry ratios (raw) from visits_long, visit-major (n, V, R).
    R = len(C.RATIOS)
    ratios = np.full((n, V, R), np.nan)
    for r in vl.itertuples(index=False):
        fid = getattr(r, C.KEY_LONG)
        if fid not in id_pos or r.visit not in vpos:
            continue
        i, j = id_pos[fid], vpos[r.visit]
        for k, col in enumerate(C.RATIOS):
            val = getattr(r, col)
            if pd.notna(val):
                ratios[i, j, k] = val
    ratios_mask = ~np.isnan(ratios)

    # Sparse raw longitudinal Doppler PI/CPR at 28s/32s (n, RV, RD).
    rvpos = {v: j for j, v in enumerate(C.RAW_DOPPLER_VISITS)}
    RV, RD = len(C.RAW_DOPPLER_VISITS), len(C.RAW_DOPPLER)
    raw_doppler = np.full((n, RV, RD), np.nan)
    for r in vl.itertuples(index=False):
        fid = getattr(r, C.KEY_LONG)
        if fid not in id_pos or r.visit not in rvpos:
            continue
        i, j = id_pos[fid], rvpos[r.visit]
        for k, col in enumerate(C.RAW_DOPPLER):
            val = getattr(r, col)
            if pd.notna(val):
                raw_doppler[i, j, k] = val
    raw_doppler_mask = ~np.isnan(raw_doppler)

    # Visit blood pressure (per-fetus) from impact_features.
    bp, bp_mask = matrix(feat, C.BP)
    mat_long = (vl.groupby(C.KEY_LONG)[C.MATERNAL].first().reindex(ids))
    mat_dis = feat.reindex(columns=C.MATERNAL_DISEASE)
    mat_cols = C.MATERNAL + C.MATERNAL_DISEASE
    mat = pd.concat([mat_long.reset_index(drop=True),
                     mat_dis.reset_index(drop=True)], axis=1)
    maternal = mat.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    maternal_mask = ~np.isnan(maternal)

    # Outcomes: birth centile + sga/severe/lga from visits_long (per-fetus constant),
    # PE/preterm/NICU from impact_outcomes. PEwithSGA is yes/no -> 1/0.
    bc = (vl.groupby(C.KEY_LONG)[["percentile_birth_pop", "sga", "severe_sga", "lga"]]
          .first().reindex(ids))
    pe = out["PEwithSGA"].map({"yes": 1, "no": 0})
    od = pd.DataFrame({
        "percentile_birth_pop": bc["percentile_birth_pop"].to_numpy(),
        "sga": bc["sga"].to_numpy(),
        "severe_sga": bc["severe_sga"].to_numpy(),
        "lga": bc["lga"].to_numpy(),
        "PEwithSGA": pe.to_numpy(),
        "PartoPret": out["PartoPret"].to_numpy(),
        "NICU": out["NICU"].to_numpy(),
    }, index=ids)[C.OUTCOMES]

    return Panel(ids=ids, biom_z=biom_z, biom_mask=biom_mask, ga_days=ga,
                 biom_cols=list(C.BIOM_Z),
                 doppler=doppler, doppler_mask=doppler_mask,
                 doppler_cols=list(C.DOPPLER_PCTL),
                 cardiac=cardiac, cardiac_mask=cardiac_mask,
                 cardiac_cols=list(C.CARDIAC_PCTL),
                 maternal=maternal, maternal_mask=maternal_mask,
                 maternal_cols=mat_cols, outcomes=od,
                 ratios=ratios, ratios_mask=ratios_mask,
                 ratios_cols=list(C.RATIOS),
                 bp=bp, bp_mask=bp_mask, bp_cols=list(C.BP),
                 raw_doppler=raw_doppler, raw_doppler_mask=raw_doppler_mask,
                 raw_doppler_cols=list(C.RAW_DOPPLER),
                 raw_doppler_visits=list(C.RAW_DOPPLER_VISITS))


def flatten(panel, include=("biom", "doppler", "maternal")):
    """Per-fetus feature matrix + observed mask for FA. Biometry flattened visit-major:
    one column per (visit, measure). Returns (X, mask, colnames). NO imputation."""
    if include == "full":
        include = ("biom", "ratios", "doppler", "cardiac", "maternal", "bp")
    blocks, masks, names = [], [], []
    if "biom" in include:
        n, V, B = panel.biom_z.shape
        blocks.append(panel.biom_z.reshape(n, V * B))
        masks.append(panel.biom_mask.reshape(n, V * B))
        names += [f"{v}:{c}" for v in C.VISITS for c in panel.biom_cols]
    if "ratios" in include:
        n, V, R = panel.ratios.shape
        blocks.append(panel.ratios.reshape(n, V * R))
        masks.append(panel.ratios_mask.reshape(n, V * R))
        names += [f"{v}:{c}" for v in C.VISITS for c in panel.ratios_cols]
    if "doppler" in include:
        blocks.append(panel.doppler); masks.append(panel.doppler_mask)
        names += [f"dop:{c}" for c in panel.doppler_cols]
    if "cardiac" in include:
        blocks.append(panel.cardiac); masks.append(panel.cardiac_mask)
        names += [f"card:{c}" for c in panel.cardiac_cols]
    if "maternal" in include:
        blocks.append(panel.maternal); masks.append(panel.maternal_mask)
        names += [f"mat:{c}" for c in panel.maternal_cols]
    if "bp" in include:
        blocks.append(panel.bp); masks.append(panel.bp_mask)
        names += [f"bp:{c}" for c in panel.bp_cols]
    if "raw_doppler" in include:
        n, RV, RD = panel.raw_doppler.shape
        blocks.append(panel.raw_doppler.reshape(n, RV * RD))
        masks.append(panel.raw_doppler_mask.reshape(n, RV * RD))
        names += [f"rdop:{v}:{c}" for v in panel.raw_doppler_visits
                  for c in panel.raw_doppler_cols]
    X = np.concatenate(blocks, axis=1)
    M = np.concatenate(masks, axis=1)
    return X, M, names
