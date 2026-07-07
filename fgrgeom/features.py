import numpy as np
import pandas as pd
from fgrgeom import config as C
from fgrgeom import panel as P


def velocity_features(panel, log=True):
    """Per-fetus growth velocity (OLS slope of each biometry z vs ga_days, in z per week)
    plus efw centile crossing. Complete-case per measure: needs >=2 visits with both
    the z value and ga present. Fetuses with <2 valid points get np.nan for that measure
    and are counted in the dropout log. NO imputation."""
    n, V, B = panel.biom_z.shape
    slopes = np.full((n, B), np.nan)
    n_pts = np.zeros((n, B), int)
    for i in range(n):
        ga = panel.ga_days[i]              # (V,)
        for b in range(B):
            y = panel.biom_z[i, :, b]
            ok = ~np.isnan(y) & ~np.isnan(ga)
            n_pts[i, b] = ok.sum()
            if ok.sum() >= 2:
                x = ga[ok] / 7.0          # weeks
                # OLS slope; guard against degenerate (all-equal) ga
                if np.ptp(x) > 0:
                    slopes[i, b] = np.polyfit(x, y[ok], 1)[0]

    slope_cols = [f"slope_{c}" for c in panel.biom_cols]

    # efw centile crossing: late minus early efw_percentile_pop, complete-case.
    # Read raw centiles from visits_long since panel carries only z.
    vl = pd.read_csv(C.DATA / "visits_long.csv")
    vl = vl[vl["visit"].isin(C.VISITS)]
    wide = vl.pivot_table(index=C.KEY_LONG, columns="visit",
                          values="efw_percentile_pop", aggfunc="first")
    wide = wide.reindex(panel.ids)
    early = wide.get("20s")
    late = wide.get("eco")
    cent_drop = (early - late).to_numpy(float)   # positive = lost centile

    df = pd.DataFrame(slopes, columns=slope_cols, index=panel.ids)
    df["efw_centile_drop"] = cent_drop

    if log:
        for b, c in enumerate(panel.biom_cols):
            drop = int((n_pts[:, b] < 2).sum())
            print(f"[features] {c}: {drop}/{n} fetuses <2 valid visits -> slope NaN")
        cd = int(np.isnan(cent_drop).sum())
        print(f"[features] efw_centile_drop: {cd}/{n} missing (need 20s & eco centile)")
    return df
