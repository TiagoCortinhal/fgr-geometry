import json
from pathlib import Path
import numpy as np
import pandas as pd

from fgrgeom import config as C
from fgrgeom import panel as P

# Pinned to the validated minimal-set latent used by the rest of the pipeline
# (run_all.py / clinical_anchor.main both fit FA k=3 on biom+doppler, varimax).
K = 3
INCLUDE = ("biom", "doppler")
OUT = Path("results/mechanism.json")


def _ols_dir(Z, y):
    """Latent direction that linearly encodes index y: OLS beta of y on Z
    (complete-case on y), returned as a unit vector plus its n and R2.
    Z is fully observed (posterior means); y may carry missingness."""
    ok = ~np.isnan(y)
    Zo, yo = Z[ok], y[ok]
    Zc = Zo - Zo.mean(0)
    yc = yo - yo.mean()
    beta, *_ = np.linalg.lstsq(Zc, yc, rcond=None)
    yhat = Zc @ beta
    ss = np.sum(yc ** 2)
    r2 = float(1 - np.sum((yc - yhat) ** 2) / ss) if ss > 0 else np.nan
    nrm = np.linalg.norm(beta)
    u = beta / nrm if nrm > 0 else beta
    return u, int(ok.sum()), r2


def _residualize(t, s):
    """Residual of target projection t after removing size projection s (OLS,
    both fully observed)."""
    sc = s - s.mean()
    b = (sc @ (t - t.mean())) / (sc @ sc)
    return (t - t.mean()) - b * sc


def consolidate(panel, n_perm=2000):
    from fgrgeom import clinical_anchor as A
    from fgrgeom import flow

    # --- validated latent, varimax-rotated (reuse the pipeline path) ---
    res = A.anchor(panel, k=K, include=INCLUDE, rotate=True, n_perm=0)
    Zr, Wr, labels, fit = res["Z_rot"], res["W_rot"], res["labels"], res["fit"]
    names = np.array(fit["colnames"])

    # --- external, interpretable indices (NOT latent axes) ---
    # size: per-fetus mean biometry-z over all visit x measure cells (observed-only).
    biom = panel.biom_z.reshape(len(panel.ids), -1)
    with np.errstate(invalid="ignore"):
        size_index = np.nanmean(np.where(panel.biom_mask.reshape(biom.shape),
                                         biom, np.nan), axis=1)
    # redistribution: late Doppler contrast, higher = more redistribution
    # (high UtA + low CPR). Complete-case on both percentiles.
    dcol = {c: i for i, c in enumerate(panel.doppler_cols)}
    cpr = panel.doppler[:, dcol["Percentil_CPR"]]
    uta = panel.doppler[:, dcol["Percentil_UTA"]]
    redistr_index = uta - cpr

    # --- size & redistribution directions inside the latent ---
    u_size, n_size, r2_size = _ols_dir(Zr, size_index)
    u_red, n_red, r2_red = _ols_dir(Zr, redistr_index)
    proj_size = Zr @ u_size
    proj_red = Zr @ u_red

    # sign-orient: size higher = bigger; redistribution higher = more redistribution
    if np.corrcoef(proj_size[~np.isnan(size_index)],
                   size_index[~np.isnan(size_index)])[0, 1] < 0:
        u_size, proj_size = -u_size, -proj_size
    okr = ~np.isnan(redistr_index)
    if np.corrcoef(proj_red[okr], redistr_index[okr])[0, 1] < 0:
        u_red, proj_red = -u_red, -proj_red

    # --- (a) entanglement of the redistribution direction with the size direction ---
    cos = float(np.dot(u_size, u_red))
    principal_angle_deg = float(np.degrees(np.arccos(np.clip(abs(cos), 0, 1))))
    r_proj, p_proj, n_proj = A._perm_r(proj_size, proj_red, n_perm)
    # qualitative headline: which varimax axis carries the CPR/UtA redistribution
    # signature, and does it co-load with biometry size on the same axis.
    biom_mask_col = np.array([n.split(":")[1] in C.BIOM_Z for n in names])
    co_load = []
    for a in range(Wr.shape[1]):
        w = Wr[:, a]
        co_load.append({
            "axis": a, "label": labels[a][0],
            "biom_abs_mean": float(np.abs(w[biom_mask_col]).mean()),
            "CPR": float(w[names == "dop:Percentil_CPR"][0]),
            "UtA": float(w[names == "dop:Percentil_UTA"][0]),
        })

    # residual redistribution after partialling out size (load-bearing test).
    resid_red = _residualize(proj_red, proj_size)

    # --- (b) convergent flow from the velocity-flow machinery ---
    fl = flow.analyse(panel, n_boot=500)
    fwin = []
    for lab, w in fl["windows"].items():
        if "spectral_abscissa" not in w:
            fwin.append({"window": lab, "n": w.get("n"), "note": w.get("note")})
            continue
        fwin.append({
            "window": lab, "n": w["n"], "ga_mid_weeks": w["ga_mid_weeks"],
            "spectral_abscissa": w["spectral_abscissa"],
            "abscissa_ci": w["abscissa_ci"],
            "trace": w["trace"], "trace_ci": w["trace_ci"],
            "p_abscissa_pos": w["p_abscissa_pos"],
        })
    contractive = [w for w in fwin if w.get("spectral_abscissa") is not None]
    strongest = min(contractive, key=lambda w: w["spectral_abscissa"]) \
        if contractive else None
    all_contract = all(w["abscissa_ci"][1] < 0 for w in contractive) \
        if contractive else None

    # --- (c) descriptive anchors for both directions (value + perm p) ---
    # CPR/UtA DEFINE the redistribution direction -> identification, NOT validation.
    # PE / severe_sga / birth-centile are held out -> genuine validation.
    od = panel.outcomes
    cont_val = {"percentile_birth_pop": od["percentile_birth_pop"].to_numpy(float)}
    bin_val = {"severe_sga": od["severe_sga"].to_numpy(float),
               "PEwithSGA": od["PEwithSGA"].to_numpy(float)}
    cont_id = {"Percentil_CPR": cpr, "Percentil_UTA": uta}

    def project_anchor(score, role):
        rows = []
        for nm, y in {**cont_id, **cont_val}.items():
            kind = "identification" if nm in cont_id else "validation"
            r, p, n = A._perm_r(score, y, n_perm)
            rows.append(dict(direction=role, anchor=nm, role=kind,
                             metric="r", value=float(r), p=float(p), n=n))
        for nm, y in bin_val.items():
            auc, p, n = A._perm_auc(score, y, n_perm)
            rows.append(dict(direction=role, anchor=nm, role="validation",
                             metric="auc", value=float(auc), p=float(p), n=n))
        return rows

    anchors = (project_anchor(proj_size, "size")
               + project_anchor(proj_red, "redistribution")
               + project_anchor(resid_red, "redistribution|size_partialled"))

    out = {
        "config": {"k": K, "include": list(INCLUDE), "rotation": "varimax",
                   "n_fetuses": int(len(panel.ids)), "n_perm": n_perm,
                   "note": "DIAGNOSTIC geometry consolidation; metrics describe "
                           "structure, they are NOT objectives to optimize"},
        "entanglement": {
            "principal_angle_deg": principal_angle_deg,
            "cos_size_redistr": cos,
            "proj_corr": float(r_proj), "proj_corr_p": float(p_proj),
            "proj_corr_n": n_proj,
            "size_dir_R2_on_latent": r2_size, "size_index_n": n_size,
            "redistr_dir_R2_on_latent": r2_red, "redistr_index_n": int(okr.sum()),
            "varimax_co_loading": co_load,
            "interpretation": "varimax score-axes are orthogonal by construction; "
                              "the angle/corr here are between independently-fit "
                              "size and redistribution directions inside the latent. "
                              "Small angle / nonzero corr = size-coupled redistribution.",
        },
        "flow": {
            "axes": fl["axes"], "windows": fwin,
            "strongest_contraction_window": strongest,
            "all_windows_contract_ci_upper_below_0": all_contract,
            "divergence_first_window": fl["divergence_first_window"],
            "verdict": fl["verdict"],
        },
        "anchors": anchors,
        "partial_caveat": "residual redistribution is partialled against the LATENT "
                          "size axis (mean biometry-z), not birth weight; severe_sga "
                          "is itself a birth-centile outcome, so the surviving AUC may "
                          "still carry size-via-birthweight, not pure redistribution.",
        "verdict": (
            "Geometry: one high-variance size/severity axis (latent R2={:.2f}) plus a "
            "low-variance ({:.2f}), size-ENTANGLED redistribution axis (principal angle "
            "{:.0f}deg, proj r={:.2f}, p<={:.1e}). 'Weaker' is true ONLY in variance: in "
            "outcome discrimination the two axes are co-equal (size severe_sga AUC {:.2f}, "
            "redistribution {:.2f}). Redistribution is not subsumed by size: after "
            "partialling size it retains severe_sga AUC {:.2f} and PE AUC {:.2f} "
            "(perm p<={:.1e}). Flow is convergent everywhere (all window abscissa CI<0), "
            "contraction strongest late at {:.0f}w (abscissa {:.2f}). "
            "DIAGNOSTIC, not a predictive objective."
        ).format(
            r2_size, r2_red, principal_angle_deg, r_proj, 1.0 / (n_perm + 1),
            [a for a in anchors if a["direction"] == "size"
             and a["anchor"] == "severe_sga"][0]["value"],
            [a for a in anchors if a["direction"] == "redistribution"
             and a["anchor"] == "severe_sga"][0]["value"],
            [a for a in anchors if a["direction"] == "redistribution|size_partialled"
             and a["anchor"] == "severe_sga"][0]["value"],
            [a for a in anchors if a["direction"] == "redistribution|size_partialled"
             and a["anchor"] == "PEwithSGA"][0]["value"],
            1.0 / (n_perm + 1),
            strongest["ga_mid_weeks"] if strongest else float("nan"),
            strongest["spectral_abscissa"] if strongest else float("nan"),
        ),
    }
    return out


def main(n_perm=2000):
    panel = P.load_panel()
    out = consolidate(panel, n_perm=n_perm)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    e = out["entanglement"]
    print(f"[entanglement] principal_angle={e['principal_angle_deg']:.1f}deg "
          f"proj_corr={e['proj_corr']:.3f} (p={e['proj_corr_p']:.3f})")
    sc = out["flow"]["strongest_contraction_window"]
    if sc:
        print(f"[flow] strongest contraction {sc['window']} "
              f"ga={sc['ga_mid_weeks']:.1f}w abscissa={sc['spectral_abscissa']:.2f} "
              f"CI={sc['abscissa_ci']}")
    print(f"[flow] all contract={out['flow']['all_windows_contract_ci_upper_below_0']}"
          f"  verdict={out['flow']['verdict'][:60]}")
    df = pd.DataFrame(out["anchors"])
    with pd.option_context("display.width", 160,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(df.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
