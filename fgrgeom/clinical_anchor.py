import numpy as np
import pandas as pd
from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L

# Held-out clinical anchors. None of these are in the (biom,doppler) FA fit, so
# correlating them with the latent is genuine validation, not circularity.
# (Doppler IS in the fit -> used only to LABEL axes, flagged as such.)
CONT_ANCHORS = ["percentile_birth_pop"]
BIN_ANCHORS = ["severe_sga", "NICU", "PartoPret", "PEwithSGA"]
MAT_BIN = ["HTAcronic"]


def _varimax(W, gamma=1.0, max_iter=100, tol=1e-6):
    """Varimax rotation of loadings W (d,k). Returns (Wrot, R)."""
    d, k = W.shape
    if k < 2:
        return W.copy(), np.eye(k)
    R = np.eye(k)
    dsum = 0
    for _ in range(max_iter):
        Lr = W @ R
        u, s, vt = np.linalg.svd(
            W.T @ (Lr ** 3 - (gamma / d) * Lr @ np.diag(np.diag(Lr.T @ Lr))))
        R = u @ vt
        ds = s.sum()
        if ds < dsum * (1 + tol):
            break
        dsum = ds
    return W @ R, R


def _auc(score, y):
    """Rank AUC of continuous score vs binary y (0/1), NaNs dropped pairwise."""
    ok = ~np.isnan(score) & ~np.isnan(y)
    s, yy = score[ok], y[ok]
    pos, neg = s[yy == 1], s[yy == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, 0
    r = pd.Series(s).rank().to_numpy()
    auc = (r[yy == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return auc, int(ok.sum())


def _perm_auc(score, y, n=2000, seed=C.SEED):
    auc, n_ok = _auc(score, y)
    if np.isnan(auc):
        return auc, np.nan, n_ok
    ok = ~np.isnan(score) & ~np.isnan(y)
    s, yy = score[ok], y[ok]
    rng = np.random.default_rng(seed)
    obs = abs(auc - 0.5)
    cnt = sum(abs(_auc(s, rng.permutation(yy))[0] - 0.5) >= obs for _ in range(n))
    return auc, (cnt + 1) / (n + 1), n_ok


def _pearson(score, y):
    ok = ~np.isnan(score) & ~np.isnan(y)
    if ok.sum() < 3:
        return np.nan, 0
    return np.corrcoef(score[ok], y[ok])[0, 1], int(ok.sum())


def _perm_r(score, y, n=2000, seed=C.SEED):
    r, n_ok = _pearson(score, y)
    if np.isnan(r):
        return r, np.nan, n_ok
    ok = ~np.isnan(score) & ~np.isnan(y)
    s, yy = score[ok], y[ok]
    rng = np.random.default_rng(seed)
    cnt = sum(abs(np.corrcoef(s, rng.permutation(yy))[0, 1]) >= abs(r)
              for _ in range(n))
    return r, (cnt + 1) / (n + 1), n_ok


def label_axes(W, colnames):
    """Label each rotated latent direction by its dominant loading block:
    'size' (biometry z), 'redistribution' (CPR/UtA Doppler), or 'mixed'.
    Returns list of (label, detail) per axis."""
    names = np.array(colnames)
    biom = np.array([n.split(":")[1] in C.BIOM_Z for n in names])
    is_cpr = names == "dop:Percentil_CPR"
    is_uta = names == "dop:Percentil_UTA"
    is_dop = np.array([n.startswith("dop:") for n in names])
    out = []
    for a in range(W.shape[1]):
        w = W[:, a]
        bmag = np.abs(w[biom]).mean() if biom.any() else 0.0
        dmag = np.abs(w[is_dop]).mean() if is_dop.any() else 0.0
        cpr = w[is_cpr][0] if is_cpr.any() else 0.0
        uta = w[is_uta][0] if is_uta.any() else 0.0
        # redistribution signature: CPR up / UtA down (or vice versa), both nonneg-trivial
        redistr = (cpr * uta < 0) and (abs(cpr) > 0.1) and (abs(uta) > 0.1)
        if dmag > bmag and redistr:
            lab = "redistribution"
        elif bmag >= dmag:
            lab = "size+redistr" if redistr else "size"
        else:
            lab = "mixed"
        out.append((lab, f"biom|w|={bmag:.2f} dop|w|={dmag:.2f} "
                         f"CPR={cpr:+.2f} UtA={uta:+.2f}"))
    return out


def anchor(panel, k=3, include=("biom", "doppler"), rotate=True, n_perm=2000):
    """Fit the fetal-growth latent, varimax-rotate, label axes, and project
    held-out clinical anchors onto each direction. Returns dict of results."""
    fl = L.fit_latent(panel, k=k, include=include)
    W, Z = fl["W"], fl["Z"]
    if rotate:
        Wr, R = _varimax(W)
        Zr = Z @ R
    else:
        Wr, Zr = W, Z
    labels = label_axes(Wr, fl["colnames"])

    od = panel.outcomes
    # maternal disease pulled from the maternal block
    mat_idx = {c: i for i, c in enumerate(panel.maternal_cols)}
    anchors = {}
    for c in CONT_ANCHORS:
        anchors[c] = ("cont", od[c].to_numpy(float))
    for c in BIN_ANCHORS:
        anchors[c] = ("bin", od[c].to_numpy(float))
    for c in MAT_BIN:
        anchors[c] = ("bin", panel.maternal[:, mat_idx[c]])

    rows = []
    for a in range(Zr.shape[1]):
        s = Zr[:, a]
        for name, (kind, y) in anchors.items():
            if kind == "cont":
                stat, p, n = _perm_r(s, y, n_perm)
                metric = "r"
            else:
                stat, p, n = _perm_auc(s, y, n_perm)
                metric = "auc"
            rows.append(dict(axis=a, label=labels[a][0], anchor=name,
                             metric=metric, value=stat, p=p, n=n))
    res = pd.DataFrame(rows)
    return {"fit": fl, "W_rot": Wr, "Z_rot": Zr, "labels": labels,
            "anchors": res}


def pole_test(panel, k=3, include=("biom", "doppler"), n_perm=2000):
    """Constitutional pole (normal Doppler + flat efw trajectory) vs placental pole
    (abnormal CPR + efw centile crossing). Test whether the latent / its
    redistribution axis separates the two, with permutation p."""
    from fgrgeom import features as F  # sibling import inside function
    r = anchor(panel, k=k, include=include, rotate=True, n_perm=0)
    Zr, labels = r["Z_rot"], r["labels"]

    dop = {c: panel.doppler[:, i] for i, c in enumerate(panel.doppler_cols)}
    cpr = dop["Percentil_CPR"]
    uta = dop["Percentil_UTA"]
    vel = F.velocity_features(panel, log=False).reindex(panel.ids)
    slope_efw = vel["slope_efw_z_ig21"].to_numpy(float)
    drop = vel["efw_centile_drop"].to_numpy(float)

    constit = (cpr >= 50) & (uta <= 50) & (np.abs(slope_efw) <= 0.1)
    placent = (cpr <= 10) & (drop >= 20)
    # require the defining variables observed
    constit &= ~np.isnan(cpr) & ~np.isnan(uta) & ~np.isnan(slope_efw)
    placent &= ~np.isnan(cpr) & ~np.isnan(drop)
    grp = np.full(len(cpr), np.nan)
    grp[constit] = 0
    grp[placent] = 1

    # Test every rotated axis for pole separation; no clean redistribution axis
    # is assumed to exist (it usually does not -> placental signal rides on the
    # severity axis). Report all so the reader sees where separation lives.
    rows = []
    for axis in range(Zr.shape[1]):
        auc, p, n = _perm_auc(Zr[:, axis], grp, n_perm)
        rows.append(dict(axis=axis, label=labels[axis][0], metric="auc",
                         value=auc, p=p, n=n))
    return {"n_constitutional": int(constit.sum()),
            "n_placental": int(placent.sum()),
            "labels": labels, "separation": pd.DataFrame(rows)}


def main():
    panel = P.load_panel()
    print("=== axis labels + held-out anchors (FA on biom+doppler, varimax) ===")
    r = anchor(panel, k=3)
    for a, (lab, det) in enumerate(r["labels"]):
        print(f"axis {a}: {lab:14s} {det}")
    print()
    with pd.option_context("display.width", 140,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(r["anchors"].sort_values(["anchor", "axis"]).to_string(index=False))
    print("\n=== constitutional vs placental pole separation ===")
    pt = pole_test(panel, k=3)
    print(f"n_constitutional={pt['n_constitutional']}  "
          f"n_placental={pt['n_placental']}")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(pt["separation"].to_string(index=False))


if __name__ == "__main__":
    main()
