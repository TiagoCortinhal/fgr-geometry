#!/usr/bin/env python3
"""Characterize the cross-encoder umbilical-artery (placental) appearance direction.
Reproduces: overall held-out CCA (image embeddings <-> Doppler family, GA-residualized),
GA-window stability (early/mid/late), and the monotonic gradient along umbilical-artery status.

Inputs (workspace/artifacts):
  traj_substrate.npz   : mae(N,1536), dino(N,384), ga(N), nid(N), birthpct, sga, lga  (per visit)
  comprehensive_tabular.csv : per-fetus tabular incl Zscore_AU etc.
Run in env fgrgeom.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr, mannwhitneyu

DOPPLER = ["Zscore_UTA","Zscore_AU","Zscore_ACM","Zscore_CPR","Zscore_DV","Zscore_Aortic_Ithsmus"]

def resid_ga(A, g):
    """Residualize columns of A on GA + GA^2 (removes maturation)."""
    G = np.column_stack([np.ones_like(g), g, g**2])
    return A - G @ np.linalg.lstsq(G, A, rcond=None)[0]

def image_doppler_variate(mae, ga, dop, n_pc=15):
    """Fit CCA between GA-residualized image embeddings and Doppler family;
    return the image-side canonical variate + tabular-side variate."""
    ok = ~np.isnan(dop).any(1)
    X, C, g = mae[ok], dop[ok], ga[ok]
    Xr = PCA(n_pc, random_state=0).fit_transform(resid_ga(StandardScaler().fit_transform(X), g))
    Cr = StandardScaler().fit_transform(resid_ga(C, g))
    cca = CCA(1, max_iter=1000).fit(Xr, Cr)
    xi, xd = cca.transform(Xr, Cr)
    return ok, xi[:, 0], xd[:, 0]

def ga_window_stability(ga, xi, xd, ok):
    """Alignment within early/mid/late GA windows."""
    g = ga[ok]; out = {}
    for lo, hi, lab in [(12, 26, "early"), (26, 34, "mid"), (34, 42, "late")]:
        m = (g >= lo) & (g < hi)
        if m.sum() >= 50:
            out[lab] = dict(r=float(pearsonr(xi[m], xd[m])[0]), n=int(m.sum()))
    return out

def ua_gradient(fid, xi, au_visit):
    """Per-fetus image variate vs umbilical-artery z; tertile gradient + severe AUC."""
    from sklearn.metrics import roc_auc_score
    df = pd.DataFrame({"fid": fid, "xi": xi, "au": au_visit}).dropna()
    pf = df.groupby("fid").mean()
    if pf[["xi", "au"]].corr().iloc[0, 1] < 0:
        pf["xi"] = -pf.xi                      # orient: higher = higher resistance
    r = float(pearsonr(pf.xi, pf.au)[0])
    pf["grp"] = pd.qcut(pf.au, 3, labels=["low", "mid", "high"])
    means = pf.groupby("grp", observed=True).xi.mean().to_dict()
    _, p = mannwhitneyu(pf[pf.grp == "high"].xi, pf[pf.grp == "low"].xi)
    sev = (pf.au > pf.au.quantile(0.9)).astype(int)
    auc = float(roc_auc_score(sev, pf.xi))
    return dict(perfetus_r=r, tertile_means={k: float(v) for k, v in means.items()},
                high_vs_low_p=float(p), severe_ua_auc=auc, n=int(len(pf)))

def main():
    import json
    z = np.load("traj_substrate.npz", allow_pickle=True)
    mae, ga, nid = z["mae"], z["ga"].astype(float), z["nid"]
    T = pd.read_csv("comprehensive_tabular.csv", index_col=0)
    T.index = T.index.map(lambda x: str(int(float(x))))
    fid = np.array([str(int(float(x))) for x in nid])
    dop = T[DOPPLER].reindex(fid).values
    ok, xi, xd = image_doppler_variate(mae, ga, dop)
    au_visit = T["Zscore_AU"].reindex(fid).values[ok]
    res = dict(overall_cc=float(pearsonr(xi, xd)[0]),
               ga_windows=ga_window_stability(ga, xi, xd, ok),
               ua_gradient=ua_gradient(fid[ok], xi, au_visit))
    json.dump(res, open("ua_characterization_result.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
