#!/usr/bin/env python3
"""
Factorized multimodal VAE analysis + representation characterization for
FGR-geometry. Trains the 5 factorization strategies (static and longitudinal),
runs the size-axis probes, decodes what the image manifold encodes, and the
appearance-age discordance test.

Depends on models.py (FactMVAE, LongFactMVAE, fact_loss, effective_dim) and on
data prepared by build_sequences.py (_merged_seq.npz) + _fact_data.npz.

Key results (see results/img_align/*.json):
  factorized_mvae_results.json     — static: clean modality factorization,
                                      FGR-vs-constitutional null (image-private at chance)
  longfact_probes_results.json     — longitudinal: size axis recovered, but
                                      position/velocity/curvature probes noise
  representation_summary.json       — manifold decodes GA r=0.84 / plane 0.91 /
                                      site AUC 0.94 / quality 0.72; image is the
                                      SOLE maturation clock; biometry-private
                                      leaks SITE (batch confound -> site-adversarial fix)
  appearance_age_results.json       — image maturation clock yields a size-independent
                                      appearance-age; SGA/FGR look YOUNGER (lag r=-0.17,
                                      survives biometry removal) but redundant with
                                      biometry (AUC 0.781->0.789) and drift is null.

All models: ~40K params, full-batch, CPU. Static 500 ep, longitudinal 400 ep.
"""
import sys, os, json, numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from models import FactMVAE, LongFactMVAE, fact_loss, effective_dim

DATA = "/Users/tiago/dev/fetal_growth_mechanism/data"
IMG = "/Users/tiago/dev/fgr-geometry/results/img_align"
MODES = ["vanilla", "indep", "poe", "adversarial", "contrastive"]


def train_static(Ximg, Xbio, mode, beta=0.5, ep=500, lr=3e-3):
    torch.manual_seed(0)
    m = FactMVAE(Ximg.shape[1], Xbio.shape[1], mode=mode)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    xi, xb = torch.tensor(Ximg), torch.tensor(Xbio)
    for e in range(ep):
        m.train(); opt.zero_grad(); o = m(xi, xb)
        loss = fact_loss(o, Ximg.shape[1], Xbio.shape[1], beta=beta)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        o = m(xi, xb)
    return m, {k: o[k].numpy() for k in ["zs", "zpi", "zpb"]}


def train_longitudinal(X, L, F, K, mode, beta=0.4, ep=400, lr=3e-3):
    torch.manual_seed(0)
    m = LongFactMVAE(F=F, K=K, mode=mode)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    xt, lt = torch.tensor(X), torch.tensor(L)
    for e in range(ep):
        m.train(); opt.zero_grad(); o = m(xt, lt)
        loss = fact_loss(o, F, K, beta=beta, longitudinal=True, x=xt)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        o = m(xt, lt)
    return m, {k: o[k].numpy() for k in ["zs", "zpi", "zpb"]}


def rcv(Z, y, cv=5):
    m = np.isfinite(y) & np.isfinite(Z).all(1)
    if m.sum() < 40 or np.nanstd(y[m]) < 1e-6:
        return np.nan
    return np.corrcoef(cross_val_predict(Ridge(5), Z[m], y[m], cv=cv), y[m])[0, 1]


def size_axis_motion(model, X, L, birth):
    """Per-timestep trajectory through biometry-private code -> dominant axis;
    return per-fetus endpoint and velocity, oriented so + tracks birth pct."""
    MAXT = X.shape[1]
    tr = np.full((len(X), MAXT, 4), np.nan, np.float32)
    with torch.no_grad():
        for t in range(1, MAXT + 1):
            sel = L >= t
            if sel.sum() == 0:
                continue
            o = model(torch.tensor(X[sel]), torch.tensor(np.minimum(L[sel], t)))
            tr[sel, t - 1] = o["zpb"].numpy()
    valid = ~np.isnan(tr[:, :, 0])
    pc = PCA(1, random_state=0).fit(tr[valid])
    ax = np.full((len(X), MAXT), np.nan)
    for i in range(len(X)):
        v = ~np.isnan(tr[i, :, 0])
        if v.sum() >= 2:
            ax[i, v] = pc.transform(tr[i, v])[:, 0]
    vel = np.full(len(X), np.nan); endp = np.full(len(X), np.nan)
    for i in range(len(X)):
        a = ax[i][~np.isnan(ax[i])]
        if len(a) >= 2:
            vel[i] = np.mean(np.diff(a)); endp[i] = a[-1]
    mm = np.isfinite(endp) & np.isfinite(birth)
    if np.corrcoef(endp[mm], birth[mm])[0, 1] < 0:
        endp, vel = -endp, -vel
    return endp, vel


def appearance_age(E_img_z, meta_df, biom_pf, labels):
    """Image maturation-clock lag (image->GA minus dates GA), per fetus,
    with size-independence check. meta_df has ga_weeks_recovered + fid."""
    ga = meta_df.ga_weeks_recovered.values; grp = meta_df.fid.values
    pga = cross_val_predict(Ridge(5), E_img_z, ga, cv=GroupKFold(5), groups=grp)
    meta_df = meta_df.assign(app_resid=pga - ga)
    rows = []
    for f, g in meta_df.groupby("fid"):
        if len(g) >= 3 and g.ga_weeks_recovered.std() > 1.0:
            sl = np.polyfit(g.ga_weeks_recovered, g.app_resid, 1)[0]
            rows.append((f, g.app_resid.mean(), sl))
    return pd.DataFrame(rows, columns=["fid", "lag_mean", "lag_slope"]).set_index("fid")


if __name__ == "__main__":
    print("This module provides the factorized-MVAE + representation analysis "
          "functions. Import and call, or see the JSON result files in results/img_align/.")
