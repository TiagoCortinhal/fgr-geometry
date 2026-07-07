"""
Train + 5-fold held-out evaluation of the three GRU-VAEs, and the
representation analyses (birth-pct reconstruction, image-vs-biometry manifold
comparison, image-vs-biometry target head-to-head).

All results in this module were computed with fgrgeom env (torch 2.12, sklearn
1.9). KL weights: BiomGRUVAE beta=0.1 (recovers held-out eff dim ~2.8),
ImgSeqVAE beta=1.0, JointGRUVAE beta=0.1.

Key findings persisted to results/img_align/*.json:
  - biometry GRU latent held-out eff dim ~2.77; birth-pct reconstruction r=0.53
  - image-trajectory GRU latent eff dim ~2.62; birth-pct r=0.00
  - joint fused latent eff dim ~5.1; birth-pct r=0.09 (fusion DILUTES growth)
  - concatenation of the two separate latents preserves it (r=0.52)
  - image vs biometry head-to-head: image WINS on GA (r 0.84 vs 0.06) and
    maternal BMI (0.45 vs 0.20); biometry wins on all size/caliper/clinical
    targets. => images are an appearance-outcome substrate, orthogonal to the
    fetal-size/FGR continuum.
"""
import numpy as np
import torch
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .models import BiomGRUVAE, ImgSeqVAE, JointGRUVAE, effective_dim

IMG = "/Users/tiago/PythonProject/fgr-geometry/results/img_align"


# ---------- training helpers ----------
def train_biom(X, L, S, beta=0.1, epochs=250, lr=3e-3):
    F = 5
    m = BiomGRUVAE(F=F, Sdim=S.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    xt, lt, st = torch.tensor(X), torch.tensor(L), torch.tensor(S)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        r, mu, lv = m(xt, lt, st)
        tgt, msk = xt[:, :, :F], xt[:, :, F:2 * F]
        loss = (((r - tgt) ** 2) * msk).sum() / msk.sum() / F - 0.5 * beta * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        loss.backward(); opt.step()
    m.eval()
    return m


def train_img(X, L, beta=1.0, epochs=200, lr=3e-3):
    K = X.shape[2] - 2
    m = ImgSeqVAE(K=K)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    xt, lt = torch.tensor(X), torch.tensor(L)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        r, mu, lv = m(xt, lt)
        tgt, msk = xt[:, :, :K], xt[:, :, K + 1:K + 2]
        loss = (((r - tgt) ** 2) * msk).sum() / msk.sum() / K - 0.5 * beta * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        loss.backward(); opt.step()
    m.eval()
    return m


def train_joint(X, L, F=5, K=32, beta=0.1, epochs=250, lr=3e-3):
    m = JointGRUVAE(F=F, K=K)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    xt, lt = torch.tensor(X), torch.tensor(L)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        rb, ri, mu, lv = m(xt, lt)
        bt, bm = xt[:, :, :F], xt[:, :, F:2 * F]
        it, im = xt[:, :, 2 * F:2 * F + K], xt[:, :, 2 * F + K:2 * F + K + 1]
        loss = ((((rb - bt) ** 2) * bm).sum() / bm.sum().clamp(min=1) / F
                + (((ri - it) ** 2) * im).sum() / im.sum().clamp(min=1) / K
                - 0.5 * beta * torch.mean(1 + lv - mu.pow(2) - lv.exp()))
        loss.backward(); opt.step()
    m.eval()
    return m


def encode_all(m, X, L, S=None):
    with torch.no_grad():
        if S is not None:
            _, mu, _ = m(torch.tensor(X), torch.tensor(L), torch.tensor(S))
        else:
            out = m(torch.tensor(X), torch.tensor(L))
            mu = out[-2]
    return mu.numpy()


# ---------- evaluation ----------
def heldout_eff_dim(X, L, train_fn, S=None, folds=5):
    Z = None
    effs = []
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        m = train_fn(X[tr], L[tr], S[tr]) if S is not None else train_fn(X[tr], L[tr])
        mu = encode_all(m, X, L, S)
        if Z is None:
            Z = np.zeros((len(X), mu.shape[1]))
        Z[te] = mu[te]
        effs.append(effective_dim(mu[te]))
    return np.mean(effs), np.std(effs), Z


def reconstruct_r(Z, y, groups=None, alpha=5):
    """CV correlation of Ridge(Z)->y. group-aware if groups given."""
    m = np.isfinite(y)
    cv = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=0)
    kw = {"groups": groups[m]} if groups is not None else {}
    p = cross_val_predict(Ridge(alpha=alpha), Z[m], y[m], cv=cv, **kw)
    return np.corrcoef(p, y[m])[0, 1]


def auc_cv(Z, y, groups=None, C=0.5):
    m = np.isfinite(y)
    if (y[m] == 1).sum() < 20:
        return None
    cv = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=0)
    kw = {"groups": groups[m]} if groups is not None else {}
    p = cross_val_predict(LogisticRegression(max_iter=2000, C=C), Z[m], y[m], cv=cv,
                          method="predict_proba", **kw)[:, 1]
    return roc_auc_score(y[m], p)
