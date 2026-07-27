"""
Does the appearance-age lag help the temporal (GRU-VAE) representation?

Three input configurations on the SAME biometry visit sequence + fused GA timeline:
  A. biom + image PCA-32        (baseline: raw image embedding channel)
  B. biom + lag scalar          (lag REPLACES the image channel)
  C. biom + image PCA-32 + lag  (lag as an EXTRA channel alongside images)

Per config, 5-fold GroupKFold: train a GRU-VAE, take held-out latent means, and report
  - held-out effective dimension of the latent (participation-ratio style)
  - latent -> birth-percentile correlation (growth axis)
  - latent -> SGA and LGA AUC (linear probe)
so we can see whether distilling images to the lag adds growth-relevant structure the raw
image channel did not (raw image trajectory was previously orthogonal to growth, r~0.00).

Inputs (results/img_align/): _merged_seq.npz, _merged_labels.npz, _lag_seq.npz, _citus_groups.csv
Output: results/img_align/lag_in_gruvae_results.json
"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

IMG = "/Users/tiago/dev/fgr-geometry/results/img_align"
torch.manual_seed(0); np.random.seed(0)


def eff_dim(M):
    M = M - M.mean(0)
    s = np.linalg.svd(M, compute_uv=False)
    v = s**2
    return float((v.sum()**2) / (v**2).sum())


class SeqVAE(nn.Module):
    def __init__(self, Din, H=32, Z=8):
        super().__init__()
        self.gru = nn.GRU(Din, H, batch_first=True)
        self.mu = nn.Linear(H, Z); self.lv = nn.Linear(H, Z)
        self.dec = nn.Sequential(nn.Linear(Z, H), nn.ReLU(), nn.Linear(H, Din))

    def encode(self, x, l):
        _, h = self.gru(x)
        h = h[-1]
        return self.mu(h), self.lv(h)

    def forward(self, x, l):
        mu, lv = self.encode(x, l)
        z = mu + torch.randn_like(lv) * (0.5 * lv).exp()
        # decode to per-timestep (broadcast): reconstruct the mean feature vector
        rec = self.dec(z)
        return rec, mu, lv


def build_inputs(cfg):
    z = np.load(f"{IMG}/_merged_seq.npz", allow_pickle=True)
    X = z["X"].astype(np.float32); L = z["L"]; fids = z["fids"]; F = int(z["F"]); K = int(z["K"])
    lz = np.load(f"{IMG}/_lag_seq.npz", allow_pickle=True)
    lag = lz["lag_seq"].astype(np.float32); lagm = lz["lag_mask"].astype(np.float32)
    biom = X[:, :, :2 * F]              # 5 z + 5 mask
    img = X[:, :, 2 * F:2 * F + K + 1]  # 32 img + img_mask
    ga = X[:, :, -1:]                   # GA_norm
    lagf = np.concatenate([lag[:, :, None], lagm[:, :, None]], -1)  # lag + lag_mask
    if cfg == "A": parts = [biom, img, ga]
    elif cfg == "B": parts = [biom, lagf, ga]
    elif cfg == "C": parts = [biom, img, lagf, ga]
    Xc = np.concatenate(parts, -1).astype(np.float32)
    return Xc, L, fids


def train_eval(cfg, epochs=250, H=32, Z=8, beta=0.1):
    Xc, L, fids = build_inputs(cfg)
    lab = np.load(f"{IMG}/_merged_labels.npz", allow_pickle=True)
    lfids = lab["fids"]; birth = pd.Series(lab["birth"], index=lfids)
    cg = pd.read_csv(f"{IMG}/_citus_groups.csv").set_index("Cod").grp_citus
    N, T, Din = Xc.shape
    Lt = np.clip(L, 1, T)
    bp = birth.reindex(fids).values
    grp = cg.reindex(fids).values
    sga = (grp == "SGA").astype(int); lga = (grp == "LGA").astype(int)
    Z_oof = np.zeros((N, Z))
    for tr, te in GroupKFold(5).split(Xc, groups=fids):
        xt = torch.tensor(Xc[tr])
        m = SeqVAE(Din, H, Z)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        # reconstruction target = mean over valid timesteps of the feature vector
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            rec, mu, lv = m(xt, L[tr])
            tgt = torch.stack([xt[i, :Lt[tr][i]].mean(0) for i in range(len(tr))])
            rl = ((rec - tgt)**2).mean()
            kl = (-0.5 * (1 + lv - mu**2 - lv.exp()).sum(1)).mean()
            (rl + beta * kl).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            mu, _ = m.encode(torch.tensor(Xc[te]), L[te])
        Z_oof[te] = mu.numpy()
    ok = np.isfinite(bp)
    ed = eff_dim(Z_oof)
    r_bp = float(np.corrcoef(Z_oof[ok] @ np.linalg.lstsq(Z_oof[ok], bp[ok], rcond=None)[0], bp[ok])[0, 1])
    def auc(y):
        yok = np.isfinite(y) & (y >= 0)
        if len(np.unique(y[yok])) < 2: return float("nan")
        p = LogisticRegression(max_iter=1000).fit(Z_oof[yok], y[yok]).predict_proba(Z_oof[yok])[:, 1]
        return float(roc_auc_score(y[yok], p))
    return {"config": cfg, "Din": int(Din), "eff_dim": round(ed, 3),
            "birthpct_r": round(r_bp, 3), "SGA_auc": round(auc(sga), 3), "LGA_auc": round(auc(lga), 3)}


if __name__ == "__main__":
    res = {c: train_eval(c) for c in ["A", "B", "C"]}
    json.dump(res, open(f"{IMG}/lag_in_gruvae_results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
