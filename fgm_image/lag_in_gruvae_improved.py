"""
Retrain the longitudinal config-B GRU-VAE with the IMPROVED appearance-lag
(multi-layer fusion clock + median per-visit aggregation) and compare head-to-head
against the ORIGINAL lag, same session (avoids seed/version drift).

Config B: per-visit seq = biometry(10) + lag(2: scalar+mask) + GA(1) = Din 13; GRU H=32; Z=8; beta=0.1;
unsupervised (reconstructs seq-mean). birth-pct / SGA / LGA are eval-only (never inputs).

Swaps only the lag channel: _lag_seq.npz (original) vs _lag_seq_improved.npz (this session's winner).
5-fold GroupKFold OOF: eff_dim, birth-pct r, SGA/LGA AUC. Full-data latent saved for the winner.
"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

IMG = "/Users/tiago/dev/fgr-geometry/results/img_align"


def eff_dim(M):
    M = M - M.mean(0); s = np.linalg.svd(M, compute_uv=False); v = s**2
    return float((v.sum()**2) / (v**2).sum())


class SeqVAE(nn.Module):
    def __init__(self, Din, H=32, Z=8):
        super().__init__()
        self.gru = nn.GRU(Din, H, batch_first=True)
        self.mu = nn.Linear(H, Z); self.lv = nn.Linear(H, Z)
        self.dec = nn.Sequential(nn.Linear(Z, H), nn.ReLU(), nn.Linear(H, Din))

    def encode(self, x):
        _, h = self.gru(x); h = h[-1]; return self.mu(h), self.lv(h)

    def forward(self, x):
        mu, lv = self.encode(x); z = mu + torch.randn_like(lv) * (0.5 * lv).exp()
        return self.dec(z), mu, lv


def build_inputs(lag_file):
    z = np.load(f"{IMG}/_merged_seq.npz", allow_pickle=True)
    X = z["X"].astype(np.float32); L = z["L"]; fids = z["fids"]; F = int(z["F"])
    lz = np.load(f"{IMG}/{lag_file}", allow_pickle=True)
    lag = lz["lag_seq"].astype(np.float32); lagm = lz["lag_mask"].astype(np.float32)
    biom = X[:, :, :2 * F]; ga = X[:, :, -1:]
    lagf = np.concatenate([lag[:, :, None], lagm[:, :, None]], -1)
    return np.concatenate([biom, lagf, ga], -1).astype(np.float32), L, fids


def train_eval(lag_file, epochs=250, H=32, Z=8, beta=0.1, extract=False, tag=""):
    Xc, L, fids = build_inputs(lag_file)
    lab = np.load(f"{IMG}/_merged_labels.npz", allow_pickle=True); birth = pd.Series(lab["birth"], index=lab["fids"])
    cg = pd.read_csv(f"{IMG}/_citus_groups.csv").set_index("Cod").grp_citus
    N, T, Din = Xc.shape; Lt = np.clip(L, 1, T)
    bp = birth.reindex(fids).values; grp = cg.reindex(fids).values
    sga = (grp == "SGA").astype(int); lga = (grp == "LGA").astype(int); Zoof = np.zeros((N, Z))
    for tr, te in GroupKFold(5).split(Xc, groups=fids):
        xt = torch.tensor(Xc[tr]); m = SeqVAE(Din, H, Z); opt = torch.optim.Adam(m.parameters(), 1e-3)
        for ep in range(epochs):
            m.train(); opt.zero_grad(); rec, mu, lv = m(xt)
            tgt = torch.stack([xt[i, :Lt[tr][i]].mean(0) for i in range(len(tr))])
            (((rec - tgt)**2).mean() + beta * (-0.5 * (1 + lv - mu**2 - lv.exp()).sum(1)).mean()).backward(); opt.step()
        m.eval()
        with torch.no_grad(): mu, _ = m.encode(torch.tensor(Xc[te])); Zoof[te] = mu.numpy()
    ok = np.isfinite(bp)
    r_bp = float(np.corrcoef(Zoof[ok] @ np.linalg.lstsq(Zoof[ok], bp[ok], rcond=None)[0], bp[ok])[0, 1])
    def auc(y):
        yk = np.isfinite(y) & (y >= 0)
        return float(roc_auc_score(y[yk], LogisticRegression(max_iter=1000).fit(Zoof[yk], y[yk]).predict_proba(Zoof[yk])[:, 1]))
    out = {"tag": tag, "lag_file": lag_file, "Din": int(Din), "eff_dim": round(eff_dim(Zoof), 3),
           "birthpct_r": round(r_bp, 3), "SGA_auc": round(auc(sga), 3), "LGA_auc": round(auc(lga), 3)}
    if extract:
        m = SeqVAE(Din, H, Z); opt = torch.optim.Adam(m.parameters(), 1e-3); xt = torch.tensor(Xc)
        for ep in range(epochs):
            m.train(); opt.zero_grad(); rec, mu, lv = m(xt)
            tgt = torch.stack([xt[i, :Lt[i]].mean(0) for i in range(N)])
            (((rec - tgt)**2).mean() + beta * (-0.5 * (1 + lv - mu**2 - lv.exp()).sum(1)).mean()).backward(); opt.step()
        m.eval(); traj = np.zeros((N, T, Z), np.float32)
        with torch.no_grad():
            for t in range(1, T + 1):
                mu, _ = m.encode(xt[:, :t]); traj[:, t - 1] = mu.numpy()
        np.save(f"{IMG}/_lagB_improved_traj.npy", traj); out["traj_saved"] = "_lagB_improved_traj.npy"
    return out


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    res = {"original_lag": train_eval("_lag_seq.npz", tag="config-B original lag"),
           "improved_lag": train_eval("_lag_seq_improved.npz", tag="config-B improved lag (fuse+median)", extract=True)}
    json.dump(res, open(f"{IMG}/lag_improved_gruvae_results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
