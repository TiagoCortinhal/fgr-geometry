"""
Can the config-B GRU-VAE latent be disentangled, and what does each axis encode?

Two questions:
  1. INTERPRET: correlate each latent dim with the known inputs (5 biometry z, lag, GA,
     birth pct) -> read what each axis carries. (Done in the notebook; heatmap saved.)
  2. DISENTANGLE:
     (a) beta-sweep: raise the KL weight and measure whether the latent dims decorrelate
         (mean |inter-dim r|) or just collapse (effective dim -> 1) at a reconstruction cost.
     (b) post-hoc ICA rotation of the trained latent -> independent, singly-loaded axes,
         then re-read what each rotated axis encodes.

Metric of "cleanliness": for each input factor, MIG-proxy = |top latent corr| - |2nd latent
corr| (gap): large gap = that factor is captured by ONE axis (disentangled).

Inputs (results/img_align/): _merged_seq.npz, _lag_seq.npz, _merged_labels.npz, _citus_groups.csv
Output: results/img_align/lag_disentangle_results.json
"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import FastICA
IMG = "/Users/tiago/dev/fgr-geometry/results/img_align"
torch.manual_seed(0); np.random.seed(0)


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


def build_B():
    z = np.load(f"{IMG}/_merged_seq.npz", allow_pickle=True)
    X = z["X"].astype(np.float32); L = z["L"]; fids = z["fids"]; F = int(z["F"])
    lz = np.load(f"{IMG}/_lag_seq.npz", allow_pickle=True)
    lag = lz["lag_seq"].astype(np.float32); lagm = lz["lag_mask"].astype(np.float32)
    biom = X[:, :, :2 * F]; ga = X[:, :, -1:]
    lagf = np.concatenate([lag[:, :, None], lagm[:, :, None]], -1)
    Xc = np.concatenate([biom, lagf, ga], -1).astype(np.float32)
    return Xc, L, fids


def train_latent(Xc, L, beta, epochs=300, H=32, Z=8):
    N, T, Din = Xc.shape; Lt = np.clip(L, 1, T)
    m = SeqVAE(Din, H, Z); opt = torch.optim.Adam(m.parameters(), 1e-3)
    xt = torch.tensor(Xc); tgt = torch.stack([xt[i, :Lt[i]].mean(0) for i in range(N)])
    for ep in range(epochs):
        m.train(); opt.zero_grad(); rec, mu, lv = m(xt)
        rl = ((rec - tgt)**2).mean(); kl = (-0.5 * (1 + lv - mu**2 - lv.exp()).sum(1)).mean()
        (rl + beta * kl).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        mu, _ = m.encode(xt); rec, _, _ = m(xt)
        recon = float(((rec - tgt)**2).mean())
    return mu.numpy(), recon


def factor_matrix(Lat, FD):
    Z = Lat.shape[1]; C = np.zeros((Z, FD.shape[1]))
    for d in range(Z):
        for k, col in enumerate(FD.columns):
            ok = np.isfinite(Lat[:, d]) & np.isfinite(FD[col].values)
            C[d, k] = np.corrcoef(Lat[ok, d], FD[col].values[ok])[0, 1]
    return C


def mig_proxy(C):
    """mean over factors of (|top| - |2nd|) latent-corr gap."""
    A = np.abs(C)
    gaps = []
    for k in range(A.shape[1]):
        s = np.sort(A[:, k])[::-1]
        gaps.append(s[0] - s[1])
    return float(np.mean(gaps))


def inter_corr(Lat):
    ok = np.isfinite(Lat).all(1); LC = np.corrcoef(Lat[ok].T)
    Z = Lat.shape[1]; od = LC[~np.eye(Z, dtype=bool)]
    return float(np.abs(od).mean()), float(np.abs(od).max())


def run():
    Xc, L, fids = build_B()
    FD = pd.read_csv(f"{IMG}/_lagB_probe_features.csv")
    out = {"beta_sweep": [], "ica": {}}
    latents = {}
    for beta in [0.1, 0.5, 1.0, 2.0, 4.0]:
        Lat, recon = train_latent(Xc, L, beta)
        latents[beta] = Lat
        mic, mac = inter_corr(Lat)
        C = factor_matrix(Lat, FD)
        out["beta_sweep"].append({"beta": beta, "recon_mse": round(recon, 4),
                                  "eff_dim": round(eff_dim(Lat), 3),
                                  "inter_corr_mean": round(mic, 3), "inter_corr_max": round(mac, 3),
                                  "mig_proxy": round(mig_proxy(C), 3)})
    # ICA on the beta=0.1 latent (the one used for the trajectory figures)
    Lat = latents[0.1]; ok = np.isfinite(Lat).all(1)
    S = FastICA(n_components=Lat.shape[1], random_state=0, max_iter=1000).fit_transform(Lat[ok])
    Cica = np.zeros((S.shape[1], FD.shape[1])); FDok = FD[ok].reset_index(drop=True)
    for d in range(S.shape[1]):
        for k, col in enumerate(FD.columns):
            v = FDok[col].values; m2 = np.isfinite(v)
            Cica[d, k] = np.corrcoef(S[m2, d], v[m2])[0, 1]
    mic2, mac2 = inter_corr(S)
    out["ica"] = {"inter_corr_mean": round(mic2, 3), "inter_corr_max": round(mac2, 3),
                  "mig_proxy": round(mig_proxy(Cica), 3),
                  "axis_top_factor": {f"ic{d}": FD.columns[np.argmax(np.abs(Cica[d]))]
                                      for d in range(S.shape[1])}}
    json.dump(out, open(f"{IMG}/lag_disentangle_results.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
