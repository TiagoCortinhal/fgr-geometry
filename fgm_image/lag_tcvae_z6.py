"""
Longitudinal beta-TCVAE at Z=6 on config-B inputs (biometry + appearance-lag + GA).

6 growth variables (ac,hc,bpd,fl,efw z-scores + appearance-lag) -> Z=6 latent.
Disentanglement via the beta-TCVAE decomposition (Chen et al. 2018): the KL is split
into index-code MI + TOTAL CORRELATION + dimension-wise KL, and the total-correlation
term is up-weighted by beta. Minibatch-weighted-sampling estimator of the aggregate
posterior. Unlike a plain KL-beta VAE, this penalizes dependence BETWEEN latent dims
specifically, so it decorrelates axes without over-penalizing reconstruction.

5-fold GroupKFold OOF -> eff-dim, birth-pct r, SGA/LGA AUC, inter-dim |corr| (disentanglement).
Full-data latent + trajectories saved for plotting.

Inputs: results/img_align/{_merged_seq.npz,_lag_seq.npz,_merged_labels.npz,_citus_groups.csv}
Output: results/img_align/lag_tcvae_z6_results.json, _lagB_tcvae_z6_traj.npy
"""
import json, math, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

IMG = "/Users/tiago/dev/fgr-geometry/results/img_align"
torch.manual_seed(0); np.random.seed(0)


def eff_dim(M):
    M = M - M.mean(0); s = np.linalg.svd(M, compute_uv=False); v = s**2
    return float((v.sum()**2) / (v**2).sum())


class SeqTCVAE(nn.Module):
    def __init__(self, Din, H=32, Z=6):
        super().__init__()
        self.gru = nn.GRU(Din, H, batch_first=True)
        self.mu = nn.Linear(H, Z); self.lv = nn.Linear(H, Z)
        self.dec = nn.Sequential(nn.Linear(Z, H), nn.ReLU(), nn.Linear(H, Din))

    def encode(self, x):
        _, h = self.gru(x); h = h[-1]; return self.mu(h), self.lv(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(lv) * (0.5 * lv).exp()
        return self.dec(z), mu, lv, z


def _log_gauss(z, mu, lv):
    return -0.5 * (lv + math.log(2 * math.pi) + (z - mu) ** 2 / lv.exp())


def tc_decomposition(z, mu, lv, N):
    """Minibatch-weighted-sampling estimate of MI, TC, dimension-wise KL."""
    B, Z = z.shape
    logqz_cond = _log_gauss(z, mu, lv).sum(1)                       # log q(z|x)
    # pairwise log q(z_i | x_j)
    mat = _log_gauss(z.unsqueeze(1), mu.unsqueeze(0), lv.unsqueeze(0))  # (B,B,Z)
    logqz_prod = (torch.logsumexp(mat, 1) - math.log(N * B)).sum(1)  # sum over dims of log prod-marginal
    logqz = torch.logsumexp(mat.sum(2), 1) - math.log(N * B)         # log q(z) joint
    logpz = _log_gauss(z, torch.zeros_like(z), torch.zeros_like(z)).sum(1)
    mi = (logqz_cond - logqz).mean()          # index-code MI
    tc = (logqz - logqz_prod).mean()          # TOTAL CORRELATION
    dwkl = (logqz_prod - logpz).mean()        # dimension-wise KL
    return mi, tc, dwkl


def train_latent(Xc, L, N, beta_tc=4.0, epochs=300, H=32, Z=6, full=False):
    xt = torch.tensor(Xc); Lt = np.clip(L, 1, Xc.shape[1])
    m = SeqTCVAE(Xc.shape[2], H, Z); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        rec, mu, lv, z = m(xt)
        tgt = torch.stack([xt[i, :Lt[i]].mean(0) for i in range(len(xt))])
        rl = ((rec - tgt) ** 2).mean()
        mi, tc, dwkl = tc_decomposition(z, mu, lv, N)
        # beta-TCVAE loss: recon + MI + beta*TC + dimwise-KL  (anneal weight into KL parts)
        loss = rl + (mi + beta_tc * tc + dwkl) * 0.1
        loss.backward(); opt.step()
    m.eval()
    return m


def build_inputs():
    z = np.load(f"{IMG}/_merged_seq.npz", allow_pickle=True)
    X = z["X"].astype(np.float32); L = z["L"]; fids = z["fids"]; F = int(z["F"])
    lz = np.load(f"{IMG}/_lag_seq.npz", allow_pickle=True)
    lag = lz["lag_seq"].astype(np.float32); lagm = lz["lag_mask"].astype(np.float32)
    biom = X[:, :, :2 * F]; ga = X[:, :, -1:]
    lagf = np.concatenate([lag[:, :, None], lagm[:, :, None]], -1)
    Xc = np.concatenate([biom, lagf, ga], -1).astype(np.float32)
    return Xc, L, fids


def run(beta_tc=4.0, Z=6, extract_full=True):
    Xc, L, fids = build_inputs()
    lab = np.load(f"{IMG}/_merged_labels.npz", allow_pickle=True); birth = pd.Series(lab["birth"], index=lab["fids"])
    cg = pd.read_csv(f"{IMG}/_citus_groups.csv").set_index("Cod").grp_citus
    N, T, Din = Xc.shape; bp = birth.reindex(fids).values; grp = cg.reindex(fids).values
    sga = (grp == "SGA").astype(int); lga = (grp == "LGA").astype(int)
    Z_oof = np.zeros((N, Z))
    for tr, te in GroupKFold(5).split(Xc, groups=fids):
        m = train_latent(Xc[tr], L[tr], len(tr), beta_tc, Z=Z)
        with torch.no_grad(): mu, _ = m.encode(torch.tensor(Xc[te]))
        Z_oof[te] = mu.numpy()
    ok = np.isfinite(bp); ed = eff_dim(Z_oof)
    r_bp = float(np.corrcoef(Z_oof[ok] @ np.linalg.lstsq(Z_oof[ok], bp[ok], rcond=None)[0], bp[ok])[0, 1])
    def auc(y):
        yk = np.isfinite(y) & (y >= 0)
        p = LogisticRegression(max_iter=1000).fit(Z_oof[yk], y[yk]).predict_proba(Z_oof[yk])[:, 1]
        return float(roc_auc_score(y[yk], p))
    # disentanglement: mean/max off-diagonal |corr| of OOF latent
    C = np.corrcoef(Z_oof.T); off = np.abs(C - np.eye(Z))
    out = {"model": "Z6 beta-TCVAE (longitudinal)", "beta_tc": beta_tc, "Din": int(Din),
           "eff_dim": round(ed, 3), "birthpct_r": round(r_bp, 3),
           "SGA_auc": round(auc(sga), 3), "LGA_auc": round(auc(lga), 3),
           "interdim_absr_mean": round(off[off > 0].mean(), 3), "interdim_absr_max": round(off.max(), 3)}
    if extract_full:
        m = train_latent(Xc, L, N, beta_tc, Z=Z)
        Lt = np.clip(L, 1, T); traj = np.zeros((N, T, Z), np.float32)
        with torch.no_grad():
            for t in range(1, T + 1):
                mu, _ = m.encode(torch.tensor(Xc[:, :t])); traj[:, t - 1] = mu.numpy()
        np.save(f"{IMG}/_lagB_tcvae_z6_traj.npy", traj)
        out["traj_saved"] = "_lagB_tcvae_z6_traj.npy"
    return out


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    res = run(beta_tc=4.0, Z=6)
    json.dump(res, open(f"{IMG}/lag_tcvae_z6_results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
