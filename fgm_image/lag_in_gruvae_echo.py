"""
Config B + static echocardio side-channel fused at the latent.

Base config B = biometry visit sequence + appearance-lag scalar + GA timeline (GRU).
NEW: static cardiac z-vector (13 z-scores + 13 mask = 26-d, one panel per fetus ~28-32wk,
from IMPACT_ecocardio_zscores_corrected.xlsx) is concatenated to the GRU's final hidden
state BEFORE the latent bottleneck — a per-fetus static context, not a fake time-course.

Reconstruction: the decoder reconstructs BOTH the mean biometry+lag+GA feature vector
(as in base B) AND the static echo vector, so the latent must encode cardiac info to
reconstruct it. Eval identical to base B: 5-fold GroupKFold OOF latent ->
eff-dim, birth-pct r, SGA/LGA AUC. Also reports cardiac-reconstruction r as a check
that the echo channel is actually used.

Inputs: results/img_align/{_merged_seq.npz,_merged_labels.npz,_lag_seq.npz,_echo_static.npz,_citus_groups.csv}
Output: results/img_align/lag_in_gruvae_echo_results.json  and  _lagBecho_traj.npy (full-data trajectories)
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


class SeqVAEecho(nn.Module):
    """GRU over the visit sequence + static echo fused at the latent."""
    def __init__(self, Din, Decho, H=32, Z=8):
        super().__init__()
        self.gru = nn.GRU(Din, H, batch_first=True)
        self.mu = nn.Linear(H + Decho, Z)
        self.lv = nn.Linear(H + Decho, Z)
        # decoder reconstructs [mean seq feature vector (Din) ; static echo (Decho)]
        self.dec = nn.Sequential(nn.Linear(Z, H), nn.ReLU(), nn.Linear(H, Din + Decho))

    def encode(self, x, e):
        _, h = self.gru(x); h = h[-1]
        h = torch.cat([h, e], dim=1)
        return self.mu(h), self.lv(h)

    def forward(self, x, e):
        mu, lv = self.encode(x, e)
        z = mu + torch.randn_like(lv) * (0.5 * lv).exp()
        return self.dec(z), mu, lv


def build_inputs():
    z = np.load(f"{IMG}/_merged_seq.npz", allow_pickle=True)
    X = z["X"].astype(np.float32); L = z["L"]; fids = z["fids"]; F = int(z["F"])
    lz = np.load(f"{IMG}/_lag_seq.npz", allow_pickle=True)
    lag = lz["lag_seq"].astype(np.float32); lagm = lz["lag_mask"].astype(np.float32)
    biom = X[:, :, :2 * F]
    ga = X[:, :, -1:]
    lagf = np.concatenate([lag[:, :, None], lagm[:, :, None]], -1)
    Xc = np.concatenate([biom, lagf, ga], -1).astype(np.float32)   # base config B
    es = np.load(f"{IMG}/_echo_static.npz", allow_pickle=True)
    echo = es["echo"].astype(np.float32)                            # (N, 26)
    assert list(es["fids"]) == list(fids), "echo fids misaligned"
    return Xc, echo, L, fids


def train_eval(epochs=250, H=32, Z=8, beta=0.1, extract_full=False):
    Xc, echo, L, fids = build_inputs()
    lab = np.load(f"{IMG}/_merged_labels.npz", allow_pickle=True)
    birth = pd.Series(lab["birth"], index=lab["fids"])
    cg = pd.read_csv(f"{IMG}/_citus_groups.csv").set_index("Cod").grp_citus
    N, T, Din = Xc.shape; Decho = echo.shape[1]
    Lt = np.clip(L, 1, T)
    bp = birth.reindex(fids).values
    grp = cg.reindex(fids).values
    sga = (grp == "SGA").astype(int); lga = (grp == "LGA").astype(int)
    Z_oof = np.zeros((N, Z)); echo_rec = np.zeros((N, Decho))
    for tr, te in GroupKFold(5).split(Xc, groups=fids):
        xt = torch.tensor(Xc[tr]); et = torch.tensor(echo[tr])
        m = SeqVAEecho(Din, Decho, H, Z); opt = torch.optim.Adam(m.parameters(), 1e-3)
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            rec, mu, lv = m(xt, et)
            tgt_seq = torch.stack([xt[i, :Lt[tr][i]].mean(0) for i in range(len(tr))])
            tgt = torch.cat([tgt_seq, et], dim=1)
            rl = ((rec - tgt)**2).mean()
            kl = (-0.5 * (1 + lv - mu**2 - lv.exp()).sum(1)).mean()
            (rl + beta * kl).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            mu, _ = m.encode(torch.tensor(Xc[te]), torch.tensor(echo[te]))
            rec, _, _ = m(torch.tensor(Xc[te]), torch.tensor(echo[te]))
        Z_oof[te] = mu.numpy(); echo_rec[te] = rec.numpy()[:, Din:]
    ok = np.isfinite(bp)
    ed = eff_dim(Z_oof)
    r_bp = float(np.corrcoef(Z_oof[ok] @ np.linalg.lstsq(Z_oof[ok], bp[ok], rcond=None)[0], bp[ok])[0, 1])
    def auc(y):
        yok = np.isfinite(y) & (y >= 0)
        if len(np.unique(y[yok])) < 2: return float("nan")
        p = LogisticRegression(max_iter=1000).fit(Z_oof[yok], y[yok]).predict_proba(Z_oof[yok])[:, 1]
        return float(roc_auc_score(y[yok], p))
    # cardiac reconstruction quality (only the z-score half, where mask=1)
    zc = echo[:, :13]; rc = echo_rec[:, :13]; mk = echo[:, 13:] > 0.5
    r_card = float(np.corrcoef(rc[mk], zc[mk])[0, 1])
    out = {"config": "B+echo", "Din": int(Din), "Decho": int(Decho), "eff_dim": round(ed, 3),
           "birthpct_r": round(r_bp, 3), "SGA_auc": round(auc(sga), 3), "LGA_auc": round(auc(lga), 3),
           "cardiac_recon_r": round(r_card, 3)}
    if extract_full:
        # full-data trajectories: per-timestep latent (encode prefix up to each t)
        m = SeqVAEecho(Din, Decho, H, Z); opt = torch.optim.Adam(m.parameters(), 1e-3)
        xt = torch.tensor(Xc); et = torch.tensor(echo)
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            rec, mu, lv = m(xt, et)
            tgt = torch.cat([torch.stack([xt[i, :Lt[i]].mean(0) for i in range(N)]), et], dim=1)
            rl = ((rec - tgt)**2).mean(); kl = (-0.5 * (1 + lv - mu**2 - lv.exp()).sum(1)).mean()
            (rl + beta * kl).backward(); opt.step()
        m.eval()
        traj = np.zeros((N, T, Z), np.float32)
        with torch.no_grad():
            for t in range(1, T + 1):
                mu, _ = m.encode(xt[:, :t], et)
                traj[:, t - 1] = mu.numpy()
        np.save(f"{IMG}/_lagBecho_traj.npy", traj)
        out["traj_saved"] = "_lagBecho_traj.npy"
    return out


if __name__ == "__main__":
    res = train_eval(extract_full=True)
    json.dump(res, open(f"{IMG}/lag_in_gruvae_echo_results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
