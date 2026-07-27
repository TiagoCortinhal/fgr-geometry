"""
Echo-fusion investigation + lag-keep decision (canonical model = frozen config-B).

Runs the B+echo static-side-channel model across latent widths Z in {6,8,16,32},
plus a no-lag ablation at Z=8, and compares against base config B (biom+lag+GA, no echo).

Conclusion (see lag_keep_decision_results.json):
  - Echo fusion NEVER beats base config B at any width (inverted-U, peak at Z=16 which
    only ties base B on SGA and stays below on LGA; collapses by Z=32).
  - Removing lag is strictly worse (birth-r 0.307->0.266, SGA/LGA both drop).
  - DECISION: keep lag; canonical = frozen config-B; echocardio -> outcome variable only.

Depends on lag_in_gruvae_echo.py (SeqVAEecho, eff_dim, train_eval) and the same inputs
in results/img_align/. Reuse train_eval(Z=...) for the sweep; train_eval_nolag below for the ablation.
"""
import numpy as np, pandas as pd, torch, importlib.util
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

IMG = "/Users/tiago/dev/fgr-geometry/results/img_align"
_spec = importlib.util.spec_from_file_location(
    "le", "/Users/tiago/dev/fgr-geometry/fgm_image/lag_in_gruvae_echo.py")
le = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(le)


def build_inputs_nolag():
    z = np.load(f"{IMG}/_merged_seq.npz", allow_pickle=True)
    X = z["X"].astype(np.float32); L = z["L"]; fids = z["fids"]; F = int(z["F"])
    biom = X[:, :, :2 * F]; ga = X[:, :, -1:]
    Xc = np.concatenate([biom, ga], -1).astype(np.float32)   # NO lag: 10 + 1 = 11
    es = np.load(f"{IMG}/_echo_static.npz", allow_pickle=True); echo = es["echo"].astype(np.float32)
    return Xc, echo, L, fids


def train_eval_nolag(epochs=250, H=32, Z=8, beta=0.1):
    Xc, echo, L, fids = build_inputs_nolag()
    lab = np.load(f"{IMG}/_merged_labels.npz", allow_pickle=True); birth = pd.Series(lab["birth"], index=lab["fids"])
    cg = pd.read_csv(f"{IMG}/_citus_groups.csv").set_index("Cod").grp_citus
    N, T, Din = Xc.shape; Decho = echo.shape[1]; Lt = np.clip(L, 1, T)
    bp = birth.reindex(fids).values; grp = cg.reindex(fids).values
    sga = (grp == "SGA").astype(int); lga = (grp == "LGA").astype(int); Z_oof = np.zeros((N, Z))
    for tr, te in GroupKFold(5).split(Xc, groups=fids):
        xt = torch.tensor(Xc[tr]); et = torch.tensor(echo[tr])
        m = le.SeqVAEecho(Din, Decho, H, Z); opt = torch.optim.Adam(m.parameters(), 1e-3)
        for ep in range(epochs):
            m.train(); opt.zero_grad(); rec, mu, lv = m(xt, et)
            tgt = torch.cat([torch.stack([xt[i, :Lt[tr][i]].mean(0) for i in range(len(tr))]), et], 1)
            (((rec - tgt) ** 2).mean() + beta * (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            mu, _ = m.encode(torch.tensor(Xc[te]), torch.tensor(echo[te])); Z_oof[te] = mu.numpy()
    ok = np.isfinite(bp); ed = le.eff_dim(Z_oof)
    r_bp = float(np.corrcoef(Z_oof[ok] @ np.linalg.lstsq(Z_oof[ok], bp[ok], rcond=None)[0], bp[ok])[0, 1])
    def auc(y):
        yok = np.isfinite(y) & (y >= 0)
        p = LogisticRegression(max_iter=1000).fit(Z_oof[yok], y[yok]).predict_proba(Z_oof[yok])[:, 1]
        return float(roc_auc_score(y[yok], p))
    return {"config": "biom+GA+echo (NO lag)", "Din": int(Din), "eff_dim": round(ed, 3),
            "birthpct_r": round(r_bp, 3), "SGA_auc": round(auc(sga), 3), "LGA_auc": round(auc(lga), 3)}


if __name__ == "__main__":
    import json
    torch.manual_seed(0); np.random.seed(0)
    res = {"sweep": {f"Z{z}": le.train_eval(Z=z) for z in [6, 8, 16, 32]},
           "no_lag_Z8": train_eval_nolag()}
    print(json.dumps(res, indent=2))
