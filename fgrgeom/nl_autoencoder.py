import json
import pathlib
import numpy as np
from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import featuresets as F
from fgrgeom import latent as L

try:
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler

SEED = C.SEED
LATENT_K = 6
N_FOLDS = 5
HIDE_FRAC = 0.30          # fraction of observed entries hidden for held-out recon
HIDDEN = 64
EPOCHS = 400
LR = 3e-3
WD = 1e-4
BETA = 1.0               # KL weight (VAE)


def _moments(X, M):
    mu = np.array([X[M[:, j], j].mean() if M[:, j].any() else 0.0
                   for j in range(X.shape[1])])
    sd = np.array([X[M[:, j], j].std() if M[:, j].sum() > 1 else 1.0
                   for j in range(X.shape[1])])
    sd[sd == 0] = 1.0
    return mu, sd


def _std(X, M, mu, sd):
    Xs = (X - mu) / sd
    Xs[~M] = 0.0
    return Xs


if HAVE_TORCH:
    class AE(nn.Module):
        def __init__(self, d, k, hidden=HIDDEN, vae=False):
            super().__init__()
            self.vae = vae
            self.enc = nn.Sequential(
                nn.Linear(2 * d, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU())
            self.head_mu = nn.Linear(hidden, k)
            if vae:
                self.head_lv = nn.Linear(hidden, k)
            self.dec = nn.Sequential(
                nn.Linear(k, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, d))

        def encode(self, x, m):
            h = self.enc(torch.cat([x, m], dim=1))
            mu = self.head_mu(h)
            if self.vae:
                return mu, self.head_lv(h)
            return mu, None

        def forward(self, x, m):
            mu, lv = self.encode(x, m)
            if self.vae and self.training:
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            else:
                z = mu
            return self.dec(z), mu, lv


def _train_torch(Xtr, Mtr, d, k, vae, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AE(d, k, vae=vae)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    x = torch.tensor(Xtr, dtype=torch.float32)
    m = torch.tensor(Mtr.astype(np.float32))
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        xhat, mu, lv = model(x, m)
        se = ((xhat - x) ** 2) * m
        recon = se.sum() / m.sum().clamp(min=1.0)
        loss = recon
        if vae:
            kl = -0.5 * torch.mean(torch.sum(1 + lv - mu ** 2 - lv.exp(), dim=1))
            loss = recon + BETA * kl / d
        loss.backward()
        opt.step()
    model.eval()
    return model


def _heldout_recon_torch(model, Xin, Min, Xtrue, Hide):
    """Min = input mask (observed and NOT hidden). Hide = entries to score on."""
    with torch.no_grad():
        x = torch.tensor(Xin, dtype=torch.float32)
        m = torch.tensor(Min.astype(np.float32))
        xhat, _, _ = model(x, m)
        xhat = xhat.numpy()
    err = ((xhat - Xtrue) ** 2)[Hide]
    return err


def _latent_torch(model, Xs, M):
    with torch.no_grad():
        mu, _ = model.encode(torch.tensor(Xs, dtype=torch.float32),
                             torch.tensor(M.astype(np.float32)))
    return mu.numpy()


def _fa_heldout_recon(fa, Xin_raw, Min, Xtrue_raw, Hide, mu_s, sd_s):
    """FA predicts hidden entries from observed via posterior, in standardized
    space, then descored. Xtrue_raw standardized for comparison consistency."""
    Z, _ = fa.transform(Xin_raw, Min)
    Xhat_s = Z @ fa.W_.T + fa.mu_
    Xtrue_s = (Xtrue_raw - mu_s) / sd_s
    err = ((Xhat_s - Xtrue_s) ** 2)[Hide]
    return err


def _outcome_eval(Ztr, Ztr_full_ids, ytr, Zte, yte, kind):
    sc = StandardScaler().fit(Ztr)
    Ztr = sc.transform(Ztr)
    Zte = sc.transform(Zte)
    ok_tr = ~np.isnan(ytr)
    ok_te = ~np.isnan(yte)
    if kind == "clf":
        if len(np.unique(ytr[ok_tr])) < 2 or ok_te.sum() < 5 \
                or len(np.unique(yte[ok_te])) < 2:
            return None
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Ztr[ok_tr], ytr[ok_tr])
        p = clf.predict_proba(Zte[ok_te])[:, 1]
        return roc_auc_score(yte[ok_te], p)
    else:
        rg = Ridge(alpha=1.0)
        rg.fit(Ztr[ok_tr], ytr[ok_tr])
        return r2_score(yte[ok_te], rg.predict(Zte[ok_te]))


def run_set(panel, set_name, k=LATENT_K, seed=SEED):
    X, M, names = F.build(panel, set_name)
    n, d = X.shape
    y_sga = panel.outcomes["severe_sga"].to_numpy(dtype=float)
    y_cent = panel.outcomes["percentile_birth_pop"].to_numpy(dtype=float)

    methods = ["MEAN", "FA", "AE"] + (["VAE"] if HAVE_TORCH else [])
    recon = {m: [] for m in methods}
    auc = {m: [] for m in methods}
    r2 = {m: [] for m in methods}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)

    for fold, (tr, te) in enumerate(kf.split(np.arange(n))):
        Xtr, Mtr = X[tr], M[tr]
        Xte, Mte = X[te], M[te]
        mu_s, sd_s = _moments(Xtr, Mtr)

        # held-out hide mask on the test rows: hide HIDE_FRAC of observed entries
        Hide = np.zeros_like(Mte)
        obs_idx = np.argwhere(Mte)
        nh = int(round(HIDE_FRAC * len(obs_idx)))
        sel = rng.choice(len(obs_idx), size=nh, replace=False)
        Hide[obs_idx[sel, 0], obs_idx[sel, 1]] = True
        Min = Mte & ~Hide  # input mask for test

        # trivial baseline: predict train mean (standardized 0) on hidden entries
        Xte_s_true = (Xte - mu_s) / sd_s
        recon["MEAN"].append((Xte_s_true ** 2)[Hide])

        # FA
        fa = L.FactorAnalysisMissing(k=k, seed=seed).fit(Xtr, Mtr)
        e = _fa_heldout_recon(fa, Xte, Min, Xte, Hide, mu_s, sd_s)
        recon["FA"].append(e)
        Ztr_fa, _ = fa.transform(Xtr, Mtr)
        Zte_fa, _ = fa.transform(Xte, Mte)
        a = _outcome_eval(Ztr_fa, None, y_sga[tr], Zte_fa, y_sga[te], "clf")
        r = _outcome_eval(Ztr_fa, None, y_cent[tr], Zte_fa, y_cent[te], "reg")
        if a is not None:
            auc["FA"].append(a)
        r2["FA"].append(r)

        if HAVE_TORCH:
            Xtr_s = _std(Xtr, Mtr, mu_s, sd_s)
            for mname, vae in [("AE", False), ("VAE", True)]:
                model = _train_torch(Xtr_s, Mtr, d, k, vae, seed + fold)
                # held-out recon (standardized space, matches FA scoring)
                Xin_s = _std(Xte, Min, mu_s, sd_s)
                Xte_s_true = (Xte - mu_s) / sd_s
                e = _heldout_recon_torch(model, Xin_s, Min, Xte_s_true, Hide)
                recon[mname].append(e)
                # latent + outcomes (full observed input)
                Ztr_z = _latent_torch(model, _std(Xtr, Mtr, mu_s, sd_s), Mtr)
                Zte_z = _latent_torch(model, _std(Xte, Mte, mu_s, sd_s), Mte)
                a = _outcome_eval(Ztr_z, None, y_sga[tr], Zte_z, y_sga[te], "clf")
                r = _outcome_eval(Ztr_z, None, y_cent[tr], Zte_z, y_cent[te], "reg")
                if a is not None:
                    auc[mname].append(a)
                r2[mname].append(r)

    out = {"set": set_name, "n": n, "d": d, "k": k,
           "n_folds": N_FOLDS, "hide_frac": HIDE_FRAC,
           "recon_mse": {}, "auc_severe_sga": {}, "r2_birth_centile": {}}
    for m in methods:
        allerr = np.concatenate(recon[m])
        out["recon_mse"][m] = {"mean": float(allerr.mean()),
                               "n_entries": int(allerr.size)}
        out["auc_severe_sga"][m] = {
            "mean": float(np.mean(auc[m])) if auc[m] else None,
            "folds": [float(v) for v in auc[m]]}
        out["r2_birth_centile"][m] = {
            "mean": float(np.mean(r2[m])) if r2[m] else None,
            "folds": [float(v) for v in r2[m]]}
    return out


def main():
    panel = P.load_panel()
    sets = ["minimal", "full"]
    results = {"have_torch": HAVE_TORCH, "latent_k": LATENT_K, "sets": {}}
    for s in sets:
        print(f"running {s} ...")
        r = run_set(panel, s)
        results["sets"][s] = r
        rm = r["recon_mse"]
        print(f"  recon MSE  " + "  ".join(
            f"{m}={rm[m]['mean']:.4f}" for m in rm))
        au = r["auc_severe_sga"]
        print(f"  AUC sevSGA " + "  ".join(
            f"{m}={au[m]['mean']:.3f}" if au[m]['mean'] is not None
            else f"{m}=NA" for m in au))
        r2d = r["r2_birth_centile"]
        print(f"  R2 centile " + "  ".join(
            f"{m}={r2d[m]['mean']:.3f}" if r2d[m]['mean'] is not None
            else f"{m}=NA" for m in r2d))
    res = pathlib.Path(__file__).resolve().parents[1] / "results" / "nl"
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "autoencoder.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", res / "autoencoder.json")
    return results


if __name__ == "__main__":
    main()
