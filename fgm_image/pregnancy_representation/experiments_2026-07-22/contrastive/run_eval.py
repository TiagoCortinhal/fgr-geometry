"""Full adversarial evaluation: 3 variants x {real, label-shuffle, GA-shuffle}, held-out CV.
Headline metric = pooled held-out birthpct-r (Ridge train->test readout).
All labels used ONLY in loss / for eval grouping, NEVER as encoder input.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "contrastive_traj")
from traj_model import build_sequences, train_fold, encode_all, set_seed
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
import torch

def eta2_by_group(x, groups):
    """% variance in 1D x explained by categorical grouping."""
    x = np.asarray(x); groups = np.asarray(groups)
    grand = x.mean(); ss_tot = ((x - grand) ** 2).sum()
    ss_bet = 0.0
    for g in np.unique(groups):
        xg = x[groups == g]
        ss_bet += len(xg) * (xg.mean() - grand) ** 2
    return ss_bet / ss_tot if ss_tot > 0 else 0.0

def run_variant(mae, ga, nid, bp, variant, mode, device="cpu",
                seed=0, epochs=120, n_pca=48, z=16, shuffle_seed=None, **hp):
    """mode in {'real','label_shuffle','ga_shuffle'}.
    Returns dict with pooled held-out arrays + per-fold silhouette."""
    set_seed(seed)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(len(nid)), groups=nid))
    pooled_bphat, pooled_true, pooled_dec = [], [], []
    fold_sil = []
    for tr_v, te_v in splits:
        # --- PCA fit on TRAIN visits only ---
        pca = PCA(n_components=n_pca, random_state=seed).fit(mae[tr_v])
        mae_p = pca.transform(mae)
        gc, gs = ga[tr_v].mean(), ga[tr_v].std()
        ga_use = ga.copy()
        if mode == 'ga_shuffle':
            # shuffle GA-order WITHIN each fetus (temporal control): permute the
            # feature-visit assignment so GA channel no longer matches biology order
            rng = np.random.default_rng((shuffle_seed or 0) + 1000)
            for f in np.unique(nid):
                idx = np.where(nid == f)[0]
                perm = rng.permutation(len(idx))
                mae_p[idx] = mae_p[idx][perm]   # scramble which feature goes with which GA
        uniq, seqs = build_sequences(mae_p, ga_use, nid, gc, gs)
        uniq = np.array(uniq)
        tr_fet = set(nid[tr_v])
        tr_mask = np.array([u in tr_fet for u in uniq])
        tr_idx = np.where(tr_mask)[0]; te_idx = np.where(~tr_mask)[0]
        # per-fetus birthpct + train-defined deciles
        bpf = pd.Series(bp, index=nid).groupby(level=0).first()
        bp_uniq = bpf.loc[uniq].values
        bp_tr = bp_uniq[tr_idx]
        # deciles from TRAIN birthpct bins, applied to all
        qs = np.quantile(bp_tr, np.linspace(0, 1, 11))
        qs[0] = -np.inf; qs[-1] = np.inf
        dec_all = np.digitize(bp_uniq, qs[1:-1])
        dec_tr = dec_all[tr_idx].copy()
        bp_tr_std = (bp_tr - bp_tr.mean()) / bp_tr.std()
        # --- label shuffle: permute train labels among train fetuses ---
        if mode == 'label_shuffle':
            rng = np.random.default_rng(shuffle_seed or 0)
            pp = rng.permutation(len(dec_tr))
            dec_tr = dec_tr[pp]
            bp_tr_std = bp_tr_std[pp]
        seqs_tr = [seqs[i] for i in tr_idx]
        model = train_fold(seqs_tr, dec_tr, bp_tr_std, n_pca + 1, variant, device,
                           seed=seed, epochs=epochs, z=z, **hp)
        Z = encode_all(model, seqs, device)
        Ztr, Zte = Z[tr_idx], Z[te_idx]
        # honest readout: Ridge train latent -> train birthpct, predict test
        sc = StandardScaler().fit(Ztr)
        rg = Ridge(alpha=1.0).fit(sc.transform(Ztr), bp_tr)
        bphat = rg.predict(sc.transform(Zte))
        pooled_bphat.append(bphat)
        pooled_true.append(bp_uniq[te_idx])
        pooled_dec.append(dec_all[te_idx])
        # per-fold silhouette in standardized latent, grouped by test deciles
        Zte_s = StandardScaler().fit_transform(Zte)
        try:
            fold_sil.append(silhouette_score(Zte_s, dec_all[te_idx]))
        except Exception:
            fold_sil.append(np.nan)
    bphat = np.concatenate(pooled_bphat); true = np.concatenate(pooled_true)
    dec = np.concatenate(pooled_dec)
    r = pearsonr(bphat, true)[0]
    eta2 = eta2_by_group(bphat, dec)
    return dict(r=float(r), eta2=float(eta2), sil=float(np.nanmean(fold_sil)),
                bphat=bphat, true=true, dec=dec)
