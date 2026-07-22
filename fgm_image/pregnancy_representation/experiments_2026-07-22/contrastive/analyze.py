"""Turn battery_progress.json into effect sizes + CIs + perm p."""
import json, numpy as np
from scipy.stats import pearsonr

def boot_ci_r(bphat, true, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    bphat = np.asarray(bphat); true = np.asarray(true)
    n_obs = len(bphat)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        rs.append(pearsonr(bphat[idx], true[idx])[0])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))

def analyze(path="contrastive_traj/battery_progress.json"):
    R = json.load(open(path))
    out = {"config": R["config"], "variants": {}}
    for v in ['a', 'b', 'c']:
        real = R["real"][v]
        real_r = np.array([x["r"] for x in real])
        real_eta = np.array([x["eta2"] for x in real])
        real_sil = np.array([x["sil"] for x in real])
        gash = R["ga_shuffle"][v]
        ga_r = np.array([x["r"] for x in gash])
        cp = R["canonical_preds"][v]
        ci = boot_ci_r(cp["bphat"], cp["true"])
        entry = {
            "real_r_mean": float(real_r.mean()), "real_r_sd": float(real_r.std()),
            "real_r_seeds": real_r.tolist(),
            "canonical_r": float(pearsonr(cp["bphat"], cp["true"])[0]),
            "canonical_r_ci95": list(ci),
            "real_eta2_mean": float(real_eta.mean()),
            "real_sil_mean": float(real_sil.mean()),
            "ga_shuffle_r_mean": float(ga_r.mean()), "ga_shuffle_r_sd": float(ga_r.std()),
        }
        if v in R["label_shuffle"] and R["label_shuffle"][v]:
            ls = np.array([x["r"] for x in R["label_shuffle"][v]])
            lse = np.array([x["eta2"] for x in R["label_shuffle"][v]])
            # perm p: real (canonical single-seed, matched: label_shuffle uses seed=0) vs null
            real_canon = entry["canonical_r"]
            p = (1 + np.sum(ls >= real_canon)) / (1 + len(ls))
            entry.update({
                "label_shuffle_n": int(len(ls)),
                "label_shuffle_r_mean": float(ls.mean()), "label_shuffle_r_sd": float(ls.std()),
                "label_shuffle_r_p95": float(np.percentile(ls, 95)),
                "label_shuffle_r_max": float(ls.max()),
                "label_shuffle_eta2_mean": float(lse.mean()),
                "perm_p_vs_labelshuffle": float(p),
            })
        out["variants"][v] = entry
    return out, R
