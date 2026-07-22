"""Adversarial battery. Saves incrementally to contrastive_traj/battery_progress.json.
- REAL: 5 seeds per variant (seed-variability) + canonical seed=0 pooled preds for bootstrap CI.
- LABEL_SHUFFLE null: N_SHUF runs per variant b,c (perm p for held-out r).
- GA_SHUFFLE: 5 seeds per variant.
"""
import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "contrastive_traj")
from run_eval import run_variant

def main(mae, ga, nid, bp, n_seeds=5, n_shuf=100, epochs=120, out="contrastive_traj/battery_progress.json"):
    res = {"config": {"n_seeds": n_seeds, "n_shuf": n_shuf, "epochs": epochs,
                      "n_fetus": int(pd.unique(nid).size), "n_visits": int(len(nid))},
           "real": {}, "label_shuffle": {}, "ga_shuffle": {}, "canonical_preds": {}}
    t0 = time.time()
    def save():
        json.dump(res, open(out, "w"))
    # REAL across seeds
    for v in ['a', 'b', 'c']:
        rs = []
        for s in range(n_seeds):
            r = run_variant(mae, ga, nid, bp, v, 'real', seed=s, epochs=epochs)
            rs.append({"r": r["r"], "eta2": r["eta2"], "sil": r["sil"]})
            if s == 0:
                res["canonical_preds"][v] = {"bphat": r["bphat"].tolist(),
                                             "true": r["true"].tolist(),
                                             "dec": r["dec"].astype(int).tolist()}
        res["real"][v] = rs
        save()
        print(f"[{time.time()-t0:.0f}s] REAL {v}: r={np.mean([x['r'] for x in rs]):.3f}")
    # GA_SHUFFLE across seeds
    for v in ['a', 'b', 'c']:
        rs = []
        for s in range(n_seeds):
            r = run_variant(mae, ga, nid, bp, v, 'ga_shuffle', seed=s, epochs=epochs, shuffle_seed=s)
            rs.append({"r": r["r"], "eta2": r["eta2"], "sil": r["sil"]})
        res["ga_shuffle"][v] = rs
        save()
        print(f"[{time.time()-t0:.0f}s] GA_SHUF {v}: r={np.mean([x['r'] for x in rs]):.3f}")
    # LABEL_SHUFFLE null for b, c (a: labels unused -> skip)
    for v in ['b', 'c']:
        rs = []
        for k in range(n_shuf):
            r = run_variant(mae, ga, nid, bp, v, 'label_shuffle', seed=0, epochs=epochs, shuffle_seed=1000 + k)
            rs.append({"r": r["r"], "eta2": r["eta2"], "sil": r["sil"]})
            if (k + 1) % 20 == 0:
                res["label_shuffle"][v] = rs; save()
                print(f"[{time.time()-t0:.0f}s] LBLSHUF {v} {k+1}/{n_shuf}: nullmean r={np.mean([x['r'] for x in rs]):.3f}")
        res["label_shuffle"][v] = rs
        save()
    print(f"DONE [{time.time()-t0:.0f}s]")
    return res
