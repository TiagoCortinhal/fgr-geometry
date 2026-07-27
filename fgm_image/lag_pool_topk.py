"""
Level-A top-k patch pooling (Phase 0, needs raw USB grids).

For each image's 196 patch tokens, instead of the plain mean, pool only the
top-k patches by informativeness. Informativeness = patch projection onto the
GA-direction w (from the block6 image-mean clock), |patch . w|. Take mean of the
top-k patches -> image vector. Sweep k in {4,8,16,32,196=mean}.

Then clock + lag + 3 metrics via lag_pool_harness. Compares against the mean baseline.
Output: results/img_align/lag_pool_topk.json
"""
import numpy as np, json, sys
sys.path.insert(0, "/Users/tiago/dev/fgr-geometry")
from fgm_image import patchgrid_io as pgio
from fgm_image.lag_pool_harness import outcomes, clock_perimg, fetus_agg, metrics, IMG

BLOCK = 5  # block6

def build_topk_features(ks=(4, 8, 16, 32)):
    # direction w: fit on the pooled mean once to rank patches
    P = np.load(f"{IMG}/_patchgrid_pooled.npz", allow_pickle=True)
    ga_all = P["ga"]; gm = np.isfinite(ga_all) & (ga_all >= 6) & (ga_all <= 42)
    from sklearn.linear_model import Ridge
    Em = P["pool_mean"][gm]; Ez = (Em - Em.mean(0)) / (Em.std(0) + 1e-8)
    w = Ridge(100).fit(Ez, ga_all[gm]).coef_          # GA direction in normalized space
    mu, sd = Em.mean(0), Em.std(0) + 1e-8

    feats = {k: [] for k in ks}; fids = []; gas = []
    for grid, meta in pgio.iter_grids(BLOCK):          # grid (B,196,768) fp16
        g = grid.astype(np.float32)
        score = np.abs(((g - mu) / sd) @ w)            # (B,196) informativeness
        order = np.argsort(-score, axis=1)             # descending
        for k in ks:
            idx = order[:, :k]
            sel = np.take_along_axis(g, idx[:, :, None], axis=1)  # (B,k,768)
            feats[k].append(sel.mean(1))
        fids.append(np.asarray(meta["fetus_id"], float))
        gas.append(np.asarray(meta["ga_weeks_recovered"], float))
    fid = np.concatenate(fids); ga = np.concatenate(gas)
    keep = np.isfinite(ga) & (ga >= 6) & (ga <= 42)
    return {k: np.concatenate(v)[keep] for k, v in feats.items()}, fid[keep].astype(int), ga[keep]


if __name__ == "__main__":
    cg, birthf = outcomes()
    feats, fid, ga = build_topk_features()
    rows = []
    for k, feat in feats.items():
        rc, pil = clock_perimg(feat, fid, ga)
        for lb in ["mean", "median", "trim"]:
            rows.append(metrics(fetus_agg(pil, fid, lb), rc, cg, birthf, f"top{k}", lb))
    json.dump(rows, open(f"{IMG}/lag_pool_topk.json", "w"), indent=2)
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))
