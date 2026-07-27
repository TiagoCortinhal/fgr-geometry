"""
Track: Plane/quality weighting (Option 4) + multi-layer fusion (Option 5).

Levers tested against the mean-patch . mean-image baseline:
  (4a) Level-A quality weighting  : weight the 196 patch tokens by a
       tissue/quality proxy before pooling to the image vector.
         - wnorm      : weight = patch L2 norm (signal strength)
         - wcenter    : Gaussian weight toward the 14x14 grid centre (fetal cone)
         - wnormcenter: product of the two
  (4b) Level-B plane-aware fetus average : weight a fetus's images toward the
       abdominal plane (the discriminative plane) using USFM plane confidence.
  (5)  Multi-layer fusion : concat block1(l0)+block6(l5)+block12(l11) patch-mean
       features (768-d each -> 2304-d) then clock.

Every variant is scored on the same 4 metrics via lag_pool_harness.

Step 1 (build_patch_pools) streams the block-5 raw grids from USB once and
writes _quality_pools.npz (weighted image vectors in pooled `fn` order).
Step 2 (run) scores all variants and writes lag_quality_multilayer_results.json.
"""
import os, sys, json
sys.path.insert(0, "/Users/tiago/dev/fgr-geometry")
import numpy as np, pandas as pd
from scipy import stats
from fgm_image import lag_pool_harness as H
from fgm_image.patchgrid_io import iter_grids

IMG = H.IMG
QPOOLS = f"{IMG}/_quality_pools.npz"


def _center_weight(sigma=3.0):
    """Gaussian weight over the 14x14 patch grid, peaked at centre. -> (196,)"""
    yy, xx = np.mgrid[0:14, 0:14].astype(np.float32)
    cy = cx = 6.5
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    w = np.exp(-d2 / (2 * sigma ** 2))
    return (w / w.sum()).reshape(196)


def build_patch_pools():
    """Stream block-5 raw grids, compute weighted image pools, scatter into
    pooled `fn` order, save _quality_pools.npz. ~30 min (USB-bound)."""
    P = np.load(f"{IMG}/_patchgrid_pooled.npz", allow_pickle=True)
    fn_pool = P["fn"].astype(str)
    idx_of = {f: i for i, f in enumerate(fn_pool)}
    N = len(fn_pool)
    cw = _center_weight()                                     # (196,)

    pools = {k: np.zeros((N, 768), np.float32)
             for k in ["mean", "wnorm", "wcenter", "wnormcenter"]}
    filled = np.zeros(N, bool)

    for si, (g, m) in enumerate(iter_grids(block=5)):        # g (n,196,768) f32
        fns = m["new_filename"].astype(str).values
        rows = np.array([idx_of.get(f, -1) for f in fns])
        keep = rows >= 0
        g = g[keep]; rows = rows[keep]

        norm = np.linalg.norm(g, axis=2)                     # (n,196) L2 per patch
        wn = norm / (norm.sum(1, keepdims=True) + 1e-8)      # normed weights
        wc = cw[None, :]                                     # (1,196)
        wnc = wn * wc; wnc = wnc / (wnc.sum(1, keepdims=True) + 1e-8)

        pools["mean"][rows]        = g.mean(1)
        pools["wnorm"][rows]       = np.einsum("np,npd->nd", wn, g)
        pools["wcenter"][rows]     = np.einsum("np,npd->nd", np.broadcast_to(wc, norm.shape), g)
        pools["wnormcenter"][rows] = np.einsum("np,npd->nd", wnc, g)
        filled[rows] = True
        print(f"shard {si:02d}: +{keep.sum()} imgs, total filled {filled.sum()}/{N}", flush=True)

    np.savez_compressed(QPOOLS, filled=filled,
                        **{f"pool_{k}": v for k, v in pools.items()})
    print("saved", QPOOLS, "filled", int(filled.sum()), "/", N)


# ---------- plane-aware Level-B (4b) ----------
def _plane_confidence():
    """Per-image abdominal-plane confidence keyed by new_filename.
    Uses _abd_worklist.csv if it carries a plane/confidence column;
    else falls back to a binary abdominal flag from _abd_features.npz fns."""
    import glob
    # abdominal worklist: images predicted abdominal + their confidence
    wl = pd.read_csv(f"{IMG}/_abd_worklist.csv")
    return wl


def run():
    pools, fid, ga = H.load_pooled()
    cg, birthf = H.outcomes()
    P = np.load(f"{IMG}/_patchgrid_pooled.npz", allow_pickle=True)
    gafull = P["ga"]; gm = np.isfinite(gafull) & (gafull >= 6) & (gafull <= 42)
    fn = P["fn"].astype(str)[gm]

    variants = []

    def score(feat, name, how="mean", levelB="mean_img", img_w=None):
        cr, pil = H.clock_perimg(feat, fid, ga)
        if img_w is None:
            fl = H.fetus_agg(pil, fid, how)
        else:                                                # weighted fetus mean
            df = pd.DataFrame({"fid": fid, "lag": pil, "w": img_w})
            fl = df.groupby("fid").apply(
                lambda x: np.average(x["lag"], weights=x["w"]) if x["w"].sum() > 0
                else x["lag"].mean())
        m = H.metrics(fl, cr, cg, birthf, name, levelB)
        variants.append(m); print(m, flush=True); return m

    # ---- baseline ----
    score(pools["pool_mean"], "mean_patch", "mean", "mean_img")

    # ---- (4a) quality-weighted Level-A ----
    if os.path.exists(QPOOLS):
        Q = np.load(QPOOLS, allow_pickle=True)
        qfilled = Q["filled"][gm]
        for key, nm in [("wnorm", "qw_norm"), ("wcenter", "qw_center"),
                        ("wnormcenter", "qw_norm_center")]:
            feat = Q[f"pool_{key}"][gm]
            score(feat, nm, "mean", "mean_img")

    # ---- (5) multi-layer fusion ----
    ml = np.load(f"{IMG}/emb_usfm_multilayer.npz", allow_pickle=True)
    l0, l5, l11 = ml["emb_l0"][gm], ml["emb_l5"][gm], ml["emb_l11"][gm]
    # z-score each block before concat so scales match
    def z(a): return (a - a.mean(0)) / (a.std(0) + 1e-8)
    fus = np.concatenate([z(l0), z(l5), z(l11)], 1)
    score(fus, "fuse_b1_b6_b12", "mean", "mean_img")
    score(np.concatenate([z(l0), z(l5)], 1), "fuse_b1_b6", "mean", "mean_img")

    # ---- (4b) plane-aware Level-B ----
    try:
        wl = _plane_confidence()
        abd = set(wl["new_filename"].astype(str))
        is_abd = np.array([f in abd for f in fn], float)
        cr0, pil0 = H.clock_perimg(pools["pool_mean"], fid, ga)
        # (i) abdominal-only, uniform fallback for fetuses w/o an abd image
        variants.append(_plane_agg(pil0, fid, is_abd, cr0, cg, birthf, H,
                                   "plane_abd_only"))
        # (ii) abdominal up-weighted 3x, all images kept
        variants.append(_plane_agg(pil0, fid, 1.0 + 2.0 * is_abd, cr0, cg, birthf, H,
                                   "plane_abd_x3"))
    except Exception as e:
        print("plane-aware skipped:", repr(e), flush=True)

    out = {"baseline": {"clock_GA_r": 0.848, "lag_SGA_r": -0.155,
                        "lag_LGA_r": 0.121, "lag_birthpct_r": 0.191},
           "variants": variants}
    with open(f"{IMG}/lag_quality_multilayer_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("saved results json")
    return out


def _plane_agg(pil, fid, img_w, cr, cg, birthf, H, levelB):
    """Weighted per-fetus mean of per-image lag; uniform fallback if a
    fetus's weights sum to 0 (no abdominal image)."""
    df = pd.DataFrame({"fid": fid, "lag": pil, "w": img_w})
    def agg(x):
        w = x["w"].values
        if w.sum() <= 0: w = np.ones_like(w)
        return np.average(x["lag"].values, weights=w)
    fl = df.groupby("fid").apply(agg)
    m = H.metrics(fl, cr, cg, birthf, "mean_patch", levelB)
    print(m, flush=True); return m


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build_patch_pools()
    else:
        run()
