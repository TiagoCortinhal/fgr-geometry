"""
Level-B learned attention over a fetus's images (lag-pooling battery, 2026-07-09).

Keeps the patch MEAN (block6 pool_mean) as the per-image embedding. Learns an
attention head over the M images belonging to each fetus:

    score_i = MLP(feat_i)        feat_i = [standardized 768-d mean emb  (+ plane soft-probs + plane_conf)]
    w_i     = softmax over the fetus's M images
    lag_fetus = sum_i w_i * per_image_lag_i

The per-image lag is fixed (predGA - datesGA from the group-CV clock, no fetus
leakage). Only the pooling weights are learned. Training is UNSUPERVISED of
outcome: split-half consistency -- weighted lag from a random half of a fetus's
images must match the weighted lag from the other half. This forces the head to
down-weight noisy/outlier images (denoise toward the within-fetus consensus)
without ever seeing SGA / birth percentile.

Compared against mean / median / trim on the 4 battery metrics. Per-fetus image
weights are saved so we can check whether the head up-weights abdominal / clean
(high plane_conf) scans.
"""
import json, numpy as np, pandas as pd, torch
from torch import nn
from fgm_image.lag_pool_harness import (
    load_pooled, outcomes, clock_perimg, fetus_agg, metrics, IMG)

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


# ---- segment ops (ragged groups, no padding) --------------------------------
def seg_softmax(scores, grp, G):
    """softmax of `scores` within each group id in `grp` (0..G-1)."""
    m = torch.full((G,), -1e30, device=scores.device)
    m = m.scatter_reduce(0, grp, scores, reduce="amax", include_self=True)
    e = torch.exp(scores - m[grp])
    s = torch.zeros(G, device=scores.device).scatter_add(0, grp, e)
    return e / (s[grp] + 1e-12)

def seg_sum(vals, grp, G):
    return torch.zeros(G, device=vals.device).scatter_add(0, grp, vals)


class AttnHead(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, X):
        return self.net(X).squeeze(-1)


def train_attn(X, lag, grp, G, n_imgs_per, epochs=400, lr=1e-3, wd=1e-4, seed=0):
    """Split-half consistency training. Returns per-fetus weighted lag (numpy),
    and full per-image weights (numpy)."""
    g = torch.Generator().manual_seed(seed)
    Xt   = torch.tensor(X, dtype=torch.float32)
    lagt = torch.tensor(lag, dtype=torch.float32)
    grpt = torch.tensor(grp, dtype=torch.long)
    head = AttnHead(X.shape[1])
    opt  = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    N = len(lag)
    for ep in range(epochs):
        head.train(); opt.zero_grad()
        s = head(Xt)
        # random half assignment per image -> composite subgroup grp*2 + half
        half = torch.randint(0, 2, (N,), generator=g)
        cg = grpt * 2 + half
        w = seg_softmax(s, cg, 2 * G)
        Lsub = seg_sum(w * lagt, cg, 2 * G)          # (2G,)  weighted lag per (fetus,half)
        LA, LB = Lsub[0::2], Lsub[1::2]              # halves 0 and 1 per fetus
        loss = ((LA - LB) ** 2).mean()
        loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        s = head(Xt)
        w_full = seg_softmax(s, grpt, G)
        Lf = seg_sum(w_full * lagt, grpt, G)
    return Lf.numpy(), w_full.numpy()


def main():
    feats, fid, ga = load_pooled()
    cg, birth = outcomes()
    emb = feats["pool_mean"]
    clock_r, pil = clock_perimg(emb, fid, ga)     # image->GA clock + per-image lag

    # contiguous group ids
    uf, grp = np.unique(fid, return_index=False, return_inverse=True)
    G = len(uf)
    n_per = np.bincount(grp)

    # standardized embedding features
    Ez = (emb - emb.mean(0)) / (emb.std(0) + 1e-8)

    # plane soft features aligned to the pooled fn order
    P = np.load(f"{IMG}/_patchgrid_pooled.npz", allow_pickle=True)
    gmask = np.isfinite(P["ga"]) & (P["ga"] >= 6) & (P["ga"] <= 42)
    fn = P["fn"][gmask]
    ic = pd.read_csv(f"{IMG}/image_clusters.csv", low_memory=False).set_index("new_filename")
    plane = ic.reindex(fn)[["p_abdominal", "p_cerebral", "p_femur", "plane_conf"]].astype(float)
    plane = plane.fillna(plane.mean()).values

    results = []
    # ---- Level-B baselines ---------------------------------------------------
    for how in ["mean", "median", "trim"]:
        fl = fetus_agg(pil, fid, how)
        results.append(metrics(fl, clock_r, cg, birth, "mean", how))

    # ---- learned attention variants -----------------------------------------
    def as_series(Lf):
        return pd.Series(Lf, index=uf.astype(int))

    variants = {
        "attn_emb":       Ez,
        "attn_emb_plane": np.hstack([Ez, plane]),
        "attn_plane":     plane,
    }
    attn_weights = {}
    for name, X in variants.items():
        Lf, w = train_attn(X, pil, grp, G, n_per, seed=SEED)
        attn_weights[name] = w
        m = metrics(as_series(Lf), clock_r, cg, birth, "mean", name)
        results.append(m)

    # ---- does the head up-weight abdominal / clean scans? --------------------
    # per-image weight normalized by fetus size (uniform baseline = 1/n_i)
    w = attn_weights["attn_emb_plane"]
    rel = w * n_per[grp]                      # >1 = up-weighted vs uniform
    diag = {
        "corr_relweight_p_abdominal": float(np.corrcoef(rel, plane[:, 0])[0, 1]),
        "corr_relweight_p_cerebral":  float(np.corrcoef(rel, plane[:, 1])[0, 1]),
        "corr_relweight_p_femur":     float(np.corrcoef(rel, plane[:, 2])[0, 1]),
        "corr_relweight_plane_conf":  float(np.corrcoef(rel, plane[:, 3])[0, 1]),
    }

    baseline = next(r for r in results if r["levelB"] == "mean")
    best_attn = max((r for r in results if r["levelB"].startswith("attn")),
                    key=lambda r: abs(r["lag_SGA_r"]))
    out = {
        "clock_GA_r": round(clock_r, 3),
        "n_fetus": int(G),
        "baseline_mean_lag_SGA_r": baseline["lag_SGA_r"],
        "variants": results,
        "weight_plane_diagnostics": diag,
        "beats_mean_on_lag_SGA": bool(abs(best_attn["lag_SGA_r"]) > abs(baseline["lag_SGA_r"])),
    }
    with open(f"{IMG}/lag_attn_images_results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    # per-fetus weights (attn_emb_plane): save fetus id + top-weighted image share
    np.savez(f"{IMG}/_lag_attn_weights.npz",
             fid=fid, fn=fn, w_emb_plane=attn_weights["attn_emb_plane"],
             w_emb=attn_weights["attn_emb"], rel_emb_plane=rel,
             p_abdominal=plane[:, 0], plane_conf=plane[:, 3])
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
