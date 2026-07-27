"""Level-A LEARNED pooling: attention head over the 196 block6 patch tokens.

Per patch: score MLP 768->64->1; softmax over 196 patches -> weights;
weighted sum of patch tokens -> 768-d image vector; a linear GA head (768->1)
regresses GA. Trained END-TO-END, group-CV 5-fold (a fetus's images never
appear in the fold that scores them). The out-of-fold attention-pooled image
vector is then run through the SHARED harness (clock_perimg/fetus_agg/metrics)
so the numbers are directly comparable to mean and top-k pooling.

Reads the GA-filtered fp16 memmap built from the USB block6 grids
(6<=GA<=42 already applied). Also dumps per-image 196-weight attention maps
for a handful of images for interpretability.

Run:  PYTHONPATH=/Users/tiago/dev/fgr-geometry \
      python fgm_image/lag_attn_patches.py
"""
import os, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from fgm_image import lag_pool_harness as H

IMG = H.IMG
TMP = "/Users/tiago/usb/_attn_tmp"                       # meta.npz lives here
GRIDS = os.environ.get("ATTN_GRIDS", f"{TMP}/grids_f16.npy")  # fp16 memmap (copy to local SSD for speed)
torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"


class AttnPool(nn.Module):
    """score MLP over patches -> softmax -> weighted sum -> linear GA head."""
    def __init__(self, d=768, h=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
        self.head = nn.Linear(d, 1)

    def forward(self, x):                         # x (B,196,768)
        s = self.score(x).squeeze(-1)             # (B,196)
        a = torch.softmax(s, dim=1)               # (B,196)
        v = (a.unsqueeze(-1) * x).sum(1)          # (B,768)
        return self.head(v).squeeze(-1), v, a


def _standardize(mu, sd, x):
    return (x - mu) / sd


def train_fold(Xtr_idx, ytr, mm, mu_g, sd_g, ga_mu, ga_sd,
               epochs=8, bs=512, lr=1e-3, wd=1e-4):
    """Train on the memmap rows in Xtr_idx. Features standardized per-channel
    using training-fold stats (mu_g,sd_g); GA standardized too."""
    model = AttnPool().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    ytr_z = (ytr - ga_mu) / ga_sd
    n = len(Xtr_idx)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            b = Xtr_idx[perm[i:i + bs]]
            order = np.argsort(b)                 # memmap fancy-index wants sorted
            xb = torch.from_numpy(mm[b[order]].astype(np.float32)).to(DEV)
            xb = _standardize(mu_g, sd_g, xb)
            yb = torch.from_numpy(ytr_z[perm[i:i + bs]][order].astype(np.float32)).to(DEV)
            pred, _, _ = model(xb)
            loss = lossf(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def infer(model, idx, mm, mu_g, sd_g, bs=1024, want_attn=False):
    vs, ats = [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        xb = torch.from_numpy(mm[b].astype(np.float32)).to(DEV)
        xb = _standardize(mu_g, sd_g, xb)
        _, v, a = model(xb)
        vs.append(v.cpu().numpy())
        if want_attn:
            ats.append(a.cpu().numpy())
    V = np.concatenate(vs)
    A = np.concatenate(ats) if want_attn else None
    return V, A


def main():
    t0 = time.time()
    meta = np.load(f"{TMP}/meta.npz", allow_pickle=True)
    fid = meta["fid"].astype(float).astype(int)
    ga = meta["ga"].astype(float)
    fn = meta["fn"].astype(str)
    N = len(ga)
    mm = np.lib.format.open_memmap(GRIDS, mode="r")
    assert mm.shape[0] == N, (mm.shape, N)

    # OOF attention-pooled image vectors (768-d) trained end-to-end for GA
    oof = np.zeros((N, 768), np.float32)
    gkf = GroupKFold(5)
    for k, (tr, te) in enumerate(gkf.split(np.arange(N), ga, groups=fid)):
        # per-channel standardization stats from TRAIN patches only (subsample for speed)
        sub = tr[np.random.permutation(len(tr))[:4000]]
        samp = mm[np.sort(sub)].astype(np.float32).reshape(-1, 768)
        mu_g = torch.from_numpy(samp.mean(0)).to(DEV)
        sd_g = torch.from_numpy(samp.std(0) + 1e-6).to(DEV)
        ga_mu, ga_sd = float(ga[tr].mean()), float(ga[tr].std())
        model = train_fold(tr, ga[tr], mm, mu_g, sd_g, ga_mu, ga_sd)
        V, _ = infer(model, te, mm, mu_g, sd_g)
        oof[te] = V
        print(f"fold {k} done, n_te={len(te)}, elapsed={time.time()-t0:.0f}s", flush=True)
        if k == 0:
            last = (model, mu_g, sd_g)             # keep a model for attn dump

    # ---- score attention-pooled vectors through the SHARED harness ----
    cg, birthf = H.outcomes()
    clock_r, perimg_lag = H.clock_perimg(oof, fid, ga)
    rows = []
    for how in ["mean", "median", "trim"]:
        fl = H.fetus_agg(perimg_lag, fid, how)
        rows.append(H.metrics(fl, clock_r, cg, birthf,
                              levelA="attn_patches", levelB=how))
    for r in rows:
        print(r, flush=True)

    # ---- interpretability: per-image 196-weight attention maps for a few images ----
    model, mu_g, sd_g = last
    # pick a spread of GA: 8 images across GA quantiles
    order = np.argsort(ga)
    picks = order[np.linspace(0, N - 1, 8).astype(int)]
    _, A = infer(model, picks, mm, mu_g, sd_g, want_attn=True)
    np.savez(f"{IMG}/lag_attn_maps.npz",
             attn=A.astype(np.float32), img_idx=picks,
             ga=ga[picks], fid=fid[picks], fn=fn[picks])

    out = {"variants": rows,
           "baseline_meanmean": {"clock_GA_r": 0.848, "lag_SGA_r": -0.15,
                                 "lag_LGA_r": 0.12, "lag_birthpct_r": 0.19},
           "n_images": int(N), "n_fetuses": int(len(np.unique(fid))),
           "elapsed_s": round(time.time() - t0, 1)}
    with open(f"{IMG}/lag_attn_patches_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("SAVED", f"{IMG}/lag_attn_patches_results.json")


if __name__ == "__main__":
    main()
