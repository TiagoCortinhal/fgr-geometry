#!/usr/bin/env python
"""Entry point for the GPU cluster. One arm per invocation.

    python run_ssl.py --arm mae        --epochs 100
    python run_ssl.py --arm contrast   --epochs 100
    python run_ssl.py --arm supervised --epochs 60 --target cardiac
    python run_ssl.py --arm frozen     # incumbent baseline, no training

Writes <out>/<arm>_embeddings.npz (per-fetus representation + fids) and
<out>/<arm>_log.json. Evaluation is deliberately a SEPARATE step (evaluate.py
via score_arms.py) so the same scoring code runs on every arm, including the
frozen incumbent.

SANITY GATES built in, because a silently-broken run that produces a null is
worse than a crash:
  * asserts the fetus-level split leaves no fetus on both sides
  * asserts the loss actually moved from its first-epoch value
  * refuses to write embeddings whose per-fetus variance has collapsed
"""
from __future__ import annotations

import argparse
import json
import os
import time

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fgm_ssl.data import FrameDataset, FrameManifest, fetus_level_folds
from fgm_ssl.models import MAE, ContrastiveNet, SupervisedNet


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   choices=["mae", "contrast", "supervised", "frozen"])
    p.add_argument("--manifest", default="data/image_clusters.csv")
    p.add_argument("--image-root", default="data/frames")
    p.add_argument("--panel", default="data/panel.npz",
                   help="npz with Z (n,25), cols, blocks, fids, ga, bmi")
    p.add_argument("--keep-csv", default=None,
                   help="optional fetal-gate csv with new_filename,keep_fetal")
    p.add_argument("--dataset-type", default="impact")
    p.add_argument("--out", default="results")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--target", default="cardiac",
                   help="supervised arm only: growth|Doppler|cardiac")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-fetuses", type=int, default=100,
                   help="guard against a wrong --image-root; lower only for smoke tests")
    p.add_argument("--amp", action="store_true", help="mixed precision")
    return p.parse_args()


def load_panel(path):
    z = np.load(path, allow_pickle=True)
    return (z["Z"], [str(c) for c in z["cols"]], [str(b) for b in z["blocks"]],
            np.array([int(f) for f in z["fids"]]), z["ga"], z["bmi"])


def embed_all(enc, byf, fids, size, device, batch=128):
    """Per-fetus representation = mean over that fetus's frames.

    Matches the frozen-feature pipeline exactly, so the comparison is about the
    encoder rather than the pooling.
    """
    from fgm_ssl.data import load_gray
    enc.eval()
    out, keep = [], []
    with torch.no_grad():
        for f in fids:
            ps = byf.get(int(f), [])
            if not ps:
                out.append(None)
                continue
            acc = []
            for i in range(0, len(ps), batch):
                x = np.stack([load_gray(p, size) for p in ps[i:i + batch]])[:, None]
                acc.append(enc(torch.from_numpy(x).to(device)).cpu().numpy())
            out.append(np.concatenate(acc).mean(0))
            keep.append(int(f))
    d = next(x.shape[0] for x in out if x is not None)
    E = np.array([x if x is not None else np.full(d, np.nan) for x in out])
    return E, keep


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={dev} arm={a.arm}", flush=True)

    Z, cols, blocks, fids, ga, bmi = load_panel(a.panel)
    man = FrameManifest(a.manifest, a.image_root, a.dataset_type, a.keep_csv).existing()
    byf = man.by_fetus()
    have = [int(f) for f in fids if int(f) in byf]
    print(f"[data] {len(man.df)} frames | {len(have)} fetuses with images "
          f"| panel n={len(fids)}", flush=True)
    assert len(have) >= a.min_fetuses, (
        f"only {len(have)} fetuses have frames (--min-fetuses={a.min_fetuses}) "
        f"-- check --image-root")

    if a.arm == "frozen":
        print("[frozen] no training; score the existing USFM embeddings with "
              "score_arms.py --arm frozen", flush=True)
        return

    tr_fids, te_fids = next(fetus_level_folds(have, n_folds=5, seed=a.seed))
    assert not (set(tr_fids) & set(te_fids)), "FETUS LEAK between train and test"
    print(f"[split] train {len(tr_fids)} fetuses | held-out {len(te_fids)}", flush=True)

    log = dict(arm=a.arm, epochs=a.epochs, n_train_fetuses=len(tr_fids),
               n_frames=int(len(man.df)), losses=[], args=vars(a))

    if a.arm == "mae":
        model = MAE(a.dim, a.width, mask_ratio=a.mask_ratio).to(dev)
        ds = FrameDataset(byf, tr_fids, a.size, mode="mae")
        dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        drop_last=True, pin_memory=True)
    elif a.arm == "contrast":
        model = ContrastiveNet(a.dim, a.width).to(dev)
        ds = FrameDataset(byf, tr_fids, a.size, mode="pair")
        dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        drop_last=True, pin_memory=True)
    else:
        tix = [i for i, b in enumerate(blocks) if b == a.target]
        assert tix, f"no columns for target block {a.target}"
        model = SupervisedNet(len(tix), a.dim, a.width).to(dev)
        ds = FrameDataset(byf, tr_fids, a.size, mode="mae")
        dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        drop_last=True, pin_memory=True)
        pos = {int(f): i for i, f in enumerate(fids)}
        Yt = np.where(np.isfinite(Z[:, tix]), Z[:, tix], 0.0).astype("float32")
        Mt = np.isfinite(Z[:, tix]).astype("float32")
        log["target_block"] = a.target
        log["target_cols"] = [cols[i] for i in tix]

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=a.amp and dev == "cuda")
    t0 = time.time()

    for ep in range(a.epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in dl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=a.amp and dev == "cuda"):
                if a.arm == "mae":
                    x, _ = batch
                    loss, _, _ = model(x.to(dev, non_blocking=True))
                elif a.arm == "contrast":
                    xa, xb, _ = batch
                    loss, _ = model(xa.to(dev), xb.to(dev))
                else:
                    x, f = batch
                    uniq, grp = torch.unique(f, return_inverse=True)
                    rows = [pos[int(u)] for u in uniq]
                    yb = torch.from_numpy(Yt[rows]).to(dev)
                    mb = torch.from_numpy(Mt[rows]).to(dev)
                    pred, _ = model(x.to(dev), grp.to(dev))
                    loss = (((pred - yb) ** 2) * mb).sum() / mb.sum().clamp(min=1)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item())
            nb += 1
        sched.step()
        log["losses"].append(tot / max(nb, 1))
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"[train] epoch {ep:3d}/{a.epochs}  loss {tot / max(nb,1):.5f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    first, last = log["losses"][0], log["losses"][-1]
    assert last < first * 0.98, (
        f"loss did not move ({first:.5f} -> {last:.5f}); a null from this run "
        f"would be meaningless -- check lr/batch before trusting any result")

    enc = model.enc
    E, keep = embed_all(enc, byf, fids, a.size, dev)
    sd = np.nanstd(E, axis=0)
    assert np.isfinite(sd).any() and np.nanmax(sd) > 1e-4, \
        "embedding variance collapsed -- refusing to write"
    np.savez_compressed(os.path.join(a.out, f"{a.arm}_embeddings.npz"),
                        E=E, fids=fids, heldout_fids=np.array(te_fids),
                        train_fids=np.array(tr_fids))
    log["loss_first"], log["loss_last"] = first, last
    log["embed_dim"] = int(E.shape[1])
    log["n_embedded"] = int(len(keep))
    log["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(log, open(os.path.join(a.out, f"{a.arm}_log.json"), "w"), indent=1)
    print(f"[done] {a.arm}: loss {first:.5f} -> {last:.5f} | "
          f"{len(keep)} fetuses embedded | {log['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
