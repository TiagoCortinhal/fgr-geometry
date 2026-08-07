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
    p.add_argument("--image-root", required=True,
                   help="IMPACT frames -- the EVALUATION cohort (has tabular targets)")
    p.add_argument("--image-root-clinical", default=None,
                   help="clinical frames -- PRETRAINING ONLY (no tabular targets). "
                        "Used by mae/contrast; ignored by the supervised arm.")
    p.add_argument("--panel", default="data/panel.npz",
                   help="npz with Z (n,25), cols, blocks, fids, ga, bmi")
    p.add_argument("--keep-csv", default=None,
                   help="optional fetal-gate csv with new_filename,keep_fetal")

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
    p.add_argument("--cv-folds", type=int, default=5,
                   help="folds for the fetus-level split / out-of-fold supervised training")
    p.add_argument("--no-oof", action="store_true",
                   help="supervised arm: single split instead of out-of-fold (faster, "
                        "but then only 1/K of fetuses are scorable)")
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
    roots = {"impact": a.image_root}
    if a.image_root_clinical:
        roots["clinical"] = a.image_root_clinical

    # EVALUATION pool: IMPACT only -- it is the cohort with tabular targets.
    ev = FrameManifest(a.manifest, roots, ["impact"], a.keep_csv).existing()
    byf = ev.by_fetus()
    have = [int(f) for f in fids if int(f) in byf]
    print(f"[data] IMPACT: {len(ev.df)} frames | {len(have)} fetuses with images "
          f"| panel n={len(fids)}", flush=True)

    # PRETRAIN pool: IMPACT + clinical for the unsupervised arms. The clinical
    # set has no tabular targets, so it cannot be used by the supervised arm --
    # but it roughly triples the frames available for representation learning,
    # which is the whole point of an SSL arm on a small cohort.
    pre_types = ["impact"] + (["clinical"] if a.image_root_clinical else [])
    if a.arm in ("mae", "contrast") and a.image_root_clinical:
        pm = FrameManifest(a.manifest, roots, pre_types, a.keep_csv).existing()
        coll = pm.id_collision()
        # namespaced keys: the two cohorts number fetuses independently
        byf_pre = pm.by_fetus(namespaced=True)
        print(f"[data] PRETRAIN pool: {dict(pm.counts())} "
              f"| ids shared across cohorts: {coll} (namespaced, not merged)",
              flush=True)
    else:
        byf_pre = None
        if a.arm == "supervised" and a.image_root_clinical:
            print("[data] clinical frames IGNORED for the supervised arm "
                  "(no tabular targets exist for them)", flush=True)
    assert len(have) >= a.min_fetuses, (
        f"only {len(have)} fetuses have frames (--min-fetuses={a.min_fetuses}) "
        f"-- check --image-root")

    if a.arm == "frozen":
        print("[frozen] no training; score the existing USFM embeddings with "
              "score_arms.py --arm frozen", flush=True)
        return

    # The supervised arm SEES THE TARGET during training, so scoring it on its
    # own training fetuses measures memorisation. It therefore runs OUT-OF-FOLD:
    # K models, each embedding only the fold it never trained on, so every fetus
    # ends up with an embedding from a model that never saw its targets.
    # MAE/contrastive never see targets, so a single pretrain is fine -- but they
    # still record their held-out fold so scoring can honour it.
    oof = (a.arm == "supervised") and not a.no_oof
    folds = list(fetus_level_folds(have, n_folds=a.cv_folds, seed=a.seed))
    for tr_, te_ in folds:
        assert not (set(tr_) & set(te_)), "FETUS LEAK between train and test"
    if oof:
        print(f"[split] OUT-OF-FOLD: {a.cv_folds} models, each embedding its own "
              f"held-out fold ({len(have)} fetuses total)", flush=True)
    else:
        tr_fids, te_fids = folds[0]
        print(f"[split] train {len(tr_fids)} fetuses | held-out {len(te_fids)}",
              flush=True)

    log = dict(arm=a.arm, epochs=a.epochs, n_frames_eval=int(len(ev.df)),
               n_frames_pretrain=int(sum(len(v) for v in byf_pre.values()))
               if byf_pre is not None else int(len(ev.df)),
               pretrain_cohorts=pre_types if byf_pre is not None else ["impact"],
               args=vars(a), out_of_fold=bool(oof), folds=[])

    def build():
        if a.arm == "mae":
            m = MAE(a.dim, a.width, mask_ratio=a.mask_ratio).to(dev)
            return m, "mae"
        if a.arm == "contrast":
            return ContrastiveNet(a.dim, a.width).to(dev), "pair"
        tix = [i for i, b in enumerate(blocks) if b == a.target]
        assert tix, f"no columns for target block {a.target}"
        return SupervisedNet(len(tix), a.dim, a.width).to(dev), "mae"

    tix = [i for i, b in enumerate(blocks) if b == a.target] if a.arm == "supervised" else []
    if tix:
        pos = {int(f): i for i, f in enumerate(fids)}
        Yt = np.where(np.isfinite(Z[:, tix]), Z[:, tix], 0.0).astype("float32")
        Mt = np.isfinite(Z[:, tix]).astype("float32")
        log["target_block"], log["target_cols"] = a.target, [cols[i] for i in tix]

    def train_one(tr_fids, tag):
        model, mode = build()
        if byf_pre is not None:
            # train on this fold's IMPACT fetuses PLUS every clinical fetus.
            # Clinical fetuses are never evaluated, so they cannot leak -- but
            # IMPACT fetuses held out for scoring must still be excluded here.
            keys = [f"impact:{int(f)}" for f in tr_fids] + \
                   [k for k in byf_pre if k.startswith("clinical:")]
            ds = FrameDataset(byf_pre, keys, a.size, mode=mode)
        else:
            ds = FrameDataset(byf, tr_fids, a.size, mode=mode)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        drop_last=True, pin_memory=True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=a.amp and dev == "cuda")
        losses = []
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
            losses.append(tot / max(nb, 1))
            if ep % 10 == 0 or ep == a.epochs - 1:
                print(f"[train {tag}] epoch {ep:3d}/{a.epochs}  loss {losses[-1]:.5f}  "
                      f"{time.time()-t0:.0f}s", flush=True)
        assert losses[-1] < losses[0] * 0.98, (
            f"{tag}: loss did not move ({losses[0]:.5f} -> {losses[-1]:.5f}); a null "
            f"from this run would be meaningless -- check lr/batch")
        return model, losses

    t0 = time.time()
    E = np.full((len(fids), a.dim), np.nan, dtype="float32")
    scored = []
    if oof:
        for k, (tr_fids, te_fids) in enumerate(folds):
            model, losses = train_one(tr_fids, f"fold{k}")
            Ek, _ = embed_all(model.enc, byf, te_fids, a.size, dev)
            ix = {int(f): i for i, f in enumerate(fids)}
            for f, row in zip(te_fids, Ek):
                E[ix[int(f)]] = row
            scored += [int(f) for f in te_fids]
            log["folds"].append(dict(fold=k, n_train=len(tr_fids), n_heldout=len(te_fids),
                                     loss_first=losses[0], loss_last=losses[-1]))
        heldout, trained = np.array(sorted(scored)), np.array([], dtype=int)
    else:
        tr_fids, te_fids = folds[0]
        model, losses = train_one(tr_fids, "single")
        Eall, _ = embed_all(model.enc, byf, fids, a.size, dev)
        E = Eall.astype("float32")
        heldout, trained = np.array(te_fids), np.array(tr_fids)
        log["folds"].append(dict(fold=0, n_train=len(tr_fids), n_heldout=len(te_fids),
                                 loss_first=losses[0], loss_last=losses[-1]))

    sd = np.nanstd(E, axis=0)
    assert np.isfinite(sd).any() and np.nanmax(sd) > 1e-4, \
        "embedding variance collapsed -- refusing to write"
    np.savez_compressed(os.path.join(a.out, f"{a.arm}_embeddings.npz"),
                        E=E, fids=fids, heldout_fids=heldout, train_fids=trained)
    log["embed_dim"] = int(E.shape[1])
    log["n_scorable"] = int(len(heldout))
    log["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(log, open(os.path.join(a.out, f"{a.arm}_log.json"), "w"), indent=1)
    print(f"[done] {a.arm}: {len(heldout)} fetuses scorable "
          f"({'out-of-fold' if oof else 'held-out fold only'}) | "
          f"{log['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
