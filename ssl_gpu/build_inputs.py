#!/usr/bin/env python
"""Build the two npz inputs the cluster needs. RUN THIS LOCALLY, not on the GPU.

    python build_inputs.py --out data

Writes:
  data/panel.npz        Z (n,25), cols, blocks, fids, ga, bmi
  data/frozen_usfm.npz  E (n,768), fids  -- the incumbent to beat

Both are small (a few MB). Ship them with the code; the frames go separately.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/tiago/dev/fgr-geometry/tools/impact_fetal_panel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--emb", default="/Users/tiago/dev/fgr-geometry/results/img_align/emb_usfm_multilayer.npz")
    ap.add_argument("--layer", default="emb_l5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import pandas as pd
    from fgm_tools import fgm_ga_at_echo, fgm_panel, fgm_setup

    fgm_setup()
    P = fgm_panel()
    Z, cols, blocks, fids = P["Z"], P["cols"], P["blocks"], P["fids"]
    ga = fgm_ga_at_echo(fids)
    bmi = Z[:, cols.index("maternal_bmi")]
    np.savez_compressed(os.path.join(a.out, "panel.npz"),
                        Z=Z, cols=np.array(cols), blocks=np.array(blocks),
                        fids=np.array([int(f) for f in fids]), ga=ga, bmi=bmi)
    print(f"[panel] {Z.shape} | blocks {sorted(set(blocks))}")

    z = np.load(a.emb, allow_pickle=True)
    imp = pd.Series(z["dataset_type"]).astype(str).values == "impact"
    fi = pd.to_numeric(pd.Series(z["fetus_id"]).astype(str), errors="coerce").values
    E = z[a.layer].astype("float32")
    m = imp & np.isfinite(fi)
    df = pd.DataFrame(E[m])
    df["fid"] = fi[m].astype(int)
    pf = df.groupby("fid").mean()
    lut = {int(k): v for k, v in zip(pf.index, pf.values)}
    d = pf.shape[1]
    Efz = np.array([lut.get(int(f), np.full(d, np.nan)) for f in fids])
    np.savez_compressed(os.path.join(a.out, "frozen_usfm.npz"),
                        E=Efz, fids=np.array([int(f) for f in fids]))
    print(f"[frozen] {Efz.shape} | {int(np.isfinite(Efz).all(1).sum())} fetuses with images")
    print(f"\nShip {a.out}/panel.npz + {a.out}/frozen_usfm.npz with the code.")
    print("Frames go separately -- see README for the expected --image-root layout.")


if __name__ == "__main__":
    main()
