"""Loader for USFM patch-token grids (no mean pooling), blocks 1/6/12.

Grids live on the USB at /Users/tiago/usb/patchgrids/ as sharded fp16 .npz
files named b{block}_grid_{shard}.npz, each holding grid (n,196,768) + metadata.
Block index is 0-indexed USFM depth: 0="block 1", 5="block 6", 11="block 12".

    from fgm_image.patchgrid_io import load_grids, iter_grids
    G, meta = load_grids(block=5)          # all block-6 grids concatenated
    for G, meta in iter_grids(block=0):    # stream shard-by-shard (memory-safe)
        ...
"""
import os, glob
import numpy as np, pandas as pd

GRID_DIR = os.environ.get("PATCHGRID_DIR", "/Users/tiago/usb/patchgrids")
META = ["new_filename", "fetus_id", "in_cohort", "ga_weeks_recovered", "dataset_type", "study_date", "same_pregnancy"]


def _shards(block):
    return sorted(glob.glob(f"{GRID_DIR}/b{block}_grid_*.npz"))


def iter_grids(block):
    """Yield (grid (n,196,768) float32, meta DataFrame) per shard."""
    shards = _shards(block)
    if not shards:
        raise FileNotFoundError(f"no grids for block {block} in {GRID_DIR} (USB mounted?)")
    for s in shards:
        z = np.load(s, allow_pickle=True)
        meta = pd.DataFrame({k: z[k] for k in META if k in z})
        yield z["grid"].astype(np.float32), meta


def load_grids(block):
    """Concatenate all shards for a block. WARNING: ~16 GB in RAM as float32."""
    gs, ms = [], []
    for g, m in iter_grids(block):
        gs.append(g); ms.append(m)
    return np.concatenate(gs), pd.concat(ms, ignore_index=True)


def saliency(grid, w):
    """Per-patch clock contribution: grid (n,196,768) @ w (768,) -> (n,14,14)."""
    return (grid @ w).reshape(-1, 14, 14)
