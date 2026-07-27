"""Render one saliency panel per GA-lag image (real + block 1/6/12), RESUMABLE.

Each panel is saved as {OUTDIR}/{new_filename}.png. On (re)start the renderer
SKIPS any image whose panel already exists, so you can stop it any time
(Ctrl-C / stop the cell) and rerun this exact command to continue.

    python -m fgm_image.render_saliency_panels          # render all remaining
    python -m fgm_image.render_saliency_panels --status  # just print progress

Reads the aligned patch grids from /Users/tiago/usb/patchgrids (b0/b5/b11),
filters to the 43,699 lag images, overlays each layer's GA-clock saliency.
"""
import os, sys, glob, json, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from PIL import Image

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = f"{WS}/results/img_align"
GD = "/Users/tiago/usb/patchgrids"
OUTDIR = f"{GD}/lag_saliency_panels"
BLOCKS = [(0, 0, "block 1 (texture)"), (5, 1, "block 6 (mid)"), (11, 2, "block 12 (semantic)")]


def raw_path(fn):
    for base in ["/Users/tiago/usb/preprocessed", "/Users/tiago/usb/clinical/preprocessed"]:
        p = f"{base}/{fn}.png"
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    lag_fns = set(json.load(open(f"{IMG}/_lag_image_fns.json")))
    done = {os.path.splitext(f)[0] for f in os.listdir(OUTDIR) if f.endswith(".png")}
    remaining = lag_fns - done
    print(f"lag images {len(lag_fns)} | done {len(done)} | remaining {len(remaining)}", flush=True)
    if args.status:
        return

    W = np.load(f"{IMG}/_W_blocks.npy")  # [b1, b6, b12] GA-clock weights
    sh0 = sorted(glob.glob(f"{GD}/b0_grid_*.npz"))
    sh5 = sorted(glob.glob(f"{GD}/b5_grid_*.npz"))
    sh11 = sorted(glob.glob(f"{GD}/b11_grid_*.npz"))
    assert len(sh0) == len(sh5) == len(sh11), "shard count mismatch"
    t0, n = time.time(), 0
    for s0, s5, s11 in zip(sh0, sh5, sh11):
        z0, z5, z11 = (np.load(s, allow_pickle=True) for s in (s0, s5, s11))
        fns = z0["new_filename"]
        g = {0: z0["grid"], 5: z5["grid"], 11: z11["grid"]}
        for i, fn in enumerate(fns):
            if fn not in remaining:
                continue
            rp = raw_path(fn)
            if rp is None:
                continue
            raw = np.array(Image.open(rp).convert("L").resize((224, 224)))
            fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
            axes[0].imshow(raw, cmap="gray"); axes[0].set_title("real ultrasound", fontsize=9)
            for bi, wi, nm in BLOCKS:
                sal = (g[bi][i].astype(np.float32) @ W[wi]).reshape(14, 14)
                sal = sal - sal.mean()
                vm = np.percentile(np.abs(sal), 95)
                ax = axes[wi + 1]; ax.imshow(raw, cmap="gray")
                im = ax.imshow(zoom(sal, 224 / 14, order=1), cmap="RdBu_r", alpha=0.5, vmin=-vm, vmax=vm)
                ax.set_title(nm, fontsize=9); fig.colorbar(im, ax=ax, fraction=0.046)
            for a in axes:
                a.set_xticks([]); a.set_yticks([])
            fig.tight_layout()
            fig.savefig(f"{OUTDIR}/{fn}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
            n += 1
            if n % 200 == 0:
                r = n / (time.time() - t0)
                print(f"{n} rendered | {r:.1f}/s | ~{(len(remaining)-n)/r/60:.0f} min left", flush=True)
    print(f"DONE this run: {n} panels rendered in {(time.time()-t0)/60:.1f} min -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
