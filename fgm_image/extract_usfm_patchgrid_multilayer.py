"""Extract USFM patch-token grids (NO mean pooling) at blocks 1/6/12 for all
cohort images, sharded, fp16, to the USB. Each grid is (196, 768) per image per
layer. Companion loader in fgm_image/patchgrid_io.py.

Usage:
    python -m fgm_image.extract_usfm_patchgrid_multilayer --blocks 0 11
(block 5 / "block 6" is already extracted in results/img_align/_patchgrid_shards)
"""
import os, sys, time, json, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from functools import partial
from torchvision import transforms
from PIL import Image

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = f"{WS}/results/img_align"
OUT = "/Users/tiago/usb/patchgrids"          # 909 GB free; git-ignored
USFM_REPO = os.environ.get("USFM_REPO", f"{os.path.expanduser('~')}/.claude-science/orgs/2e50fc88-f384-4a2e-9328-c60d613fd12a/workspaces/50e875d8-edc0-4647-b8d5-b6df7858c9cb/USFM-master")
WEIGHTS = "/Users/tiago/Downloads/USFM_latest.pth"
BATCH, SHARD_EVERY = 32, 2048
META = ["new_filename", "fetus_id", "in_cohort", "ga_weeks_recovered", "dataset_type", "study_date", "same_pregnancy"]


def build_model():
    sys.path.insert(0, USFM_REPO)
    from usdsgen.modules.backbone.vision_transformer import VisionTransformer
    m = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.0, init_values=0.1,
        use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=True,
        use_mean_pooling=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    ck = torch.load(WEIGHTS, map_location="cpu")
    m.load_state_dict(ck.get("model", ck.get("state_dict", ck)), strict=False)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 11])  # 0-indexed: 0=b1,5=b6,11=b12
    args = ap.parse_args()
    assert os.path.isdir("/Users/tiago/usb/preprocessed"), "USB not mounted"
    os.makedirs(OUT, exist_ok=True)
    model = build_model()
    caps = {}
    def mk(bi):
        def h(mod, i, o): caps[bi] = o[:, 1:, :].half().cpu()  # (n,196,768) all patches, fp16
        return h
    for bi in args.blocks:
        model.blocks[bi].register_forward_hook(mk(bi))
    pre = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # worklist: cohort images with recovered GA (same set the mean features used)
    z = np.load(f"{IMG}/emb_usfm_multilayer.npz", allow_pickle=True)
    wl = pd.DataFrame({k: z[k] for k in META})
    wl = wl[wl.in_cohort.astype(bool)].reset_index(drop=True)
    def path_of(fn):
        for base in ["/Users/tiago/usb/preprocessed", "/Users/tiago/usb/clinical/preprocessed"]:
            p = f"{base}/{fn}.png"
            if os.path.exists(p): return p
        return None
    wl["png"] = wl.new_filename.map(path_of)
    wl = wl[wl.png.notna()].reset_index(drop=True)
    print(f"worklist {len(wl)} images, blocks {args.blocks}", flush=True)

    buf_img, buf_meta = [], []
    bufs = {bi: [] for bi in args.blocks}
    metas, shard_i, n, t0 = [], {bi: 0 for bi in args.blocks}, 0, time.time()

    def dump():
        nonlocal metas
        if not metas: return
        M = {k: np.array([m[k] for m in metas], dtype=object) for k in META}
        for bi in args.blocks:
            G = np.concatenate(bufs[bi]).astype(np.float16)
            np.savez_compressed(f"{OUT}/b{bi}_grid_{shard_i[bi]:04d}.npz", grid=G, **M)
            shard_i[bi] += 1; bufs[bi] = []
        metas = []

    for idx, r in wl.iterrows():
        try: img = pre(Image.open(r.png).convert("RGB"))
        except Exception: continue
        buf_img.append(img); buf_meta.append(r)
        if len(buf_img) == BATCH or idx == len(wl) - 1:
            with torch.no_grad(): model.forward_features(torch.stack(buf_img))
            for bi in args.blocks: bufs[bi].append(caps[bi].numpy())
            metas.extend(buf_meta); n += len(buf_img); buf_img, buf_meta = [], []
            if len(metas) >= SHARD_EVERY: dump()
            if n % 4096 < BATCH:
                print(f"{n}/{len(wl)} {n/(time.time()-t0):.1f} img/s", flush=True)
    dump()
    json.dump({"blocks": args.blocks, "n_images": int(n), "out": OUT,
               "grid_shape": [196, 768], "dtype": "float16"},
              open(f"{OUT}/manifest_blocks_{'_'.join(map(str,args.blocks))}.json", "w"), indent=2)
    print(f"DONE {n} images x blocks {args.blocks} -> {OUT} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
