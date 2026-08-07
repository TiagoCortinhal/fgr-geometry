# Terminal commands — you are already on the GPU node

## 0. Get the code — it is in the repo

```bash
git clone https://github.com/TiagoCortinhal/fgr-geometry.git
cd fgr-geometry/ssl_gpu
mkdir -p data results logs
```

(Already cloned? `git pull` — the package is under `ssl_gpu/`.)

## 1. The two data files

These are NOT in the repo — `.gitignore` excludes `data/` and `*.npz`, which is
correct: they are cohort data. Copy them across once:

| file | size | what it is |
|---|---|---|
| `panel.npz` | 156 KB | the 977×25 tabular panel + fids + GA + BMI |
| `frozen_usfm.npz` | 3.0 MB | per-fetus frozen USFM embedding — **the incumbent to beat** |

```bash
# from your laptop
scp panel.npz frozen_usfm.npz USER@HOST:~/fgr-geometry/ssl_gpu/data/
```

Both are regenerable on any machine with the cohort mounted:
`python build_inputs.py --out data`.

You also need the **image manifest** (`image_clusters.csv`, ~9 MB, the
new_filename → fetus_id map) in `data/`. If it is already on the HPC, symlink it
rather than copying:

```bash
ln -s /path/on/hpc/to/image_clusters.csv data/image_clusters.csv
```

**The frames stay where they are** — you only point `--image-root` at them.

## 2. Point at the frames and CHECK before training

Set `FRAMES` to wherever the PNGs live on the HPC:

```bash
export FRAMES=/path/on/hpc/to/preprocessed     # <-- tell me the path and I'll fix this line
```

Frames may be flat or nested; if nested, `--image-root` must be the directory
such that `<image-root>/<new_filename>` resolves.

**Run this first — it costs 30 seconds and catches a wrong path before you burn
GPU hours:**

```bash
python - <<'EOF'
import sys; sys.path.insert(0,'.')
from fgm_ssl.data import FrameManifest
import os
m = FrameManifest("data/image_clusters.csv", os.environ["FRAMES"], "impact")
n_rows = len(m.df); m.existing()
print(f"manifest rows: {n_rows} | files found on disk: {len(m.df)}")
print(f"fetuses with frames: {m.df.fid.nunique()}")
assert len(m.df) > 1000, "PATH IS WRONG -- fix $FRAMES before training"
print("OK -- ready to train")
EOF
```

Expect roughly 21,000 frames / ~950 fetuses for IMPACT.

## 3. Environment

```bash
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
# if missing:
# pip install torch numpy pandas pillow scikit-learn
```

## 4. Train — three commands, run them one at a time

```bash
# arm 1: masked autoencoder            ~2-4 h
python run_ssl.py --arm mae --epochs 100 --batch 64 --amp --workers 8 \
  --manifest data/image_clusters.csv --image-root $FRAMES \
  --panel data/panel.npz --out results 2>&1 | tee logs/mae.log

# arm 2: contrastive (same-fetus)      ~2-4 h
python run_ssl.py --arm contrast --epochs 100 --batch 64 --amp --workers 8 \
  --manifest data/image_clusters.csv --image-root $FRAMES \
  --panel data/panel.npz --out results 2>&1 | tee logs/contrast.log

# arm 3: supervised, THE DECISIVE ONE  ~5x longer: it trains 5 models
#        (out-of-fold, so every fetus is scored by a model that never saw it)
python run_ssl.py --arm supervised --epochs 60 --batch 64 --amp --workers 8 \
  --target cardiac --manifest data/image_clusters.csv --image-root $FRAMES \
  --panel data/panel.npz --out results 2>&1 | tee logs/supervised.log
```

Run them in the background and detach if your session may drop:

```bash
nohup python run_ssl.py --arm mae ... > logs/mae.log 2>&1 &
tail -f logs/mae.log
```

## 5. Score — CPU, ~10 minutes, run after all three finish

```bash
python score_arms.py --panel data/panel.npz --results results \
  --frozen data/frozen_usfm.npz --out results/scores.json 2>&1 | tee logs/score.log
```

## 6. What to send back

`results/scores.json` and `logs/` are what matter — a few hundred KB. Either
scp them back, or commit the small ones (`results/` is gitignored, so use `-f`
deliberately and only for scores.json + logs):

```bash
scp -r USER@HOST:~/fgr-geometry/ssl_gpu/results/scores.json .
scp -r USER@HOST:~/fgr-geometry/ssl_gpu/logs .
```

The `*_embeddings.npz` files are a few MB each; keep them on the HPC unless we
need to re-score.

## Reading the result

The last line of `score_arms.py` is the answer:

- **`BEATS the frozen incumbent`** → the null is overturned; SSL found what
  frozen features missed. This becomes the paper.
- **`no arm beats the frozen incumbent`** → the multimodal null survives a
  cohort-trained encoder, which is a much stronger negative than we have now.
- **`INCONCLUSIVE`** → something broke. The script refuses to call a null when
  no trained arm passed its positive control (image → maternal BMI must be
  strong; we measured 0.512 with frozen features). Send me the logs.

## Two things the code will not let you do

**Score a trained encoder on its own training fetuses.** `score_arms.py` reads
`heldout_fids` from each npz and scores only those; the supervised arm runs
out-of-fold so every fetus is embedded by a model that never saw its targets.
A smoke test showed why this matters: scored on its own training fetuses the
supervised arm reported +0.851 on the block it was trained on and ~0 on
everything else — pure memorisation.

**Trust a run that did not train.** `run_ssl.py` aborts if the loss did not move
or the embedding variance collapsed, rather than writing embeddings that would
produce a meaningless null.

## If something fails

| symptom | cause |
|---|---|
| `only N fetuses have frames` | `$FRAMES` is wrong — rerun step 2 |
| `loss did not move` | raise `--lr` or `--epochs`; do not trust the run |
| `CUDA out of memory` | `--batch 32` (or 16); `--size 160` also helps |
| `INCONCLUSIVE` in the verdict | send `logs/` — do not read it as a null |
