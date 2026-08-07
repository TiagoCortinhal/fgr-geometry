# Terminal commands — you are already on the GPU node

## 0. Get the code — it is in the repo

```bash
git clone https://github.com/TiagoCortinhal/fgr-geometry.git
cd fgr-geometry/ssl_gpu
mkdir -p data results logs      # logs/ MUST exist before any `| tee logs/...`
```

(Already cloned? `git pull` — the package is under `ssl_gpu/`.)

## 1. The four data files — NOT in the repo

**The repo is public**, so cohort data stays out of it (`.gitignore` excludes
`data/` and `*.npz`). Copy these four into `ssl_gpu/data/` once:

| file | size | what it is |
|---|---|---|
| `image_clusters.csv` | 5.3 MB | manifest: new_filename -> fetus_id, dataset_type |
| `panel.npz` | 156 KB | 977x25 tabular panel + fids + GA + BMI |
| `frozen_usfm.npz` | 3.0 MB | per-fetus frozen USFM embedding — **the incumbent to beat** |
| `clinical_fetal_gate.csv` | 2.8 MB | drops the ~5,142 non-fetal clinical frames (optional but recommended) |

```bash
# from your laptop, one command
scp image_clusters.csv panel.npz frozen_usfm.npz clinical_fetal_gate.csv \
  tiago.fernandes@HOST:~/PyCharmProjects/fgr-geometry/ssl_gpu/data/
```

`panel.npz` and `frozen_usfm.npz` are regenerable on any machine with the cohort
mounted: `python build_inputs.py --out data`.

## 2. The two frame paths

Both cohorts have the same layout; `preprocessed/` holds the PNGs.

```bash
export IMPACT=/mnt/beegfs/groups/collage/data/IMPACT_FULL/processed/IMPACT_FULL/preprocessed
export CLINICAL=/mnt/beegfs/groups/collage/data/IMPACT_CLINICAL/processed/IMPACT_CLINICAL/preprocessed
```

**Why both, and what each is for:**

| cohort | role | why |
|---|---|---|
| **IMPACT** (~21k frames, ~950 fetuses) | **evaluation** — and pretraining | the only cohort with tabular targets, so every endpoint is scored here |
| **clinical** (~47k frames) | **pretraining only** | no tabular targets exist for it, but it roughly triples the frames available for representation learning |

The clinical set is what makes the SSL arms worth running at all: the objection
to our frozen-feature nulls is that USFM was never tuned on this data, and
68k frames is a far better answer to that than 21k.

The supervised arm ignores `--image-root-clinical` automatically — it needs
targets, which clinical does not have. It says so in its log.

**One subtlety the code handles for you:** the two cohorts number fetuses
independently, so IMPACT fetus 293 and clinical fetus 293 are different
pregnancies. Frames are keyed `<cohort>:<id>` during pretraining; merging them
would treat two unrelated pregnancies as one fetus and corrupt contrastive
training. The loader reports how many ids collide so you can see it working.

**Run this before training — 30 seconds, and it catches a wrong path before you
burn GPU hours:**

```bash
python check_paths.py --image-root $IMPACT --image-root-clinical $CLINICAL \
  --keep-csv data/clinical_fetal_gate.csv
```

It prints what resolved and exits non-zero if anything is wrong. Expect roughly
21,000 IMPACT frames / ~950 fetuses, and a pretrain pool near 68,000 once
clinical is included.

`check_paths.py` (like every script here) puts its own directory on `sys.path`,
so it works from any working directory — including inside a container. If you
are not in `ssl_gpu/`, call it by full path and point at the inputs:

```bash
python /path/to/fgr-geometry/ssl_gpu/check_paths.py \
  --image-root $IMPACT --image-root-clinical $CLINICAL \
  --manifest /path/to/ssl_gpu/data/image_clusters.csv \
  --panel    /path/to/ssl_gpu/data/panel.npz \
  --frozen   /path/to/ssl_gpu/data/frozen_usfm.npz
```

**Filenames:** the manifest stores stems without an extension
(`IMP0324_20171030_1.2.276...`) while the frames on disk are `.png`. The loader
probes a few extensions and reports which it resolved
(`extension resolved: {'impact': '.png'}`). If nothing matches it prints the
manifest's expected name next to what is actually in the directory, so you can
see whether it is a suffix problem or the wrong directory.

If the clinical PNGs are not under `preprocessed/`, try `inpainted/` — both
cohorts have that directory too, and it was byte-identical to `preprocessed/`
for IMPACT.

## 2b. Running inside a container (Apptainer)

`cd` into the package first, or the `fgm_ssl` import will fail:

```bash
apptainer exec --nv \
  --bind /mnt/beegfs/groups/collage/data:/mnt/beegfs/groups/collage/data \
  YOUR_IMAGE.sif bash -lc '
    cd /path/to/fgr-geometry/ssl_gpu && \
    python check_paths.py --image-root $IMPACT --image-root-clinical $CLINICAL'
```

Bind-mount the data directory so the frames are visible inside the container,
and export `IMPACT`/`CLINICAL` inside the shell (or pass them as flags).

## 2c. Swapping the frozen encoder (optional)

The incumbent normally comes from `data/frozen_usfm.npz`. To extract it online
instead — so a different encoder is a flag rather than a regenerate-and-scp
round trip:

```bash
export USFM_WEIGHTS=/path/on/hpc/to/USFM_latest.pth
export USFM_REPO=/path/on/hpc/to/USFM          # the source tree, not the weights

# reproduce the stored incumbent, and CHECK that it does
python run_ssl.py --arm frozen --encoder usfm:5 \
  --image-root $IMPACT --manifest data/image_clusters.csv \
  --panel data/panel.npz --out results \
  --verify-against data/frozen_usfm.npz

# then any other encoder is one flag
python run_ssl.py --arm frozen --encoder usfm:11 ...
python run_ssl.py --arm frozen --encoder resnet50 ...
```

**Run the `--verify-against` check once before trusting a swap.** If `usfm:5`
does not reproduce `frozen_usfm.npz`, the baseline has silently moved and every
comparison in the project is against a different incumbent — the script warns
loudly rather than continuing quietly.

Adding an encoder is one entry in `ENCODERS` in `fgm_ssl/encoders.py`: a
callable returning `(model, preprocess, dim)`. Pooling, evaluation and the stop
rule are untouched, which is the point — only the representation varies.

`usfm:*` imports `usdsgen.modules.backbone.vision_transformer` from the USFM
source tree — point `--usfm-repo` (or `$USFM_REPO`) at it, or put it on
`PYTHONPATH`. Locally that tree is `/Users/tiago/dev/USFM`, so on the HPC look
for the directory containing `usdsgen/`.

**There is deliberately no fallback to a similar ViT.** An earlier version fell
back to timm's Beit, which failed loudly on a signature mismatch — and had it
*not* failed, it would have built a different architecture and produced features
silently unlike the incumbent. If the import fails, the script tells you what it
tried and stops.

`resnet50` needs only torchvision and is a useful cheap check — it should
reproduce the maternal-BMI confound and little else.

## 2d. Container CPU limits

The container may expose far fewer CPUs than the host. `--workers` now defaults
to auto (`sched_getaffinity`, which respects the cgroup) and prints what it
chose. Passing `--workers 8` inside a 2-CPU container triggers
"excessive worker creation might get DataLoader running slow or even freeze" —
believe it; that warning can cost a GPU job hours.

Override only if you know better: `--workers 2`.

**`mkdir -p logs results` before any command that pipes to `tee`.** The shell
starts `tee` at the same moment as python, so `tee` opens `logs/x.log` long
before python could create the directory -- a `makedirs` inside the script does
NOT save you here. Step 0 already creates both; if you cloned fresh, run it
again.

## 2e. Frame cache — do this if an epoch takes minutes

On a CPU-starved node PNG decode, not the GPU, is the bottleneck: one dataloader
worker manages ~107 frames/s, so a 17k-frame epoch takes ~158 s while the network
itself needs ~5 s. Over 60 epochs x 5 folds that is **13 hours of mostly-idle GPU**.

`--cache` decodes every frame ONCE into a uint8 memmap and trains from RAM:

```bash
python run_ssl.py --arm supervised --epochs 60 --batch 64 --amp \
  --target Doppler --manifest data/image_clusters.csv --image-root $IMPACT \
  --panel data/panel.npz --out results --cache cache_impact224.npy \
  2>&1 | tee logs/supervised_doppler.log
```

~1.1 GB for 21k frames at 224x224 (0.5 GB at 160). The cache is built once and
reused by later runs, including the other arms — so arms 2 and 3 start training
immediately. Add clinical frames and it grows proportionally (~3.5 GB for 68k).

The cache changes speed only: augmentation and sampling are unchanged, and a
cached run reproduces an uncached one exactly.

## 3. Environment

```bash
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
# if missing:
# pip install torch numpy pandas pillow scikit-learn
```

## 4. Train — three commands, run them one at a time

```bash
mkdir -p logs results           # no-op if they exist; tee needs logs/ NOW

# arm 1: masked autoencoder -- pretrains on BOTH cohorts    ~4-8 h
python run_ssl.py --arm mae --epochs 100 --batch 64 --amp --workers 8 \
  --manifest data/image_clusters.csv \
  --image-root $IMPACT --image-root-clinical $CLINICAL \
  --keep-csv data/clinical_fetal_gate.csv \
  --panel data/panel.npz --out results 2>&1 | tee logs/mae.log

# arm 2: contrastive, same-fetus positives -- BOTH cohorts  ~4-8 h
python run_ssl.py --arm contrast --epochs 100 --batch 64 --amp --workers 8 \
  --manifest data/image_clusters.csv \
  --image-root $IMPACT --image-root-clinical $CLINICAL \
  --keep-csv data/clinical_fetal_gate.csv \
  --panel data/panel.npz --out results 2>&1 | tee logs/contrast.log

# arm 3: supervised, THE DECISIVE ONE -- IMPACT only (needs targets)
#        trains 5 models out-of-fold, so ~5x one arm's time
python run_ssl.py --arm supervised --epochs 60 --batch 64 --amp --workers 8 \
  --target cardiac --manifest data/image_clusters.csv --image-root $IMPACT \
  --panel data/panel.npz --out results 2>&1 | tee logs/supervised.log
```

Arms 1-2 take longer than the earlier estimate because the pretrain pool is
~68k frames rather than 21k. Run them in the background if your session may
drop:

```bash
nohup python run_ssl.py --arm mae --epochs 100 --batch 64 --amp --workers 8 \
  --manifest data/image_clusters.csv \
  --image-root $IMPACT --image-root-clinical $CLINICAL \
  --panel data/panel.npz --out results > logs/mae.log 2>&1 &
tail -f logs/mae.log
```

`--keep-csv` drops the 5,142 clinical frames the fetal gate rejected
(transvaginal, gynaecological, Doppler traces — ~10% of the clinical set),
leaving 46,956. Without it the encoder spends capacity on anatomy that is not
the fetus.

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
| `only N fetuses have frames` | `$IMPACT` is wrong — rerun step 2 |
| pretrain pool == eval pool | `$CLINICAL` is wrong; check `preprocessed/` vs `inpainted/` |
| `loss did not move` | raise `--lr` or `--epochs`; do not trust the run |
| `CUDA out of memory` | `--batch 32` (or 16); `--size 160` also helps |
| `INCONCLUSIVE` in the verdict | send `logs/` — do not read it as a null |
