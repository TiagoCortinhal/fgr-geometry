# Pregnancy-representation image-derivation pipeline

New image encoders (beyond USFM/radiomics) for the longitudinal pregnancy-representation paper.

## Encoders
- **DINOv2-S** (`torch.hub facebookresearch/dinov2 dinov2_vits14`) — natural-image SSL ViT, 384-d.
- **USF-MAE ViT-B/16** — ultrasound masked-autoencoder; checkpoint NOT in repo (user-supplied
  `USF-MAE_full_pretrain_43dataset_100epochs.pt`, ~448 MB). Encoder 768-d, embed 1536-d (cls‖mean-patch),
  decoder intact → per-patch reconstruction error.

## `extract_image_derivations.py`
Per frame computes and shards (`results/imgderiv/`, git-ignored):
- `dino` (384) — DINOv2 embedding
- `mae` (1536) — USF-MAE embedding (cls ‖ mean patch token)
- `err_raw` — mean per-patch recon error, 75% mask, 4-seed avg (NEGATIVE CONTROL: dominated by framing/plane)
- `err_roi` — recon error over tissue patches only (cone & ~caliper ROI)
- `err_patch` (196) + `tissue` (196) — saved raw so the **GA+plane-conditioned atypicality** scalar
  (the candidate signal) can be built post-hoc.

**Masks (ROI = cone & ~caliper):**
- clinical: `cone_mask/<name>.png` & `inpaint_mask/<name>.png` (curated).
- IMPACT: `cone_mask/<name>.png` & `overlay_masks_impact/<IMP+DDMMYYYY>/<YYYYMMDD>/<UID>_caliper_mask.png`.

**Confound note:** MAE reconstructs fetal tissue well (low error) and dark background/cone poorly (high
error), so raw error ≈ framing/zoom + plane. ROI-restriction removes background but plane still dominates
(raw-vs-ROI r≈0.88); only GA+plane-conditioning isolates tissue atypicality. Hence the error ladder.

## Run (resumable, daemonized)
```
python _daemon_imgderiv.py     # double-fork; logs to handoff/imgderiv.log
```
Resumable: rebuilds done-set from existing shard `names`, skips them, continues. Sharded every 2000
frames; kill loses at most the in-flight buffer. ~2.2 frames/s on 10 CPU threads (~6 h for 50,670 frames).

Frame index (`full_frame_index.csv`, git-ignored) columns: `nid,new_filename,png,cone,calip,cohort,
plane_prop,ga_weeks_recovered,dataset_type,study_date`; gated to fetal frames (fetal-gate keep_fetal==1),
GA 6–42 wk.
