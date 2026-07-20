# Artifact-store manifest — region_image_sweep_2026-07-20
# Every deliverable's durable artifact version_id (open in app via /artifacts/<id>).

## Protocols
region_quality_prereg.md      f3696d92-43bb-444a-9b01-36f76537cfa2
image_dl_prereg.md            097765d8-05db-4118-9668-ac9b1cc1a71c

## Results
image_dl_results.json         55916d50-a139-46ef-ac87-c240d12210b0
fgr_geometry_status.md (v44)  6fee8147-2e55-4c04-9e03-9fd754a3c149

## 3D interactive latents (store-only, git-ignored)
latent3d_gbtm_trajectory.html   c8eb75c2-32cd-4234-ac33-2b3435fcd62a   # most distinctive region model
latent3d_joint_vae.html         f1b8a0fa-4257-4645-af28-c79a98ac51fb   # joint biometry+image (unchanged)
latent3d_biometry_vae.html      f424dc21-458d-4544-8b30-1aa0eea69977   # biometry-only VAE
gbtm_trajectory_paths_3d.html   d40035e9-a59d-446a-bb90-d1e722d7e88c   # trajectory PATHS over GA

## Plots
image_dl_comparison.png       7104723d-58b1-48fd-a49d-1f2c1ba703c1

## Per-fetus latents + regions (908 fetuses)
joint_vae_latent_regions.csv       20f363bf-0ed0-44fd-917e-dd5db8017808
fusion_clustering_regions_908.csv  da875de9-1fff-4bc9-bd15-ee938d5beee1
pca_image_channel_perfetus.csv     88bc4550-bce8-4c18-bf1a-580287852fcc
gbmtm_plus_image.csv               9061021a-7c0a-4ecd-a674-3a156793df90

## Fitted models (store-only, git-ignored)
gbmtm_multivariate.rds        12193d0b-c30a-499d-b8fc-e4c253460693
gbmtm_plus_image.rds          f2acb389-2a98-40ad-9f56-03ffeabcf692
joint_vae_latent.npz          bf8d6c86-420c-41e8-a7b9-b90d2f986c60

## Substrate (checkpoints)
joint_traj_substrate.csv      4307644d-6d8b-4767-ad0c-e2c6b744ac0f
image_substrate.npz           fe15dc7c-f31f-4ce0-b433-13f86c2b14dc
