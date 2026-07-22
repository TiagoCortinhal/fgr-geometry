# From-scratch GA clock (learned-from-pixels) — IMPACT all planes

Motivation: the frozen-USF-MAE clock's saliency was a LINEAR PROJECTION (diffuse, possibly
depth-confounded). A CNN trained from pixels enables true gradient saliency (Grad-CAM) and
bakes confound control (random zoom/crop defeats scale/pixel-spacing shortcuts) into training.

## Model / setup
- ResNet-18 from scratch (weights=None), final fc -> 1 (GA in weeks), Huber loss (delta=2).
- Input 160x160, grayscale->3ch. Train aug: RandomResizedCrop(0.7-1.0) + hflip + intensity jitter
  (so the net cannot cheat on absolute scale / zoom / pixel spacing).
- Cohort: all-plane IMPACT, 20,413 frames / 908 fetuses, GA 26-41wk.
- Split: FETUS-GROUPED 70/15/15 (train 635 / val 136 / test 137 fetuses) — no fetus in two splits.
- Device: auto MPS (Apple Silicon GPU) if available, else CPU (8 threads).

## Data index
The training index (handoff/ga_cnn_index.csv: path, ga_weeks_recovered, plane_prop, split) is
NOT committed (references USB image paths / is data). Rebuild it from image_clusters.csv +
/Users/tiago/usb/{preprocessed,inpainted} with the snippet in the session notes, or ask the agent.

## RUN
Training (writes handoff/ga_cnn/best.pt + log.json):
    python train_ga_cnn.py
    # if DataLoader workers get killed on your setup: GA_CNN_WORKERS=0 python train_ga_cnn.py
Evaluation + Grad-CAM (after training):
    python eval_ga_cnn.py
    # -> handoff/ga_cnn/test_results.json (test MAE/r, per-plane) + gradcam/*.png

Benchmark for the frozen clock to beat: USF-MAE cerebral GA-r=0.941 (abdominal 0.890, femur 0.885).
