#!/bin/bash
#SBATCH --job-name=fgm_ssl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
# Fill in --partition/--account for your cluster before submitting.

set -euo pipefail
mkdir -p logs results

# module load cuda/12.1          # uncomment/adjust for your cluster
# source ~/venvs/fgmssl/bin/activate
# pip install torch numpy pandas pillow scikit-learn

ARMS=(mae contrast supervised)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
EPOCHS=100
[ "$ARM" = "supervised" ] && EPOCHS=60

echo "=== arm=$ARM epochs=$EPOCHS on $(hostname) ==="
python run_ssl.py \
  --arm "$ARM" \
  --epochs "$EPOCHS" \
  --batch 64 \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --amp \
  --manifest data/image_clusters.csv \
  --image-root data/frames \
  --panel data/panel.npz \
  --out results \
  ${ARM:+$([ "$ARM" = "supervised" ] && echo "--target cardiac")}

# Score once, after the whole array finishes:
#   sbatch --dependency=afterok:$SLURM_ARRAY_JOB_ID --wrap \
#     "python score_arms.py --panel data/panel.npz --results results \
#                           --frozen data/frozen_usfm.npz"
