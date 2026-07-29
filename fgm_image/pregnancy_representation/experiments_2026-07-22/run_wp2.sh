#!/usr/bin/env bash
# WP2 axis annotation with visual words -- runs every step in order, stops on first failure.
#
#   bash fgm_image/pregnancy_representation/experiments_2026-07-22/run_wp2.sh
#
# Steps 0 and 1 are idempotent: step 0 rewrites the axis csv, step 1 skips nothing but is only
# ~10 min. Step 2's assignment pass SKIPS encoders already assigned, so re-running is cheap.
# Everything logs to out_probe/run_wp2.log as well as the terminal.
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
K="${K:-16}"
mkdir -p "$S/out_probe"
LOG="$S/out_probe/run_wp2.log"
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "WP2 annotation run  $(date)  K=$K"
echo "=================================================================="

echo
echo "--- STEP 0: tabular WP2 axis (CPU, seconds) ---------------------"
echo "    builds handoff/wp2_axis.csv with a provenance guard that REFUSES"
echo "    an image-contaminated axis (asserts |rho| with the image lag < 0.5)"
"$PY" "$S/wp2_axis_build.py"

echo
echo "--- STEP 1: seed-stability CEILING = the kill test (GPU, ~10 min) "
echo "    K=$K, 4 encoders x 5 seeds. If mean AMI < 0.50 the design FAILS"
echo "    and you should STOP -- cross-encoder agreement would be meaningless."
"$PY" "$S/hpc_seed_ceiling.py" --K "$K" --seeds 5

CEIL=$("$PY" - <<PYEOF
import json,os
p=os.path.join("$S","out_probe","seed_ceiling_K$K.json")
print(json.load(open(p)).get("mean_ceiling_across_encoders",0.0))
PYEOF
)
echo "    measured ceiling: $CEIL"
STOP=$("$PY" -c "print(1 if float('$CEIL')<0.50 else 0)")
if [ "$STOP" = "1" ]; then
  echo
  echo "!!! CEILING BELOW 0.50 -- DESIGN FAILS. Stopping before the annotation."
  echo "!!! Two fits of the SAME encoder differing only by seed do not agree, so"
  echo "!!! cross-encoder agreement cannot be evidence about fetal ultrasound."
  exit 1
fi

echo
echo "--- STEPS 2+3: annotation + mandatory nulls (GPU assign, then CPU) "
echo "    A: assign ALL 20,413 frames to the FROZEN centroids (no refit)"
echo "    B: plane-balanced, budget-equalised per-fetus histograms"
echo "    C: per-code partial Spearman vs the tabular WP2 axis, GA spline + sex adjusted"
echo "    D: stratified conditional permutation null (NOT label shuffle)"
echo "    E: SEX calibration -- if this fails, a null on WP2 is UNINFORMATIVE"
"$PY" "$S/hpc_wp2_annotate.py" --K "$K"

echo
echo "=================================================================="
echo "DONE. Read in this order:"
echo "  1. out_probe/seed_ceiling_K$K.json      -> the ceiling that bounds everything"
echo "  2. out_probe/wp2_annotate_K$K.json      -> per_encoder[].calibration_sex_max_abs_r FIRST"
echo "     (if sex is undetectable, the WP2 null means underpowered, not absent)"
echo "     then meta_analysis_PRIMARY / headline"
echo "  full log: out_probe/run_wp2.log"
echo "=================================================================="
