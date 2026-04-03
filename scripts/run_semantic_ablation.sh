#!/usr/bin/env bash
# Regenerate 22-feature (production) and 38-feature (semantic ablation) matrices, then train both.
# Plots for the ablation go to lab_outputs/ (not outputs/). Production defaults stay in outputs/
# when you run trainer.py alone without this script.
#
# Optional: wipe stale artifacts first — same as: bash scripts/clean_router_artifacts.sh
#   RUN_CLEAN=1 bash scripts/run_semantic_ablation.sh
#
# Run from repo root: bash scripts/run_semantic_ablation.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${RUN_CLEAN:-0}" == "1" ]]; then
  bash "$ROOT/scripts/clean_router_artifacts.sh"
fi

mkdir -p lab_outputs outputs

python src/prepare_dataset.py --input data/qwen_mcq_results.pkl --output data/router_training_matrix.pkl
python src/prepare_dataset.py --semantic-pca --input data/qwen_mcq_results.pkl --output data/router_training_matrix_semantic.pkl

echo "=== Trainer: 22 features (production baseline for comparison) → lab_outputs/ ==="
python src/trainer.py \
  --data data/router_training_matrix.pkl \
  --feature-importance-out lab_outputs/feature_importance_22feat.png \
  --calibration-out lab_outputs/calibration_22feat.png

echo "=== Trainer: 38 features (semantic ablation) → lab_outputs/ ==="
python src/trainer.py \
  --data data/router_training_matrix_semantic.pkl \
  --feature-importance-out lab_outputs/feature_importance_38feat.png \
  --calibration-out lab_outputs/calibration_38feat.png
