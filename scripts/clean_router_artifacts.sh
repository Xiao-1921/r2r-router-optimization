#!/usr/bin/env bash
# Remove generated router training artifacts so the next run cannot accidentally read stale files.
#
# DELETES:
#   - data/router_training_matrix.{pkl,csv,feat_list.txt}
#   - data/router_training_matrix_semantic.{pkl,csv,feat_list.txt}
#   - outputs/calibration.png, outputs/feature_importance.png, and matching *.feat_list.txt
#   - Deprecated names: outputs/calibration_plot.png, outputs/feature_importance_*.png, calibration_*.png
#   - lab_outputs/* (semantic ablation plots and sidecar lists)
#
# DOES NOT delete: data/qwen_mcq_results.pkl, inference CSV, or source code.
#
# Usage (from repo root): bash scripts/clean_router_artifacts.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p outputs lab_outputs

rm -f \
  data/router_training_matrix.pkl \
  data/router_training_matrix.csv \
  data/router_training_matrix.feat_list.txt \
  data/router_training_matrix_semantic.pkl \
  data/router_training_matrix_semantic.csv \
  data/router_training_matrix_semantic.feat_list.txt

rm -f \
  outputs/calibration.png \
  outputs/feature_importance.png \
  outputs/calibration.feat_list.txt \
  outputs/feature_importance.feat_list.txt

rm -f \
  outputs/calibration_plot.png \
  outputs/feature_importance_22feat.png \
  outputs/feature_importance_38feat.png \
  outputs/calibration_22feat.png \
  outputs/calibration_38feat.png

find lab_outputs -maxdepth 1 -type f ! -name '.gitkeep' -delete 2>/dev/null || true

echo "Cleaned router matrix files, outputs/*.png (known names), and lab_outputs/ (keeps .gitkeep)."
