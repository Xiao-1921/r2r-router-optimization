#!/usr/bin/env bash
# USC CARC smoke test: Qwen2.5-3B on A40 / debug partition
# Goal: validate CARC CUDA + model loading + output writing with a small sample
#
# Submit from repo root:
#   sbatch scripts/run_qwen_3b_smoke_a40.slurm

#SBATCH --job-name=qwen25-3b-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=robinjia_1822
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home1/xli29203/r2r-router-optimization}"
cd "$ROOT"

mkdir -p logs

echo "=== Slurm / node ==="
echo "Job ID:               ${SLURM_JOB_ID:-n/a}"
echo "Submit directory:     ${SLURM_SUBMIT_DIR:-n/a}"
echo "Resolved ROOT:        ${ROOT}"
echo "Hostname:             $(hostname)"
echo "PWD:                  $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

CPUS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"
export TOKENIZERS_PARALLELISM=false

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$HF_HOME/transformers" "$HF_HOME/hub" "$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

if [[ ! -f "${ROOT}/venv_carc/bin/activate" ]]; then
  echo "ERROR: Expected virtualenv at ${ROOT}/venv_carc/bin/activate" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${ROOT}/venv_carc/bin/activate"

MODEL_NAME="qwen2.5-3B-smoke"
HF_ID="Qwen/Qwen2.5-3B-Instruct"

echo "=== Inference smoke test: ${MODEL_NAME} (${HF_ID}) ==="

python src/inference_qwen_mcq.py \
  --model_name "$MODEL_NAME" \
  --input_path data/raw/Train.csv \
  --hf-model-id "$HF_ID" \
  --device cuda \
  --limit 20

echo "=== Smoke test done ==="
