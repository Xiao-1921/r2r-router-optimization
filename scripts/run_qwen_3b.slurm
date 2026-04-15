#!/usr/bin/env bash
# USC CARC: Qwen2.5-3B MCQ inference on Train / Validation / Test CSVs (three separate runs).
# Submit from repo root, e.g.: sbatch scripts/run_qwen_3b.slurm
#
# Requires: venv_carc/ at repo root with torch (CUDA), transformers, pandas, tqdm, etc.
#
#SBATCH --job-name=qwen25-3b-mcq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --account=robinjia_1822
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Use SLURM_SUBMIT_DIR instead of BASH_SOURCE to resolve the repo root.
# BASH_SOURCE[0] can resolve incorrectly on compute nodes, causing cd to land
# in a read-only path and making mkdir -p logs fail immediately under set -e.
ROOT="${SLURM_SUBMIT_DIR}"
cd "$ROOT"
mkdir -p logs

echo "=== Slurm / node ==="
echo "Job ID:               ${SLURM_JOB_ID:-n/a}"
echo "Submit directory:     ${SLURM_SUBMIT_DIR:-n/a}"
echo "Hostname:             $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# ── Thread / parallelism limits ──────────────────────────────────────────────
CPUS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"
export TOKENIZERS_PARALLELISM=false

# ── Hugging Face cache directories ───────────────────────────────────────────
# Override HF_HOME before calling sbatch if you want a non-default location,
# e.g.: HF_HOME=/scratch/your_cache sbatch scripts/run_qwen_3b.slurm
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$HF_HOME/transformers" "$HF_HOME/hub" "$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

# ── Activate virtual environment ─────────────────────────────────────────────
if [[ ! -f "${ROOT}/venv_carc/bin/activate" ]]; then
  echo "ERROR: Expected virtualenv at ${ROOT}/venv_carc/bin/activate" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${ROOT}/venv_carc/bin/activate"

# ── Inference ─────────────────────────────────────────────────────────────────
MODEL_NAME="qwen2.5-3B"
HF_ID="Qwen/Qwen2.5-3B-Instruct"

echo "=== Inference: ${MODEL_NAME} (${HF_ID}) ==="

# Three separate invocations (reloads model each time — simple and matches local workflow).
python src/inference_qwen_mcq.py \
  --model_name "$MODEL_NAME" \
  --input_path data/raw/Train.csv \
  --hf-model-id "$HF_ID" \
  --device cuda

python src/inference_qwen_mcq.py \
  --model_name "$MODEL_NAME" \
  --input_path data/raw/Validation.csv \
  --hf-model-id "$HF_ID" \
  --device cuda

python src/inference_qwen_mcq.py \
  --model_name "$MODEL_NAME" \
  --input_path data/raw/Test.csv \
  --hf-model-id "$HF_ID" \
  --device cuda

echo "=== Done ==="
