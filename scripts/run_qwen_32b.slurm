#!/usr/bin/env bash

#SBATCH --job-name=qwen25-32b-mcq

#SBATCH --partition=gpu

#SBATCH --gres=gpu:p100:2

#SBATCH --cpus-per-task=8

#SBATCH --mem=48G

#SBATCH --time=24:00:00

#SBATCH --account=zfan0804_712

#SBATCH --output=logs/%x-%j.out

#SBATCH --error=logs/%x-%j.err



set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"

cd "$ROOT"

mkdir -p logs



export HF_HOME=/scratch1/gosar/hf_cache

export TRANSFORMERS_CACHE=$HF_HOME/transformers

export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

export HF_DATASETS_CACHE=$HF_HOME/datasets

export TOKENIZERS_PARALLELISM=false



source /home1/gosar/CSCI544/r2r-router-optimization/venv_carc/bin/activate



python src/inference_qwen_mcq.py --model_name qwen2.5-32B --input_path data/raw/Train.csv --hf-model-id Qwen/Qwen2.5-32B-Instruct --load-in-4bit --device cuda

python src/inference_qwen_mcq.py --model_name qwen2.5-32B --input_path data/raw/Validation.csv --hf-model-id Qwen/Qwen2.5-32B-Instruct --load-in-4bit --device cuda

python src/inference_qwen_mcq.py --model_name qwen2.5-32B --input_path data/raw/Test.csv --hf-model-id Qwen/Qwen2.5-32B-Instruct --load-in-4bit --device cuda



echo Done

