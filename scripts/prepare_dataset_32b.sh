#!/bin/bash
#SBATCH --job-name=r2r_feat_32b
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prepare_dataset_32b_%j.out
#SBATCH --error=logs/prepare_dataset_32b_%j.err

source /home1/gosar/CSCI544/r2r-router-optimization/venv_carc/bin/activate

cd /home1/gosar/CSCI544/r2r-router-optimization

echo "=== Step 2: Feature Engineering (train) ==="
python src/prepare_dataset.py --model_name qwen2.5-32B --split train

echo "=== Step 2: Feature Engineering (validation) ==="
python src/prepare_dataset.py --model_name qwen2.5-32B --split validation

echo "=== Step 2: Feature Engineering (test) ==="
python src/prepare_dataset.py --model_name qwen2.5-32B --split test

echo "=== Done ==="
