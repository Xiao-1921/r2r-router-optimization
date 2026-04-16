#!/bin/bash
#SBATCH --job-name=r2r_train_7b
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/trainer_7b_%j.out
#SBATCH --error=logs/trainer_7b_%j.err

source ~/r2r-router-optimization/venv_carc/bin/activate
cd ~/r2r-router-optimization

echo "=== Step 3: Train XGBoost Router (7B) ==="
python src/trainer.py \
  --model_name qwen2.5-7B \
  --input_path data/processed/qwen2.5-7B/router_training_matrix_train.pkl \
  --test_input_path data/processed/qwen2.5-7B/router_training_matrix_test.pkl \
  --reports_dir outputs/qwen2.5-7B \
  --print-importance-ranking

echo "=== Done ==="
