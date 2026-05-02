# R2R Router Optimization

## How to Set Up the Environment

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn tqdm transformers torch joblib spacy
python -m spacy download en_core_web_sm
```

Optional dependency for semantic PCA ablation only:

```bash
pip install sentence-transformers
```

Notes:
- The inference script downloads the Hugging Face model specified by `--hf-model-id`.
- If you want to control the Hugging Face cache location, set `HF_HOME` before running inference.

## Device or System Used to Run the Code

This codebase is set up for two main environments:

- Apple Silicon macOS with PyTorch MPS acceleration. This is the default local inference path (`--device mps`).
- Linux GPU nodes with NVIDIA CUDA. The repository includes Slurm scripts for USC CARC in [scripts/run_qwen_3b.slurm](/Users/castro/JUST_DO_IT/production/study/USC/2026Spring/csci544/project/r2r-router-optimization/scripts/run_qwen_3b.slurm) and [scripts/run_qwen_7b.slurm](/Users/castro/JUST_DO_IT/production/study/USC/2026Spring/csci544/project/r2r-router-optimization/scripts/run_qwen_7b.slurm).

The inference script also supports CPU with `--device cpu`.

## Instructions for Running the Code

Run the pipeline from the repository root.

### 1. Generate model inference outputs

```bash
python src/inference_qwen_mcq.py \
  --model_name qwen2.5-0.5b \
  --input_path data/raw/Train.csv \
  --hf-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device mps

python src/inference_qwen_mcq.py \
  --model_name qwen2.5-0.5b \
  --input_path data/raw/Test.csv \
  --hf-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device mps
```

This step writes inference results to:

- `data/processed/qwen2.5-0.5b/inference_results_train.pkl`
- `data/processed/qwen2.5-0.5b/inference_results_test.pkl`

### 2. Build router feature matrices

```bash
python src/prepare_dataset.py --model_name qwen2.5-0.5b --split train
python src/prepare_dataset.py --model_name qwen2.5-0.5b --split test
```

This step writes:

- `data/processed/qwen2.5-0.5b/router_training_matrix_train.pkl`
- `data/processed/qwen2.5-0.5b/router_training_matrix_test.pkl`

### 3. Train and evaluate the router

```bash
python src/trainer.py \
  --model_name qwen2.5-0.5b \
  --input_path data/processed/qwen2.5-0.5b/router_training_matrix_train.pkl \
  --test_input_path data/processed/qwen2.5-0.5b/router_training_matrix_test.pkl
```

This step writes:

- `models/qwen2.5-0.5b/router_model.joblib`
- `outputs/qwen2.5-0.5b/calibration.png`
- `outputs/qwen2.5-0.5b/feature_importance.png`

### 4. Optional report figures

```bash
python src/generate_report_visuals.py
python src/report_charts.py
```

## How the Results Are Generated

The project generates results in three stages:

1. `src/inference_qwen_mcq.py` runs a Hugging Face causal language model on each multiple-choice question and stores the predicted answer together with uncertainty signals such as entropy, perplexity, chosen-token probability, and top-5 first-token log-probabilities.
2. `src/prepare_dataset.py` converts those inference outputs into a router dataset. It creates `target_label = 1` when the base model is wrong and `target_label = 0` when the base model is correct, then builds the default 22 `feat_*` features from text patterns, spaCy features, and model-confidence features.
3. `src/trainer.py` trains and compares routing baselines, including Logistic Regression, Random Forest, and a tuned XGBoost router. It evaluates them with F1 and ROC-AUC, produces the calibration and feature-importance figures, and saves the best router model.

In short, the final reported outputs come from predicting when the base LLM is likely to fail, so the router can decide when retrieval should be triggered.

## Running 14B and 32B on USC CARC

Parth Gosar extended the pipeline to Qwen2.5-14B-Instruct and Qwen2.5-32B-Instruct. Submit from the repo root after activating the venv:

```bash
source venv_carc/bin/activate
```

### Step 1 — Inference

**14B** (~10 hrs for Train split, runs on V100):
```bash
sbatch scripts/run_qwen_14b.slurm
```

**32B** (~3 hrs per split, uses 4-bit quantization on p100:2):
```bash
sbatch scripts/run_qwen_32b.slurm
```

> Note: For 32B, set `HF_HOME` to a scratch directory to avoid home quota issues:
> `export HF_HOME=/scratch1/<your_username>/hf_cache`

Outputs: `data/processed/qwen2.5-{14B,32B}/inference_results_{train,validation,test}.{pkl,csv}`

### Step 2 — Feature Engineering
```bash
sbatch scripts/prepare_dataset_14b.sh
sbatch scripts/prepare_dataset_32b.sh
```

Outputs: `data/processed/qwen2.5-{14B,32B}/router_training_matrix_{split}.{pkl,csv}`

### Step 3 — Router Training
```bash
sbatch scripts/trainer_14b.sh
sbatch scripts/trainer_32b.sh
```

Outputs: `models/qwen2.5-{14B,32B}/router_model.joblib`, `outputs/qwen2.5-{14B,32B}/`
