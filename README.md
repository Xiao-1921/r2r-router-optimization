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

Additional dependencies for the RAG pipeline and R2R baseline comparison:

```bash
pip install wikipedia-api sentence-transformers bitsandbytes accelerate
```

Notes:
- The inference script downloads the Hugging Face model specified by `--hf-model-id`.
- If you want to control the Hugging Face cache location, set `HF_HOME` before running inference.

## Device or System Used to Run the Code

This codebase is set up for two main environments:

- Apple Silicon macOS with PyTorch MPS acceleration. This is the default local inference path (`--device mps`).
- Linux GPU nodes with NVIDIA CUDA. The repository includes Slurm scripts for USC CARC in [scripts/run_qwen_3b.slurm](/scripts/run_qwen_3b.slurm) and [scripts/run_qwen_7b.slurm](/scripts/run_qwen_7b.slurm).

The inference script also supports CPU with `--device cpu`.

The R2R baseline comparison (`src/r2r_baseline_comparison.py`) requires a CUDA GPU with at least 16 GB VRAM (A100 recommended). It uses 4-bit quantization via bitsandbytes.

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

### 4. Run R2R baseline comparison (Router + RAG)

This runs all three conditions (No RAG, Always RAG, Router RAG) on the test set using the trained router and the RAG retrieval pipeline. Requires a CUDA GPU.

```bash
python src/r2r_baseline_comparison.py \
  --model_name qwen2.5-3B \
  --hf_model_id Qwen/Qwen2.5-3B-Instruct \
  --router_path models/qwen2.5-3B/router_model.joblib \
  --feature_matrix data/processed/qwen2.5-3B/router_training_matrix_test.csv \
  --test_csv data/raw/Test.csv
```

For other model sizes, swap the paths:

```bash
# 7B
python src/r2r_baseline_comparison.py \
  --model_name qwen2.5-7B \
  --hf_model_id Qwen/Qwen2.5-7B-Instruct \
  --router_path models/qwen2.5-7B/router_model.joblib \
  --feature_matrix data/processed/qwen2.5-7B/router_training_matrix_test.csv \
  --test_csv data/raw/Test.csv

# 14B
python src/r2r_baseline_comparison.py \
  --model_name qwen2.5-14B \
  --hf_model_id Qwen/Qwen2.5-14B-Instruct \
  --router_path models/qwen2.5-14B/router_model.joblib \
  --feature_matrix data/processed/qwen2.5-14B/router_training_matrix_test.csv \
  --test_csv data/raw/Test.csv

# 32B
python src/r2r_baseline_comparison.py \
  --model_name qwen2.5-32B \
  --hf_model_id Qwen/Qwen2.5-32B-Instruct \
  --router_path models/qwen2.5-32B/router_model.joblib \
  --feature_matrix data/processed/qwen2.5-32B/router_training_matrix_test.csv \
  --test_csv data/raw/Test.csv
```

This step writes:

- `outputs/<model_name>/r2r_baseline_comparison.csv`

### 5. Optional report figures

```bash
python src/generate_report_visuals.py
python src/report_charts.py
```

### 6. Generate a new evaluation dataset

To sample a fresh set of questions with deduplication against the existing training pool:

```bash
python scripts/generate_dataset.py \
  --existing_csv data/raw/RouterReflector_Combined_Dataset.csv \
  --samples_per_dataset 500 \
  --output_dir data/raw/ \
  --seed 42
```

This produces three stratified splits with zero overlap against the existing dataset:

- `data/raw/Train.csv` (2,400 questions, 400 per benchmark)
- `data/raw/Validation.csv` (300 questions, 50 per benchmark)
- `data/raw/Test.csv` (300 questions, 50 per benchmark)

## How the Results Are Generated

The project generates results in four stages:

1. `src/inference_qwen_mcq.py` runs a Hugging Face causal language model on each multiple-choice question and stores the predicted answer together with uncertainty signals such as entropy, perplexity, chosen-token probability, and top-5 first-token log-probabilities.
2. `src/prepare_dataset.py` converts those inference outputs into a router dataset. It creates `target_label = 1` when the base model is wrong and `target_label = 0` when the base model is correct, then builds the default 22 `feat_*` features from text patterns, spaCy features, and model-confidence features.
3. `src/trainer.py` trains and compares routing baselines, including Logistic Regression, Random Forest, and a tuned XGBoost router. It evaluates them with F1 and ROC-AUC, produces the calibration and feature-importance figures, and saves the best router model.
4. `src/r2r_baseline_comparison.py` runs the end-to-end R2R evaluation. For each test question, it runs three conditions: No RAG (vanilla), Always RAG (raw unfiltered Wikipedia context on every question), and Router RAG (retrieval triggered only when the router predicts failure, with similarity filtering). The RAG retrieval pipeline (`src/rag_pipeline.py`) handles Wikipedia search and passage filtering.

### RAG Pipeline Details

The RAG module (`src/rag_pipeline.py`) retrieves external context from Wikipedia in two stages:

**Query formulation** uses a cascading strategy:
1. Named entities extracted via capitalized phrase detection (e.g., "Mount Kilimanjaro")
2. Content keywords after stopword removal
3. Cleaned question text as fallback

Each query is passed to Wikipedia's opensearch API, which returns pages ranked by relevance. Up to 3 passages are retrieved per question.

**Relevance filtering** uses Sentence-BERT (all-MiniLM-L6-v2) to compute cosine similarity between the question and each retrieved passage. Passages below a threshold of 0.3 are discarded. In practice, over 90% of questions receive zero passages after filtering.

For the Always RAG baseline, the similarity filter is removed to show the true cost of indiscriminate retrieval. For Router RAG, the filter is applied and retrieval is only triggered when the router predicts failure.

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

### Step 4 — R2R Baseline Comparison

After router training, run the three-way comparison for each model size:

```bash
python src/r2r_baseline_comparison.py \
  --model_name qwen2.5-14B \
  --hf_model_id Qwen/Qwen2.5-14B-Instruct \
  --router_path models/qwen2.5-14B/router_model.joblib \
  --feature_matrix data/processed/qwen2.5-14B/router_training_matrix_test.csv \
  --test_csv data/raw/Test.csv
```
