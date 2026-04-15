# R2R-Router: Model-Aware Routing for Reliable LLM Generation

Repository for the CSCI 544 Applied NLP project: **When Should a Language Model Trust Itself?**  
The **router** predicts whether the base LLM is likely to answer a multiple-choice question **wrong** (`target_label = 1`), so the system can selectively trigger Retrieval-Augmented Generation (RAG).

---

## What This Project Does (Recap)

1. **Inference** — Run Qwen2.5 (or another HF causal LM) on MCQ rows from **CSV** (Apple Silicon **MPS** or **CPU**), record first-token logprobs, entropy, perplexity, and top-5 first-token logprobs for downstream **model-aware** features.
2. **Dataset preparation** — Build `target_label` from correctness, extract **22** default features (linguistic + regex + spaCy + uncertainty + top-5–derived metrics). Optional **Direction 3** adds **16** semantic PCA features (`feat_pc_*`) for ablation only (`--semantic-pca`).
3. **Training & evaluation** — Compare baselines (Always-RAG, Logistic Regression, Random Forest) with a **GridSearchCV-tuned XGBoost** router; report F1, ROC-AUC, calibration (ECE), and optional Platt scaling. Saves the best pipeline to **`models/{model_name}/router_model.joblib`**.
4. **Reporting** — Generate figures for the mid-term report: benchmark progression, AUC/ECE charts, reliability diagrams, feature importance, and optional ablation tables.

---

## End-to-End Pipeline (Script Flow)

```text
data/raw/*.csv  (MCQ: question, choice_labels, choice_texts, gold_label; optional split)
        │
        ▼
  inference_qwen_mcq.py  ──►  data/processed/{model_name}/inference_results_*.pkl (+ .csv)
        │
        ▼
  prepare_dataset.py       ──►  data/processed/{model_name}/router_training_matrix_*.pkl
                             (+ .csv, .feat_list.txt next to each matrix)
        │
        ▼
  trainer.py               ──►  outputs/{model_name}/calibration.png
                             outputs/{model_name}/feature_importance.png
                             outputs/{model_name}/feature_importance.feat_list.txt
                             models/{model_name}/router_model.joblib
        │
        ├── generate_report_visuals.py  ──►  outputs/report/ (figures + ablation .md/.tex)
        ├── report_charts.py            ──►  outputs/charts/ (static summary PNGs)
        │
        └── scripts/run_semantic_ablation.sh  ──►  lab_outputs/ (may need flag updates for new CLI)
             (optional: scripts/clean_router_artifacts.sh — paths may lag the new layout)
```

**Utilities (not in the main training path):** `mps_gpu_check.py` (MPS verification), `data_utils.py` / `feature_engineering.py` (lightweight helpers / legacy stubs).

---

## Repository Layout

| Path | Purpose |
|------|---------|
| **`data/raw/`** | Source MCQ **CSVs** (`Train.csv`, `Validation.csv`, `Test.csv`, etc.). List columns `choice_labels` / `choice_texts` are Python-list strings parsed with `ast.literal_eval`. |
| **`data/processed/{model_name}/`** | Per–HF-model artifacts: inference PKL/CSV, `router_training_matrix_*.{pkl,csv,feat_list.txt}`. Typical contents are **gitignored** except `data/.gitkeep`. |
| **`outputs/{model_name}/`** | Default **trainer** plots and `feature_importance.feat_list.txt` (see `trainer.py --reports_dir`). |
| **`models/{model_name}/`** | Saved router **`router_model.joblib`** (GridSearchCV best XGBoost pipeline). |
| **`src/`** | All Python entrypoints and modules (see table below). |
| **`scripts/`** | Shell helpers: semantic ablation, artifact cleanup. |
| **`outputs/report/`** | Figures from `generate_report_visuals.py` (benchmark, boxplot, reliability, top-15 importance, ablation tables). |
| **`outputs/charts/`** | Static slides-style charts from `report_charts.py` (F1 progression, AUC, ECE). |
| **`lab_outputs/`** | Ablation-only outputs when you point `trainer.py` at **`--reports_dir lab_outputs`** (or similar). |
| **`temp/`** | Gitignored scratch / intermediate exports. |

---

## `src/` Scripts

| Script | Role |
|--------|------|
| **`inference_qwen_mcq.py`** | Loads MCQ **CSV** via `pandas.read_csv`, parses list columns with **`ast.literal_eval`**, runs **`--hf-model-id`** with the chat template on **MPS** or **CPU**, writes **`data/processed/{model_name}/inference_results_{split_or_stem}.{pkl,csv}`**. Key flags: **`--model_name`**, **`--input_path`**, **`--output_dir`**, optional **`--split`**. Uses **`mps_gpu_check`**. |
| **`mps_gpu_check.py`** | Verifies PyTorch **MPS** is usable (small GPU op); imported before heavy inference. |
| **`prepare_dataset.py`** | Reads inference PKL, sets **`target_label`**, builds **22** `feat_*` by default. Optional **`--semantic-pca`** → **38** `feat_*`. Writes under **`data/processed/{model_name}/`**. Key flags: **`--model_name`**, **`--input_path`**, **`--output_dir`**, optional **`--split`** / **`--output_stem`**. |
| **`trainer.py`** | Loads a training-matrix PKL from **`data/processed/...`** (default **`--data_dir`**), stratified train/test split or **`--test_input_path`**, baselines + **GridSearchCV XGBoost**, calibration plots to **`outputs/{model_name}/`** (**`--reports_dir`**), saves **`models/{model_name}/router_model.joblib`**. Optional **`--exclude-feat-prefix`** for ablation. |
| **`generate_report_visuals.py`** | Re-trains LR/XGB and writes report figures under **`outputs/report/`**. |
| **`report_charts.py`** | Writes static summary PNGs to **`outputs/charts/`**. |
| **`data_utils.py`** | Small **`DataLoader`** for legacy JSON (optional exploratory use). |
| **`feature_engineering.py`** | Legacy **`RouterFeatureExtractor`** stub; main feature logic is in **`prepare_dataset.py`**. |

---

## `scripts/` Helpers

| Script | Role |
|--------|------|
| **`run_semantic_ablation.sh`** | Intended to regenerate 22- and 38-feature matrices and run **`trainer.py`** twice into **`lab_outputs/`**. **Note:** the script may still use older paths/flags; mirror the **Quick Commands** below with **`--reports_dir lab_outputs`** and explicit **`--input_path`** / **`--output_stem`** until the shell script is updated. |
| **`clean_router_artifacts.sh`** | Removes legacy flat **`data/`** matrix names and some **`outputs/`** plots. Extend or run manually if you need to wipe **`data/processed/`** or per-model **`outputs/`** / **`models/`** trees. |

---

## Setup

1. **Python** 3.10+ recommended (project was developed with 3.10.x).
2. **Core dependencies:**  
   `pip install pandas numpy scikit-learn xgboost matplotlib seaborn tqdm transformers torch joblib`  
   plus **spaCy** English model: `python -m spacy download en_core_web_sm`
3. **Inference-only:** PyTorch with **MPS** on Apple Silicon (see `mps_gpu_check.py`).
4. **Optional semantic PCA** (`prepare_dataset.py --semantic-pca`):  
   `pip install sentence-transformers`

---

## Quick Commands

Replace `MODEL` (e.g. `qwen2.5-0.5b`) and paths to match your CSVs and Hugging Face model id.

```bash
# 1) Inference → data/processed/{MODEL}/
python src/inference_qwen_mcq.py \
  --model_name MODEL \
  --input_path data/raw/Train.csv \
  --hf-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu

# Optional: validation / test CSVs (separate files or use --split if a combined CSV has a split column)
python src/inference_qwen_mcq.py --model_name MODEL --input_path data/raw/Test.csv --device cpu

# 2) Features → training matrix under data/processed/{MODEL}/
python src/prepare_dataset.py --model_name MODEL --split train
python src/prepare_dataset.py --model_name MODEL --split test

# 3) Train / evaluate → outputs/{MODEL}/ and models/{MODEL}/
python src/trainer.py \
  --model_name MODEL \
  --input_path data/processed/MODEL/router_training_matrix_train.pkl \
  --test_input_path data/processed/MODEL/router_training_matrix_test.pkl

# Report figures (paths inside script may still assume legacy data/ filenames)
python src/generate_report_visuals.py

# Static slide charts
python src/report_charts.py
```

**Trainer path flags (summary):** **`--data_dir`** — default root for the training PKL if **`--input_path`** is omitted; **`--reports_dir`** — PNGs + `feature_importance.feat_list.txt` (default **`outputs/{model_name}/`**); **`--model_output_path`** — override for **`router_model.joblib`**. **`--output_dir`** is a deprecated alias for **`--reports_dir`** when **`--reports_dir`** is not set.

---

## Historical Baselines (Report)

- **Previous baseline (6 linguistic + LR):** F1 ≈ **0.435** (early milestone; not produced by the current 22-feature script without modification).
- **Current stack (22 multimodal features, same split):** LR Test F1 ≈ **0.697**, tuned XGBoost Test F1 ≈ **0.713**, Test AUC ≈ **0.82** (see `trainer.py` logs for exact runs).
