# R2R-Router: Model-Aware Routing for Reliable LLM Generation

Repository for the CSCI 544 Applied NLP project: **When Should a Language Model Trust Itself?**  
The **router** predicts whether the base LLM is likely to answer a multiple-choice question **wrong** (`target_label = 1`), so the system can selectively trigger Retrieval-Augmented Generation (RAG).

---

## What This Project Does (Recap)

1. **Inference** — Run Qwen2.5 on MCQ items (Apple Silicon MPS supported), record first-token logprobs, entropy, perplexity, and top-5 first-token logprobs for downstream **model-aware** features.
2. **Dataset preparation** — Build `target_label` from correctness, extract **22** default features (linguistic + regex + spaCy + uncertainty + top-5–derived metrics). Optional **Direction 3** adds **16** semantic PCA features (`feat_pc_*`) for ablation only (`--semantic-pca`).
3. **Training & evaluation** — Compare baselines (Always-RAG, Logistic Regression, Random Forest) with a **GridSearchCV-tuned XGBoost** router; report F1, ROC-AUC, calibration (ECE), and optional Platt scaling.
4. **Reporting** — Generate figures for the mid-term report: benchmark progression, AUC/ECE charts, reliability diagrams, feature importance, and optional ablation tables.

---

## End-to-End Pipeline (Script Flow)

```text
combined_dataset.json (or similar)
        │
        ▼
  inference_qwen_mcq.py  ──►  data/qwen_mcq_results.pkl (+ .csv)
        │
        ▼
  prepare_dataset.py     ──►  data/router_training_matrix.pkl (+ .csv, .feat_list.txt)
        │                    [optional: --semantic-pca → router_training_matrix_semantic.pkl]
        ▼
  trainer.py             ──►  metrics, outputs/calibration.png, outputs/feature_importance.png
        │
        ├── generate_report_visuals.py  ──►  outputs/report/ (figures + ablation .md/.tex)
        ├── report_charts.py            ──►  outputs/charts/ (static summary PNGs)
        │
        └── scripts/run_semantic_ablation.sh  ──►  lab_outputs/ (22 vs 38 ablation plots)
             (optional: scripts/clean_router_artifacts.sh before a full rerun)
```

**Utilities (not in the main training path):** `mps_gpu_check.py` (MPS verification), `data_utils.py` / `feature_engineering.py` (lightweight helpers / legacy stubs).

---

## Repository Layout

| Path | Purpose |
|------|---------|
| **`data/`** | Datasets and generated matrices. Contents are **gitignored** except `data/.gitkeep`; typical files include `combined_dataset.json`, `qwen_mcq_results.pkl`, `router_training_matrix.pkl` / `.csv`, optional `router_training_matrix_semantic.*`, and `*.feat_list.txt` sidecars. |
| **`src/`** | All Python entrypoints and modules (see table below). |
| **`scripts/`** | Shell helpers: full semantic ablation pipeline, artifact cleanup. |
| **`outputs/`** | Default **production** trainer plots (`calibration.png`, `feature_importance.png`, `*.feat_list.txt`). |
| **`outputs/report/`** | Figures from `generate_report_visuals.py` (benchmark, boxplot, reliability, top-15 importance, ablation tables). |
| **`outputs/charts/`** | Static slides-style charts from `report_charts.py` (F1 progression, AUC, ECE). |
| **`lab_outputs/`** | **Ablation-only** outputs (e.g. 22-feature vs 38-feature calibration and importance plots) from `run_semantic_ablation.sh`. |
| **`temp/`** | Gitignored scratch / intermediate exports. |
| **`models/`** | Reserved for saved model binaries (gitignored patterns in `.gitignore`). |

---

## `src/` Scripts

| Script | Role |
|--------|------|
| **`inference_qwen_mcq.py`** | Loads MCQ JSON, runs **Qwen2.5** with chat template on **MPS** (or CPU), computes first-token and sequence uncertainty metrics, top-5 first-token logprobs, writes **`qwen_mcq_results.pkl`** (+ CSV). Uses **`mps_gpu_check`**. |
| **`mps_gpu_check.py`** | Verifies PyTorch **MPS** is usable (small GPU op); imported before heavy inference. |
| **`prepare_dataset.py`** | Reads inference PKL, sets **`target_label`**, builds **22** `feat_*` by default (basic/regex/spaCy + model-aware + top-5 features). Optional **`--semantic-pca`** adds **38** `feat_*` (SentenceTransformer + PCA). Writes **`router_training_matrix.pkl`**, CSV, and **`.feat_list.txt`**. |
| **`trainer.py`** | Loads a training-matrix PKL, stratified train/test split, **SimpleImputer** + baselines (Always-RAG, LR, RF), **GridSearchCV XGBoost**, calibration curves, optional **Platt** (`CalibratedClassifierCV`), ECE, feature-importance plot. Optional **`--exclude-feat-prefix`** for ablation. |
| **`generate_report_visuals.py`** | Re-trains LR/XGB (same split as trainer) and writes **report figures** under **`outputs/report/`** (benchmark bar, `feat_top2_margin` boxplot, top-15 importance with readable names, reliability Base LLM vs Platt, ablation **.md/.tex** if semantic PKL exists). |
| **`report_charts.py`** | Writes **static** summary PNGs to **`outputs/charts/`** (F1 progression, AUC comparison, ECE bars) using fixed headline numbers for slides. |
| **`data_utils.py`** | Small **`DataLoader`** for `combined_dataset.json` (optional exploratory use). |
| **`feature_engineering.py`** | Legacy **`RouterFeatureExtractor`** stub (simple linguistic placeholders); **main** feature logic lives in **`prepare_dataset.py`**. |

---

## `scripts/` Helpers

| Script | Role |
|--------|------|
| **`run_semantic_ablation.sh`** | Regenerates 22-feature and 38-feature matrices, runs **`trainer.py`** twice, writes plots to **`lab_outputs/`** (optional **`RUN_CLEAN=1`** to run **`clean_router_artifacts.sh`** first). |
| **`clean_router_artifacts.sh`** | Removes known generated router matrices, default **`outputs/`** plots, and **`lab_outputs/`** files (does **not** delete `qwen_mcq_results.pkl`). |

---

## Setup

1. **Python** 3.10+ recommended (project was developed with 3.10.x).
2. **Core dependencies:**  
   `pip install pandas numpy scikit-learn xgboost matplotlib seaborn tqdm transformers torch`  
   plus **spaCy** English model: `python -m spacy download en_core_web_sm`
3. **Inference-only:** PyTorch with **MPS** on Apple Silicon (see `mps_gpu_check.py`).
4. **Optional semantic PCA** (`prepare_dataset.py --semantic-pca`):  
   `pip install sentence-transformers`

---

## Quick Commands

```bash
# 1) Inference → PKL
python src/inference_qwen_mcq.py --input data/combined_dataset.json --output data/qwen_mcq_results.pkl

# 2) Features → training matrix (22 feat_* by default)
python src/prepare_dataset.py --input data/qwen_mcq_results.pkl --output data/router_training_matrix.pkl

# 3) Train / evaluate → outputs/
python src/trainer.py --data data/router_training_matrix.pkl

# Report figures (needs both matrices for ablation table if 38-feature path exists)
python src/generate_report_visuals.py

# Static slide charts
python src/report_charts.py
```

---

## Historical Baselines (Report)

- **Previous baseline (6 linguistic + LR):** F1 ≈ **0.435** (early milestone; not produced by the current 22-feature script without modification).
- **Current stack (22 multimodal features, same split):** LR Test F1 ≈ **0.697**, tuned XGBoost Test F1 ≈ **0.713**, Test AUC ≈ **0.82** (see `trainer.py` logs for exact runs).

---

## License / Course Context

Course project (USC CSCI 544). Adapt paths and credentials for your environment; **`data/*`** is excluded from version control by default.
