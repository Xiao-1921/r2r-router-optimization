# R2R-Router: Model-Aware Routing for Reliable LLM Generation

This repository contains the **Router** component for the CSCI 544 Applied NLP project: *When Should a Language Model Trust Itself?*.

## Overview
The Router is designed to detect "knowledge gaps" in Large Language Models (LLMs) before generation. It selectively triggers Retrieval-Augmented Generation (RAG) based on linguistic features and model uncertainty (Logprobs/Entropy).

## Project Mission
- **Baseline F1:** 0.435 (Logistic Regression)
- **Target F1:** > 0.55 (XGBoost + Uncertainty Features)
- **Key Innovation:** Moving from surface-level text features to internal model-awareness.

## Repository Structure
- `data/`: Raw and processed datasets (e.g., `combined_dataset.json`).
- `src/`: Core Python modules for feature extraction and training.
- `models/`: Trained model artifacts (.pkl).
- `notebooks/`: Data exploration and Mid-term report visualizations.

## Setup
1. Use Python 3.10.14.
2. Install dependencies: `pip install pandas xgboost scikit-learn transformers matplotlib seaborn`