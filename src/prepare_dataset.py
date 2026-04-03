"""
Prepare router training matrix for R2R-Router.

Inputs:
  - data/qwen_mcq_results.pkl
    Expected keys per record:
      original_question, is_correct,
      first_token_entropy, avg_entropy, perplexity, chosen_token_prob,
      first_token_top5_logprobs (list of 5 floats, descending, nats)

Outputs:
  - data/router_training_matrix.pkl
  - data/router_training_matrix.csv

This script builds:
  - target_label: 1 if model is wrong (is_correct == False), else 0
  - Extensive linguistic features (feat_*) from the question text
  - Model-aware uncertainty features (feat_*) from inference outputs
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Regex feature definitions
# -----------------------------------------------------------------------------

_RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

_RE_NEGATION = re.compile(
    r"\b(?:not|n't|no|never|none|neither|except|unless|without)\b",
    flags=re.IGNORECASE,
)
_RE_COMPARISON = re.compile(
    r"\b(?:most|least|best|worst|better|worse|than|more|less)\b",
    flags=re.IGNORECASE,
)
_RE_QUANT = re.compile(
    r"\b(?:how\s+many|how\s+much|percent(?:age)?|ratio|average|mean|median|"
    r"total|sum|increase|decrease|twice|half|double)\b",
    flags=re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Text feature helpers
# -----------------------------------------------------------------------------


def safe_text(x: Any) -> str:
    """Coerce to a usable string (never None)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def basic_text_features(text: str) -> dict[str, float]:
    """Compute lightweight, model-free textual statistics."""
    t = text.strip()
    words = [w for w in re.split(r"\s+", t) if w]

    char_count = len(t)
    word_count = len(words)
    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count > 0 else 0.0
    punct_count = len(_RE_PUNCT.findall(t))

    return {
        "feat_char_count": float(char_count),
        "feat_word_count": float(word_count),
        "feat_avg_word_len": float(avg_word_len),
        "feat_punct_count": float(punct_count),
    }


def regex_text_features(text: str) -> dict[str, float]:
    """Boolean/ count features from regex keyword detection."""
    t = text
    neg = _RE_NEGATION.findall(t)
    comp = _RE_COMPARISON.findall(t)
    quant = _RE_QUANT.findall(t)

    return {
        "feat_has_negation": float(len(neg) > 0),
        "feat_negation_count": float(len(neg)),
        "feat_has_comparison": float(len(comp) > 0),
        "feat_comparison_count": float(len(comp)),
        "feat_has_quant": float(len(quant) > 0),
        "feat_quant_count": float(len(quant)),
    }


# -----------------------------------------------------------------------------
# spaCy features
# -----------------------------------------------------------------------------


def compute_spacy_features(
    texts: list[str],
    *,
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Compute NER entity counts and POS density features using spaCy.

    POS density is computed as:
      noun_ratio = (# NOUN + # PROPN) / (# non-space tokens)
      verb_ratio = (# VERB + # AUX) / (# non-space tokens)
    """
    try:
        import spacy  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "spaCy is required for prepare_dataset.py. Install with `pip install spacy`."
        ) from e

    try:
        nlp = spacy.load(spacy_model, disable=())  # keep NER+tagger
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"Could not load spaCy model {spacy_model!r}. "
            "Install it with `python -m spacy download en_core_web_sm` "
            "or pass --spacy-model to a model you have installed."
        ) from e

    rows: list[dict[str, float]] = []
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size),
        total=len(texts),
        desc="spaCy features",
    ):
        n_tokens = sum(1 for t in doc if not t.is_space)
        if n_tokens == 0:
            noun_ratio = 0.0
            verb_ratio = 0.0
        else:
            noun_count = sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN"))
            verb_count = sum(1 for t in doc if t.pos_ in ("VERB", "AUX"))
            noun_ratio = noun_count / n_tokens
            verb_ratio = verb_count / n_tokens

        ent_count = float(len(doc.ents))
        rows.append(
            {
                "feat_entity_count": ent_count,
                "feat_noun_ratio": float(noun_ratio),
                "feat_verb_ratio": float(verb_ratio),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Top-5 logprob–based model features (from inference_qwen_mcq.py)
# -----------------------------------------------------------------------------


def _nan_top5_feats() -> dict[str, float]:
    nan = float("nan")
    return {
        "feat_top2_margin": nan,
        "feat_top5_entropy": nan,
        "feat_dist_kurtosis": nan,
        "feat_dist_std": nan,
    }


def top5_logprob_features(lp: Any) -> dict[str, float]:
    """
    Derive scalar features from the first-token top-k log-probabilities (nats, descending).

    - feat_top2_margin: p(1) - p(2) where p(i) = exp(logprob_i) (absolute probability mass).
    - feat_top5_entropy: Shannon entropy of the normalized top-5 distribution (sum of masses = 1).
    - feat_dist_kurtosis: excess kurtosis (Fisher) of the five logprob values.
    - feat_dist_std: standard deviation of the five logprob values (ddof=0).
    """
    if lp is None:
        return _nan_top5_feats()
    if isinstance(lp, str):
        try:
            lp = json.loads(lp)
        except (json.JSONDecodeError, TypeError):
            return _nan_top5_feats()
    if not isinstance(lp, (list, tuple)):
        return _nan_top5_feats()

    arr = np.asarray(lp, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if len(finite) < 2:
        return _nan_top5_feats()

    finite = np.sort(finite)[::-1]

    # Top-2 margin in probability space
    p1 = float(np.exp(finite[0]))
    p2 = float(np.exp(finite[1]))
    feat_top2_margin = p1 - p2

    # Normalized entropy over the top-5 candidates (softmax then H in nats)
    shifted = finite - np.max(finite)
    masses = np.exp(shifted)
    q = masses / masses.sum()
    q = np.clip(q, 1e-30, 1.0)
    feat_top5_entropy = float(-np.sum(q * np.log(q)))

    feat_dist_std = float(np.std(finite, ddof=0))

    # Excess kurtosis of the logprob sample (needs variance > 0; stable for n>=4)
    m = float(np.mean(finite))
    s = float(np.std(finite, ddof=0))
    if s > 0 and len(finite) >= 4:
        z = (finite - m) / s
        feat_dist_kurtosis = float(np.mean(z**4) - 3.0)
    else:
        feat_dist_kurtosis = float("nan")

    return {
        "feat_top2_margin": feat_top2_margin,
        "feat_top5_entropy": feat_top5_entropy,
        "feat_dist_kurtosis": feat_dist_kurtosis,
        "feat_dist_std": feat_dist_std,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare router training matrix from Qwen MCQ inference results")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/qwen_mcq_results.pkl"),
        help="Input inference results PKL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/router_training_matrix.pkl"),
        help="Output training matrix PKL path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path (default: same basename as --output with .csv extension)",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name for NLP features")
    parser.add_argument("--spacy-batch-size", type=int, default=64, help="spaCy nlp.pipe batch size")
    args = parser.parse_args()

    csv_path = args.csv if args.csv is not None else args.output.with_suffix(".csv")

    # -------------------------------------------------------------------------
    # Load inference results
    # -------------------------------------------------------------------------
    with open(args.input, "rb") as f:
        records = pickle.load(f)
    if not isinstance(records, list) or (len(records) > 0 and not isinstance(records[0], dict)):
        raise RuntimeError("Expected input PKL to be a list[dict].")

    df = pd.DataFrame(records)
    if "original_question" not in df.columns or "is_correct" not in df.columns:
        raise RuntimeError("Input PKL is missing required columns: original_question, is_correct.")

    # -------------------------------------------------------------------------
    # Task 1: Labeling (router triggers on model mistakes)
    # -------------------------------------------------------------------------
    df["target_label"] = (~df["is_correct"].astype(bool)).astype(int)

    # -------------------------------------------------------------------------
    # Task 2: Extensive linguistic features (feat_*)
    # -------------------------------------------------------------------------
    texts = df["original_question"].map(safe_text).tolist()

    tqdm.pandas(desc="Basic text features")
    basic_feats = df["original_question"].progress_apply(lambda x: basic_text_features(safe_text(x)))
    basic_df = pd.DataFrame(list(basic_feats))

    tqdm.pandas(desc="Regex keyword features")
    regex_feats = df["original_question"].progress_apply(lambda x: regex_text_features(safe_text(x)))
    regex_df = pd.DataFrame(list(regex_feats))

    spacy_df = compute_spacy_features(
        texts,
        spacy_model=args.spacy_model,
        batch_size=args.spacy_batch_size,
    )

    # -------------------------------------------------------------------------
    # Task 3: Model-aware features (feat_*)
    # -------------------------------------------------------------------------
    model_aware = pd.DataFrame(
        {
            "feat_first_token_entropy": df["first_token_entropy"].astype(float),
            "feat_avg_entropy": df["avg_entropy"].astype(float),
            "feat_perplexity": df["perplexity"].astype(float),
            "feat_chosen_token_prob": df["chosen_token_prob"].astype(float),
        }
    )
    model_aware["feat_prob_margin"] = 1.0 - model_aware["feat_chosen_token_prob"]

    if "first_token_top5_logprobs" in df.columns:
        tqdm.pandas(desc="Top-5 logprob features")
        top5_extra = df["first_token_top5_logprobs"].progress_apply(top5_logprob_features)
        top5_df = pd.DataFrame(top5_extra.tolist(), index=df.index)
    else:
        top5_df = pd.DataFrame(
            [_nan_top5_feats() for _ in range(len(df))],
            index=df.index,
        )

    # Merge all features
    out_df = pd.concat([df, basic_df, regex_df, spacy_df, model_aware, top5_df], axis=1)

    # -------------------------------------------------------------------------
    # Persist outputs
    # -------------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(out_df, f)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"Wrote training matrix PKL: {args.output} (rows={len(out_df)})")
    print(f"Wrote training matrix CSV: {csv_path} (rows={len(out_df)})")


if __name__ == "__main__":
    main()

