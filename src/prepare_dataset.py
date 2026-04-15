"""
Prepare router training matrix for R2R-Router.

Inputs:
  - Inference PKL under ``data/processed/{model_name}/`` (from ``inference_qwen_mcq.py``).
    Expected keys per record:
      original_question, is_correct,
      first_token_entropy, avg_entropy, perplexity, chosen_token_prob,
      first_token_top5_logprobs (list of 5 floats, descending, nats)

Outputs:
  - ``{output_dir}/router_training_matrix[_{split}].pkl``
  - matching ``.csv`` and ``.feat_list.txt`` (sorted feat_* names, one per line)

This script builds:
  - target_label: 1 if model is wrong (is_correct == False), else 0
  - Extensive linguistic features (feat_*) from the question text
  - Model-aware uncertainty features (feat_*) from inference outputs
  - By default **22** feat_* columns (production router).

Optional ablation (``--semantic-pca``): SentenceTransformer (all-MiniLM-L6-v2) + PCA to
feat_pc_01 … feat_pc_16 → **38** feat_* total. Use a separate ``--output`` path for the
38-feature matrix and compare with ``trainer.py`` vs the default 22-feature run.
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

# Expected feat_* count (keep in sync with feature blocks below).
EXPECTED_FEAT_BASE = 22
EXPECTED_FEAT_SEMANTIC_PCA = 16


def _sanitize_model_dirname(model_name: str) -> str:
    cleaned = model_name.strip().replace("/", "_").replace("\\", "_")
    if not cleaned:
        raise ValueError("model_name must be a non-empty string after sanitization.")
    return cleaned


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
# Semantic embeddings + PCA (Direction 3)
# -----------------------------------------------------------------------------


def pca_column_names(n_components: int) -> list[str]:
    return [f"feat_pc_{i + 1:02d}" for i in range(n_components)]


def compute_semantic_pca_features(
    texts: list[str],
    target: pd.Series,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    n_components: int = 16,
    pca_train_size: float = 0.8,
    random_state: int = 42,
    encode_batch_size: int = 32,
    fit_on_full: bool = False,
) -> pd.DataFrame:
    """
    Encode ``original_question`` texts with SentenceTransformer, then reduce to ``n_components``
    via PCA. By default PCA is **fitted on a stratified train split** and applied to all rows.

    Returns a DataFrame with columns ``feat_pc_01`` … ``feat_pc_{n_components:02d}``.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required for semantic features. "
            "Install with `pip install sentence-transformers`."
        ) from e

    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb = np.asarray(embeddings, dtype=np.float32)
    n_samples, d = emb.shape
    if n_components > min(n_samples, d):
        raise ValueError(
            f"n_components={n_components} too large for n_samples={n_samples}, dim={d}."
        )

    idx = np.arange(n_samples)
    if fit_on_full:
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(emb)
        reduced = pca.transform(emb)
    else:
        idx_train, idx_test = train_test_split(
            idx,
            train_size=pca_train_size,
            random_state=random_state,
            stratify=target.astype(int).values,
        )
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(emb[idx_train])
        reduced = np.empty((n_samples, n_components), dtype=np.float64)
        reduced[idx_train] = pca.transform(emb[idx_train])
        reduced[idx_test] = pca.transform(emb[idx_test])

    cols = pca_column_names(n_components)
    return pd.DataFrame(reduced, columns=cols, index=target.index)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare router training matrix from Qwen MCQ inference results",
        epilog=(
            "Default: 22 feat_* columns. Ablation (38 features): "
            "python src/prepare_dataset.py --semantic-pca "
            "--output_stem router_training_matrix_semantic\n"
            "Full ablation train + plots in lab_outputs/: bash scripts/run_semantic_ablation.sh"
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen2.5-3b",
        help="Short model tag; defaults align paths under data/processed/{model_name}/.",
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=None,
        help="Input inference results PKL (defaults under output_dir based on --split).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for training matrix PKL/CSV/feat_list (default: data/processed/{model_name}/).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split tag appended to default filenames (e.g. train → router_training_matrix_train.pkl).",
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default=None,
        help="Override output basename without extension (default: router_training_matrix[_split]).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path (defaults alongside the PKL basename in output_dir)",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name for NLP features")
    parser.add_argument("--spacy-batch-size", type=int, default=64, help="spaCy nlp.pipe batch size")
    parser.add_argument(
        "--sentence-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model for question embeddings",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=16,
        help="Number of PCA dimensions for semantic features (feat_pc_01 …)",
    )
    parser.add_argument(
        "--pca-train-size",
        type=float,
        default=0.8,
        help="Fraction of rows used to fit PCA (when not using --pca-fit-on full)",
    )
    parser.add_argument(
        "--pca-fit-on",
        choices=("train", "full"),
        default="train",
        help="Fit PCA on stratified train split (train) or on all rows (full)",
    )
    parser.add_argument("--pca-random-state", type=int, default=42, help="Random seed for PCA / split")
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=32,
        help="Batch size for SentenceTransformer.encode",
    )
    parser.add_argument(
        "--semantic-pca",
        action="store_true",
        help=(
            "Add 16 semantic PCA columns (feat_pc_01 …). "
            "Without this flag, the matrix has 22 feat_* columns (default production)."
        ),
    )
    args = parser.parse_args()

    model_dir = _sanitize_model_dirname(args.model_name)
    out_dir = Path(args.output_dir) if args.output_dir is not None else Path("data/processed") / model_dir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Could not create output directory {out_dir}: {exc}") from exc

    in_path = args.input_path
    if in_path is None:
        if args.split:
            in_path = out_dir / f"inference_results_{args.split.strip().lower()}.pkl"
        else:
            in_path = out_dir / "inference_results.pkl"

    if args.output_stem:
        stem = args.output_stem.strip()
        if not stem:
            raise ValueError("--output_stem must be non-empty when provided.")
    elif args.split:
        stem = f"router_training_matrix_{args.split.strip().lower()}"
    else:
        stem = "router_training_matrix"

    out_pkl = out_dir / f"{stem}.pkl"
    csv_path = args.csv if args.csv is not None else out_dir / f"{stem}.csv"
    print(f"Loading inference PKL: {in_path.resolve()} | Writing matrix stem={stem!r} -> {out_dir.resolve()}")

    # -------------------------------------------------------------------------
    # Load inference results
    # -------------------------------------------------------------------------
    try:
        with open(in_path, "rb") as f:
            records = pickle.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Inference PKL not found: {in_path.resolve()}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read inference PKL {in_path}: {exc}") from exc
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

    # -------------------------------------------------------------------------
    # Task 4 (optional ablation): Semantic embeddings + PCA → feat_pc_01 … feat_pc_16
    # -------------------------------------------------------------------------
    parts = [df, basic_df, regex_df, spacy_df, model_aware, top5_df]
    if args.semantic_pca:
        pca_df = compute_semantic_pca_features(
            texts,
            df["target_label"],
            model_name=args.sentence_model,
            n_components=args.pca_components,
            pca_train_size=args.pca_train_size,
            random_state=args.pca_random_state,
            encode_batch_size=args.encode_batch_size,
            fit_on_full=(args.pca_fit_on == "full"),
        )
        parts.append(pca_df)

    # Default: 22 feat_*; with --semantic-pca: 22 + 16 = 38 feat_*
    out_df = pd.concat(parts, axis=1)

    # -------------------------------------------------------------------------
    # Persist outputs
    # -------------------------------------------------------------------------
    try:
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_pkl, "wb") as f:
            pickle.dump(out_df, f)
    except OSError as exc:
        raise RuntimeError(f"Failed to write training matrix PKL {out_pkl}: {exc}") from exc

    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(csv_path, index=False, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write training matrix CSV {csv_path}: {exc}") from exc

    feat_names = sorted(c for c in out_df.columns if c.startswith("feat_"))
    n_feat = len(feat_names)
    expected_n = EXPECTED_FEAT_BASE + (EXPECTED_FEAT_SEMANTIC_PCA if args.semantic_pca else 0)
    if n_feat != expected_n:
        raise RuntimeError(
            f"feat_* count mismatch: got {n_feat}, expected {expected_n} "
            f"(semantic_pca={args.semantic_pca}). Check feature engineering blocks."
        )

    feat_list_path = out_pkl.with_suffix(".feat_list.txt")
    try:
        feat_list_path.write_text("\n".join(feat_names) + "\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write feature list {feat_list_path}: {exc}") from exc
    print(f"Wrote training matrix PKL: {out_pkl} (rows={len(out_df)}, feat_* columns={n_feat})")
    print(f"Wrote training matrix CSV: {csv_path} (rows={len(out_df)})")
    print(f"Wrote feat_* list: {feat_list_path}")


if __name__ == "__main__":
    main()

