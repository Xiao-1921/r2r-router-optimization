"""
Qwen2.5-3B-Instruct MCQ inference on Apple Silicon (MPS) with token logprobs,
entropy, perplexity, and the top-5 first-token log-probabilities (for margins / kurtosis).

Uses the tokenizer's built-in Qwen2.5 chat template (apply_chat_template).
Loads MCQ items from CSV (``data/raw`` by convention) via :func:`pandas.read_csv`.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mps_gpu_check import verify_mps_gpu_ready

# -----------------------------------------------------------------------------
# Data loading & prompt construction
# -----------------------------------------------------------------------------


def _sanitize_model_dirname(model_name: str) -> str:
    cleaned = model_name.strip().replace("/", "_").replace("\\", "_")
    if not cleaned:
        raise ValueError("model_name must be a non-empty string after sanitization.")
    return cleaned


def _parse_list_cell(value: Any, *, column: str) -> list[Any]:
    """Parse a CSV cell that stores a Python list (e.g. ``\"['A', 'B']\"``) via ``ast.literal_eval``."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError(f"Missing or NaN value in column {column!r}.")
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError, MemoryError) as exc:
            raise ValueError(
                f"Could not parse column {column!r} as a Python literal list: {exc}"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError(
                f"Column {column!r} must evaluate to a list, got {type(parsed).__name__}."
            )
        return parsed
    raise TypeError(f"Unsupported type for column {column!r}: {type(value).__name__}.")


def _coerce_gold_label(value: Any, choice_labels: list[Any]) -> str:
    """Map CSV ``gold_label`` onto the uppercase token set implied by ``choice_labels``."""
    allowed_upper = {str(lab).strip().upper() for lab in choice_labels}
    raw = str(value).strip()
    if raw.upper() in allowed_upper:
        return raw.upper()
    m = re.search(r"[A-Za-z]", raw)
    if m:
        ch = m.group(0).upper()
        if ch in allowed_upper:
            return ch
    return ""


def load_mcq_csv(
    path: Path,
    *,
    split: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Load MCQ rows from CSV into dict records for :func:`format_mcq_user_message`.

    If the CSV has a ``split`` column with multiple distinct values, ``split`` must be
    provided (e.g. ``train``, ``validation``, ``test``) so one split is processed per run.

    Returns:
        (items, split_tag) where ``split_tag`` is used for default output filenames
        (``None`` when no split column is present).
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset CSV not found: {path.resolve()}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Failed to read CSV file {path}: {exc}") from exc

    required = {"question", "choice_labels", "choice_texts", "gold_label"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"CSV missing required columns {sorted(missing)}. Found columns: {list(df.columns)}."
        )

    split_tag: str | None = None
    if "split" in df.columns:
        norm = df["split"].astype(str).str.strip().str.lower()
        unique_splits = sorted(norm.unique().tolist())
        if split is not None:
            want = split.strip().lower()
            mask = norm == want
            df = df.loc[mask].reset_index(drop=True)
            if len(df) == 0:
                raise RuntimeError(
                    f"No rows with split={split!r} (normalized: {want!r}). "
                    f"Available splits in file: {unique_splits}."
                )
            split_tag = want
        elif len(unique_splits) > 1:
            raise RuntimeError(
                "CSV has a 'split' column with multiple values; pass --split "
                "(e.g. train, validation, test) to process one split per run. "
                f"Found splits: {unique_splits}"
            )
        else:
            split_tag = unique_splits[0] if unique_splits else None
    elif split is not None:
        raise RuntimeError("CSV has no 'split' column but --split was provided.")

    items: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        try:
            choice_texts = _parse_list_cell(row["choice_texts"], column="choice_texts")
            choice_labels = _parse_list_cell(row["choice_labels"], column="choice_labels")
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"Invalid list fields at CSV row index {idx}.") from exc
        if len(choice_labels) != len(choice_texts):
            raise RuntimeError(
                f"Row {idx}: choice_labels length ({len(choice_labels)}) != "
                f"choice_texts length ({len(choice_texts)})."
            )
        gold = _coerce_gold_label(row["gold_label"], choice_labels)
        allowed = frozenset(str(lab).strip().upper() for lab in choice_labels)
        if not gold or gold not in allowed:
            raise RuntimeError(
                f"Row {idx}: gold_label {row['gold_label']!r} is not compatible with choice_labels {choice_labels}."
            )

        items.append(
            {
                "question": str(row["question"]).strip(),
                "choice_labels": choice_labels,
                "choice_texts": choice_texts,
                "gold_label": gold,
            }
        )

    return items, split_tag


def format_mcq_user_message(item: dict[str, Any]) -> str:
    """Build MCQ text from dataset fields."""
    # Turn JSON MCQ fields into a single user message for the chat template
    q = item["question"].strip()
    labels = item["choice_labels"]
    texts = item["choice_texts"]
    lines = [f"{lab}. {txt}" for lab, txt in zip(labels, texts, strict=True)]
    choices_block = "\n".join(lines)
    label_syms = [str(lab).strip().upper() for lab in labels]
    label_hint = ", ".join(label_syms)
    return (
        f"{q}\n\n"
        f"Choices:\n{choices_block}\n\n"
        f"Reply with only the letter of the correct answer ({label_hint})."
    )


def extract_choice_letter(text: str, *, allowed: frozenset[str] | None = None) -> str | None:
    """Parse the first multiple-choice letter from model output (default: A–Z)."""
    if not text:
        return None
    t = text.strip().upper()
    if allowed:
        allowed_union = "|".join(re.escape(a) for a in sorted(allowed))
        m = re.search(rf"\b({allowed_union})\b", t)
        if m:
            return m.group(1)
        for ch in t:
            if ch in allowed:
                return ch
        return None
    m = re.search(r"\b([A-Z])\b", t)
    if m:
        return m.group(1)
    for ch in t:
        if "A" <= ch <= "Z":
            return ch
    return None


# -----------------------------------------------------------------------------
# Uncertainty metrics from generation logits (entropy, NLL, perplexity)
# -----------------------------------------------------------------------------


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of softmax(logits), shape (...)."""
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def first_token_topk_logprobs(logits_batch1: torch.Tensor, k: int = 5) -> list[float]:
    """
    Return the k largest log-probabilities (nats) for the first generated token distribution.
    Sorted descending (highest logprob first). Pads with nan if vocabulary size < k.
    """
    logits_f = logits_batch1[0].float()
    log_p = F.log_softmax(logits_f, dim=-1)
    vocab = log_p.numel()
    kk = min(k, vocab)
    top = torch.topk(log_p, k=kk, largest=True, sorted=True)
    values = top.values.detach().cpu().tolist()
    if len(values) < k:
        values = values + [float("nan")] * (k - len(values))
    return [float(v) for v in values[:k]]


def analyze_generation_scores(
    scores: tuple[torch.Tensor, ...],
    generated_token_ids: torch.Tensor,
    *,
    top_k_first: int = 5,
) -> dict[str, Any]:
    """
    scores[i]: logits predicting generated_token_ids[i] (batch=1).

    Returns first-token entropy, mean entropy, mean NLL, perplexity, chosen-token probability,
    and ``first_token_top5_logprobs``: the top-``top_k_first`` log-probabilities at the first
    generated step (for Top-k margin, kurtosis, etc.).
    """
    nan_list = [float("nan")] * top_k_first

    if len(scores) == 0:
        return {
            "first_token_entropy": float("nan"),
            "avg_entropy": float("nan"),
            "perplexity": float("nan"),
            "chosen_token_prob": float("nan"),
            "first_token_top5_logprobs": nan_list,
        }

    entropies: list[torch.Tensor] = []
    nlls: list[torch.Tensor] = []

    for i, logits in enumerate(scores):
        logits_f = logits[0].float()
        tid = int(generated_token_ids[i].item())
        log_p = F.log_softmax(logits_f, dim=-1)
        nlls.append(-log_p[tid])
        entropies.append(entropy_from_logits(logits_f))

    ent_stack = torch.stack([e if e.dim() == 0 else e.squeeze() for e in entropies])
    first_ent = entropies[0]
    if first_ent.dim() > 0:
        first_ent = first_ent.squeeze()
    avg_ent = ent_stack.mean()
    mean_nll = torch.stack(nlls).mean()
    perplexity = torch.exp(mean_nll)

    tid0 = int(generated_token_ids[0].item())
    log_p0 = F.log_softmax(scores[0][0].float(), dim=-1)
    p0 = log_p0[tid0].exp()

    top5_lp = first_token_topk_logprobs(scores[0], k=top_k_first)

    return {
        "first_token_entropy": float(first_ent.item()),
        "avg_entropy": float(avg_ent.item()),
        "perplexity": float(perplexity.item()),
        "chosen_token_prob": float(p0.item()),
        "first_token_top5_logprobs": top5_lp,
    }


# -----------------------------------------------------------------------------
# Qwen chat template (official tokenizer template)
# -----------------------------------------------------------------------------


def build_prompt(tokenizer: AutoTokenizer, user_content: str) -> str:
    """Official Qwen2.5 Instruct chat template via tokenizer."""
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# -----------------------------------------------------------------------------
# CLI entry point: load model, run inference, persist results (PKL + CSV)
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5 MCQ inference with MPS + uncertainty metrics")
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen2.5-3b",
        help="Short model tag for processed output layout (directory name under data/processed/).",
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("data/raw/Train.csv"),
        help="Input MCQ CSV path (expects columns: question, choice_labels, choice_texts, gold_label; "
        "optional split).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for inference PKL/CSV (default: data/processed/{model_name}/).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="When the CSV has a split column with multiple values, select one: train, validation, test, etc.",
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default=None,
        help="Override output file basename (default: inference_results or inference_results_{split}).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path (defaults alongside the PKL in output_dir)",
    )
    parser.add_argument(
        "--hf-model-id",
        dest="hf_model_id",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model id for AutoModelForCausalLM / AutoTokenizer",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Only first N items (debug)")
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help="Inference device (default: mps for Apple Silicon)",
    )
    args = parser.parse_args()

    model_dir = _sanitize_model_dirname(args.model_name)
    out_dir = Path(args.output_dir) if args.output_dir is not None else Path("data/processed") / model_dir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Could not create output directory {out_dir}: {exc}") from exc

    data, split_tag = load_mcq_csv(args.input_path, split=args.split)
    if args.output_stem:
        stem = args.output_stem.strip()
        if not stem:
            raise ValueError("--output_stem must be a non-empty string when provided.")
    elif split_tag is not None:
        stem = f"inference_results_{split_tag}"
    else:
        # Per-split CSV files (e.g. Train.csv / Test.csv) without a split column get distinct names.
        stem = f"inference_results_{args.input_path.stem.lower()}"

    pkl_path = out_dir / f"{stem}.pkl"
    csv_path = args.csv if args.csv is not None else out_dir / f"{stem}.csv"
    print(
        f"Loading MCQ CSV: {args.input_path.resolve()} | "
        f"model_name={model_dir!r} | artifacts: {pkl_path.name}, {csv_path.name} -> {out_dir.resolve()}"
    )

    # -------------------------------------------------------------------------
    # Device selection (MPS on Apple Silicon, or CPU for debugging)
    # -------------------------------------------------------------------------
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is not available. Use --device cpu for non-Mac testing, "
                "or run on an Apple Silicon Mac with PyTorch MPS enabled."
            )
        device = torch.device("mps")
        # ---------------------------------------------------------------------
        # Confirm MPS (Metal GPU) is usable before loading the model
        # ---------------------------------------------------------------------
        verify_mps_gpu_ready()
    else:
        device = torch.device("cpu")

    # -------------------------------------------------------------------------
    # Model & tokenizer (bf16 on MPS, float32 on CPU)
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch_dtype = torch.bfloat16 if device.type == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_id,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Dataset (optional truncation via --limit)
    # -------------------------------------------------------------------------
    if args.limit is not None:
        data = data[: args.limit]

    results: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Per-item generation: greedy decode + score-based uncertainty metrics
    # -------------------------------------------------------------------------
    for item in tqdm(data, desc="MCQ inference"):
        user_msg = format_mcq_user_message(item)
        prompt_text = build_prompt(tokenizer, user_msg)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        gen_ids = out.sequences[0, prompt_len:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        allowed_pick = frozenset(str(lab).strip().upper() for lab in item["choice_labels"])
        letter = extract_choice_letter(raw_text, allowed=allowed_pick)
        gold = item["gold_label"]

        scores = out.scores
        if scores is None or len(scores) == 0:
            metrics = {
                "first_token_entropy": float("nan"),
                "avg_entropy": float("nan"),
                "perplexity": float("nan"),
                "chosen_token_prob": float("nan"),
                "first_token_top5_logprobs": [float("nan")] * 5,
            }
        else:
            n_scores = len(scores)
            n_gen = gen_ids.shape[0]
            # Align scores to generated tokens (trim if EOS ended early)
            use = min(n_scores, n_gen)
            metrics = analyze_generation_scores(scores[:use], gen_ids[:use])

        results.append(
            {
                "original_question": item["question"],
                "gold_label": gold,
                "model_answer": raw_text.strip(),
                "is_correct": letter == gold if letter is not None else False,
                "first_token_entropy": metrics["first_token_entropy"],
                "avg_entropy": metrics["avg_entropy"],
                "perplexity": metrics["perplexity"],
                "chosen_token_prob": metrics["chosen_token_prob"],
                "first_token_top5_logprobs": metrics["first_token_top5_logprobs"],
            }
        )

    # -------------------------------------------------------------------------
    # Persist results: pickle (full Python objects) + CSV (tabular export)
    # -------------------------------------------------------------------------
    try:
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
    except OSError as exc:
        raise RuntimeError(f"Failed to write pickle results to {pkl_path}: {exc}") from exc
    print(f"Wrote {len(results)} records to {pkl_path}")

    df = pd.DataFrame(results)
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_csv = df.copy()
        if "first_token_top5_logprobs" in df_csv.columns:
            df_csv["first_token_top5_logprobs"] = df_csv["first_token_top5_logprobs"].apply(json.dumps)
        df_csv.to_csv(csv_path, index=False, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write CSV results to {csv_path}: {exc}") from exc
    print(f"Wrote {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    main()
