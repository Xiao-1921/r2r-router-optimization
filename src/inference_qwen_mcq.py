"""
Qwen2.5-3B-Instruct MCQ inference on Apple Silicon (MPS) with token logprobs,
entropy, and perplexity for router / uncertainty analysis.

Uses the tokenizer's built-in Qwen2.5 chat template (apply_chat_template).
"""

from __future__ import annotations

import argparse
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


def load_json_dataset(path: Path) -> list[dict[str, Any]]:
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_mcq_user_message(item: dict[str, Any]) -> str:
    """Build MCQ text from dataset fields."""
    # Turn JSON MCQ fields into a single user message for the chat template
    q = item["question"].strip()
    labels = item["choice_labels"]
    texts = item["choice_texts"]
    lines = [f"{lab}. {txt}" for lab, txt in zip(labels, texts, strict=True)]
    choices_block = "\n".join(lines)
    return (
        f"{q}\n\n"
        f"Choices:\n{choices_block}\n\n"
        "Reply with only the letter of the correct answer (A, B, C, or D)."
    )


def extract_choice_letter(text: str) -> str | None:
    """Parse the first A/B/C/D from model output."""
    if not text:
        return None
    t = text.strip().upper()
    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)
    # Leading letter
    for ch in t:
        if ch in "ABCD":
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


def analyze_generation_scores(
    scores: tuple[torch.Tensor, ...],
    generated_token_ids: torch.Tensor,
) -> dict[str, float]:
    """
    scores[i]: logits predicting generated_token_ids[i] (batch=1).
    Returns first-token entropy, mean entropy, mean NLL, perplexity, first-token prob.
    """
    if len(scores) == 0:
        return {
            "first_token_entropy": float("nan"),
            "avg_entropy": float("nan"),
            "perplexity": float("nan"),
            "chosen_token_prob": float("nan"),
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

    return {
        "first_token_entropy": float(first_ent.item()),
        "avg_entropy": float(avg_ent.item()),
        "perplexity": float(perplexity.item()),
        "chosen_token_prob": float(p0.item()),
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
        "--data",
        type=Path,
        default=Path("data/combined_dataset.json"),
        help="Path to combined_dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/qwen_mcq_results.pkl"),
        help="Output .pkl path (list of dicts)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path (default: same basename as --output with .csv extension)",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model id",
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

    # Resolve CSV path: default mirrors the pickle filename with a .csv suffix
    csv_path = args.csv if args.csv is not None else args.output.with_suffix(".csv")

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch_dtype = torch.bfloat16 if device.type == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Dataset (optional truncation via --limit)
    # -------------------------------------------------------------------------
    data = load_json_dataset(args.data)
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
        letter = extract_choice_letter(raw_text)
        gold = item["gold_label"]

        scores = out.scores
        if scores is None or len(scores) == 0:
            metrics = {
                "first_token_entropy": float("nan"),
                "avg_entropy": float("nan"),
                "perplexity": float("nan"),
                "chosen_token_prob": float("nan"),
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
            }
        )

    # -------------------------------------------------------------------------
    # Persist results: pickle (full Python objects) + CSV (tabular export)
    # -------------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Wrote {len(results)} records to {args.output}")

    df = pd.DataFrame(results)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    main()
