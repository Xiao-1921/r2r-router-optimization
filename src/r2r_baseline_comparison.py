"""
R2R Baseline Comparison: No RAG vs Always RAG vs Router-RAG
==============================================================
Compares three settings on the 300 test questions:
  1) No RAG (vanilla direct)
  2) Always RAG (raw unfiltered Wikipedia on every question)
  3) Router-RAG (Router decides, similarity filter applied)

Matches the prompt format from the Reflector notebook (system + user
message, 1-token generation, 4-bit quantization, logit-based answer).

Usage:
    python src/r2r_baseline_comparison.py \
        --model_name qwen2.5-3B \
        --hf_model_id Qwen/Qwen2.5-3B-Instruct \
        --router_path models/qwen2.5-3B/router_model.joblib \
        --feature_matrix data/processed/qwen2.5-3B/router_training_matrix_test.csv \
        --test_csv data/raw/Test.csv
"""

import argparse
import json, re, ast, time
import requests, numpy as np, torch, pandas as pd, joblib
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from rag_pipeline import RAGRetriever, SimilarityFilter

SIMILARITY_THRESHOLD = 0.3
TOP_K_PASSAGES = 3
MAX_CONTEXT_CHARS = 1500
CONTEXT_SOURCES = {"ReClor"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model label, e.g. qwen2.5-3B")
    parser.add_argument("--hf_model_id", type=str, required=True,
                        help="HuggingFace model ID, e.g. Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--router_path", type=str, required=True,
                        help="Path to router_model.joblib")
    parser.add_argument("--feature_matrix", type=str, required=True,
                        help="Path to router_training_matrix_test.csv")
    parser.add_argument("--test_csv", type=str, default="data/raw/Test.csv",
                        help="Path to Test.csv")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/<model_name>/)")
    return parser.parse_args()


# ============================================================
# Prompt builders (matching Reflector notebook)
# ============================================================
def normalize_label(text):
    text = str(text).strip().upper()
    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1)
    if text and text[0].isalpha():
        return text[0]
    return text


def get_passage_for_row(source, raw_context):
    if source not in CONTEXT_SOURCES:
        return None
    if raw_context is None:
        return None
    text = str(raw_context).strip()
    if text in {"", "nan", "None"}:
        return None
    return text


def build_system_prompt(valid_labels, with_rag=False):
    label_list = ", ".join(valid_labels)
    rag_instr = ("Use the provided reference information if relevant, "
                 "otherwise rely on your own knowledge. ") if with_rag else ""
    return (
        "You are a multiple-choice question answering system. "
        f"{rag_instr}"
        "Read the question and options carefully. "
        f"Return ONLY the final answer label, one of {label_list}. "
        "Do not explain your reasoning. Do not write anything else."
    )


def build_user_prompt(question, options, passage=None, rag_context=None):
    option_lines = [f"{label}. {text}" for label, text in options.items()]
    rag_block = f"Reference Information:\n{rag_context}\n\n" if rag_context else ""
    passage_block = f"Passage:\n{passage}\n\n" if passage else ""
    return (
        f"{rag_block}{passage_block}"
        f"Question:\n{question}\n\n"
        "Options:\n" + "\n".join(option_lines) +
        "\n\nAnswer with only the label."
    )


# ============================================================
# LLM inference (1-token, logit-based)
# ============================================================
def get_label_token_ids(label, tok):
    token_ids = []
    for candidate in [label, f" {label}"]:
        ids = tok(candidate, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            token_ids.append(ids[0])
    return list(dict.fromkeys(token_ids))


def infer_label_only(messages, options, mdl, tok):
    prompt_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt_text, return_tensors="pt").to(mdl.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        gen_out = mdl.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.eos_token_id,
        )

    step_scores = gen_out.scores[0][0]
    option_labels = list(options.keys())
    log_probs = torch.log_softmax(step_scores, dim=-1)

    label_scores = {}
    for label in option_labels:
        candidate_ids = get_label_token_ids(label, tok)
        if candidate_ids:
            candidate_lps = torch.tensor(
                [log_probs[tid].item() for tid in candidate_ids],
                dtype=torch.float32
            )
            label_scores[label] = torch.logsumexp(candidate_lps, dim=0).item()
        else:
            label_scores[label] = float("-inf")

    score_tensor = torch.tensor(
        [label_scores[l] for l in option_labels], dtype=torch.float32
    )
    label_probs = torch.softmax(score_tensor, dim=0)
    return option_labels[label_probs.argmax().item()]


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/{args.model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load LLM
    print(f"[*] Loading {args.hf_model_id} (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        args.hf_model_id, device_map="auto",
        quantization_config=quant_config, trust_remote_code=True,
    )
    llm.eval()
    print("    LLM loaded!")

    # Load embedding model + RAG
    print("[*] Loading embedding model...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    retriever = RAGRetriever(top_k=TOP_K_PASSAGES)
    sim_filter = SimilarityFilter(embed_model, threshold=SIMILARITY_THRESHOLD)

    # Load Router
    print("[*] Loading Router model...")
    router = joblib.load(args.router_path)

    # Load test data
    print("[*] Loading test data...")
    test_matrix = pd.read_csv(args.feature_matrix)
    test_csv = pd.read_csv(args.test_csv)

    feat_cols = router["imputer"].feature_names_in_.tolist()
    router_preds = router.predict(test_matrix[feat_cols])
    print(f"    {len(test_matrix)} questions | Router flags {sum(router_preds)} for RAG")

    # Run all 3 conditions
    results = []
    start_time = time.time()

    for i in range(len(test_csv)):
        row_csv = test_csv.iloc[i]

        source = str(row_csv.get("source", ""))
        question = str(row_csv["question"])
        gold_label = normalize_label(str(row_csv["gold_label"]))
        r_pred = int(router_preds[i])

        try:
            labels = ast.literal_eval(row_csv["choice_labels"])
            texts = ast.literal_eval(row_csv["choice_texts"])
            options = dict(zip(labels, texts))
        except:
            continue

        valid_labels = list(options.keys())
        passage = get_passage_for_row(source, row_csv.get("context"))

        # Condition 1: No RAG
        messages_vanilla = [
            {"role": "system", "content": build_system_prompt(valid_labels)},
            {"role": "user", "content": build_user_prompt(question, options, passage)},
        ]
        vanilla_pred = infer_label_only(messages_vanilla, options, llm, tokenizer)
        vanilla_correct = (vanilla_pred == gold_label)

        # Retrieve (raw + filtered)
        full_q = question if not passage else f"{passage}\n\n{question}"
        raw_passages = retriever.retrieve(full_q)
        if raw_passages:
            parts = [f"[{title}]\n{text}" for title, text in raw_passages]
            rag_context_raw = "\n\n".join(parts)[:MAX_CONTEXT_CHARS]
        else:
            rag_context_raw = ""
        filtered_passages, _ = sim_filter.filter_passages(full_q, raw_passages)
        if filtered_passages:
            parts = [f"[{title}]\n{text}" for title, text in filtered_passages]
            rag_context_filtered = "\n\n".join(parts)[:MAX_CONTEXT_CHARS]
        else:
            rag_context_filtered = ""

        # Condition 2: Always RAG (raw, no filter)
        messages_rag = [
            {"role": "system", "content": build_system_prompt(valid_labels, with_rag=True)},
            {"role": "user", "content": build_user_prompt(question, options, passage, rag_context_raw)},
        ]
        rag_pred = infer_label_only(messages_rag, options, llm, tokenizer)
        rag_correct = (rag_pred == gold_label)

        # Condition 3: Router-RAG (filtered)
        if r_pred == 1 and rag_context_filtered:
            messages_r2r = [
                {"role": "system", "content": build_system_prompt(valid_labels, with_rag=True)},
                {"role": "user", "content": build_user_prompt(question, options, passage, rag_context_filtered)},
            ]
            router_rag_pred = infer_label_only(messages_r2r, options, llm, tokenizer)
            router_rag_correct = (router_rag_pred == gold_label)
        else:
            router_rag_pred = vanilla_pred
            router_rag_correct = vanilla_correct

        results.append({
            "idx": i,
            "source": source,
            "question": question[:120],
            "gold_label": gold_label,
            "vanilla_pred": vanilla_pred,
            "vanilla_correct": vanilla_correct,
            "rag_pred": rag_pred,
            "rag_correct": rag_correct,
            "router_pred": r_pred,
            "router_rag_pred": router_rag_pred,
            "router_rag_correct": router_rag_correct,
            "rag_context_raw_used": bool(rag_context_raw),
            "rag_context_filtered_used": bool(rag_context_filtered),
        })

        v = "\u2713" if vanilla_correct else "\u2717"
        r = "\u2713" if rag_correct else "\u2717"
        rr = "\u2713" if router_rag_correct else "\u2717"
        tag = "RAG" if r_pred == 1 else "skip"
        print(f"[{i+1}/{len(test_csv)}] {source[:8]:8s} V:{v} RAG:{r} R2R:{rr} [{tag}] | {question[:45]}...")

        if (i + 1) % 50 == 0:
            print(f"\n--- Checkpoint {i+1}/{len(test_csv)} ({(time.time()-start_time)/60:.1f}min) ---\n")

    # Report
    df = pd.DataFrame(results)
    total = len(df)
    elapsed = time.time() - start_time

    v_acc = df["vanilla_correct"].sum()
    r_acc = df["rag_correct"].sum()
    rr_acc = df["router_rag_correct"].sum()
    rag_calls = df["router_pred"].sum()

    print(f"\n{'='*70}")
    print(f"R2R BASELINE COMPARISON (No RAG vs Always RAG vs Router-RAG)")
    print(f"{'='*70}")
    print(f"Questions: {total} | Runtime: {elapsed/60:.1f} min | Model: {args.hf_model_id}\n")
    print(f"  No RAG (vanilla)  : {v_acc}/{total} ({100*v_acc/total:.1f}%)")
    print(f"  Always RAG        : {r_acc}/{total} ({100*r_acc/total:.1f}%)")
    print(f"  Router-RAG (R2R)  : {rr_acc}/{total} ({100*rr_acc/total:.1f}%)")
    print(f"  RAG calls saved   : {total - rag_calls}/{total} ({100*(total-rag_calls)/total:.1f}%)")
    print()

    print(f"--- PER DATASET ---")
    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        n = len(sub)
        v = sub["vanilla_correct"].sum()
        r = sub["rag_correct"].sum()
        rr = sub["router_rag_correct"].sum()
        rc = sub["router_pred"].sum()
        print(f"  {src:<12} {v:>3}/{n} ({100*v/n:4.0f}%)  {r:>3}/{n} ({100*r/n:4.0f}%)  {rr:>3}/{n} ({100*rr/n:4.0f}%)  {rc:>4}/{n}")

    csv_path = output_dir / "r2r_baseline_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
