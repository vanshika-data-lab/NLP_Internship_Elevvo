"""
Task 6: Question Answering with Transformers
Dataset: SQuAD v1.1 (Kaggle)
https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset
File needed: train-v1.1.json
Topics: Question answering, Span extraction, Transformer-based NLP
Models: DistilBERT, BERT, RoBERTa
"""

import json
import re
import string
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# FIX: Import AutoTokenizer and AutoModelForQuestionAnswering for fallback
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

# ══════════════════════════════════════════════
# ▶▶  SET YOUR DATASET PATH HERE  ◀◀
# ══════════════════════════════════════════════
DATASET_PATH   = r"C:\Users\Vanshika\Downloads\SQuADv1.1_load\train-v1.1.json"
COMPARE_MODELS = True   # set False to run only DistilBERT (faster)
# ══════════════════════════════════════════════


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_squad(filepath: str) -> list:
    print(f"[INFO] Loading SQuAD from: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        squad = json.load(f)

    samples = []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                if not qa.get("is_impossible", False) and qa["answers"]:
                    samples.append({
                        "context"     : ctx,
                        "question"    : qa["question"],
                        "answers"     : [a["text"] for a in qa["answers"]],
                        "answer_start": qa["answers"][0]["answer_start"],
                    })

    print(f"[INFO] Loaded {len(samples)} QA pairs from SQuAD")
    return samples


# ─────────────────────────────────────────────
# 2. EVALUATION METRICS
# ─────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def exact_match(pred: str, gold_list: list) -> float:
    pred_norm = normalize_answer(pred)
    return float(any(normalize_answer(g) == pred_norm for g in gold_list))


def token_f1(pred: str, gold_list: list) -> float:
    pred_tokens = normalize_answer(pred).split()
    best_f1 = 0.0
    for gold in gold_list:
        gold_tokens = normalize_answer(gold).split()
        common      = Counter(pred_tokens) & Counter(gold_tokens)
        num_same    = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall    = num_same / len(gold_tokens)
        f1        = 2 * precision * recall / (precision + recall)
        best_f1   = max(best_f1, f1)
    return best_f1


# ─────────────────────────────────────────────
# 3. MODEL INFERENCE  (FIX: pipeline fallback)
# ─────────────────────────────────────────────

def load_qa_pipeline(model_name: str):
    """
    FIX: Try standard pipeline first.
    If it raises KeyError (transformers version mismatch),
    fall back to loading AutoModel directly — works on any version.
    """
    print(f"[INFO] Loading model: {model_name} ...")
    try:
        qa = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )
        print(f"[INFO] Loaded via pipeline successfully")
        return qa, "pipeline"

    except KeyError:
        print(f"[WARN] pipeline() raised KeyError — loading model directly as fallback")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForQuestionAnswering.from_pretrained(model_name)
        model.eval()
        print(f"[INFO] Loaded via AutoModel successfully")
        return (tokenizer, model), "manual"


def _manual_inference(tokenizer, model, question: str, context: str) -> dict:
    """Direct inference without pipeline — fallback method."""
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx  = torch.argmax(outputs.start_logits).item()
    end_idx    = torch.argmax(outputs.end_logits).item() + 1
    start_conf = torch.softmax(outputs.start_logits, dim=1)[0][start_idx].item()
    end_conf   = torch.softmax(outputs.end_logits,   dim=1)[0][end_idx - 1].item()
    score      = (start_conf + end_conf) / 2

    answer = tokenizer.decode(
        inputs["input_ids"][0][start_idx:end_idx],
        skip_special_tokens=True
    )
    answer_start = context.find(answer) if answer in context else 0

    return {
        "answer": answer,
        "score" : score,
        "start" : answer_start,
        "end"   : answer_start + len(answer),
    }


def run_inference(qa_obj: tuple, samples: list) -> list:
    """Works with both pipeline and manual (tokenizer, model) modes."""
    pipe_or_tuple, mode = qa_obj
    results = []
    for s in samples:
        if mode == "pipeline":
            out = pipe_or_tuple(question=s["question"], context=s["context"])
        else:
            tokenizer, model = pipe_or_tuple
            out = _manual_inference(tokenizer, model, s["question"], s["context"])

        results.append({
            "question"  : s["question"],
            "context"   : s["context"][:120] + "...",
            "gold"      : s["answers"],
            "prediction": out["answer"],
            "score"     : out["score"],
            "em"        : exact_match(out["answer"], s["answers"]),
            "f1"        : token_f1(out["answer"],   s["answers"]),
        })
    return results


def evaluate(results: list) -> dict:
    return {
        "exact_match": np.mean([r["em"] for r in results]) * 100,
        "f1"         : np.mean([r["f1"] for r in results]) * 100,
        "n_samples"  : len(results),
    }


# ─────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_confidence_distribution(results: list, model_name: str, filename: str):
    scores = [r["score"] for r in results]
    plt.figure(figsize=(7, 4))
    plt.hist(scores, bins=20, color="#3498db", edgecolor="white", alpha=0.85)
    plt.axvline(np.mean(scores), color="#e74c3c", lw=2, linestyle="--",
                label=f"Mean = {np.mean(scores):.3f}")
    plt.xlabel("Confidence Score"); plt.ylabel("Count")
    plt.title(f"Answer Confidence Distribution - {model_name}", fontweight="bold")
    plt.legend(); plt.tight_layout()
    plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def plot_em_f1_per_sample(results: list, model_name: str, filename: str):
    df  = pd.DataFrame(results)
    x   = range(len(df))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, df["f1"], label="F1",          color="#3498db", alpha=0.7)
    ax.bar(x, df["em"], label="Exact Match", color="#2ecc71", alpha=0.7)
    ax.set_xlabel("Sample"); ax.set_ylabel("Score")
    ax.set_title(f"Per-Sample EM & F1 - {model_name}", fontweight="bold")
    ax.legend(); plt.tight_layout()
    plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def plot_model_comparison(metrics_dict: dict):
    models  = list(metrics_dict.keys())
    em_vals = [metrics_dict[m]["exact_match"] for m in models]
    f1_vals = [metrics_dict[m]["f1"]          for m in models]
    x       = np.arange(len(models))
    width   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, em_vals, width, label="Exact Match", color="#3498db")
    bars2 = ax.bar(x + width/2, f1_vals, width, label="F1 Score",    color="#e74c3c")
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 110)
    ax.set_title("Model Comparison - EM & F1", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150); plt.close()
    print("[SAVED] model_comparison.png")


def plot_answer_length(results: list):
    pred_lens = [len(r["prediction"].split()) for r in results]
    gold_lens = [len(r["gold"][0].split())    for r in results]
    fig, ax   = plt.subplots(figsize=(7, 4))
    ax.scatter(gold_lens, pred_lens, alpha=0.7, color="#9b59b6")
    lim = max(max(gold_lens), max(pred_lens)) + 2
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="Perfect match")
    ax.set_xlabel("Gold Answer Length (tokens)")
    ax.set_ylabel("Predicted Answer Length (tokens)")
    ax.set_title("Gold vs Predicted Answer Length", fontweight="bold")
    ax.legend(); plt.tight_layout()
    plt.savefig("answer_length.png", dpi=150); plt.close()
    print("[SAVED] answer_length.png")


# ─────────────────────────────────────────────
# 5. RESULTS TABLE
# ─────────────────────────────────────────────

def print_predictions_table(results: list):
    print(f"\n{'─'*90}")
    print(f"{'Question':<40} {'Gold':<20} {'Prediction':<20} {'EM':>4} {'F1':>6}")
    print(f"{'─'*90}")
    for r in results:
        q = r["question"][:38] + ".." if len(r["question"]) > 40 else r["question"]
        g = r["gold"][0][:18]  + ".." if len(r["gold"][0])  > 20 else r["gold"][0]
        p = r["prediction"][:18] + ".." if len(r["prediction"]) > 20 else r["prediction"]
        print(f"{q:<40} {g:<20} {p:<20} {r['em']:>4.0f} {r['f1']:>6.3f}")
    print(f"{'─'*90}")


# ─────────────────────────────────────────────
# 6. MODELS CONFIG
# ─────────────────────────────────────────────

MODELS = {
    "DistilBERT": "distilbert-base-cased-distilled-squad",
    "BERT-large": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "RoBERTa"   : "deepset/roberta-base-squad2",
}


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main(squad_path: str = DATASET_PATH, compare_models: bool = COMPARE_MODELS):
    samples      = load_squad(squad_path)
    eval_samples = samples[:100]
    print(f"[INFO] Evaluating on {len(eval_samples)} samples")

    all_metrics  = {}
    last_results = []

    for model_label, model_name in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Model : {model_label}")
        print(f"  Name  : {model_name}")
        print(f"{'='*60}")
        try:
            qa_obj      = load_qa_pipeline(model_name)
            results     = run_inference(qa_obj, eval_samples)
            metrics     = evaluate(results)
            all_metrics[model_label] = metrics
            last_results = results

            print(f"  Exact Match : {metrics['exact_match']:.2f}%")
            print(f"  F1 Score    : {metrics['f1']:.2f}%")
            print_predictions_table(results[:10])

            safe = model_label.replace(" ", "_").replace("-", "_")
            plot_confidence_distribution(results, model_label,
                                          f"confidence_{safe}.png")
            plot_em_f1_per_sample(results, model_label,
                                   f"em_f1_{safe}.png")
            pd.DataFrame(results).to_csv(f"predictions_{safe}.csv", index=False)
            print(f"[SAVED] predictions_{safe}.csv")

        except Exception as e:
            print(f"[WARN] Could not run {model_label}: {e}")
            all_metrics[model_label] = {"exact_match": 0, "f1": 0, "n_samples": 0}

        if not compare_models:
            break

    if last_results:
        plot_answer_length(last_results)

    if len(all_metrics) > 1:
        plot_model_comparison(all_metrics)

    summary = pd.DataFrame(all_metrics).T
    print("\n── Final Summary ──")
    print(summary.to_string())
    summary.to_csv("model_summary.csv")
    print("[SAVED] model_summary.csv")

    print("\n✅ Task 6 complete!")


if __name__ == "__main__":
    main()

