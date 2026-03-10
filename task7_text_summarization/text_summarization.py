"""
Task 7: Text Summarization Using Pre-trained Models
Dataset: CNN-DailyMail News (Kaggle)
https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail
Topics: Abstractive summarization, NLP with deep learning, ROUGE evaluation
Models: BART, T5, Pegasus + extractive TextRank (bonus)
"""

import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
warnings.filterwarnings("ignore")

from transformers import pipeline
import torch
from rouge_score import rouge_scorer

# ══════════════════════════════════════════════
# ▶▶  SET YOUR DATASET PATH HERE  ◀◀
# ══════════════════════════════════════════════
DATASET_PATH = r"C:\Users\Vanshika\Downloads\cnn_dailymail\train.csv"
N_SAMPLES    = 20    # number of articles to process (increase for more results)
# ══════════════════════════════════════════════


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_cnn_dailymail(filepath: str, n_samples: int) -> pd.DataFrame:
    print(f"[INFO] Loading dataset from : {filepath}")
    df = pd.read_csv(filepath, nrows=n_samples)
    print(f"[INFO] Columns found        : {list(df.columns)}")
    print(f"[INFO] Rows loaded          : {len(df)}")

    # CNN-DailyMail standard columns: id, article, highlights
    if "article" not in df.columns:
        text_col = next((c for c in df.columns
                         if any(k in c.lower()
                                for k in ["article","text","content","body"])),
                        df.columns[1])
        high_col = next((c for c in df.columns
                         if any(k in c.lower()
                                for k in ["highlight","summary","target"])),
                        df.columns[2])
        df.rename(columns={text_col: "article", high_col: "highlights"}, inplace=True)

    df = df[["article", "highlights"]].dropna().reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 2. ABSTRACTIVE SUMMARIZATION MODELS
# ─────────────────────────────────────────────

SUMMARIZER_CONFIGS = {
    "BART"   : {
        "model"    : "facebook/bart-large-cnn",
        "max_input": 1024,
        "min_len"  : 40,
        "max_len"  : 150,
    },
    "T5"     : {
        "model"    : "t5-small",
        "max_input": 512,
        "min_len"  : 30,
        "max_len"  : 120,
        "prefix"   : "summarize: ",
    },
    "Pegasus": {
        "model"    : "google/pegasus-cnn_dailymail",
        "max_input": 1024,
        "min_len"  : 40,
        "max_len"  : 150,
    },
}


def load_summarizer(config: dict):
    print(f"[INFO] Loading model: {config['model']} ...")
    return pipeline(
        "summarization",
        model=config["model"],
        tokenizer=config["model"],
        device=0 if torch.cuda.is_available() else -1,
        framework="pt",
    )


def summarize_batch(summarizer, articles: list, config: dict) -> list:
    prefix  = config.get("prefix", "")
    inputs  = [(prefix + a)[:config["max_input"] * 5] for a in articles]
    outputs = summarizer(
        inputs,
        max_length=config["max_len"],
        min_length=config["min_len"],
        do_sample=False,
        truncation=True,
        batch_size=4,
    )
    return [o["summary_text"] for o in outputs]


# ─────────────────────────────────────────────
# 3. EXTRACTIVE SUMMARIZATION — TextRank (bonus)
# ─────────────────────────────────────────────

def textrank_summary(text: str, n_sentences: int = 3) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if len(s.split()) > 5]
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    tfidf = TfidfVectorizer(stop_words="english")
    try:
        mat = tfidf.fit_transform(sentences).toarray()
    except ValueError:
        return " ".join(sentences[:n_sentences])

    sim     = cosine_similarity(mat)
    np.fill_diagonal(sim, 0)
    scores  = sim.sum(axis=1)
    top_idx = sorted(np.argsort(scores)[-n_sentences:])
    return " ".join([sentences[i] for i in top_idx])


# ─────────────────────────────────────────────
# 4. ROUGE EVALUATION
# ─────────────────────────────────────────────

def compute_rouge(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "ROUGE-1": np.mean(r1) * 100,
        "ROUGE-2": np.mean(r2) * 100,
        "ROUGE-L": np.mean(rL) * 100,
    }


# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_rouge_comparison(rouge_results: dict):
    models  = list(rouge_results.keys())
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    x       = np.arange(len(models))
    width   = 0.25
    colors  = ["#3498db", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [rouge_results[m][metric] for m in models]
        bars = ax.bar(x + i * width, vals, width,
                      label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=10, ha="right")
    ax.set_ylabel("ROUGE Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("ROUGE Score Comparison Across Models", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("rouge_comparison.png", dpi=150); plt.close()
    print("[SAVED] rouge_comparison.png")


def plot_summary_lengths(articles: list, summaries_dict: dict, references: list):
    fig, ax = plt.subplots(figsize=(9, 5))
    data = {
        "Original" : [len(a.split()) for a in articles],
        "Reference": [len(r.split()) for r in references],
    }
    data.update({k: [len(s.split()) for s in v]
                 for k, v in summaries_dict.items()})
    df = pd.DataFrame({k: pd.Series(v).describe()
                       for k, v in data.items()}).T
    df[["mean", "50%"]].rename(columns={"50%": "median"}).plot(
        kind="bar", ax=ax,
        color=["#3498db", "#e74c3c"], alpha=0.85)
    ax.set_title("Average Word Count: Article vs Summaries", fontweight="bold")
    ax.set_ylabel("Words"); ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("summary_lengths.png", dpi=150); plt.close()
    print("[SAVED] summary_lengths.png")


def plot_compression_ratio(articles: list, summaries_dict: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for (model, summaries), color in zip(summaries_dict.items(), colors):
        ratios = [len(s.split()) / max(len(a.split()), 1)
                  for a, s in zip(articles, summaries)]
        ax.hist(ratios, bins=10, alpha=0.6, label=model, color=color)
    ax.set_xlabel("Compression Ratio (summary / article word count)")
    ax.set_ylabel("Count")
    ax.set_title("Compression Ratio Distribution", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("compression_ratio.png", dpi=150); plt.close()
    print("[SAVED] compression_ratio.png")


def print_example_summaries(article: str, summaries: dict, reference: str):
    print(f"\n{'='*70}")
    print(f"ARTICLE (first 300 chars):\n{article[:300]}...\n")
    print(f"REFERENCE SUMMARY:\n{reference}\n{'─'*70}")
    for model, summary in summaries.items():
        print(f"{model}:\n{summary}\n{'─'*70}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main(filepath: str = DATASET_PATH, n_samples: int = N_SAMPLES):
    df         = load_cnn_dailymail(filepath, n_samples)
    articles   = df["article"].tolist()
    references = df["highlights"].tolist()

    rouge_results = {}
    all_summaries = {}

    # ── Abstractive models ────────────────────
    for model_name, config in SUMMARIZER_CONFIGS.items():
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        try:
            summarizer = load_summarizer(config)
            summaries  = summarize_batch(summarizer, articles, config)
            rouge      = compute_rouge(summaries, references)
            rouge_results[model_name] = rouge
            all_summaries[model_name] = summaries

            print(f"ROUGE-1: {rouge['ROUGE-1']:.2f}  "
                  f"ROUGE-2: {rouge['ROUGE-2']:.2f}  "
                  f"ROUGE-L: {rouge['ROUGE-L']:.2f}")

            pd.DataFrame({
                "article"  : articles,
                "reference": references,
                "summary"  : summaries,
            }).to_csv(f"summaries_{model_name}.csv", index=False)
            print(f"[SAVED] summaries_{model_name}.csv")

        except Exception as e:
            print(f"[WARN] {model_name} failed: {e}")

    # ── Extractive TextRank (bonus) ────────────
    print("\n── TextRank (Extractive Baseline) ──")
    tr_summaries              = [textrank_summary(a) for a in articles]
    tr_rouge                  = compute_rouge(tr_summaries, references)
    rouge_results["TextRank"] = tr_rouge
    all_summaries["TextRank"] = tr_summaries
    print(f"ROUGE-1: {tr_rouge['ROUGE-1']:.2f}  "
          f"ROUGE-2: {tr_rouge['ROUGE-2']:.2f}  "
          f"ROUGE-L: {tr_rouge['ROUGE-L']:.2f}")

    # ── Visualisations ────────────────────────
    if rouge_results:
        plot_rouge_comparison(rouge_results)
        plot_summary_lengths(articles, all_summaries, references)
        plot_compression_ratio(articles, all_summaries)

    # ── Example outputs ───────────────────────
    if all_summaries:
        sample_sums = {k: v[0] for k, v in all_summaries.items()}
        print_example_summaries(articles[0], sample_sums, references[0])

    # ── ROUGE summary table ───────────────────
    rouge_df = pd.DataFrame(rouge_results).T
    print("\n── ROUGE Summary Table ──")
    print(rouge_df.to_string())
    rouge_df.to_csv("rouge_summary.csv")
    print("[SAVED] rouge_summary.csv")

    print("\n✅ Task 7 complete!")


if __name__ == "__main__":
    main()
