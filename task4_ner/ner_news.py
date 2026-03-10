"""
Task 4: Named Entity Recognition (NER) from News Articles
Dataset: CoNLL-2003 (Kaggle)
https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion
File needed: train.txt
Topics: Sequence labeling, NER, spaCy rule-based + model-based
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

import spacy
from spacy import displacy

# ══════════════════════════════════════════════
# ▶▶  SET YOUR DATASET PATH HERE  ◀◀
# ══════════════════════════════════════════════
DATASET_PATH = r"C:\Users\Vanshika\Downloads\conll2003\train.txt"
# ══════════════════════════════════════════════


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_conll(filepath: str) -> list:
    """
    Parse CoNLL-2003 format.
    Each line: token  POS  chunk  NER
    Blank lines separate sentences.
    """
    print(f"[INFO] Loading CoNLL-2003 from: {filepath}")
    sentences, tokens, tags = [], [], []
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-") or not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                parts = line.split()
                tokens.append(parts[0])
                tags.append(parts[-1])
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    print(f"[INFO] Loaded {len(sentences)} sentences")
    return sentences


def sentences_to_text(sentences: list) -> list:
    return [" ".join(s["tokens"]) for s in sentences]


# ─────────────────────────────────────────────
# 2. spaCy MODEL-BASED NER
# ─────────────────────────────────────────────

def load_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        import subprocess, sys
        print(f"[INFO] Downloading {model_name} ...")
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name],
                       check=True)
        return spacy.load(model_name)


def extract_entities(texts: list, nlp) -> list:
    results = []
    for doc in nlp.pipe(texts, batch_size=32):
        results.append([(ent.text, ent.label_, ent.start_char, ent.end_char)
                         for ent in doc.ents])
    return results


# ─────────────────────────────────────────────
# 3. RULE-BASED NER (EntityRuler — bonus)
# ─────────────────────────────────────────────

def build_rule_based_nlp(base_nlp):
    """Add EntityRuler with custom patterns on top of the base model."""
    ruler = base_nlp.add_pipe("entity_ruler", before="ner", name="custom_ruler")
    patterns = [
        {"label": "ORG",    "pattern": "United Nations"},
        {"label": "ORG",    "pattern": "European Union"},
        {"label": "ORG",    "pattern": "NATO"},
        {"label": "ORG",    "pattern": "FIFA"},
        {"label": "ORG",    "pattern": "WHO"},
        {"label": "GPE",    "pattern": "United States"},
        {"label": "GPE",    "pattern": "United Kingdom"},
        {"label": "GPE",    "pattern": "New York"},
        {"label": "GPE",    "pattern": "Washington"},
        {"label": "GPE",    "pattern": "London"},
    ]
    ruler.add_patterns(patterns)
    return base_nlp


# ─────────────────────────────────────────────
# 4. EVALUATION vs CoNLL GOLD LABELS
# ─────────────────────────────────────────────

def iob_to_spans(tokens: list, tags: list) -> set:
    spans, current, current_type = set(), [], None
    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current:
                spans.add((tuple(current), current_type))
            current = [tok]; current_type = tag[2:]
        elif tag.startswith("I-") and current:
            current.append(tok)
        else:
            if current:
                spans.add((tuple(current), current_type))
            current, current_type = [], None
    if current:
        spans.add((tuple(current), current_type))
    return spans


def evaluate_ner(sentences: list, nlp, max_sents: int = 500) -> dict:
    tp = fp = fn = 0
    for sent in sentences[:max_sents]:
        text = " ".join(sent["tokens"])
        gold = iob_to_spans(sent["tokens"], sent["ner_tags"])
        doc  = nlp(text)
        pred = {(tuple(ent.text.split()), ent.label_) for ent in doc.ents}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0
    recall    = tp / (tp + fn) if tp + fn else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall else 0)
    return {"precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────

ENTITY_COLORS = {
    "PER" : "#AED6F1",
    "ORG" : "#A9DFBF",
    "LOC" : "#FAD7A0",
    "MISC": "#D7BDE2",
    "PERSON": "#AED6F1",
    "GPE"   : "#FAD7A0",
}


def plot_entity_frequency(all_entities: list, title: str, filename: str):
    flat = [(ent, lbl) for sent in all_entities for ent, lbl, *_ in sent]
    if not flat:
        print(f"[WARN] No entities found for {title}")
        return
    df  = pd.DataFrame(flat, columns=["entity", "label"])
    top = df["entity"].value_counts().head(20)
    plt.figure(figsize=(9, 5))
    top.sort_values().plot(kind="barh", color="#3498db")
    plt.title(f"Top Entities – {title}", fontweight="bold")
    plt.xlabel("Frequency")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def plot_entity_type_distribution(all_entities: list, title: str, filename: str):
    labels = [lbl for sent in all_entities for _, lbl, *_ in sent]
    if not labels:
        return
    counts = Counter(labels)
    df = pd.DataFrame(counts.items(),
                      columns=["Entity Type", "Count"]).sort_values(
                      "Count", ascending=False)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(df["Entity Type"], df["Count"],
                   color=[ENTITY_COLORS.get(t, "#ccc") for t in df["Entity Type"]])
    plt.title(f"Entity Type Distribution – {title}", fontweight="bold")
    plt.ylabel("Count")
    for bar, val in zip(bars, df["Count"]):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                 str(val), ha="center")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def render_html_displacy(texts: list, nlp, filename: str, n_samples: int = 10):
    docs = list(nlp.pipe(texts[:n_samples]))
    html = displacy.render(docs, style="ent", page=True, jupyter=False,
                           options={"colors": ENTITY_COLORS})
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[SAVED] {filename}  ← open in browser to view highlighted entities")


def plot_model_comparison(comp_df: pd.DataFrame):
    melted = comp_df[["sentence_id","model_entities","rule_entities"]].melt(
        id_vars="sentence_id", var_name="Model", value_name="Entities Found")
    plt.figure(figsize=(12, 4))
    sns.barplot(x="sentence_id", y="Entities Found", hue="Model",
                data=melted, palette=["#3498db","#e74c3c"])
    plt.title("Model vs Rule-Based: Entities per Sentence", fontweight="bold")
    plt.xlabel("Sentence ID"); plt.ylabel("# Entities")
    plt.tight_layout()
    plt.savefig("model_vs_rule_comparison.png", dpi=150); plt.close()
    print("[SAVED] model_vs_rule_comparison.png")


def plot_evaluation_scores(scores_dict: dict):
    models  = list(scores_dict.keys())
    metrics = ["precision", "recall", "f1"]
    x       = np.arange(len(models))
    width   = 0.25
    colors  = ["#3498db", "#2ecc71", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [scores_dict[m][metric] for m in models]
        bars = ax.bar(x + i * width, vals, width,
                      label=metric.capitalize(), color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_title("NER Evaluation: Precision / Recall / F1", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("evaluation_scores.png", dpi=150); plt.close()
    print("[SAVED] evaluation_scores.png")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main(conll_path: str = DATASET_PATH):
    # ── Load dataset ──────────────────────────
    sentences = load_conll(conll_path)
    texts     = sentences_to_text(sentences)

    # Use first 500 sentences for speed (remove [:500] for full dataset)
    sentences = sentences[:500]
    texts     = texts[:500]

    # ── Load spaCy models ─────────────────────
    print("[INFO] Loading spaCy models ...")
    nlp_sm = load_spacy_model("en_core_web_sm")
    try:
        nlp_md = load_spacy_model("en_core_web_md")
    except Exception:
        nlp_md = nlp_sm
        print("[INFO] en_core_web_md not available — using sm as second model")

    nlp_rule = load_spacy_model("en_core_web_sm")
    nlp_rule = build_rule_based_nlp(nlp_rule)

    # ── Extract entities ──────────────────────
    print("[INFO] Running NER ...")
    ents_sm   = extract_entities(texts, nlp_sm)
    ents_md   = extract_entities(texts, nlp_md)
    ents_rule = extract_entities(texts, nlp_rule)

    # ── Visualisations ────────────────────────
    plot_entity_frequency(ents_sm,   "en_core_web_sm", "entity_freq_sm.png")
    plot_entity_frequency(ents_rule, "Rule-Based",     "entity_freq_rule.png")

    plot_entity_type_distribution(ents_sm,   "en_core_web_sm", "entity_types_sm.png")
    plot_entity_type_distribution(ents_rule, "Rule-Based",     "entity_types_rule.png")

    render_html_displacy(texts, nlp_sm,   "ner_displacy_sm.html")
    render_html_displacy(texts, nlp_rule, "ner_displacy_rule.html")

    # ── Side-by-side comparison ───────────────
    comp_rows = []
    for i, (ents_m, ents_r) in enumerate(zip(ents_sm, ents_rule)):
        comp_rows.append({
            "sentence_id"   : i,
            "model_entities": len(ents_m),
            "rule_entities" : len(ents_r),
        })
    comp_df = pd.DataFrame(comp_rows)
    plot_model_comparison(comp_df)
    comp_df.to_csv("ner_comparison.csv", index=False)
    print("[SAVED] ner_comparison.csv")

    # ── Evaluation vs gold labels ─────────────
    print("\n── Evaluation on CoNLL gold labels ──")
    eval_scores = {}
    for model_name, nlp in [("en_core_web_sm", nlp_sm),
                              ("en_core_web_md", nlp_md),
                              ("Rule-Based",     nlp_rule)]:
        scores = evaluate_ner(sentences, nlp)
        eval_scores[model_name] = scores
        print(f"{model_name:20s}  "
              f"P={scores['precision']:.3f}  "
              f"R={scores['recall']:.3f}  "
              f"F1={scores['f1']:.3f}")

    plot_evaluation_scores(eval_scores)

    # ── Sample output ─────────────────────────
    print("\n── Sample Entities (first 5 sentences) ──")
    for i, (text, ents) in enumerate(zip(texts[:5], ents_sm[:5])):
        print(f"\nSentence {i+1}: {text}")
        for ent, lbl, *_ in ents:
            print(f"  [{lbl}] {ent}")

    print("\n✅ Task 4 complete! Open ner_displacy_sm.html in your browser.")


if __name__ == "__main__":
    main()
