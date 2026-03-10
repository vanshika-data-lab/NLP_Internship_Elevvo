"""
Task 1: Sentiment Analysis on Product Reviews
Dataset: IMDb Dataset (Kaggle)
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re, string, warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ══════════════════════════════════════════════
# ▶▶  SET YOUR DATASET PATH HERE  ◀◀
# ══════════════════════════════════════════════
DATASET_PATH = r"C:\Users\Vanshika\Downloads\IMDB Dataset.csv"
# ══════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    print(f"[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Columns : {list(df.columns)}")
    print(f"[INFO] Shape   : {df.shape}")

    if "review" not in df.columns:
        df.rename(columns={df.columns[0]: "review"}, inplace=True)
    if "sentiment" not in df.columns and "label" in df.columns:
        df["sentiment"] = df["label"].map({1: "positive", 0: "negative"})

    return df


lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def plot_class_distribution(df):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="sentiment", data=df, palette=["#e74c3c", "#2ecc71"])
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width()/2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150); plt.close()
    print("[SAVED] class_distribution.png")


def plot_wordclouds(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sentiment, color in zip(axes, ["positive","negative"], ["Greens","Reds"]):
        text = " ".join(df[df["sentiment"] == sentiment]["cleaned_review"])
        wc = WordCloud(width=600, height=400, background_color="white",
                       colormap=color, max_words=100).generate(text)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(f"{sentiment.capitalize()} Reviews", fontsize=13, fontweight="bold")
    plt.suptitle("Most Frequent Words", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("wordclouds.png", dpi=150); plt.close()
    print("[SAVED] wordclouds.png")


def plot_top_words(df, n=20):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, sentiment, color in zip(axes, ["positive","negative"],
                                    ["#2ecc71","#e74c3c"]):
        text = " ".join(df[df["sentiment"] == sentiment]["cleaned_review"])
        freq = pd.Series(text.split()).value_counts().head(n)
        freq.sort_values().plot(kind="barh", ax=ax, color=color)
        ax.set_title(f"Top {n} {sentiment.capitalize()} Words", fontweight="bold")
    plt.tight_layout()
    plt.savefig("top_words.png", dpi=150); plt.close()
    print("[SAVED] top_words.png")


def build_pipeline(vectorizer, classifier):
    return Pipeline([("vec", vectorizer), ("clf", classifier)])


def evaluate_model(model, X_test, y_test, label):
    preds = model.predict(X_test)
    print(f"\n{'='*50}\n  {label}\n{'='*50}")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))
    return preds


def plot_confusion_matrix(y_test, preds, title, filename):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative","Positive"],
                yticklabels=["Negative","Positive"])
    plt.title(title, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def plot_roc(model, X_test, y_test, label, filename):
    try:
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#3498db", lw=2)
        plt.plot([0,1],[0,1],"k--")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC – {label}", fontweight="bold")
        plt.legend(); plt.tight_layout()
        plt.savefig(filename, dpi=150); plt.close()
        print(f"[SAVED] {filename}")
    except Exception:
        pass


def main():
    # ── Load ──────────────────────────────────
    df = load_data(DATASET_PATH)
    print(df["sentiment"].value_counts())

    # ── Preprocess ────────────────────────────
    print("\n[INFO] Preprocessing text …")
    df["cleaned_review"] = df["review"].astype(str).apply(preprocess_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    # ── EDA Plots ─────────────────────────────
    plot_class_distribution(df)
    plot_wordclouds(df)
    plot_top_words(df)

    # ── Split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_review"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    # ── Models ────────────────────────────────
    models = {
        "LR + TF-IDF" : build_pipeline(
            TfidfVectorizer(max_features=10_000, ngram_range=(1,2)),
            LogisticRegression(max_iter=1000, C=1.0)),
        "LR + BoW"    : build_pipeline(
            CountVectorizer(max_features=10_000, ngram_range=(1,2)),
            LogisticRegression(max_iter=1000)),
        "NB + TF-IDF" : build_pipeline(
            TfidfVectorizer(max_features=10_000),
            MultinomialNB()),
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds   = evaluate_model(pipe, X_test, y_test, name)
        results[name] = accuracy_score(y_test, preds)
        safe = name.replace(" ","_").replace("+","").strip()
        plot_confusion_matrix(y_test, preds, f"CM – {name}", f"cm_{safe}.png")
        plot_roc(pipe, X_test, y_test, name, f"roc_{safe}.png")

    # ── Summary ───────────────────────────────
    plt.figure(figsize=(8, 4))
    bars = plt.barh(list(results.keys()), list(results.values()),
                    color=["#3498db","#2ecc71","#e74c3c"])
    plt.xlim(0.5, 1.0)
    for bar, val in zip(bars, results.values()):
        plt.text(val+0.002, bar.get_y()+bar.get_height()/2,
                 f"{val:.4f}", va="center")
    plt.title("Model Accuracy Comparison", fontweight="bold")
    plt.xlabel("Accuracy")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150); plt.close()
    print("[SAVED] model_comparison.png")
    print("\n✅ Task 1 complete!")


if __name__ == "__main__":
    main()
