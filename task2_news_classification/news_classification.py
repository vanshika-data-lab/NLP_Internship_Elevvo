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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ══════════════════════════════════════════════
# ▶▶  SET YOUR DATASET PATH HERE  ◀◀
# ══════════════════════════════════════════════
DATASET_PATH = r"C:\Users\Vanshika\Downloads\agnews\train.csv"
# ══════════════════════════════════════════════

CATEGORIES = {1: "World", 2: "Sports", 3: "Business", 4: "Technology"}


# ─────────────────────────────────────────────
# 1. DATA LOADING  
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    print(f"[INFO] Loading: {filepath}")

    # AG News from Kaggle sometimes has a header row, sometimes not
    # Read without header first, then detect and drop if needed
    df = pd.read_csv(filepath, header=None,
                     names=["label", "title", "description"])

    # If first row looks like a header, drop it
    if str(df["label"].iloc[0]).strip().lower() in [
            "class index", "label", "class_index", "classlabel", "0"]:
        print("[INFO] Header row detected and removed")
        df = df.iloc[1:].reset_index(drop=True)

    # Convert label to numeric safely
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Map to category names
    df["category"] = df["label"].map(CATEGORIES)
    df = df.dropna(subset=["category"]).reset_index(drop=True)

    # Combine title + description
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

    print(f"[INFO] Shape: {df.shape}")
    print(df["category"].value_counts())
    return df[["text", "category"]]


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|<.*?>|\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


# ─────────────────────────────────────────────
# 3. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_class_distribution(df):
    plt.figure(figsize=(7, 4))
    order = df["category"].value_counts().index
    ax = sns.countplot(y="category", data=df, order=order, palette="Set2")
    ax.set_title("Article Category Distribution", fontweight="bold")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_width())}",
                    (p.get_width() + 10, p.get_y() + p.get_height() / 2),
                    va="center")
    plt.tight_layout()
    plt.savefig("category_distribution.png", dpi=150)
    plt.close()
    print("[SAVED] category_distribution.png")


def plot_wordclouds_per_category(df):
    # BUG 2 FIXED — df["cleaned"] now exists before this is called
    categories = df["category"].unique()
    fig, axes  = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    cmaps = ["Blues", "Greens", "Oranges", "Purples"]
    for ax, cat, cmap in zip(axes, categories, cmaps):
        text = " ".join(df[df["category"] == cat]["cleaned"])
        if not text.strip():
            ax.axis("off")
            continue
        wc = WordCloud(width=500, height=350, background_color="white",
                       colormap=cmap, max_words=80).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(cat, fontsize=13, fontweight="bold")
    # hide unused axes if fewer than 4 categories
    for ax in axes[len(categories):]:
        ax.axis("off")
    plt.suptitle("Word Clouds per Category", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("wordclouds_per_category.png", dpi=150)
    plt.close()
    print("[SAVED] wordclouds_per_category.png")


def plot_top_words_per_category(df, n=15):
    categories = df["category"].unique()
    fig, axes  = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    for ax, cat, color in zip(axes, categories, colors):
        text = " ".join(df[df["category"] == cat]["cleaned"])
        freq = pd.Series(text.split()).value_counts().head(n)
        freq.sort_values().plot(kind="barh", ax=ax, color=color)
        ax.set_title(f"Top Words – {cat}", fontweight="bold")
    for ax in axes[len(categories):]:
        ax.axis("off")
    plt.suptitle("Most Frequent Words per Category", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("top_words_per_category.png", dpi=150)
    plt.close()
    print("[SAVED] top_words_per_category.png")


def plot_text_length_distribution(df):
    df = df.copy()
    df["text_len"] = df["text"].apply(lambda x: len(x.split()))
    plt.figure(figsize=(9, 4))
    for cat, color in zip(df["category"].unique(),
                           ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]):
        subset = df[df["category"] == cat]["text_len"]
        plt.hist(subset, bins=40, alpha=0.5, label=cat, color=color)
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.title("Text Length Distribution per Category", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("text_length_distribution.png", dpi=150)
    plt.close()
    print("[SAVED] text_length_distribution.png")


def plot_confusion_matrix(y_test, preds, classes, title, filename):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[SAVED] {filename}")


def plot_model_comparison(results: dict):
    plt.figure(figsize=(8, 4))
    bars = plt.barh(list(results.keys()), list(results.values()),
                    color=["#3498db", "#e74c3c", "#2ecc71"])
    plt.xlim(0.5, 1.0)
    for bar, val in zip(bars, results.values()):
        plt.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center")
    plt.title("Model Accuracy Comparison", fontweight="bold")
    plt.xlabel("Accuracy")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()
    print("[SAVED] model_comparison.png")


# ─────────────────────────────────────────────
# 4. MODEL PIPELINE
# ─────────────────────────────────────────────

def build_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2),
                                   sublinear_tf=True)),
        ("clf", classifier),
    ])


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load ──────────────────────────────────
    df = load_data(DATASET_PATH)

    # ── Preprocess FIRST  ─────────
    print("\n[INFO] Preprocessing ...")
    df["cleaned"] = df["text"].astype(str).apply(preprocess)

    # ── EDA Plots ─
    plot_class_distribution(df)
    plot_wordclouds_per_category(df)
    plot_top_words_per_category(df)
    plot_text_length_distribution(df)

    # ── Encode Labels (drop NaN) ─
    df = df.dropna(subset=["category"]).reset_index(drop=True)
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["category"])
    classes = le.classes_
    print(f"\n[INFO] Classes: {classes}")

    # ── Train / Test Split ────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label_enc"],
        test_size=0.2, random_state=42, stratify=df["label_enc"]
    )
    print(f"[INFO] Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Train Models ──────────────────────────
    models = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, C=5)),
        "LinearSVC"          : build_pipeline(
            LinearSVC(C=1.0, max_iter=2000)),
        "Random Forest"      : build_pipeline(
            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
    }

    results = {}
    for name, pipe in models.items():
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=classes))
        plot_confusion_matrix(
            y_test, preds, classes,
            f"Confusion Matrix – {name}",
            f"cm_{name.replace(' ', '_')}.png"
        )

    # ── Summary Chart ─────────────────────────
    plot_model_comparison(results)

    print("\n✅ Task 2 complete!")


if __name__ == "__main__":
    main()
