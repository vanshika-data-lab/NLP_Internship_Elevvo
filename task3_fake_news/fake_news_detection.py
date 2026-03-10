"""
Task 3: Fake News Detection
Dataset: Fake and Real News Dataset (Kaggle)
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
Files needed: Fake.csv  +  True.csv
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ══════════════════════════════════════════════
# ▶▶  SET YOUR DATASET PATHS HERE  ◀◀
# ══════════════════════════════════════════════
FAKE_PATH = r"C:\Users\Vanshika\Downloads\fakeandrealnews\Fake.csv"
TRUE_PATH = r"C:\Users\Vanshika\Downloads\fakeandrealnews\True.csv"
# ══════════════════════════════════════════════


def load_data(fake_path: str, true_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading Fake.csv from : {fake_path}")
    print(f"[INFO] Loading True.csv from : {true_path}")

    fake = pd.read_csv(fake_path); fake["label"] = 0; fake["label_name"] = "fake"
    real = pd.read_csv(true_path); real["label"] = 1; real["label_name"] = "real"
    df   = pd.concat([fake, real], ignore_index=True)

    print(f"[INFO] Fake columns: {list(fake.columns)}")
    df["text"] = (df.get("title", pd.Series([""] * len(df))).fillna("") + " " +
                  df.get("text", df.get("content", pd.Series([""] * len(df)))).fillna(""))
    print(f"[INFO] Total rows: {len(df)}")
    print(df["label_name"].value_counts())
    return df[["text", "label", "label_name"]]


lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|<.*?>|\[.*?\]|\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def plot_distribution(df):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="label_name", data=df,
                       palette={"fake": "#e74c3c", "real": "#2ecc71"})
    ax.set_title("Fake vs Real News", fontweight="bold")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("label_distribution.png", dpi=150); plt.close()
    print("[SAVED] label_distribution.png")


def plot_wordclouds(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, lbl, cmap, title in zip(
        axes,
        ["fake", "real"],
        ["Reds", "Greens"],
        ["Fake News", "Real News"]
    ):
        text = " ".join(df[df["label_name"] == lbl]["cleaned"])
        wc = WordCloud(width=600, height=400, background_color="white",
                       colormap=cmap, max_words=100).generate(text)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold")
    plt.suptitle("Common Terms: Fake vs Real", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("wordclouds_fake_real.png", dpi=150); plt.close()
    print("[SAVED] wordclouds_fake_real.png")


def plot_text_length(df):
    df["text_len"] = df["text"].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 4))
    for lbl, color in [("fake","#e74c3c"), ("real","#2ecc71")]:
        plt.hist(df[df["label_name"]==lbl]["text_len"],
                 bins=50, alpha=0.6, label=lbl.capitalize(), color=color)
    plt.xlabel("Word Count"); plt.ylabel("Frequency")
    plt.title("Text Length Distribution", fontweight="bold")
    plt.legend(); plt.tight_layout()
    plt.savefig("text_length_dist.png", dpi=150); plt.close()
    print("[SAVED] text_length_dist.png")


def plot_cm(y_test, preds, title, filename):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake","Real"], yticklabels=["Fake","Real"])
    plt.title(title, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def plot_roc(clf, X_test_vec, y_test, name, filename):
    try:
        proba = clf.predict_proba(X_test_vec)[:, 1]
    except AttributeError:
        proba = clf.decision_function(X_test_vec)
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, color="#3498db", label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC – {name}", fontweight="bold")
    plt.legend(); plt.tight_layout()
    plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def main():
    df = load_data(FAKE_PATH, TRUE_PATH)

    print("\n[INFO] Preprocessing …")
    df["cleaned"] = df["text"].astype(str).apply(preprocess)

    plot_distribution(df)
    plot_wordclouds(df)
    plot_text_length(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    tfidf = TfidfVectorizer(max_features=20_000, ngram_range=(1,2), sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5),
        "LinearSVC"          : LinearSVC(C=1.0, max_iter=2000),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc, f1 = accuracy_score(y_test, preds), f1_score(y_test, preds)
        results[name] = {"accuracy": acc, "f1": f1}
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        print(f"Accuracy: {acc:.4f}   F1: {f1:.4f}")
        print(classification_report(y_test, preds, target_names=["Fake","Real"]))
        plot_cm(y_test, preds, f"CM – {name}", f"cm_{name.replace(' ','_')}.png")
        plot_roc(clf, X_test_vec, y_test, name, f"roc_{name.replace(' ','_')}.png")

    metrics = pd.DataFrame(results).T
    print("\n── Summary ──"); print(metrics.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["accuracy","f1"]):
        vals = metrics[metric]
        bars = ax.bar(vals.index, vals.values, color=["#3498db","#e74c3c"])
        ax.set_ylim(0, 1); ax.set_title(f"{metric.capitalize()} Comparison", fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig("metrics_comparison.png", dpi=150); plt.close()
    print("[SAVED] metrics_comparison.png")
    print("\n✅ Task 3 complete!")


if __name__ == "__main__":
    main()
