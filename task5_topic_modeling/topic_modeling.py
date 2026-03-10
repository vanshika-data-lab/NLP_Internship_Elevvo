"""
Task 5: Topic Modeling on News Articles
Dataset: BBC News Dataset (Kaggle)
https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category
Topics: Unsupervised NLP, LDA, NMF, Gensim coherence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re, string, warnings, os
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

import gensim
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ══════════════════════════════════════════════
# ▶▶  SET YOUR DATASET PATH HERE  ◀◀
# ══════════════════════════════════════════════
DATASET_PATH = r"C:\Users\Vanshika\Downloads\bbc-text.csv"
N_TOPICS     = 5    # number of topics to discover
# ══════════════════════════════════════════════


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_bbc(filepath: str) -> pd.DataFrame:
    """
    Load BBC News CSV.
    Expected columns: 'category' (optional) and 'text' (or similar).
    Also supports folder structure via topic_modeling_folders.py
    """
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Columns found        : {list(df.columns)}")
    print(f"[INFO] Shape                : {df.shape}")

    # Auto-detect text column
    if "text" not in df.columns:
        text_col = next(
            (c for c in df.columns
             if any(k in c.lower() for k in ["text","content","article","body"])),
            df.columns[-1]
        )
        df["text"] = df[text_col].astype(str)

    if "category" in df.columns:
        print(df["category"].value_counts())

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    print(f"[INFO] Total articles       : {len(df)}")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))
EXTRA_STOPS = {"said", "would", "could", "also", "one", "two", "new",
               "year", "make", "get", "use", "time", "may", "first", "last"}
stop_words.update(EXTRA_STOPS)


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|<.*?>|\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 3]
    return " ".join(tokens)


# ─────────────────────────────────────────────
# 3. sklearn LDA & NMF
# ─────────────────────────────────────────────

def fit_lda_sklearn(dtm, n_topics: int):
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=20,
        learning_method="online", random_state=42, n_jobs=-1)
    lda.fit(dtm)
    return lda


def fit_nmf_sklearn(tfidf_matrix, n_topics: int):
    nmf = NMF(n_components=n_topics, max_iter=300,
               random_state=42, init="nndsvda")
    nmf.fit(tfidf_matrix)
    return nmf


def get_top_words(model, feature_names: list, n: int = 10) -> list:
    return [
        [feature_names[i] for i in topic.argsort()[:-n - 1:-1]]
        for topic in model.components_
    ]


# ─────────────────────────────────────────────
# 4. Gensim LDA + coherence
# ─────────────────────────────────────────────

def fit_lda_gensim(tokenized_texts: list, n_topics: int):
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_texts]
    lda = LdaMulticore(corpus=corpus, num_topics=n_topics,
                        id2word=dictionary, passes=10,
                        workers=2, random_state=42)
    return lda, corpus, dictionary


def compute_coherence(lda_model, tokenized_texts, dictionary) -> float:
    cm = CoherenceModel(model=lda_model, texts=tokenized_texts,
                         dictionary=dictionary, coherence="c_v")
    return cm.get_coherence()


def find_optimal_topics(tokenized_texts, start=2, stop=8) -> pd.DataFrame:
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    corpus  = [dictionary.doc2bow(doc) for doc in tokenized_texts]
    records = []
    for k in range(start, stop + 1):
        lda = LdaMulticore(corpus=corpus, num_topics=k,
                            id2word=dictionary, passes=5,
                            workers=2, random_state=42)
        c = CoherenceModel(model=lda, texts=tokenized_texts,
                            dictionary=dictionary,
                            coherence="c_v").get_coherence()
        records.append({"n_topics": k, "coherence": c})
        print(f"  k={k}  coherence={c:.4f}")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_top_words_grid(top_words_list: list, title: str, filename: str):
    n    = len(top_words_list)
    cols = min(n, 5); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for i, (words, ax) in enumerate(zip(top_words_list, axes)):
        freqs = {w: n - j for j, w in enumerate(words)}
        wc = WordCloud(width=200, height=150, background_color="white",
                       colormap="tab10",
                       prefer_horizontal=1.0).generate_from_frequencies(freqs)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(f"Topic {i+1}", fontweight="bold", fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def plot_topic_word_bar(top_words_list: list, title: str, filename: str):
    n    = len(top_words_list)
    cols = min(n, 5); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3.5, rows * 3.5))
    axes   = np.array(axes).flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (words, ax) in enumerate(zip(top_words_list, axes)):
        weights = list(range(len(words), 0, -1))
        ax.barh(words[::-1], weights[::-1], color=colors[i])
        ax.set_title(f"Topic {i+1}", fontweight="bold", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def plot_doc_topic_heatmap(doc_topic: np.ndarray, n_show: int = 50,
                            filename: str = "doc_topic_heatmap.png"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(doc_topic[:n_show].T, cmap="YlOrRd",
                xticklabels=False,
                yticklabels=[f"T{i+1}" for i in range(doc_topic.shape[1])])
    plt.title(f"Document–Topic Distribution (first {n_show} docs)",
              fontweight="bold")
    plt.xlabel("Documents"); plt.ylabel("Topics")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def plot_coherence_curve(coherence_df: pd.DataFrame,
                          filename: str = "coherence_curve.png"):
    plt.figure(figsize=(7, 4))
    plt.plot(coherence_df["n_topics"], coherence_df["coherence"],
             marker="o", color="#3498db", lw=2)
    best = coherence_df.loc[coherence_df["coherence"].idxmax()]
    plt.axvline(x=best["n_topics"], color="#e74c3c", linestyle="--",
                label=f"Best k={int(best['n_topics'])} "
                      f"(c={best['coherence']:.3f})")
    plt.xlabel("Number of Topics"); plt.ylabel("Coherence Score (c_v)")
    plt.title("Optimal Number of Topics", fontweight="bold")
    plt.legend(); plt.tight_layout()
    plt.savefig(filename, dpi=150); plt.close()
    print(f"[SAVED] {filename}")


def plot_lda_nmf_comparison(lda_words: list, nmf_words: list, n_topics: int):
    fig, axes = plt.subplots(n_topics, 2,
                              figsize=(12, n_topics * 1.8))
    for i in range(n_topics):
        for j, (words, model_name) in enumerate(
                [(lda_words[i], "LDA"), (nmf_words[i], "NMF")]):
            axes[i, j].barh(range(len(words)),
                             range(len(words), 0, -1),
                             color="#3498db" if j == 0 else "#e74c3c",
                             alpha=0.8)
            axes[i, j].set_yticks(range(len(words)))
            axes[i, j].set_yticklabels(words, fontsize=8)
            axes[i, j].set_title(f"{model_name} – Topic {i+1}",
                                  fontsize=9, fontweight="bold")
            axes[i, j].tick_params(axis="x", labelbottom=False)
    plt.suptitle("LDA vs NMF: Top Words per Topic",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("lda_vs_nmf_topics.png", dpi=150); plt.close()
    print("[SAVED] lda_vs_nmf_topics.png")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main(filepath: str = DATASET_PATH, n_topics: int = N_TOPICS):
    df = load_bbc(filepath)

    print("\n[INFO] Preprocessing ...")
    df["cleaned"] = df["text"].astype(str).apply(preprocess)
    df = df[df["cleaned"].str.strip() != ""].reset_index(drop=True)
    tokenized = [doc.split() for doc in df["cleaned"]]

    # ── Vectorize ─────────────────────────────
    cv    = CountVectorizer(max_features=5_000, min_df=3, max_df=0.9)
    tfidf = TfidfVectorizer(max_features=5_000, min_df=3, max_df=0.9)
    dtm       = cv.fit_transform(df["cleaned"])
    tfidf_mat = tfidf.fit_transform(df["cleaned"])
    cv_vocab    = cv.get_feature_names_out()
    tfidf_vocab = tfidf.get_feature_names_out()

    # ── sklearn LDA ───────────────────────────
    print(f"\n[INFO] Fitting sklearn LDA (k={n_topics}) ...")
    lda_sk    = fit_lda_sklearn(dtm, n_topics)
    lda_words = get_top_words(lda_sk, cv_vocab, n=15)
    print("LDA Topics:")
    for i, words in enumerate(lda_words):
        print(f"  Topic {i+1}: {', '.join(words[:8])}")

    plot_top_words_grid(lda_words, "LDA Topics", "lda_wordclouds.png")
    plot_topic_word_bar(lda_words, "LDA Top Words per Topic", "lda_top_words.png")
    doc_topic_lda = lda_sk.transform(dtm)
    plot_doc_topic_heatmap(doc_topic_lda, filename="lda_doc_topic_heatmap.png")

    # ── sklearn NMF ───────────────────────────
    print(f"\n[INFO] Fitting NMF (k={n_topics}) ...")
    nmf_sk    = fit_nmf_sklearn(tfidf_mat, n_topics)
    nmf_words = get_top_words(nmf_sk, tfidf_vocab, n=15)
    print("NMF Topics:")
    for i, words in enumerate(nmf_words):
        print(f"  Topic {i+1}: {', '.join(words[:8])}")

    plot_top_words_grid(nmf_words, "NMF Topics", "nmf_wordclouds.png")
    plot_topic_word_bar(nmf_words, "NMF Top Words per Topic", "nmf_top_words.png")

    # ── LDA vs NMF comparison ─────────────────
    plot_lda_nmf_comparison(lda_words, nmf_words, n_topics)

    # ── Gensim LDA + coherence ────────────────
    print(f"\n[INFO] Fitting Gensim LDA ...")
    lda_gen, corpus, dictionary = fit_lda_gensim(tokenized, n_topics)
    c_score = compute_coherence(lda_gen, tokenized, dictionary)
    print(f"Gensim LDA coherence (c_v): {c_score:.4f}")

    # Coherence curve to find best k
    print("\n[INFO] Scanning coherence over k=2..8 ...")
    coh_df = find_optimal_topics(tokenized, start=2, stop=8)
    plot_coherence_curve(coh_df)
    coh_df.to_csv("coherence_scores.csv", index=False)
    print("[SAVED] coherence_scores.csv")

    # ── Dominant topic per document ───────────
    df["dominant_topic"] = np.argmax(doc_topic_lda, axis=1) + 1
    topic_dist = df["dominant_topic"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    topic_dist.plot(kind="bar",
                     color=plt.cm.Set2(np.linspace(0, 1, n_topics)))
    plt.title("Document Count per Dominant Topic", fontweight="bold")
    plt.xlabel("Topic"); plt.ylabel("Count"); plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("dominant_topic_dist.png", dpi=150); plt.close()
    print("[SAVED] dominant_topic_dist.png")

    df[["text","cleaned","dominant_topic"]].to_csv(
        "articles_with_topics.csv", index=False)
    print("[SAVED] articles_with_topics.csv")

    print("\n✅ Task 5 complete!")


if __name__ == "__main__":
    main()
