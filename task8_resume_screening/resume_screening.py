import re
import string
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ══════════════════════════════════════════════
# ▶▶  SET YOUR DATASET PATHS HERE  ◀◀
# ══════════════════════════════════════════════
RESUME_PATH = r"C:\Users\Vanshika\Downloads\UpdatedResumeDataSet.csv"
JOB_PATH    = r"C:\Users\Vanshika\Downloads\job_descriptions.csv"
# ══════════════════════════════════════════════


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_resumes(filepath: str) -> pd.DataFrame:
    print(f"[INFO] Loading resumes from : {filepath}")
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
    print(f"[INFO] Columns found        : {list(df.columns)}")
    print(f"[INFO] Shape                : {df.shape}")

    # Format 1: 'Category', 'Resume'  (jillanisofttech / gauravduttakiit)
    if "Resume" in df.columns:
        df["text"]     = df["Resume"].astype(str)
        df["category"] = df["Category"].astype(str) if "Category" in df.columns else "Unknown"
        df["name"]     = [f"Candidate_{i+1}" for i in range(len(df))]

    # Format 2: 'ID', 'Resume_str', 'Category'
    elif "Resume_str" in df.columns:
        df["text"]     = df["Resume_str"].astype(str)
        df["category"] = df["Category"].astype(str) if "Category" in df.columns else "Unknown"
        df["name"]     = df.get("ID", pd.Series(
            [f"Candidate_{i+1}" for i in range(len(df))])).astype(str)

    # Format 3: generic
    else:
        text_col = next((c for c in df.columns
                         if any(k in c.lower()
                                for k in ["resume", "text", "content", "description"])),
                        df.columns[-1])
        name_col = next((c for c in df.columns
                         if any(k in c.lower()
                                for k in ["name", "id", "candidate"])), None)
        cat_col  = next((c for c in df.columns
                         if "category" in c.lower() or "label" in c.lower()), None)
        df["text"]     = df[text_col].astype(str)
        df["category"] = df[cat_col].astype(str) if cat_col else "Unknown"
        df["name"]     = df[name_col].astype(str) if name_col else \
                         [f"Candidate_{i+1}" for i in range(len(df))]

    # ── Clean special characters from text  ──────────────────────
    df["text"] = df["text"].apply(lambda x:
        x.encode("ascii", errors="replace").decode("ascii")
         .replace("?", " "))

    df = df[["name", "category", "text"]].dropna().reset_index(drop=True)
    print(f"[INFO] Total resumes loaded : {len(df)}")
    print(f"[INFO] Categories found     : {df['category'].value_counts().to_dict()}")
    return df


def load_jobs(filepath: str) -> pd.DataFrame:
    print(f"[INFO] Loading jobs from    : {filepath}")
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
    print(f"[INFO] Job columns found    : {list(df.columns)}")


    # Title column 
    title_col = None
    for exact in ["Job Title", "job_title", "JobTitle", "Title", "Role", "Position"]:
        if exact in df.columns:
            title_col = exact
            break
    if title_col is None:
        title_col = next((c for c in df.columns
                          if any(k in c.lower()
                                 for k in ["job title", "title", "role", "position"])
                          and "id" not in c.lower()), None)

    # Description column 
    desc_col = None
    for exact in ["Job Description", "job_description", "JobDescription",
                  "Description", "Job Summary"]:
        if exact in df.columns:
            desc_col = exact
            break
    if desc_col is None:
        desc_col = next((c for c in df.columns
                         if any(k in c.lower()
                                for k in ["description", "summary", "detail",
                                          "requirement", "content"])), None)

    print(f"[INFO] Using title col      : {title_col}")
    print(f"[INFO] Using description col: {desc_col}")

    if desc_col is None:
        raise ValueError(f"Could not find a job description column. "
                         f"Available columns: {list(df.columns)}")

    df["text"]  = df[desc_col].astype(str)
    df["title"] = df[title_col].astype(str) if title_col else \
                  [f"Job_{i+1}" for i in range(len(df))]

    # Clean special chars
    df["text"]  = df["text"].apply(lambda x:
        x.encode("ascii", errors="replace").decode("ascii").replace("?", " "))
    df["title"] = df["title"].apply(lambda x:
        x.encode("ascii", errors="replace").decode("ascii").replace("?", " "))

    df = df[["title", "text"]].dropna()
    df = df[df["text"].str.strip().str.len() > 50]   # drop empty/short descriptions
    df = df.head(6).reset_index(drop=True)
    print(f"[INFO] Total jobs loaded    : {len(df)}")
    print(f"[INFO] Job titles           : {df['title'].tolist()}")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
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
# 3. SKILL EXTRACTION
# ─────────────────────────────────────────────

TECH_SKILLS = {
    "python", "sql", "java", "scala", "tensorflow", "pytorch", "keras",
    "scikit-learn", "sklearn", "pandas", "numpy", "spark", "hadoop", "kafka",
    "aws", "gcp", "azure", "docker", "kubernetes", "bert", "gpt", "nlp",
    "machine learning", "deep learning", "data science", "statistics",
    "tableau", "powerbi", "excel", "airflow", "dbt", "snowflake",
    "huggingface", "transformers", "spacy", "nltk", "flask", "django",
    "fastapi", "react", "node", "typescript", "git", "opencv",
    "communication", "leadership", "management", "excel", "word",
    "photoshop", "illustrator", "autocad", "solidworks", "matlab",
    "recruitment", "payroll", "negotiation", "marketing", "seo","python", "sql", "r", "java", "c", "c++", "c#", "javascript", "typescript", "go", "rust", "swift", "kotlin",
"html", "css", "react", "angular", "vue", "nodejs", "express", "django", "flask", "fastapi",
"spring boot", "asp.net", "rest api", "graphql",
"machine learning", "deep learning", "nlp", "computer vision", "reinforcement learning",
"data science", "data analysis", "data visualization", "statistics", "feature engineering",
"model evaluation", "model deployment", "mlops",
"tensorflow", "pytorch", "scikit-learn", "keras", "xgboost", "lightgbm",
"pandas", "numpy", "matplotlib", "seaborn",
"spacy", "nltk", "huggingface", "transformers", "bert", "gpt",
"tableau", "powerbi", "excel",
"spark", "hadoop", "kafka", "airflow", "snowflake", "databricks",
"mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle",
"aws", "gcp", "azure", "firebase",
"docker", "kubernetes", "terraform", "jenkins", "ci/cd",
"linux", "bash", "shell scripting",
"git", "github", "gitlab",
"web scraping", "beautifulsoup", "scrapy",
"api integration", "microservices", "distributed systems", "cloud computing",
"cybersecurity", "cryptography", "ethical hacking", "penetration testing",
"blockchain", "solidity",
"iot", "embedded systems", "robotics",
"system design", "data structures", "algorithms",
"agile", "scrum", "project management",
"problem solving", "analytical thinking", "communication", "teamwork"
}


def extract_skills(text: str) -> set:
    t = text.lower()
    return {skill for skill in TECH_SKILLS if skill in t}


def skill_match_score(resume_text: str, job_text: str) -> float:
    job_skills    = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)
    if not job_skills:
        return 0.0
    return len(job_skills & resume_skills) / len(job_skills)


# ─────────────────────────────────────────────
# 4. MATCHING METHODS
# ─────────────────────────────────────────────

def tfidf_cosine_match(resumes: list, job_desc: str) -> np.ndarray:
    corpus = [job_desc] + resumes
    vec    = TfidfVectorizer(max_features=5_000, ngram_range=(1, 2))
    mat    = vec.fit_transform([preprocess(t) for t in corpus])
    return cosine_similarity(mat[0:1], mat[1:]).flatten()


def sbert_cosine_match(resumes: list, job_desc: str,
                        model: SentenceTransformer) -> np.ndarray:
    job_emb     = model.encode([job_desc], normalize_embeddings=True)
    resume_embs = model.encode(resumes,    normalize_embeddings=True)
    return (resume_embs @ job_emb.T).flatten()


def ensemble_score(tfidf_scores: np.ndarray,
                   sbert_scores: np.ndarray,
                   skill_scores: np.ndarray,
                   weights: tuple = (0.25, 0.50, 0.25)) -> np.ndarray:
    scaler = MinMaxScaler()
    t = scaler.fit_transform(tfidf_scores.reshape(-1, 1)).flatten()
    s = scaler.fit_transform(sbert_scores.reshape(-1, 1)).flatten()
    return weights[0] * t + weights[1] * s + weights[2] * skill_scores


def rank_resumes(scores: np.ndarray, candidates: list) -> pd.DataFrame:
    df = pd.DataFrame({"candidate": candidates, "score": scores})
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_score_heatmap(score_matrix: pd.DataFrame, title: str, filename: str):
    plt.figure(figsize=(max(6, len(score_matrix.columns) * 1.5),
                         max(4, len(score_matrix) * 0.4)))
    sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap="YlGn",
                vmin=0, vmax=1, linewidths=0.5, linecolor="white")
    plt.title(title, fontweight="bold", pad=12)
    plt.xlabel("Job Description")
    plt.ylabel("Candidate")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[SAVED] {filename}")


def plot_ranked_candidates(ranked_df: pd.DataFrame, job_title: str,
                            method: str, filename: str):
    top    = ranked_df.head(8)
    colors = ["#f1c40f" if i == 0 else "#95a5a6" if i == 1
              else "#cd7f32" if i == 2 else "#3498db"
              for i in range(len(top))]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top["candidate"][::-1], top["score"][::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, top["score"][::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_title(f"Candidate Rankings - {job_title} ({method})", fontweight="bold")
    ax.set_xlabel("Match Score")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[SAVED] {filename}")


def plot_skill_coverage(resumes_df: pd.DataFrame, job_text: str,
                         job_title: str, filename: str):
    job_skills = extract_skills(job_text)
    if not job_skills:
        print(f"[WARN] No known skills found in job: {job_title} — skipping skill plot")
        return
    rows = []
    for _, row in resumes_df.iterrows():
        cand_skills = extract_skills(row["text"])
        for skill in sorted(job_skills):
            rows.append({"Candidate": row["name"], "Skill": skill,
                          "Has Skill": int(skill in cand_skills)})
    pivot = pd.DataFrame(rows).pivot(
        index="Candidate", columns="Skill", values="Has Skill")
    plt.figure(figsize=(max(8, len(job_skills) * 1.2),
                         max(4, len(resumes_df) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, linecolor="white", cbar=False)
    plt.title(f"Skill Coverage vs '{job_title}'", fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[SAVED] {filename}")


def plot_method_comparison(tfidf_df, sbert_df, ensemble_df,
                            job_title: str, filename: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (df, title, color) in zip(
        axes,
        [(tfidf_df,    "TF-IDF",   "#3498db"),
         (sbert_df,    "SBERT",    "#e74c3c"),
         (ensemble_df, "Ensemble", "#2ecc71")]
    ):
        bars = ax.barh(df["candidate"], df["score"], color=color, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Score")
        for bar, val in zip(bars, df["score"]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)
    fig.suptitle(f"Method Comparison - {job_title}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[SAVED] {filename}")


def plot_category_distribution(resumes_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    order = resumes_df["category"].value_counts().index
    ax = sns.countplot(y="category", data=resumes_df, order=order, palette="Set2")
    ax.set_title("Resume Category Distribution", fontweight="bold")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_width())}",
                    (p.get_width() + 0.3, p.get_y() + p.get_height() / 2),
                    va="center")
    plt.tight_layout()
    plt.savefig("resume_category_distribution.png", dpi=150)
    plt.close()
    print("[SAVED] resume_category_distribution.png")


# ─────────────────────────────────────────────
# 6. REPORT GENERATION  
# ─────────────────────────────────────────────

def generate_report(ranked_df: pd.DataFrame, job: dict,
                     resumes_df: pd.DataFrame, top_n: int = 3) -> str:
    # Clean job text for report — remove non-ascii chars
    job_excerpt = job["text"][:200].encode("ascii", errors="replace").decode("ascii")

    lines = [
        "=" * 65,
        "  RESUME SCREENING REPORT",
        f"  Job: {job['title']}",
        "=" * 65,
        "",
        "JOB DESCRIPTION (excerpt):",
        f"  {job_excerpt}...",
        "",
        f"TOP {top_n} CANDIDATES:",
        "",
    ]
    for i, row in ranked_df.head(top_n).iterrows():
        match = resumes_df[resumes_df["name"] == row["candidate"]]
        if match.empty:
            continue
        cand_row = match.iloc[0]
        matched  = extract_skills(cand_row["text"]) & extract_skills(job["text"])
        # Clean resume excerpt
        excerpt = cand_row["text"][:120].encode(
            "ascii", errors="replace").decode("ascii")
        lines += [
            f"  #{row['rank']}  {row['candidate']}  (Score: {row['score']:.3f})",
            f"      Category       : {cand_row.get('category', 'N/A')}",
            f"      Matched Skills : {', '.join(sorted(matched)) if matched else 'None detected'}",
            f"      Resume excerpt : {excerpt}...",
            "",
        ]
    lines += [
        "",
        "ALL CANDIDATES RANKED:",
        ranked_df[["rank", "candidate", "score"]].to_string(index=False),
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main(resume_path: str = RESUME_PATH, job_path: str = JOB_PATH):
    # ── Load data ─────────────────────────────
    resumes_df = load_resumes(resume_path)
    jobs_df    = load_jobs(job_path)

    print(f"\nResumes : {len(resumes_df)}")
    print(f"Jobs    : {len(jobs_df)}")

    # ── EDA plot ──────────────────────────────
    plot_category_distribution(resumes_df)

    # ── Load SBERT ────────────────────────────
    print("\n[INFO] Loading Sentence-BERT (all-MiniLM-L6-v2) ...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # The dataset has many duplicate resumes in same category
    # Using all 962 gives identical scores for same-category candidates
    # Instead sample up to 5 unique resumes per category = ~120 diverse candidates
    resumes_sample = (
        resumes_df
        .groupby("category", group_keys=False)
        .apply(lambda x: x.drop_duplicates(subset=["text"]).head(5))
        .reset_index(drop=True)
    )
    print(f"\n[INFO] Using {len(resumes_sample)} unique resumes "
          f"({resumes_df['category'].nunique()} categories x up to 5 each)")

    # ── Score matrices ────────────────────────
    n_resumes = len(resumes_sample)
    n_jobs    = len(jobs_df)
    tfidf_matrix    = np.zeros((n_resumes, n_jobs))
    sbert_matrix    = np.zeros((n_resumes, n_jobs))
    ensemble_matrix = np.zeros((n_resumes, n_jobs))

    for j_idx, job_row in jobs_df.iterrows():
        job        = {"title": job_row["title"], "text": job_row["text"]}
        resumes    = resumes_sample["text"].tolist()
        candidates = resumes_sample["name"].tolist()

        print(f"\n[INFO] Screening for: {job['title']}")

        tfidf_scores = tfidf_cosine_match(resumes, job["text"])
        sbert_scores = sbert_cosine_match(resumes, job["text"], sbert)
        skill_scores = np.array([skill_match_score(r, job["text"]) for r in resumes])
        ens_scores   = ensemble_score(tfidf_scores, sbert_scores, skill_scores)

        tfidf_matrix[:, j_idx]    = tfidf_scores
        sbert_matrix[:, j_idx]    = sbert_scores
        ensemble_matrix[:, j_idx] = ens_scores

        tfidf_rank = rank_resumes(tfidf_scores, candidates)
        sbert_rank = rank_resumes(sbert_scores, candidates)
        ens_rank   = rank_resumes(ens_scores,   candidates)

        # Safe filename — strip special chars from job title
        safe = re.sub(r"[^\w]", "_", job["title"])[:40]

        plot_ranked_candidates(ens_rank, job["title"], "Ensemble",
                                f"ranked_{safe}_ensemble.png")
        plot_method_comparison(tfidf_rank, sbert_rank, ens_rank,
                                job["title"], f"methods_{safe}.png")
        plot_skill_coverage(resumes_sample, job["text"], job["title"],
                             f"skills_{safe}.png")

        report = generate_report(ens_rank, job, resumes_sample)
        print(report)

        with open(f"report_{safe}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[SAVED] report_{safe}.txt")

        ens_rank.to_csv(f"rankings_{safe}.csv", index=False, encoding="utf-8")
        print(f"[SAVED] rankings_{safe}.csv")

    # ── Heatmaps ──────────────────────────────
    job_titles = jobs_df["title"].tolist()
    plot_score_heatmap(
        pd.DataFrame(tfidf_matrix,
                     index=resumes_sample["name"], columns=job_titles),
        "TF-IDF Match Scores", "heatmap_tfidf.png"
    )
    plot_score_heatmap(
        pd.DataFrame(sbert_matrix,
                     index=resumes_sample["name"], columns=job_titles),
        "SBERT Semantic Match Scores", "heatmap_sbert.png"
    )
    plot_score_heatmap(
        pd.DataFrame(ensemble_matrix,
                     index=resumes_sample["name"], columns=job_titles),
        "Ensemble Match Scores", "heatmap_ensemble.png"
    )

    print("\n✅ Task 8 complete!")


if __name__ == "__main__":
    main()

