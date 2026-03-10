"""
Task 8 BONUS: Interactive Resume Screening Streamlit App
Run: streamlit run app_resume.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import PyPDF2
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Resume Screener – Task 8", page_icon="📄", layout="wide")
st.title("📄 AI-Powered Resume Screener For Tech Positions")
st.markdown("Upload resumes and a job description to get ranked candidates with skill matching.")

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

TECH_SKILLS = {
    "python", "sql", "r", "java", "c", "c++", "c#", "javascript", "typescript", "go", "rust", "swift", "kotlin",
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

def extract_skills(text):
    t = text.lower()
    return {s for s in TECH_SKILLS if s in t}

def read_pdf(file) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    return " ".join(page.extract_text() or "" for page in reader.pages)

def read_file(uploaded_file) -> str:
    if uploaded_file.name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    return uploaded_file.read().decode("utf-8", errors="ignore")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
top_n      = st.sidebar.slider("Show top N candidates", 3, 10, 5)
use_sbert  = st.sidebar.checkbox("Use Semantic Matching (SBERT)", value=True)
w_tfidf    = st.sidebar.slider("TF-IDF weight",   0.0, 1.0, 0.25, 0.05)
w_sbert    = st.sidebar.slider("SBERT weight",     0.0, 1.0, 0.50, 0.05)
w_skills   = st.sidebar.slider("Skill match weight", 0.0, 1.0, 0.25, 0.05)

# ─────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 Upload Resumes")
    resume_files = st.file_uploader("Upload .txt or .pdf resume files",
                                     type=["txt", "pdf"], accept_multiple_files=True)
    st.caption("Or paste resume text below (one per text area)")
    manual_resumes = st.text_area("Paste resumes (separate with '---')", height=150)

with col2:
    st.subheader("💼 Job Description")
    job_title = st.text_input("Job Title", value="Senior Data Scientist")
    job_desc  = st.text_area("Job Description", height=200,
        value=(
            "Looking for a Data Scientist with Python, machine learning, TensorFlow/PyTorch, "
            "SQL and NLP experience. 4+ years required. AWS or GCP exposure preferred. "
            "Must be able to deploy production ML models."
        ))

run = st.button("🔍 Screen Resumes", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────
if run:
    # Collect resume texts
    candidates, texts = [], []
    for f in (resume_files or []):
        candidates.append(f.name.rsplit(".", 1)[0])
        texts.append(read_file(f))
    if manual_resumes.strip():
        for i, block in enumerate(manual_resumes.split("---")):
            if block.strip():
                candidates.append(f"Candidate {i+1}")
                texts.append(block.strip())

    if not texts:
        st.warning("Please upload at least one resume or paste text above.")
        st.stop()

    if not job_desc.strip():
        st.warning("Please enter a job description.")
        st.stop()

    with st.spinner("Analysing resumes …"):
        # TF-IDF
        corpus      = [job_desc] + texts
        tfidf       = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        mat         = tfidf.fit_transform(corpus)
        tfidf_scores = cosine_similarity(mat[0:1], mat[1:]).flatten()

        # SBERT
        if use_sbert:
            model       = load_sbert()
            job_emb     = model.encode([job_desc], normalize_embeddings=True)
            res_embs    = model.encode(texts, normalize_embeddings=True)
            sbert_scores = (res_embs @ job_emb.T).flatten()
        else:
            sbert_scores = tfidf_scores.copy()
            w_tfidf, w_sbert = 0.6, 0.0

        # Skill match
        job_skills    = extract_skills(job_desc)
        skill_scores  = np.array([
            len(extract_skills(t) & job_skills) / max(len(job_skills), 1) for t in texts
        ])

        # Ensemble
        total = w_tfidf + w_sbert + w_skills
        if total == 0: total = 1
        ens = (w_tfidf * tfidf_scores + w_sbert * sbert_scores + w_skills * skill_scores) / total

        df = pd.DataFrame({
            "Rank"       : range(1, len(candidates) + 1),
            "Candidate"  : candidates,
            "TF-IDF"     : tfidf_scores,
            "SBERT"      : sbert_scores,
            "Skills"     : skill_scores,
            "Match Score": ens,
        }).sort_values("Match Score", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1

    # ── Results ───────────────────────────────
    st.divider()
    st.subheader(f"🏆 Top {min(top_n, len(df))} Candidates for: {job_title}")

    # Cards for top candidates
    top_cols = st.columns(min(top_n, len(df)))
    medals = ["🥇", "🥈", "🥉"] + ["🎖️"] * 10
    for i, col in enumerate(top_cols):
        row = df.iloc[i]
        matched = extract_skills(texts[candidates.index(row["Candidate"])]) & job_skills
        col.metric(f"{medals[i]} {row['Candidate']}", f"{row['Match Score']:.1%}")
        col.caption(f"Skills: {len(matched)}/{len(job_skills)}")

    st.divider()

    # Full table
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("📊 Full Rankings")
        display_df = df[["Rank", "Candidate", "Match Score", "TF-IDF", "SBERT", "Skills"]].copy()
        display_df[["Match Score", "TF-IDF", "SBERT", "Skills"]] = \
            display_df[["Match Score", "TF-IDF", "SBERT", "Skills"]].round(3)
        st.dataframe(display_df, use_container_width=True)

    with col_b:
        st.subheader("📈 Score Breakdown")
        fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.5)))
        bar_data = df.set_index("Candidate")[["TF-IDF", "SBERT", "Skills"]].head(8)
        bar_data.plot(kind="barh", stacked=False, ax=ax,
                       color=["#3498db", "#e74c3c", "#2ecc71"], alpha=0.8)
        ax.set_xlim(0, 1); ax.set_title("Score by Method", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Skill gap analysis
    if job_skills:
        st.divider()
        st.subheader("🎯 Skill Coverage Matrix")
        skill_rows = []
        for cand, text in zip(candidates, texts):
            cand_skills = extract_skills(text)
            for skill in sorted(job_skills):
                skill_rows.append({"Candidate": cand, "Skill": skill,
                                    "Has Skill": int(skill in cand_skills)})
        skill_df = pd.DataFrame(skill_rows)
        if not skill_df.empty:
            pivot = skill_df.pivot(index="Candidate", columns="Skill", values="Has Skill")
            fig2, ax2 = plt.subplots(figsize=(max(6, len(job_skills)), max(3, len(candidates) * 0.5)))
            sns.heatmap(pivot, annot=True, fmt="d", cmap="RdYlGn",
                         vmin=0, vmax=1, linewidths=0.5, cbar=False, ax=ax2)
            ax2.set_title("Skill Coverage vs Job Requirements", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    # Download button
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Rankings CSV", csv,
                        file_name=f"rankings_{job_title.replace(' ','_')}.csv",
                        mime="text/csv")
