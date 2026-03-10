# Task 8: Resume Screening Using NLP (Industry Level)

## 📌 Objective
Automatically screen and rank resumes against job descriptions using NLP-powered similarity scoring.

## 🗂️ Dataset
- [Resume Dataset – Kaggle](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset)
- [Job Description Dataset – Kaggle](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset)

## 🔧 Approach
Three complementary scoring methods are combined into an **ensemble**:

| Method | Weight | Description |
|--------|--------|-------------|
| TF-IDF Cosine | 25% | Lexical overlap between resume and JD |
| SBERT Semantic | 50% | Semantic similarity via `all-MiniLM-L6-v2` |
| Skill Match | 25% | Gazetteer-based tech skill coverage |

### Pipeline
1. Load resumes & job descriptions
2. Preprocess text (lemmatize, remove stopwords)
3. Compute TF-IDF cosine, SBERT semantic, and skill-match scores
4. Ensemble with configurable weights
5. Rank and report top candidates with justifications

## 📊 Outputs
| File | Description |
|------|-------------|
| `heatmap_tfidf.png` | TF-IDF score matrix (all candidates × jobs) |
| `heatmap_sbert.png` | SBERT score matrix |
| `heatmap_ensemble.png` | Final ensemble score matrix |
| `ranked_*_ensemble.png` | Ranked bar chart per job |
| `methods_*.png` | TF-IDF vs SBERT vs Ensemble comparison |
| `skills_*.png` | Skill coverage heatmap per job |
| `report_*.txt` | Human-readable screening report |
| `rankings_*.csv` | Ranked candidate CSV per job |

## ▶️ Run
```bash
# With datasets
python resume_screening.py /path/to/resumes.csv /path/to/jobs.csv

# Demo mode (built-in resumes + job descriptions)
python resume_screening.py

# Interactive Streamlit app (bonus – upload PDFs, paste text)
pip install streamlit PyPDF2
streamlit run app_resume.py
```

## 📚 Topics Covered
- Document similarity with cosine / TF-IDF
- Semantic search with Sentence-BERT
- Skill extraction (NLP entity matching)
- Ensemble scoring
- Named entity extraction from resumes (bonus)
- Streamlit file-upload interface (bonus)
