# 🧠 NLP Internship Tasks – Elevvo

A collection of **8 NLP projects** completed as part of the Elevvo 1-month internship program.  
Each task is self-contained with its own dataset, preprocessing pipeline, models, and visualisations.

---

## 📁 Repository Structure

```
nlp_internship/
├── task1_sentiment/           # Sentiment Analysis on Product Reviews
├── task2_news_classification/ # News Category Classification
├── task3_fake_news/           # Fake News Detection
├── task4_ner/                 # Named Entity Recognition
├── task5_topic_modeling/      # Topic Modeling on News Articles
├── task6_question_answering/  # QA with Transformers + Streamlit app
├── task7_text_summarization/  # Abstractive Summarization (BART/T5/Pegasus)
├── task8_resume_screening/    # Resume Screening + Streamlit app
├── requirements.txt
└── README.md
```

---

## 🗂️ Tasks Overview

| # | Task | Level | Dataset | Topics |
|---|------|-------|---------|--------|
| 1 | Sentiment Analysis | 1 | IMDb / Amazon Reviews | Binary classification, TF-IDF |
| 2 | News Classification | 1 | AG News | Multiclass, TF-IDF, SVM |
| 3 | Fake News Detection | 2 | Fake & Real News | Binary clf, F1, TF-IDF |
| 4 | Named Entity Recognition | 2 | CoNLL-2003 | spaCy, NER, displaCy |
| 5 | Topic Modeling | 2 | BBC News | LDA, NMF, Gensim, coherence |
| 6 | Question Answering | 3 | SQuAD v1.1 | BERT/DistilBERT/RoBERTa, EM & F1 |
| 7 | Text Summarization | 3 | CNN-DailyMail | BART/T5/Pegasus, ROUGE scores |
| 8 | Resume Screening | Industry | Resume + Job Dataset | SBERT, cosine similarity, ensemble |

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/nlp_internship.git
cd nlp_internship
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md   # optional – Task 4 bonus
```

---

## 🚀 Running Each Task

Every script accepts an **optional** path to the Kaggle dataset.  
If no path is given, it falls back to **built-in synthetic data** so you can run it instantly.

### Task 1 – Sentiment Analysis
```bash
cd task1_sentiment
# With dataset (IMDB CSV from Kaggle):
python sentiment_analysis.py /path/to/IMDB_Dataset.csv
# Without dataset (demo mode):
python sentiment_analysis.py
```

### Task 2 – News Classification
```bash
cd task2_news_classification
python news_classification.py /path/to/train.csv   # AG News
```

### Task 3 – Fake News Detection
```bash
cd task3_fake_news
# Kaggle gives two files: Fake.csv and True.csv
python fake_news_detection.py /path/to/Fake.csv /path/to/True.csv
```

### Task 4 – NER
```bash
cd task4_ner
python ner_news.py /path/to/train.txt   # CoNLL-2003 format
# Open ner_displacy_sm.html in a browser for highlighted entities
```

### Task 5 – Topic Modeling
```bash
cd task5_topic_modeling
python topic_modeling.py /path/to/bbc_news.csv 5   # 5 topics
```

---

## 📊 Outputs

Each task generates:
- **PNG plots** – confusion matrices, word clouds, frequency bars, ROC curves, coherence curves
- **CSV files** – predictions, topic assignments, comparison tables
- **HTML files** (Task 4) – displaCy entity visualisation

---

## 📦 Datasets (Kaggle links)

| Task | Dataset |
|------|---------|
| 1 | [IMDb Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| 2 | [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) |
| 3 | [Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| 4 | [CoNLL-2003](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion) |
| 5 | [BBC News](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category) |
| 6 | [SQuAD v1.1](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset) |
| 7 | [CNN-DailyMail](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail) |
| 8 | [Resume Dataset](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset) + [Job Descriptions](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) |

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NLTK` · `spaCy` · `scikit-learn` · `Gensim` · `HuggingFace Transformers` · `Sentence-Transformers` · `rouge-score` · `Matplotlib` · `Seaborn` · `WordCloud` · `Streamlit`

---

## 👤 Author

**[Vanshika Aggarwal]**  
Elevvo NLP Internship · 2026
