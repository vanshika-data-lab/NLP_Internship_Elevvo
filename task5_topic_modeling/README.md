# Task 5: Topic Modeling on News Articles

## 📌 Objective
Discover **hidden topics** in a corpus of news articles using unsupervised NLP.

## 🗂️ Dataset
[BBC News Dataset – Kaggle](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category)  
Columns: `category`, `text` (5 categories, ~2 225 articles)

## 🔧 Approach
1. **Preprocessing** – tokenization, stopword removal, lemmatization
2. **Vectorization** – CountVectorizer (for LDA) and TF-IDF (for NMF)
3. **Models**
   - **LDA** (sklearn + Gensim) ← primary
   - **NMF** (sklearn) ← bonus comparison
4. **Coherence sweep** – find optimal `k` from 2–8 using Gensim c_v score
5. **Visualisation** – word clouds, bar charts, doc–topic heatmap, coherence curve

## 📊 Outputs
| File | Description |
|------|-------------|
| `lda_wordclouds.png` | Word cloud per LDA topic |
| `lda_top_words.png` | Bar chart – top words per LDA topic |
| `nmf_wordclouds.png` | Word cloud per NMF topic |
| `nmf_top_words.png` | Bar chart – top words per NMF topic |
| `lda_vs_nmf_topics.png` | Side-by-side LDA vs NMF comparison (bonus) |
| `lda_doc_topic_heatmap.png` | Document–topic probability heatmap |
| `coherence_curve.png` | Coherence score vs number of topics |
| `dominant_topic_dist.png` | Article count per dominant topic |
| `articles_with_topics.csv` | Articles tagged with dominant topic |
| `coherence_scores.csv` | k vs coherence table |

## ▶️ Run
```bash
python topic_modeling.py /path/to/bbc_news.csv 5   # second arg = n_topics
# or demo mode:
python topic_modeling.py
```

## 📚 Topics Covered
- Unsupervised NLP
- LDA (Latent Dirichlet Allocation)
- NMF (Non-negative Matrix Factorization) ← bonus
- Gensim coherence evaluation
- pyLDAvis / word clouds visualisation (bonus)
