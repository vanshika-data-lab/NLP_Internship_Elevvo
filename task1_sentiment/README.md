# Task 1: Sentiment Analysis on Product Reviews

## 📌 Objective
Classify product reviews as **positive** or **negative** using traditional ML classifiers.

## 🗂️ Dataset
[IMDb Movie Reviews – Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
Columns: `review`, `sentiment` (positive / negative)

## 🔧 Approach
1. **Preprocessing** – lowercase, strip HTML/URLs, remove punctuation & stopwords, lemmatize
2. **Vectorization** – TF-IDF (unigrams + bigrams) and CountVectorizer (BoW)
3. **Models**
   - Logistic Regression + TF-IDF
   - Logistic Regression + BoW
   - Naive Bayes + TF-IDF *(bonus)*
4. **Evaluation** – accuracy, precision, recall, F1, confusion matrix, ROC-AUC

## 📊 Outputs
| File | Description |
|------|-------------|
| `class_distribution.png` | Positive vs Negative count |
| `wordclouds.png` | Most frequent words per sentiment |
| `top_words.png` | Bar chart of top 20 words per sentiment |
| `cm_*.png` | Confusion matrices per model |
| `roc_*.png` | ROC curves per model |
| `model_comparison.png` | Accuracy bar chart across models |

## ▶️ Run
```bash
python sentiment_analysis.py /path/to/IMDB_Dataset.csv
# or demo mode (no download needed):
python sentiment_analysis.py
```

## 📚 Topics Covered
- Binary text classification
- TF-IDF & CountVectorizer
- Naive Bayes (bonus)
- ROC-AUC evaluation
