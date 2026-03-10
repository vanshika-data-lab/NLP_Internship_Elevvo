# Task 3: Fake News Detection

## 📌 Objective
Classify news articles as **fake** or **real** using binary classifiers.

## 🗂️ Dataset
[Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
Two files: `Fake.csv` and `True.csv` with columns: title, text, subject, date

## 🔧 Approach
1. **Preprocessing** – combine title + content, lowercase, remove stopwords, lemmatize
2. **Vectorization** – TF-IDF (unigrams + bigrams, sublinear TF)
3. **Models**
   - Logistic Regression
   - LinearSVC
4. **Evaluation** – accuracy, F1-score, confusion matrix, ROC-AUC

## 📊 Outputs
| File | Description |
|------|-------------|
| `label_distribution.png` | Fake vs Real counts |
| `wordclouds_fake_real.png` | Common terms (bonus word cloud) |
| `text_length_dist.png` | Text length distribution |
| `cm_*.png` | Confusion matrices |
| `roc_*.png` | ROC curves |
| `metrics_comparison.png` | Accuracy & F1 side-by-side |

## ▶️ Run
```bash
python fake_news_detection.py /path/to/Fake.csv /path/to/True.csv
```

## 📚 Topics Covered
- Binary classification
- TF-IDF preprocessing
- F1-score evaluation
- Word cloud visualisation (bonus)
