# Task 2: News Category Classification

## 📌 Objective
Classify news articles into 4 categories: **World, Sports, Business, Technology**.

## 🗂️ Dataset
[AG News – Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  
Columns: class index (1–4), title, description

## 🔧 Approach
1. **Preprocessing** – tokenization, stopword removal, lemmatization
2. **Vectorization** – TF-IDF with sublinear TF scaling (unigrams + bigrams)
3. **Models**
   - Logistic Regression
   - LinearSVC
   - Random Forest
4. **Evaluation** – accuracy, per-class precision/recall/F1, confusion matrix

## 📊 Outputs
| File | Description |
|------|-------------|
| `category_distribution.png` | Article count per category |
| `wordclouds_per_category.png` | Word clouds for each category |
| `top_words_per_category.png` | Top 15 words per category |
| `cm_*.png` | Confusion matrices |
| `model_comparison.png` | Accuracy comparison |

## ▶️ Run
```bash
python news_classification.py /path/to/train.csv
```

## 📚 Topics Covered
- Multiclass text classification
- TF-IDF feature engineering
- LinearSVC, Random Forest
- Word cloud visualisation (bonus)
