# Task 6: Question Answering with Transformers

## 📌 Objective
Build a system that answers questions from a given context using pre-trained transformer models.

## 🗂️ Dataset
[SQuAD v1.1 – Kaggle](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset)  
JSON format: context paragraphs with question-answer pairs.

## 🔧 Approach
1. Parse SQuAD JSON into context–question–answer triples
2. Run inference via HuggingFace `pipeline("question-answering")`
3. Evaluate with **Exact Match** and **Token F1** metrics
4. Compare 3 models: **DistilBERT**, **BERT-large**, **RoBERTa** (bonus)
5. Interactive **Streamlit app** (bonus)

## 📊 Outputs
| File | Description |
|------|-------------|
| `confidence_*.png` | Answer confidence distribution per model |
| `em_f1_*.png` | Per-sample EM & F1 |
| `model_comparison.png` | EM & F1 across all models |
| `answer_length.png` | Gold vs predicted answer length scatter |
| `predictions_*.csv` | Full prediction results per model |
| `model_summary.csv` | Final EM/F1 summary table |

## ▶️ Run
```bash
# Main script
python question_answering.py /path/to/train-v1.1.json

# Demo mode (no dataset needed)
python question_answering.py

# Streamlit interactive app (bonus)
streamlit run app_qa.py
```

## 📚 Topics Covered
- Transformer-based QA (span extraction)
- DistilBERT, BERT, RoBERTa (bonus comparison)
- Exact Match & Token F1 evaluation
- HuggingFace Transformers & Tokenizers
- Streamlit UI (bonus)
