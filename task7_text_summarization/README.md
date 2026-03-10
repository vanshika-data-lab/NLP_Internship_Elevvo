# Task 7: Text Summarization Using Pre-trained Models

## 📌 Objective
Generate concise abstractive summaries of news articles using encoder-decoder transformer models.

## 🗂️ Dataset
[CNN-DailyMail – Kaggle](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)  
Columns: `id`, `article`, `highlights`

## 🔧 Approach
1. Load and preprocess articles (truncate to model input limits)
2. **Abstractive models**: BART, T5-small, Pegasus
3. **Extractive baseline**: custom TextRank (bonus)
4. Evaluate with **ROUGE-1, ROUGE-2, ROUGE-L** scores
5. Compare compression ratios and summary lengths

## 📊 Outputs
| File | Description |
|------|-------------|
| `rouge_comparison.png` | ROUGE scores across all models |
| `summary_lengths.png` | Word count: article vs summaries |
| `compression_ratio.png` | Compression ratio distributions |
| `summaries_*.csv` | Article + reference + predicted summaries |
| `rouge_summary.csv` | ROUGE score table |

## ▶️ Run
```bash
# With dataset
python text_summarization.py /path/to/cnn_dailymail.csv 20

# Demo mode (5 built-in articles)
python text_summarization.py
```
Second argument = number of articles to process (default: 5).

## 📚 Topics Covered
- Abstractive summarization with BART, T5, Pegasus
- ROUGE-1/2/L evaluation
- Extractive TextRank (bonus)
- Encoder–decoder architecture (deep learning)
- HuggingFace Transformers pipeline
