# Task 4: Named Entity Recognition (NER) from News Articles

## 📌 Objective
Identify and categorise named entities — **people, locations, organisations** — from news text.

## 🗂️ Dataset
[CoNLL-2003 – Kaggle](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion)  
Standard CoNLL format: one token per line with IOB NER tags.

## 🔧 Approach
1. **Model-based NER** – `spaCy en_core_web_sm` (and `en_core_web_md` for comparison)
2. **Rule-based NER** – spaCy `EntityRuler` with custom gazetteer patterns
3. **Evaluation** – precision, recall, F1 vs CoNLL gold labels (when data available)
4. **Visualisation** – displaCy HTML, entity frequency & type distribution plots

## 📊 Outputs
| File | Description |
|------|-------------|
| `ner_displacy_sm.html` | **Open in browser** – entity highlights (model-based) |
| `ner_displacy_rule.html` | Entity highlights (rule-based) |
| `entity_freq_*.png` | Top entities by frequency |
| `entity_types_*.png` | Entity type distribution |
| `model_vs_rule_comparison.png` | Entities per sentence comparison |
| `ner_comparison.csv` | Per-sentence entity counts |

## ▶️ Run
```bash
python ner_news.py /path/to/train.txt   # CoNLL-2003
# or demo mode:
python ner_news.py
# Then open ner_displacy_sm.html in your browser
```

## 📚 Topics Covered
- Sequence labeling & NER
- spaCy model-based NER
- Rule-based EntityRuler (bonus)
- displaCy visualisation (bonus)
- Two-model comparison (bonus)
