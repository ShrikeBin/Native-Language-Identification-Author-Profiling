# Datasets Overview

### All datasets are divided into TRAIN/TEST roughly split 90-10 respectively.
---

### Age
- **COMBINED**:
  - Blog Authorship Corpus

### Gender
- **COMBINED**:
  - Blog Authorship Corpus
  - Enron Email Dataset

### Language
- **COMBINED**:
  - italki Corpus
  - lang8 Corpus
  - Data found on [GitHub](https://github.com/Tejas-Nanaware/Native-Language-Identification/tree/master) with similar project (undetermined source)

### MBTI
- **COMBINED**:
  - Cleaned, augmented Kaggle dataset (HuggingFace)
  - Reddit scraped data (HuggingFace)

### Political View
- **LONG**:
  - Politically biased articles from allsides.com (HuggingFace) (unused due to too little about author personal belief (too hard to find bias))
- **SHORT**:
  - Synthetic data created using GPT-4 (HuggingFace)
  - Scraped American political tweets (HuggingFace)

---

## Tools and Techniques Used
---
- **Regex**: Cleansed and normalized text (deleted links, emails, normalized to UTF-8).  
- **Pandas**: Data manipulation, filtering, and aggregation.  
- **fastparquet**: Converted CSVs to Parquet for faster I/O.  
- **spaCy (large model)**: NER and NLP tasks, differentiating spam, promotional content, and genuine emails in the massive Enron dataset.
- **Distilbert Tokenizer**: Used to normalize token amount in data (etc make distribution of tokens among examples plausible, split examles with more than 256 tokens, combine examples with 10 tokesn to reach at least 50 etc.)
---