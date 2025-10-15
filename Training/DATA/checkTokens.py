# token_distribution_progress.py
import pandas as pd
from transformers import DistilBertTokenizerFast
import sys

# --- Config ---
text_col = "text"
file_path = sys.argv[1]
bucket_size = 20      # tokens per bucket
batch_size = 500      # number of lines per batch
output_file = "token_distribution"

# --- Load dataset ---
if file_path.endswith(".parquet"):
    df = pd.read_parquet(file_path)
elif file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    raise ValueError("File must be CSV or Parquet")

# --- Initialize tokenizer ---
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# --- Compute token lengths with progress ---
n = len(df)
n_tokens = []

for start in range(0, n, batch_size):
    batch_texts = df[text_col].iloc[start:start+batch_size].tolist()
    batch_enc = tokenizer(batch_texts, add_special_tokens=True, truncation=False, padding=False)
    batch_lengths = [len(ids) for ids in batch_enc['input_ids']]
    n_tokens.extend(batch_lengths)
    # progress bar
    print(f"\rProcessed {min(start+batch_size, n)} / {n} lines", end="")

df['n_tokens'] = n_tokens
print("\nTokenization done!")

# --- Bucket tokens ---
max_token = df['n_tokens'].max()
bins = list(range(0, max_token + bucket_size, bucket_size))
labels = [f"{b}-{b+bucket_size}" for b in bins[:-1]]

df['token_bucket'] = pd.cut(df['n_tokens'], bins=bins, labels=labels, right=False)

# --- Compute percentage per bucket ---
distribution = df['token_bucket'].value_counts(normalize=True).sort_index() * 100

# --- Print and save ---
with open(output_file, "w") as f:
    f.write(f"Token distribution (% of lines) in {file_path}:\n\n")
    for bucket, pct in distribution.items():
        line = f"{bucket} tokens: {pct:.2f}%\n"
        print(line, end="")
        f.write(line)

print(f"\nDistribution saved to {output_file}")
