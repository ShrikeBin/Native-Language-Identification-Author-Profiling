import pandas as pd
import os

# Folder containing your CSVs
csv_folder = "."  # replace with your folder path
parquet_folder = "."

os.makedirs(parquet_folder, exist_ok=True)

# Loop through all CSVs in the folder
for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        csv_path = os.path.join(csv_folder, filename)
        parquet_path = os.path.join(parquet_folder, filename.replace(".csv", ".parquet"))
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Save as Parquet
        df.to_parquet(parquet_path, index=False, compression="snappy")
        print(f"Converted {filename} → {parquet_path}")
