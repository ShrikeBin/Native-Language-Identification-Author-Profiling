import pandas as pd
import os

# Folder containing your Parquet files
parquet_folder = "."  # replace with your folder path
csv_folder = "."

os.makedirs(csv_folder, exist_ok=True)

# Loop through all Parquet files in the folder
for filename in os.listdir(parquet_folder):
    if filename.endswith(".parquet"):
        parquet_path = os.path.join(parquet_folder, filename)
        csv_path = os.path.join(csv_folder, filename.replace(".parquet", ".csv"))
        
        # Load Parquet
        df = pd.read_parquet(parquet_path)
        
        # Save as CSV
        df.to_csv(csv_path, index=False)
        print(f"Converted {filename} → {csv_path}")
