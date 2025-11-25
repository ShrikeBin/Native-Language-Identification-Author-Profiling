
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

log_folder = Path(".")

logs = {}
for file_path in log_folder.glob("*.json"):
    with open(file_path, "r") as f:
        logs[file_path.stem] = json.load(f)["log_history"]

plt.figure(figsize=(13, 8))
plt.xlim(1, 7)

log_data = []
for name, entries in logs.items():
    eval_entries = [e for e in entries if "eval_mse" in e]
    if eval_entries:
        last_acc = eval_entries[-1]["eval_mse"]
        log_data.append((name, last_acc, eval_entries))

# Sorting to make more readable
log_data.sort(key=lambda x: x[1], reverse=True)

for name, _, entries in log_data:
    epochs = [e["epoch"] for e in entries if "eval_mse" in e]
    accuracies = [np.sqrt(e["eval_mse"]) for e in entries if "eval_mse" in e]
    
    plt.plot(epochs, accuracies, marker='o', label=name)

plt.xlabel("Epoch")
plt.ylabel("Eval Root Mean Square Loss")
plt.title("Eval RMSE vs Epoch for all Models")
plt.legend(loc='upper right', fontsize=16)
plt.grid(True)
plt.savefig("../political.png")