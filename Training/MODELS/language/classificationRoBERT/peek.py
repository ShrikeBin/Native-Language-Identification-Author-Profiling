from safetensors.torch import safe_open

path = "./model/model.safetensors"

with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
    print("\n".join(keys))