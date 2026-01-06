import mlx.core as mx
import json
from huggingface_hub import HfApi, create_repo

# 1. Setup your Model and Hyperparameters
repo_id = "dzur658/mlx-unet-mnist"  # Change this!
config = {
    "T": 1000,
    "img_ch": 3,
    "img_size": 64,
    "down_chs": (64, 64, 128),
    "t_embed_dim": 8,
    "c_embed_dim": 10, # N_CLASSES
    "framework": "mlx"
}

# 2. Save config locally
with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

# 3. Create Repo and Upload
api = HfApi()

# Create the repo if it doesn't exist
create_repo(repo_id, exist_ok=True, repo_type="model")

# Upload all files in the current directory that match these patterns
print(f"Uploading to {repo_id}...")
api.upload_file(
    path_or_fileobj="mlx_unet_mnist_weights.safetensors",
    path_in_repo="mlx_unet_mnist_weights.safetensors",
    repo_id=repo_id,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj="config.json",
    path_in_repo="config.json",
    repo_id=repo_id,
    repo_type="model",
)

# Upload your python file so others can load the class!
api.upload_file(
    path_or_fileobj="mlx_unet.ipynb", # Whatever your file is named
    path_in_repo="mlx_unet.ipynb",
    repo_id=repo_id,
    repo_type="model",
)

print("Done!")