#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/has4in5i6/GNR_PROJECT.git"
BRANCH="main"
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_DIR="models/qwen2_5_vl_3b"

if [[ "$REPO_URL" == *"YOUR_USERNAME"* || "$REPO_URL" == *"YOUR_REPO"* ]]; then
  echo "Edit setup.bash and set REPO_URL to your public GitHub repository."
  exit 1
fi

git init .
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git fetch --depth 1 origin "$BRANCH"
git checkout -f FETCH_HEAD

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n gnr_project_env python=3.11 -y
conda activate gnr_project_env

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install git+https://github.com/huggingface/transformers accelerate
python -m pip install -r requirements.txt

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_ID', local_dir='$MODEL_DIR', local_dir_use_symlinks=False)"

echo "Setup complete. inference.py and model weights are ready in the current directory."
