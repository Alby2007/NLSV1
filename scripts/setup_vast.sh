#!/bin/bash
set -e

echo "=== Neuralese v1 Phase 5 Setup ==="

# Clone repo
cd /workspace
if [ ! -d "NLSV1" ]; then
    git clone https://github.com/Alby2007/NLSV1.git
fi
cd NLSV1

# Install Python deps
pip install -q torch transformers datasets peft accelerate bitsandbytes tqdm lark scikit-learn huggingface_hub

# Create required dirs
mkdir -p data/corpus outputs/checkpoints

# Download corpus from HuggingFace dataset (private)
# Usage: HF_TOKEN=<your_token> bash setup_vast.sh
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Export your HuggingFace token first:"
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

echo "Downloading corpus from HuggingFace..."
python3 -c "
from huggingface_hub import hf_hub_download
import os, shutil

token = os.environ['HF_TOKEN']
path = hf_hub_download(
    repo_id='ALBYJ07/neuralese-v1-corpus',
    filename='train.jsonl',
    repo_type='dataset',
    token=token,
    local_dir='data/corpus'
)
print(f'Corpus downloaded to {path}')
"

# Count examples
echo "Corpus lines: $(wc -l < data/corpus/train.jsonl)"

# Start training in background with nohup
echo "Starting training..."
nohup python phase5_finetune/train.py > outputs/train.log 2>&1 &
echo "Training PID: $!"
echo "Monitor: tail -f /workspace/NLSV1/outputs/train.log"
