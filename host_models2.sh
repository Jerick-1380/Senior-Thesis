#!/bin/sh 
#SBATCH --gres=gpu:L40S:4            # Request 4 GPUs
#SBATCH --partition=general
#SBATCH --mem=128GB                   # Set memory allocation
#SBATCH --time=1-23:55:00 
#SBATCH --job-name=optimized_vllm
#SBATCH --error=logs/optimized_vllm.err
#SBATCH --output=logs/optimized_vllm.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=junkais@andrew.cmu.edu

# -------------------------------
# 1. Setup Directories and Environment
# -------------------------------
LOG_DIR="logs"
mkdir -p $LOG_DIR

SCRATCH_DIR="/scratch/junkais/test"
CACHE_DIR="$SCRATCH_DIR/hf_cache"
mkdir -p $SCRATCH_DIR
mkdir -p $CACHE_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# Ensure Hugging Face authentication
HUGGINGFACE_TOKEN="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"
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

# -------------------------------
# 2. Run vLLM Server with Optimizations
# -------------------------------
python -m vllm.entrypoints.openai.api_server \
    --model "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf" \
    --port 8082 \
    --download-dir "$CACHE_DIR" \
    --load-format bitsandbytes \
    --quantization bitsandbytes \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --device cuda &

# Wait for the server to start
wait