#!/bin/bash
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --partition=preempt
#SBATCH --mem=128GB
#SBATCH --time=29-23:55:00
#SBATCH --job-name=host_llama3_8b
#SBATCH --error=logs/host_llama2_13b.err
#SBATCH --output=logs/host_llama2_13b.out

nvidia-smi
export CUDA_VISIBLE_DEVICES=0

# Set Hugging Face environment variables per Babel guidelines
export VLLM_LOG_LEVEL=WARNING
export HF_HOME=/data/user_data/$USER/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1  # optional, if models are pre-cached

export PYTHONWARNINGS="ignore"
export LOGLEVEL=WARNING

# Activate your Conda env that has vLLM installed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# Model repo ID (must match Hugging Face identifier)
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# Port to expose the OpenAI-compatible API on
PORT=8000

# If you want to require an API key, set it here; otherwise, you can pass --disable-auth.
VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"

# Start vLLM’s OpenAI-compatible server
echo "Starting vLLM serve for $MODEL_ID on port $PORT…"
vllm serve $MODEL_ID \
    --port $PORT \
    --dtype auto \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.9 \
    --uvicorn-log-level warning \
    --max-model-len 8192 \
    --download-dir /data/hf_cache/hub \
    --disable-log-requests &
  #  2>&1 | grep "Starting vLLM serve" &

wait



# # Activate conda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate vllm


# # Define model repo ID and port (model ID must be a HF identifier, not path)
# MODELS=("meta-llama/Llama-3.1-8B-Instruct")
# PORTS=(8082)

# # Ensure model-port pairs are consistent
# if [ ${#MODELS[@]} -ne ${#PORTS[@]} ]; then
#     echo "The number of models and ports must be the same. Exiting..."
#     exit 1
# fi

# # Start each model server
# for i in "${!MODELS[@]}"; do
#     MODEL=${MODELS[$i]}
#     PORT=${PORTS[$i]}

#     if ss -tulwn | grep -q ":$PORT "; then
#         echo "Port $PORT is already in use. Skipping..."
#         continue
#     fi

#     echo "Starting vLLM server for $MODEL on port $PORT"
#     python -m vllm.entrypoints.openai.api_server \
#         --model "$MODEL" \
#         --port "$PORT" \
#         --max-num-batched-tokens 65536 \
#         --max-num-seqs 256 \
#         --gpu-memory-utilization 0.9 \
#         --uvicorn-log-level warning \
#         --max-model-len 8192 \
#         --enable-chunked-prefill=True \
#         --download-dir /data/hf_cache/hub \
#         2>&1 | grep "Avg prompt throughput" &
# done


# wait
