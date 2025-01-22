#!/bin/sh 
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=128GB
#SBATCH --time 1-23:55:00 
#SBATCH --job-name=host_llama2_13b
#SBATCH --error=logs/host_llama2_13b.err
#SBATCH --output=logs/host_llama2_13b.out

nvidia-smi
export CUDA_VISIBLE_DEVICES=0,2,3
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

mkdir -p /scratch/junkais/test
source ~/miniconda3/etc/profile.d/conda.sh

export /scratch/junkais/test/hf_cache
#source ~/.bashrc

HUGGINGFACE_TOKEN="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"
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

conda activate vllm

# Define models and ports
#"/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf"
#"meta-llama/Meta-Llama-3-8B-Instruct"
MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
)
PORTS=(
    8082
)

# Ensure that the number of models matches the number of ports
if [ ${#MODELS[@]} -ne ${#PORTS[@]} ]; then
    echo "The number of models and ports must be the same. Exiting..."
    exit 1
fi

# Start servers for each model and port
for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[$i]}
    PORT=${PORTS[$i]}

    if ss -tulwn | grep -q ":$PORT "; then
        echo "Port $PORT is already in use. Skipping..."
        continue
    else
        echo "Starting server for model $MODEL on port $PORT"
        python -m vllm.entrypoints.openai.api_server \
            --model $MODEL \
            --port $PORT \
            --download-dir /scratch/junkais/test/cache &
    fi
done

wait