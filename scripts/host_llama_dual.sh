#!/bin/bash
#SBATCH --gres=gpu:8                        
#SBATCH --constraint='A100_80GB|A100_40GB'  
#SBATCH --partition=general
#SBATCH --mem=128GB
#SBATCH --time=1-23:55:00
#SBATCH --job-name=host_llama_dual
#SBATCH --error=logs/host_llama_dual.err
#SBATCH --output=logs/host_llama_dual.out

nvidia-smi

#————————————————————————————————————————
#  Activate vLLM environment
#————————————————————————————————————————
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export VLLM_LOG_LEVEL=WARNING
export HF_HOME=/data/user_data/$USER/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1


MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
NUM_GPUS=8
BASE_PORT=9000

for (( i=0; i<NUM_GPUS; i++ )); do
  export CUDA_VISIBLE_DEVICES=$i
  port=$(( BASE_PORT + i ))
  echo "Starting vLLM on GPU $i → port $port …"

  vllm serve "$MODEL_ID" \
      --port ${port} \
      --dtype auto \
      --max-num-batched-tokens 65536 \
      --max-num-seqs 32 \
      --gpu-memory-utilization 0.9 \
      --uvicorn-log-level warning \
      --max-model-len 8192 \
      --download-dir /data/hf_cache/hub \
      --disable-log-requests \
      &
done

wait