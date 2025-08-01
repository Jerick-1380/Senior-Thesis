#!/bin/bash
# pipeline_config.sh - Configuration for vLLM pipeline

#————————————————————————————————————————
# Model Configuration
#————————————————————————————————————————
export MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
export MODEL_DTYPE="auto"
export MAX_MODEL_LEN=8192

#————————————————————————————————————————
# MAIN CONFIGURATION
#————————————————————————————————————————
# Number of GPUs to use for vLLM hosting
export NUM_GPUS=8

# Number of parallel analysis jobs
# Should typically equal NUM_GPUS for optimal load distribution
export NUM_PARALLEL=$NUM_GPUS

# Base port for vLLM servers (GPU 0 = BASE_PORT, GPU 1 = BASE_PORT+1, etc.)
export BASE_PORT=9000

# Host node constraints
export HOST_GPU_CONSTRAINT='A100_80GB|A100_40GB'
export HOST_PARTITION="general"
export HOST_MEM="64GB"

# Analysis node constraints
export ANALYSIS_GPU_CONSTRAINT='6000Ada|A6000|L40S|L40'
export ANALYSIS_PARTITION="array"
export ANALYSIS_MEM="32GB"
export ANALYSIS_CPUS=4

#————————————————————————————————————————
# Job Configuration
#————————————————————————————————————————
export ANALYSIS_TIME="1-23:00:00"
export HOST_TIME="1-23:55:00"
export USER_EMAIL="junkais@andrew.cmu.edu"

#————————————————————————————————————————
# Hardware Configuration
#————————————————————————————————————————
export GPU_MEMORY_UTILIZATION=0.9

#————————————————————————————————————————
# vLLM Server Configuration
#————————————————————————————————————————
export MAX_NUM_BATCHED_TOKENS=65536
export MAX_NUM_SEQS=32
export UVICORN_LOG_LEVEL="warning"
export VLLM_LOG_LEVEL="WARNING"

#————————————————————————————————————————
# Analysis Configuration
#————————————————————————————————————————
export ANALYSIS_SCRIPT="/home/junkais/test/src/analysis/pipeline_analysis/prediction_analysis_parallel.py"
export ANALYSIS_MODE="--baseline --adaptive-conv"
# export ANALYSIS_MODE="--baseline --extended-conv --extended-rounds 100 --track-brier-rounds"

#————————————————————————————————————————
# Environment Configuration
#————————————————————————————————————————
export HF_HOME="/data/user_data/$USER/.hf_cache"
export HF_HUB_CACHE="/data/hf_cache/hub"
export HF_DATASETS_CACHE="/data/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export TMPDIR="${TMPDIR:-/data/user_data/$USER/tmp}"
export MKL_SERVICE_FORCE_INTEL=1

#————————————————————————————————————————
# Logging Configuration
#————————————————————————————————————————
export LOG_DIR="logs"
mkdir -p "$LOG_DIR"

#————————————————————————————————————————
# Safety Configuration
#————————————————————————————————————————
export MAX_WAIT_TIME=600  # Maximum time to wait for vLLM to start (seconds)
export HEALTH_CHECK_INTERVAL=5  # How often to check vLLM health
export STARTUP_STAGGER=0.02  # Delay between starting analysis jobs

#————————————————————————————————————————
# Utility Functions
#————————————————————————————————————————
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_prerequisites() {
    # Check if conda environment exists
    if ! conda env list | grep -q "vllm"; then
        echo "ERROR: vllm conda environment not found"
        echo "Please create it with: conda create -n vllm python=3.10"
        return 1
    fi
    
    # Check if analysis script exists
    if [[ ! -f "$ANALYSIS_SCRIPT" ]]; then
        echo "ERROR: Analysis script not found: $ANALYSIS_SCRIPT"
        return 1
    fi
    
    # Check if log directory is writable
    if [[ ! -w "$LOG_DIR" ]]; then
        echo "ERROR: Cannot write to log directory: $LOG_DIR"
        return 1
    fi
    
    return 0
}