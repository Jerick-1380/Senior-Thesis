#!/bin/bash
#!/bin/bash
# host_llama_dual.sh - Enhanced vLLM hosting with health checks
# This script should be called from main_pipeline.sh which sets up the SLURM headers

nvidia-smi

#————————————————————————————————————————
# Load Configuration
#————————————————————————————————————————
# Configuration should be loaded from environment (set by main_pipeline.sh)
# If not found, try to load it
if [[ -z "$NUM_GPUS" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$SCRIPT_DIR/pipeline_config.sh" ]]; then
        source "$SCRIPT_DIR/pipeline_config.sh"
    else
        echo "ERROR: Configuration not loaded and pipeline_config.sh not found!"
        exit 1
    fi
fi

#————————————————————————————————————————
#  Activate vLLM environment
#————————————————————————————————————————
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# Export environment variables from config
export VLLM_LOG_LEVEL
export HF_HOME
export HF_HUB_CACHE
export HF_DATASETS_CACHE
export HF_HUB_OFFLINE

# Array to store process IDs
declare -a PIDS

# Function to kill all vLLM processes on exit
cleanup() {
    echo "Shutting down vLLM servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
        fi
    done
    wait
    echo "All vLLM servers stopped."
}

# Set trap to cleanup on exit
trap cleanup EXIT SIGINT SIGTERM

# Start vLLM servers
for (( i=0; i<NUM_GPUS; i++ )); do
    export CUDA_VISIBLE_DEVICES=$i
    port=$(( BASE_PORT + i ))
    echo "Starting vLLM on GPU $i → port $port …"

    vllm serve "$MODEL_ID" \
        --port ${port} \
        --dtype ${MODEL_DTYPE} \
        --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --uvicorn-log-level ${UVICORN_LOG_LEVEL} \
        --max-model-len ${MAX_MODEL_LEN} \
        --download-dir ${HF_HUB_CACHE} \
        --disable-log-requests \
        &
    
    PIDS[$i]=$!
done

# Health check loop
echo "Waiting for all vLLM servers to be ready..."
ALL_READY=false
ELAPSED=0

while [[ $ELAPSED -lt $MAX_WAIT_TIME ]]; do
    ALL_READY=true
    
    for (( i=0; i<NUM_GPUS; i++ )); do
        port=$(( BASE_PORT + i ))
        
        # Check if process is still running
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            echo "ERROR: vLLM server on port $port (PID ${PIDS[$i]}) died"
            ALL_READY=false
            break
        fi
        
        # Check health endpoint
        if ! curl -s -f "http://localhost:${port}/health" >/dev/null 2>&1; then
            ALL_READY=false
        fi
    done
    
    if [[ "$ALL_READY" == "true" ]]; then
        echo "All vLLM servers are ready!"
        break
    fi
    
    sleep $HEALTH_CHECK_INTERVAL
    ELAPSED=$((ELAPSED + HEALTH_CHECK_INTERVAL))
done

if [[ "$ALL_READY" != "true" ]]; then
    echo "ERROR: Not all vLLM servers became ready within $MAX_WAIT_TIME seconds"
    exit 1
fi

# Write a marker file to indicate readiness
echo "READY" > /tmp/vllm_ready_${SLURM_JOB_ID}

# Keep the job running
echo "vLLM servers running. Will terminate when analysis jobs complete."
wait