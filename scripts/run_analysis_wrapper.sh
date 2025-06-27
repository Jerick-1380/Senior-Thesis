#!/bin/bash
#SBATCH --job-name=parallel_analysis
#SBATCH --partition=array
#SBATCH --gres=gpu:1
#SBATCH --constraint='6000Ada|A6000|L40S|L40'
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=1-23:00:00
#SBATCH --output=logs/analysis_%A_%a.log
#SBATCH --error=logs/analysis_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=junkais@andrew.cmu.edu
#SBATCH --array=0-7%8

export TMPDIR=${TMPDIR:-/data/user_data/$USER/tmp}
export MKL_SERVICE_FORCE_INTEL=1

# Get the host node from the host job
HOST_JOB_ID=5162916
VLLM_HOST_NODE=$(squeue -j $HOST_JOB_ID -h -o "%N" 2>/dev/null | head -1)

if [[ -z "$VLLM_HOST_NODE" ]]; then
    echo "ERROR: Could not determine host node for job $HOST_JOB_ID"
    exit 1
fi

echo "Using vLLM host node: $VLLM_HOST_NODE"

# Wait for vLLM servers to be ready
NUM_GPUS=8
BASE_PORT=9000
MAX_WAIT=300
ELAPSED=0

echo "Giving host job time to initialize vLLM servers..."
sleep 120

echo "Waiting for vLLM servers to be ready on $VLLM_HOST_NODE..."
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    ALL_READY=true
    for (( i=0; i<NUM_GPUS; i++ )); do
        PORT=$(( BASE_PORT + i ))
        if ! timeout 5 curl -s "http://${VLLM_HOST_NODE}:${PORT}/health" >/dev/null 2>&1; then
            ALL_READY=false
            break
        fi
    done
    
    if [[ "$ALL_READY" == "true" ]]; then
        echo "All vLLM servers ready!"
        break
    fi
    
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

if [[ "$ALL_READY" != "true" ]]; then
    echo "ERROR: vLLM servers not ready within $MAX_WAIT seconds"
    exit 1
fi

# Rest of analysis script
NUM_PARALLEL=8

SLOT=$(( SLURM_ARRAY_TASK_ID % NUM_PARALLEL ))
sleep $(awk "BEGIN {printf \"%.3f\", $SLOT * 0.02}")

GPU_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_GPUS ))
LLAMA_PORT=$(( BASE_PORT + GPU_IDX ))

echo "Analysis job ${SLURM_ARRAY_TASK_ID} (slot ${SLOT}) → using GPU ${GPU_IDX}, vLLM port ${LLAMA_PORT}"

export LLM_API_BASE="http://${VLLM_HOST_NODE}:${LLAMA_PORT}/v1"

python /home/junkais/test/src/simulations/prediction_analysis_parallel.py \
    --baseline --adaptive-conv\
    --split-id ${SLURM_ARRAY_TASK_ID} \
    --total-splits ${NUM_PARALLEL} \
    --model-url "${LLM_API_BASE}" \
    --output-prefix "analysis_split_${SLURM_ARRAY_TASK_ID}"
