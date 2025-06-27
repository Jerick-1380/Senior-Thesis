#!/bin/bash
# main_pipeline.sh - Automated pipeline for vLLM hosting and analysis

#————————————————————————————————————————
# Load Configuration
#————————————————————————————————————————
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/pipeline_config.sh" ]]; then
    source "$SCRIPT_DIR/pipeline_config.sh"
else
    echo "ERROR: pipeline_config.sh not found!"
    exit 1
fi

# Validate configuration
if ! check_prerequisites; then
    exit 1
fi

#————————————————————————————————————————
# Helper Functions
#————————————————————————————————————————
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

cleanup_jobs() {
    log "Cleaning up jobs..."
    if [[ -n "$HOST_JOB_ID" ]]; then
        scancel $HOST_JOB_ID 2>/dev/null || true
    fi
    if [[ -n "$ANALYSIS_JOB_ID" ]]; then
        scancel $ANALYSIS_JOB_ID 2>/dev/null || true
    fi
}

# Don't cleanup automatically - let the cleanup job handle it
# trap cleanup_jobs EXIT

#————————————————————————————————————————
# Step 1: Submit vLLM hosting job
#————————————————————————————————————————
log "Submitting vLLM hosting job..."

# Create temporary host script with current configuration
TEMP_HOST_SCRIPT=$(mktemp /tmp/host_llama_XXXXXX.sh)
cat > "$TEMP_HOST_SCRIPT" << EOF
#!/bin/bash
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --constraint='${HOST_GPU_CONSTRAINT}'
#SBATCH --partition=${HOST_PARTITION}
#SBATCH --mem=${HOST_MEM}
#SBATCH --time=${HOST_TIME}
#SBATCH --job-name=host_llama_dual
#SBATCH --error=${LOG_DIR}/host_llama_dual.err
#SBATCH --output=${LOG_DIR}/host_llama_dual.out

# Copy the improved host script content with configuration
$(cat host_llama_dual.sh | sed '1,/^nvidia-smi/d')
EOF

chmod +x "$TEMP_HOST_SCRIPT"
HOST_JOB_ID=$(sbatch --parsable "$TEMP_HOST_SCRIPT")
rm -f "$TEMP_HOST_SCRIPT"
log "Host job submitted with ID: $HOST_JOB_ID"

#————————————————————————————————————————
# Step 2: Get host node (will be determined by SLURM dependency)
#————————————————————————————————————————
log "Host job submitted, analysis will start after host job begins running..."
HOST_NODE="\${SLURM_JOB_NODELIST}"  # This will be resolved in the analysis job

#————————————————————————————————————————
# Step 3: Wait for host job to start, then monitor for readiness
#————————————————————————————————————————
log "Waiting for host job to start..."
while ! squeue -j $HOST_JOB_ID -h -o "%t" 2>/dev/null | grep -q "R"; do
    sleep 5
done

log "Host job is running. Monitoring for vLLM readiness..."
LOG_FILE="${LOG_DIR}/host_llama_dual.out"
MAX_WAIT=600  # 10 minutes max wait
ELAPSED=0

# Wait for log file to be created
while [[ ! -f "$LOG_FILE" && $ELAPSED -lt $MAX_WAIT ]]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [[ ! -f "$LOG_FILE" ]]; then
    log "ERROR: Log file not found after $MAX_WAIT seconds"
    exit 1
fi

log "Log file found. Monitoring for throughput metrics..."
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    # Check if we can find the throughput line in the log
    if tail -n 20 "$LOG_FILE" | grep -q "Avg prompt throughput:.*tokens/s.*Avg generation throughput:.*tokens/s"; then
        log "vLLM servers appear ready based on throughput metrics!"
        break
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    log "WARNING: Timeout waiting for throughput metrics. Proceeding anyway..."
fi

log "Submitting analysis job..."

# Create a wrapper script that includes the host node
cat > run_analysis_wrapper.sh << EOF
#!/bin/bash
#SBATCH --job-name=parallel_analysis
#SBATCH --partition=array
#SBATCH --gres=gpu:1
#SBATCH --constraint='6000Ada|A6000|L40S|L40'
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=${ANALYSIS_TIME}
#SBATCH --output=logs/analysis_%A_%a.log
#SBATCH --error=logs/analysis_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER_EMAIL}
#SBATCH --array=0-$((NUM_GPUS-1))%${NUM_PARALLEL}

export TMPDIR=\${TMPDIR:-/data/user_data/\$USER/tmp}
export MKL_SERVICE_FORCE_INTEL=1

# Get the host node from the host job
HOST_JOB_ID=${HOST_JOB_ID}
VLLM_HOST_NODE=\$(squeue -j \$HOST_JOB_ID -h -o "%N" 2>/dev/null | head -1)

if [[ -z "\$VLLM_HOST_NODE" ]]; then
    echo "ERROR: Could not determine host node for job \$HOST_JOB_ID"
    exit 1
fi

echo "Using vLLM host node: \$VLLM_HOST_NODE"

# Wait for vLLM servers to be ready
NUM_GPUS=${NUM_GPUS}
BASE_PORT=${BASE_PORT}
MAX_WAIT=300
ELAPSED=0

echo "Giving host job time to initialize vLLM servers..."
sleep 120

echo "Waiting for vLLM servers to be ready on \$VLLM_HOST_NODE..."
while [[ \$ELAPSED -lt \$MAX_WAIT ]]; do
    ALL_READY=true
    for (( i=0; i<NUM_GPUS; i++ )); do
        PORT=\$(( BASE_PORT + i ))
        if ! timeout 5 curl -s "http://\${VLLM_HOST_NODE}:\${PORT}/health" >/dev/null 2>&1; then
            ALL_READY=false
            break
        fi
    done
    
    if [[ "\$ALL_READY" == "true" ]]; then
        echo "All vLLM servers ready!"
        break
    fi
    
    sleep 10
    ELAPSED=\$((ELAPSED + 10))
done

if [[ "\$ALL_READY" != "true" ]]; then
    echo "ERROR: vLLM servers not ready within \$MAX_WAIT seconds"
    exit 1
fi

# Rest of analysis script
NUM_PARALLEL=${NUM_PARALLEL}

SLOT=\$(( SLURM_ARRAY_TASK_ID % NUM_PARALLEL ))
sleep \$(awk "BEGIN {printf \"%.3f\", \$SLOT * 0.02}")

GPU_IDX=\$(( SLURM_ARRAY_TASK_ID % NUM_GPUS ))
LLAMA_PORT=\$(( BASE_PORT + GPU_IDX ))

echo "Analysis job \${SLURM_ARRAY_TASK_ID} (slot \${SLOT}) → using GPU \${GPU_IDX}, vLLM port \${LLAMA_PORT}"

export LLM_API_BASE="http://\${VLLM_HOST_NODE}:\${LLAMA_PORT}/v1"

python /home/junkais/test/src/simulations/prediction_analysis_parallel.py \\
    --baseline --adaptive-conv\\
    --split-id \${SLURM_ARRAY_TASK_ID} \\
    --total-splits \${NUM_PARALLEL} \\
    --model-url "\${LLM_API_BASE}" \\
    --output-prefix "analysis_split_\${SLURM_ARRAY_TASK_ID}"
EOF

chmod +x run_analysis_wrapper.sh

ANALYSIS_JOB_ID=$(sbatch --parsable --dependency=after:$HOST_JOB_ID run_analysis_wrapper.sh)
log "Analysis job submitted with ID: $ANALYSIS_JOB_ID"

#————————————————————————————————————————
# Step 4: Submit cleanup job
#————————————————————————————————————————
log "Submitting cleanup job..."

cat > cleanup_job.sh << EOF
#!/bin/bash
#SBATCH --job-name=cleanup_vllm
#SBATCH --partition=general
#SBATCH --time=00:05:00
#SBATCH --output=logs/cleanup_%j.log
#SBATCH --error=logs/cleanup_%j.err
#SBATCH --dependency=afterany:${ANALYSIS_JOB_ID}

echo "Cleaning up vLLM host job ${HOST_JOB_ID}"
scancel ${HOST_JOB_ID} 2>/dev/null || true
echo "Cleanup completed"
EOF

chmod +x cleanup_job.sh
CLEANUP_JOB_ID=$(sbatch --parsable cleanup_job.sh)
log "Cleanup job submitted with ID: $CLEANUP_JOB_ID"

#————————————————————————————————————————
# Summary
#————————————————————————————————————————
log "Pipeline submitted successfully!"
log "  Host Job ID: $HOST_JOB_ID (Node: $HOST_NODE)"
log "  Analysis Job ID: $ANALYSIS_JOB_ID"
log "  Cleanup Job ID: $CLEANUP_JOB_ID"
log ""
log "Monitor progress with:"
log "  squeue -u $USER"
log "  tail -f logs/host_llama_dual.out"
log "  tail -f logs/analysis_*.log"