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
#SBATCH --array=0-7%8          # 8 parallel jobs, one per port

export TMPDIR=${TMPDIR:-/data/user_data/$USER/tmp}
export MKL_SERVICE_FORCE_INTEL=1

# Configuration - adjust these if you change GPUs/ports
NUM_GPUS=8        # how many vLLM servers/GPU ports you have
BASE_PORT=9000    # port for GPU 0; GPU 1 will be BASE_PORT+1, etc.
NUM_PARALLEL=8    # how many array jobs run concurrently

# Stagger launch times to avoid overwhelming vLLM
SLOT=$(( SLURM_ARRAY_TASK_ID % NUM_PARALLEL ))
sleep $(awk "BEGIN {printf \"%.3f\", $SLOT * 0.02}")

# Pick which GPU/port to hit, based on (task_id mod NUM_GPUS)
GPU_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_GPUS ))
LLAMA_PORT=$(( BASE_PORT + GPU_IDX ))

echo "Analysis job ${SLURM_ARRAY_TASK_ID} (slot ${SLOT}) → using GPU ${GPU_IDX}, vLLM port ${LLAMA_PORT}"

# Point prediction_analysis_parallel.py at the chosen vLLM instance
export LLM_API_BASE="http://babel-15-24:${LLAMA_PORT}/v1"

#python /home/junkais/test/src/analysis/pipeline_analysis/prediction_analysis_parallel.py \
 #   --baseline --extended-conv \
 #   --extended-rounds 100 \
 #   --track-brier-rounds \
 #   --split-id ${SLURM_ARRAY_TASK_ID} \
 #   --total-splits ${NUM_PARALLEL} \
 #   --model-url "${LLM_API_BASE}" \
 #   --output-prefix "analysis_split_f100_${SLURM_ARRAY_TASK_ID}"




# Run the parallel prediction analysis with dataset split
python /home/junkais/test/src/analysis/pipeline_analysis/prediction_analysis_parallel.py \
    --baseline --adaptive-conv \
   --split-id ${SLURM_ARRAY_TASK_ID} \
    --total-splits ${NUM_PARALLEL} \
    --model-url "${LLM_API_BASE}" \
    --output-prefix "analysis_split_${SLURM_ARRAY_TASK_ID}"