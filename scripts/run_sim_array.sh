#!/bin/bash
#SBATCH --job-name=small_sims
#SBATCH --partition=array
#SBATCH --gres=gpu:1                        
#SBATCH --constraint='6000Ada|A6000|L40S|L40'           # each sim job gets one GPU slot, but note: the Python code does NOT use that GPU for model inference (we’re still talking to vLLM over HTTP)
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=1-23:00:00
#SBATCH --output=logs/sim_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@domain.com
#SBATCH --array=0-49%8          # 8 total array tasks, at most 8 running concurrently

export TMPDIR=${TMPDIR:-/data/user_data/$USER/tmp}
export MKL_SERVICE_FORCE_INTEL=1

# —————————————————————————————————————————————
#  Configuration: adjust these two if you ever change GPUs/ports
# —————————————————————————————————————————————
NUM_GPUS=8        # ← how many vLLM servers/GPU ports you have
BASE_PORT=9000    # ← port for GPU 0; GPU 1 will be BASE_PORT+1, etc.
NUM_PARALLEL=8    # ← how many array jobs run concurrently (the “%8” in #SBATCH --array)

# —————————————————————————————————————————————
#  1) Stagger “launch” times within each group of NUM_PARALLEL
#     so that vLLM can fill large internal batches instead of spiking
# —————————————————————————————————————————————
SLOT=$(( SLURM_ARRAY_TASK_ID % NUM_PARALLEL ))
sleep $(awk "BEGIN {printf \"%.3f\", SLOT * 0.02}")

# —————————————————————————————————————————————
#  2) Pick which GPU/port to hit, based on (task_id mod NUM_GPUS)
# —————————————————————————————————————————————
GPU_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_GPUS ))
LLAMA_PORT=$(( BASE_PORT + GPU_IDX ))

echo "Job ${SLURM_ARRAY_TASK_ID} (slot ${SLOT}) → using GPU ${GPU_IDX}, vLLM port ${LLAMA_PORT}"

# —————————————————————————————————————————————
#  3) Point small_sim.py at the chosen vLLM instance
# —————————————————————————————————————————————
export LLM_API_BASE="http://babel-15-20:${LLAMA_PORT}/v1"


python ../src/simulations/small_sim.py \
    --folder "../results/simulations/output7013" \
    --sim_id ${SLURM_ARRAY_TASK_ID} \
    --epsilon 1 \
    --num_conversations 500 \
    --args_length 4 \
    --init_args 4 \
    --topic president \
    --initial_condition moderate \
    --num_pairs 10 \
    --host_model "${LLM_API_BASE}"

python ../src/simulations/small_sim.py \
    --folder "../results/simulations/output7014" \
    --sim_id ${SLURM_ARRAY_TASK_ID} \
    --epsilon 1 \
    --num_conversations 500 \
    --args_length 4 \
    --init_args 4 \
    --topic president \
    --initial_condition none \
    --num_pairs 10 \
    --host_model "${LLM_API_BASE}"


