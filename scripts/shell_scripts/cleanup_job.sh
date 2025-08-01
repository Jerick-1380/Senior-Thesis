#!/bin/bash
#SBATCH --job-name=cleanup_vllm
#SBATCH --partition=general
#SBATCH --time=00:05:00
#SBATCH --output=logs/cleanup_%j.log
#SBATCH --error=logs/cleanup_%j.err
#SBATCH --dependency=afterany:5162917

echo "Cleaning up vLLM host job 5162916"
scancel 5162916 2>/dev/null || true
echo "Cleanup completed"
