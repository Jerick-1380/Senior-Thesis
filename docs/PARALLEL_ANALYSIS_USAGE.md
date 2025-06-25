# Parallel Prediction Analysis Usage Guide

## Overview

The optimized prediction analysis system splits your dataset across 8 different model servers/ports and uses advanced batching to dramatically improve efficiency. This follows the same parallelization patterns as `run_sim_array.sh` and `small_sim.py`.

## Key Optimizations

1. **Dataset Splitting**: Questions are divided across 8 parallel workers
2. **Multi-Port Usage**: Each worker uses a different vLLM server port (9000-9007)
3. **Batching**: Agent predictions are batched together within each scenario
4. **Serial Scenarios**: Different prediction methods run serially, but maximize parallelism within each

## Files Created

1. `scripts/run_analysis_parallel.sh` - SLURM batch script for parallel execution
2. `src/simulations/prediction_analysis_parallel.py` - Optimized analysis script
3. `src/simulations/aggregate_analysis_results.py` - Results aggregation script

## Usage

### Step 1: Submit Parallel Jobs

```bash
cd /home/junkais/test
sbatch scripts/run_analysis_parallel.sh
```

This will launch 8 parallel jobs, each processing a different subset of your questions.

### Step 2: Wait for Completion

Monitor job progress:
```bash
squeue -u $USER
```

Check logs:
```bash
tail -f logs/analysis_*.log
```

### Step 3: Aggregate Results

Once all splits complete:
```bash
cd /home/junkais/test
python src/simulations/aggregate_analysis_results.py \
    --input-pattern "analysis_split_*.json" \
    --results-dir "." \
    --output-file "final_prediction_analysis.json"
```

## Configuration

### Adjusting Number of Parallel Jobs

Edit `scripts/run_analysis_parallel.sh`:
- Change `--array=0-7%8` to `--array=0-N%M` where N+1 is total jobs, M is max concurrent
- Update `NUM_PARALLEL=8` accordingly

### Modifying Port Configuration

Edit the port settings in `run_analysis_parallel.sh`:
```bash
NUM_GPUS=8        # Number of vLLM servers
BASE_PORT=9000    # Starting port number
```

### Prediction Method Selection

The parallel script supports the same flags as the original:
- `--baseline` - Community baseline predictions
- `--basic` - Basic LLM predictions  
- `--argument-based` - Argument-informed predictions
- `--adaptive-conv` - Conversational agent predictions
- `--extended-conv` - Extended conversational predictions

## Performance Benefits

**Before (Sequential)**:
- Single port usage
- One question at a time
- Individual agent predictions
- Estimated time: ~8-12 hours for full dataset

**After (Parallel)**:
- 8-port distribution
- Batched processing
- Parallel agent conversations
- Estimated time: ~1-2 hours for full dataset

## Example Output

Each split will produce files like:
```
analysis_split_0_baseline_basic_arg_adaptive_extended_args4_rounds4_exchanges6_extrounds4_temp0.7_20241225_143022.json
analysis_split_1_baseline_basic_arg_adaptive_extended_args4_rounds4_exchanges6_extrounds4_temp0.7_20241225_143025.json
...
```

The aggregation script combines these into:
```
aggregated_prediction_analysis_20241225_150000.json
```

## Troubleshooting

### Port Connection Issues
- Verify all 8 vLLM servers are running on ports 9000-9007
- Check `LLM_API_BASE` environment variable in logs

### Memory Issues  
- Reduce batch sizes in the parallel script
- Adjust `--mem=32GB` in the SLURM script

### Split Failures
- Individual splits can fail without affecting others
- Re-run failed splits manually with specific `--split-id`

## Manual Execution

For testing or manual runs:
```bash
# Run a specific split
python src/simulations/prediction_analysis_parallel.py \
    --baseline --basic --argument-based \
    --split-id 0 \
    --total-splits 8 \
    --model-url "http://babel-5-31:9000/v1" \
    --output-prefix "test_split_0"
```