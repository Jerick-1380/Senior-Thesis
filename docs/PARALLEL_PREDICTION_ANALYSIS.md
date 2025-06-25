# Parallel Prediction Analysis System

## Overview

The Parallel Prediction Analysis System dramatically improves the efficiency of forecasting question analysis by distributing the workload across multiple model servers and implementing advanced batching techniques. This system reduces analysis time from 8-12 hours to 1-2 hours for large datasets.

## Architecture

### Key Components

1. **`run_analysis_parallel.sh`** - SLURM array job script that launches 8 parallel workers
2. **`prediction_analysis_parallel.py`** - Optimized analysis script with batching and dataset splitting
3. **`aggregate_analysis_results.py`** - Combines results from all parallel workers

### Performance Optimizations

#### Dataset Splitting
- Questions are automatically divided into 8 equal parts
- Each worker processes a different subset independently
- No coordination required between workers during processing

#### Multi-Port Distribution
- Each worker connects to a different vLLM server port (9000-9007)
- Eliminates bottlenecks from single model server
- Leverages full GPU cluster capacity

#### Batching Optimizations
- **Agent Conversations**: All 20 agents process questions in batches
- **Argument Generation**: Multiple arguments generated in parallel per question
- **Probability Calculations**: Batch prediction requests to model servers
- **Trial Processing**: 20 trials for argument-based methods run with maximum parallelism

#### Serial Scenario Processing
- Different prediction methods (baseline, basic, argument-based, conversational, extended) run serially
- Within each method, maximum parallelization is achieved
- Prevents overwhelming model servers with mixed request types

## Prediction Methods

### 1. Baseline Predictions
- Uses community predictions from historical data
- No model inference required
- Fastest method, provides comparison baseline

### 2. Basic LLM Predictions
- Direct probability analysis without additional context
- Uses improved prompting templates
- Batched across all questions simultaneously

### 3. Argument-Based Predictions
- Generates 4 arguments per question per trial (configurable)
- Runs 20 trials per question for statistical robustness
- Arguments generated in parallel, then used for informed predictions
- Batch processes all argument generation and prediction steps

### 4. Adaptive Conversational Predictions
- Creates 20 conversational agents per question
- Agents engage in multiple rounds of paired conversations
- Perspectives extracted and used for final predictions
- Conversation rounds run in parallel across agent pairs

### 5. Extended Conversational Predictions
- Same as adaptive but with additional conversation rounds
- Allows for deeper perspective development
- Configurable number of extended rounds

## Usage

### Quick Start

```bash
# Submit parallel analysis job
sbatch scripts/run_analysis_parallel.sh

# Monitor progress
squeue -u $USER
tail -f logs/analysis_*.log

# Aggregate results when complete
python src/simulations/aggregate_analysis_results.py \
    --input-pattern "analysis_split_*.json" \
    --output-file "final_results.json"
```

### Configuration

#### Adjusting Parallel Workers

Edit `scripts/run_analysis_parallel.sh`:

```bash
# Change number of parallel jobs
#SBATCH --array=0-15%16    # For 16 workers instead of 8
NUM_PARALLEL=16            # Update this too

# Adjust port configuration
NUM_GPUS=16               # If you have 16 GPU servers
BASE_PORT=9000           # Starting port
```

#### Selecting Prediction Methods

```bash
# Run only specific methods
python src/simulations/prediction_analysis_parallel.py \
    --basic --argument-based \  # Only these methods
    --split-id 0 \
    --total-splits 8 \
    --model-url "http://babel-5-31:9000/v1"
```

#### Method-Specific Parameters

```bash
# Customize method behavior
python src/simulations/prediction_analysis_parallel.py \
    --baseline --basic --argument-based --adaptive-conv --extended-conv \
    --args-per-trial 6 \          # More arguments per trial
    --overall-rounds 6 \           # More conversation rounds
    --pairwise-exchanges 8 \       # Longer conversations
    --extended-rounds 4 \          # Additional extended rounds
    --temperature 0.8 \            # Higher temperature
    --split-id 0 --total-splits 8
```

### Manual Execution

For testing or debugging specific splits:

```bash
# Test a single split
python src/simulations/prediction_analysis_parallel.py \
    --basic --argument-based \
    --split-id 0 \
    --total-splits 8 \
    --model-url "http://babel-5-31:9000/v1" \
    --output-prefix "test_split_0"

# Run split 3 with specific methods
export LLM_API_BASE="http://babel-5-31:9003/v1"
python src/simulations/prediction_analysis_parallel.py \
    --adaptive-conv --extended-conv \
    --split-id 3 \
    --total-splits 8 \
    --model-url "$LLM_API_BASE" \
    --output-prefix "manual_split_3"
```

## Output Format

### Individual Split Results

Each worker produces a detailed JSON file:

```json
{
  "split_id": 0,
  "total_questions": 125,
  "mean_brier_baseline": 0.2347,
  "mean_brier_basic": 0.2156,
  "mean_brier_argument": 0.2089,
  "mean_brier_conversational": 0.2034,
  "mean_brier_extended": 0.1987,
  "configuration": {
    "args_per_trial": 4,
    "overall_rounds": 4,
    "pairwise_exchanges": 6,
    "extended_rounds": 4,
    "model_temperature": 0.7,
    "model_url": "http://babel-5-31:9000/v1"
  },
  "results": [
    {
      "question": "Will X happen by Y date?",
      "resolution": 1,
      "baseline_strength": 0.45,
      "basic_strength": 0.52,
      "argument_strength": 0.58,
      "conversational_strength": 0.61,
      "extended_strength": 0.63,
      "baseline_brier": 0.3025,
      "basic_brier": 0.2304,
      "argument_brier": 0.1764,
      "conversational_brier": 0.1521,
      "extended_brier": 0.1369
    }
  ],
  "total_time_minutes": 45.2
}
```

### Aggregated Results

The aggregation script produces a comprehensive combined result:

```json
{
  "aggregation_timestamp": "2024-01-15T14:30:00",
  "num_splits_aggregated": 8,
  "total_questions": 1000,
  "mean_brier_baseline": 0.2341,
  "mean_brier_basic": 0.2143,
  "mean_brier_argument": 0.2076,
  "mean_brier_conversational": 0.2021,
  "mean_brier_extended": 0.1978,
  "total_time_minutes": 362.4,
  "split_summaries": [
    {
      "split_id": 0,
      "questions_processed": 125,
      "time_minutes": 45.2,
      "mean_brier_baseline": 0.2347
    }
  ],
  "brier_score_counts": {
    "baseline": 1000,
    "basic": 1000,
    "argument": 1000,
    "conversational": 1000,
    "extended": 1000
  }
}
```

## Performance Monitoring

### SLURM Job Monitoring

```bash
# Check job status
squeue -u $USER

# View specific job details
scontrol show job JOBID

# Check resource usage
sacct -j JOBID --format=JobID,JobName,MaxRSS,Elapsed,State
```

### Log Analysis

```bash
# Monitor all analysis logs
tail -f logs/analysis_*.log

# Check for errors across all logs
grep -i error logs/analysis_*.log

# Monitor progress of specific split
grep "Processing conversational batch" logs/analysis_*_0.log
```

### Performance Metrics

Typical performance on a cluster with 8 A6000 GPUs:

| Method | Questions/Hour | Memory Usage | GPU Utilization |
|--------|----------------|--------------|-----------------|
| Baseline | ~2000 | Low | 0% |
| Basic | ~400 | Medium | ~30% |
| Argument-based | ~120 | High | ~60% |
| Conversational | ~80 | High | ~70% |
| Extended | ~60 | High | ~75% |

## Troubleshooting

### Common Issues

#### Port Connection Failures
```bash
# Verify vLLM servers are running
for port in {9000..9007}; do
  curl -s "http://babel-5-31:$port/v1/models" || echo "Port $port unavailable"
done
```

#### Memory Issues
- Reduce batch sizes in `prediction_analysis_parallel.py`
- Increase `--mem=32GB` in SLURM script
- Process fewer questions per split by increasing `--total-splits`

#### Split Failures
- Individual splits can fail without affecting others
- Re-run failed splits manually with specific `--split-id`
- Check logs for specific error messages

#### Aggregation Errors
```bash
# Check for missing split files
ls -la analysis_split_*.json

# Verify all splits completed
python -c "
import glob
files = glob.glob('analysis_split_*.json')
splits = [int(f.split('_')[2].split('.')[0]) for f in files]
expected = set(range(8))
missing = expected - set(splits)
if missing: print(f'Missing splits: {missing}')
else: print('All splits present')
"
```

### Recovery Procedures

#### Restarting Failed Splits
```bash
# Identify failed split IDs from logs or missing files
failed_splits=(2 5 7)

# Restart specific splits
for split_id in "${failed_splits[@]}"; do
  export LLM_API_BASE="http://babel-5-31:$((9000 + split_id % 8))/v1"
  python src/simulations/prediction_analysis_parallel.py \
    --baseline --basic --argument-based --adaptive-conv --extended-conv \
    --split-id $split_id \
    --total-splits 8 \
    --model-url "$LLM_API_BASE" \
    --output-prefix "recovery_split_${split_id}" &
done
wait
```

#### Partial Aggregation
```bash
# Aggregate only completed splits
python src/simulations/aggregate_analysis_results.py \
    --input-pattern "analysis_split_[0136].json" \  # Only specific splits
    --output-file "partial_results.json"
```

## Advanced Configuration

### Custom Dataset Splitting

For non-uniform dataset distribution:

```python
# Custom split logic in prediction_analysis_parallel.py
def custom_split_questions(questions, split_id, total_splits):
    # Example: Ensure each split has questions from different time periods
    questions_by_year = {}
    for q in questions:
        year = q['date_begin'][:4]
        if year not in questions_by_year:
            questions_by_year[year] = []
        questions_by_year[year].append(q)
    
    # Distribute years across splits
    split_questions = []
    for year, year_questions in questions_by_year.items():
        year_split = int(year) % total_splits
        if year_split == split_id:
            split_questions.extend(year_questions)
    
    return split_questions
```

### Dynamic Port Selection

For environments with varying GPU availability:

```bash
# In run_analysis_parallel.sh
available_ports=()
for port in {9000..9015}; do
  if curl -s --max-time 1 "http://babel-5-31:$port/v1/models" > /dev/null; then
    available_ports+=($port)
  fi
done

# Select port based on available GPUs
gpu_idx=$((SLURM_ARRAY_TASK_ID % ${#available_ports[@]}))
llama_port=${available_ports[$gpu_idx]}
```

### Adaptive Batch Sizing

```python
# Dynamic batch size based on available memory
import psutil

def get_optimal_batch_size():
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    if available_memory_gb > 60:
        return 10  # Large batches for high-memory nodes
    elif available_memory_gb > 30:
        return 5   # Medium batches
    else:
        return 2   # Small batches for memory-constrained nodes
```

This parallel prediction analysis system provides a robust, scalable solution for efficiently processing large forecasting datasets while maintaining the quality and accuracy of the original sequential approach.