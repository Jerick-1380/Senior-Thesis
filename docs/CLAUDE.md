# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This codebase implements a **multi-agent opinion dynamics simulation system** that studies how AI agents with different initial perspectives converge (or diverge) on controversial topics through conversation. The system runs large-scale simulations where agents engage in back-and-forth discussions and gradually update their opinion "strength" based on token-level probability analysis of their responses.

## Core Architecture

### Main Simulation Entry Points
- **`small_sim.py`** - Main simulation runner for single or parallel simulations
- **`one_agent.py`** - Single-agent analysis tool for testing argument strength distributions
- **`prediction_analysis.py`** - Original prediction analysis for forecasting questions (sequential)
- **`prediction_analysis_parallel.py`** - Optimized parallel prediction analysis with 8-way dataset splitting
- **`aggregate_analysis_results.py`** - Aggregates results from parallel prediction analysis splits

### Agent System (`helpers/bots.py`)
- **`Agent`** - Core agent class that maintains perspective arguments, calculates opinion strength via LLM token probabilities, and engages in conversations
- **`UselessAgent`** - Special agent type designed to distract conversations (for control experiments)
- **`MatchMaker`** - Facilitates agent pairing based on opinion similarity/difference

### Conversation Management (`helpers/conversation.py`)
- **`ConversationCreator`** - Orchestrates multi-round conversations between agent pairs
- **Batch processing** - Uses async batch operations for efficient LLM API calls
- **Memory management** - Agents maintain limited conversation history for context

### Model Interfaces (`helpers/model.py`)
- **`Llama`** - OpenAI-compatible API client for Llama models via vLLM
- **`GPT4o`** - OpenAI GPT-4o client with rate limiting and backoff
- **Token probability analysis** - Core feature for measuring opinion strength

### Opinion Data Structure (`opinions/*.json`)
Each topic has:
- `claims`: Pro/con statements for strength calculation
- `connector`: Template for probability measurement
- `initial_posts`: Seed arguments with pro/con labels
- `intro`: Conversation starter prompt

## Common Commands

### Running Simulations

**Single simulation:**
```bash
python src/simulations/small_sim.py --topic drugs --num_conversations 150 --args_length 4 --init_args 4 --initial_condition moderate --folder results/simulations/test_output
```

**Batch simulations (SLURM):**
```bash
cd scripts && sbatch run_sim_array.sh
```

**Host vLLM model server:**
```bash
cd scripts && sbatch host_models.sh
```

### Analysis and Visualization

**Aggregate simulation results:**
```bash
python src/analysis/aggregate_and_plot_stats.py --folder results/simulations/output5034 --bins 20
```

**Single agent strength distribution:**
```bash
python src/simulations/one_agent.py --topic president --num_conversations 500 --init_args 6
```

**Parallel prediction analysis:**
```bash
# Submit parallel analysis job (8-way dataset splitting)
sbatch scripts/run_analysis_parallel.sh

# Aggregate results when complete
python src/simulations/aggregate_analysis_results.py --input-pattern "analysis_split_*.json" --output-file "final_results.json"
```

### Key Parameters

- `--initial_condition`: Controls agent initialization (`none`, `moderate`, `extreme`, `random`, `bounded`)
- `--epsilon`: Conversation pairing strategy (0=similar agents, 1=random)
- `--args_length`: Maximum arguments each agent can hold
- `--remove_irrelevant`: Whether agents intelligently remove low-relevance arguments
- `--host_model`: URL of the LLM API server

## SLURM Configuration

The system is designed for high-performance computing clusters:
- **`host_models.sh`** - Launches vLLM model servers on GPUs
- **`run_sim_array.sh`** - Runs distributed simulations across compute nodes
- **`run_analysis_parallel.sh`** - Runs parallel prediction analysis across 8 ports
- Load balancing across multiple GPU ports (9000-9007)
- Automatic staggering to optimize vLLM batch utilization

## Parallel Prediction Analysis System

### Overview
The parallel prediction analysis system optimizes forecasting question analysis by:
- **Dataset Splitting**: Questions divided across 8 parallel workers
- **Multi-Port Distribution**: Each worker uses a different vLLM server port (9000-9007)
- **Batching**: Agent predictions processed in batches within each scenario
- **Serial Scenarios**: Different prediction methods run serially but maximize internal parallelism

### Performance Improvements
- **Before**: Sequential processing, single port, individual predictions (~8-12 hours)
- **After**: 8-way parallel processing, batched operations, multi-port usage (~1-2 hours)

### Prediction Methods Supported
- **Baseline**: Community baseline predictions from historical data
- **Basic**: Basic LLM predictions without additional context
- **Argument-based**: Generate arguments first, then make informed predictions
- **Adaptive-Conv**: Use adaptive conversations between agents to develop perspectives
- **Extended-Conv**: Use extended adaptive conversations for deeper perspective development

### Usage Workflow

1. **Submit Parallel Jobs**:
   ```bash
   sbatch scripts/run_analysis_parallel.sh
   ```
   
2. **Monitor Progress**:
   ```bash
   squeue -u $USER
   tail -f logs/analysis_*.log
   ```
   
3. **Aggregate Results**:
   ```bash
   python src/simulations/aggregate_analysis_results.py \
       --input-pattern "analysis_split_*.json" \
       --results-dir "." \
       --output-file "final_prediction_analysis.json"
   ```

### Configuration Options

**Adjust number of parallel jobs** (edit `run_analysis_parallel.sh`):
```bash
#SBATCH --array=0-7%8    # Change to 0-N%M for N+1 jobs, M max concurrent
NUM_PARALLEL=8           # Update accordingly
```

**Modify port configuration**:
```bash
NUM_GPUS=8        # Number of vLLM servers
BASE_PORT=9000    # Starting port number
```

**Select prediction methods**:
```bash
python prediction_analysis_parallel.py \
    --baseline --basic --argument-based --adaptive-conv --extended-conv \
    --split-id 0 --total-splits 8 --model-url "http://babel-5-31:9000/v1"
```

## Opinion Strength Calculation

The core innovation is measuring agent opinion strength through **token-level probability analysis**:

1. Agent builds context from current arguments + claim connector
2. Model calculates probabilities for pro/con completion tokens
3. Strength = P(pro) / (P(pro) + P(con))
4. Range: 0.0 (strongly con) to 1.0 (strongly pro)

## Development Workflow

1. **Test single agents**: Use `one_agent.py` to verify argument strength distributions
2. **Small-scale simulations**: Run `small_sim.py` with low conversation counts
3. **Add new topics**: Create JSON files in `opinions/` following existing structure
4. **Scale up**: Use SLURM scripts for large parallel simulations
5. **Analysis**: Use `aggregate_and_plot_stats.py` and plotting tools in `helpers/graph.py`

## Environment Setup

**Required Dependencies:**
```bash
pip install python-dotenv openai aiohttp asyncio numpy matplotlib networkx imageio tqdm argparse backoff aiolimiter
```

**Environment Configuration:**
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your actual values:
   ```bash
   # Required: OpenAI API key for GPT models
   OPENAI_API_KEY=your_actual_openai_api_key_here
   
   # Optional: Custom cache locations
   HF_HOME=/your/path/to/hf_cache
   TRANSFORMERS_CACHE=/your/path/to/hf_cache/transformers
   
   # Optional: vLLM server configuration
   VLLM_API_KEY=your_vllm_token
   LLM_API_BASE=http://your_server:8000/v1
   ```

3. **Security**: Never commit the `.env` file to git. It's already excluded in `.gitignore`.

## Important Notes

- **Security**: All API keys are now managed via environment variables
- **Model hosting**: Requires vLLM server setup for Llama models or OpenAI API access
- **Memory requirements**: Large simulations need substantial RAM for batch processing
- **Async operations**: Most LLM calls use async/await for performance
- **Data persistence**: Each simulation saves detailed logs as JSON and CSV files

## Output Structure

Simulations generate:
- **Strength trajectories**: `past_strengths_sim_*.json` 
- **Conversation logs**: `simulation_log.csv`
- **Visualizations**: Line plots, variance plots, convergence analysis
- **Summary statistics**: Agent distributions, final states

The system tracks opinion dynamics over time to study phenomena like polarization, consensus formation, and the effect of different conversation strategies on group opinion evolution.