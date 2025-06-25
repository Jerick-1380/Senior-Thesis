# Multi-Agent Opinion Dynamics Simulation

A research platform for studying how AI agents with different initial perspectives converge (or diverge) on controversial topics through conversation.

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r config/requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and paths
   ```

3. **Test with a simple simulation:**
   ```bash
   python src/simulations/small_sim.py --topic drugs --num_conversations 10 --folder results/simulations/test_output
   ```

## Security Notes

- **Never commit `.env` files** - they contain your API keys
- All secrets are managed via environment variables
- The `.gitignore` file excludes sensitive files automatically

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Detailed architecture, usage instructions, and development workflow
- **[PARALLEL_PREDICTION_ANALYSIS.md](PARALLEL_PREDICTION_ANALYSIS.md)** - Comprehensive guide to the parallel prediction analysis system
- **[PARALLEL_ANALYSIS_USAGE.md](PARALLEL_ANALYSIS_USAGE.md)** - Quick usage guide for parallel analysis
- **[DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)** - Project organization and file structure

## Key Features

- **Multi-agent conversations** with opinion strength tracking
- **Token-level probability analysis** for measuring agent beliefs  
- **Distributed computing support** via SLURM
- **Parallel prediction analysis** with 8-way dataset splitting and batching
- **Comprehensive visualization** of opinion dynamics over time
- **Multiple LLM backends** (Llama via vLLM, OpenAI GPT models)

## Quick Examples

```bash
# Run a moderate initialization simulation
python src/simulations/small_sim.py --topic president --initial_condition moderate --num_conversations 100

# Generate argument strength histograms
python src/simulations/one_agent.py --topic drugs --num_conversations 500 --init_args 4

# Run parallel prediction analysis across 8 ports
sbatch scripts/run_analysis_parallel.sh

# Aggregate and plot multiple simulation results  
python src/analysis/aggregate_and_plot_stats.py --folder results/simulations/output5034
```