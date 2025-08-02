# Senior Thesis - LLM Analysis Framework

This repository contains a comprehensive framework for analyzing Large Language Model behavior, argument tracking, and conversational dynamics.

## Structure

- `src/core/` - Core functionality (bots, conversation handling, models)
- `src/analysis/` - Analysis tools and scripts
  - `embedding_analysis/` - Argument and embedding analysis
  - `statistical_analysis/` - Statistical processing tools  
  - `visualization/` - Plotting and visualization tools
  - `pipeline_analysis/` - Analysis pipeline components
- `scripts/` - Analysis scripts and shell utilities
  - `analysis_results/` - Generated analysis results
  - `shell_scripts/` - Shell scripts for pipeline execution
- `docs/` - Documentation
- `analysis_outputs/` - Combined analysis results and statistics

## Requirements

See `requirements.txt` for Python dependencies.

## Usage

### Running the Main Simulation

The main simulation pipeline is orchestrated by `scripts/shell_scripts/main_pipeline.sh`, which:

1. **Sets up vLLM hosting** - Launches multiple GPU instances for model serving
2. **Runs parallel analysis** - Executes prediction analysis across multiple workers
3. **Manages resources** - Handles job scheduling and cleanup automatically

**Quick Start:**
```bash
# Navigate to the shell scripts directory
cd scripts/shell_scripts/

# Run the main pipeline (uses default configuration)
./main_pipeline.sh
```

**Configuration:**
- Edit `scripts/pipeline_config.sh` to customize:
  - Model settings (MODEL_ID, MAX_MODEL_LEN)
  - GPU allocation (NUM_GPUS, NUM_PARALLEL)
  - SLURM constraints (partitions, memory, time limits)
  - Analysis parameters

**Prerequisites:**
- SLURM cluster environment
- GPU nodes with vLLM capabilities
- Environment variables: OPENAI_API_KEY (if using OpenAI models)

**Monitoring:**
The pipeline creates monitoring jobs that track progress and handle cleanup automatically.

### Other Analysis Tools

Refer to the documentation in the `docs/` directory for additional analysis scripts and detailed usage instructions.