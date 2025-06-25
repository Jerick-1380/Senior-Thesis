# Directory Structure

This document describes the organized file structure of the multi-agent opinion dynamics simulation repository.

## 🏗️ **Clean Directory Structure**

```
/home/junkais/test/
├── src/                          # Source code
│   ├── simulations/              # Main simulation scripts
│   │   ├── small_sim.py         # Primary multi-agent simulation runner
│   │   ├── one_agent.py         # Single-agent analysis tool
│   │   ├── prediction_analysis.py        # Original prediction analysis
│   │   ├── prediction_analysis_parallel.py # Optimized parallel prediction analysis
│   │   └── aggregate_analysis_results.py   # Aggregates parallel analysis results
│   ├── analysis/                 # Analysis and plotting scripts
│   │   ├── aggregate_and_plot_stats.py
│   │   ├── argument_analysis.py
│   │   ├── compare_simulation_stats.py
│   │   ├── embedding_creator.py
│   │   ├── embedding_plotter.py
│   │   ├── image_compiler.py
│   │   ├── plotter.py
│   │   └── summarize_simulation_stats.py
│   └── utils/                    # Utility scripts
│       ├── combine_files.py
│       ├── temp.py
│       └── test_cuda.py
├── scripts/                      # SLURM and execution scripts
│   ├── run_sim_array.sh         # SLURM array job script
│   ├── run_analysis.sh          # Original analysis job script
│   ├── run_analysis_parallel.sh # Optimized parallel analysis script
│   ├── host_models.sh           # Model server hosting script
│   ├── host_llama_dual.sh       # Dual Llama server script
│   └── gpu_usage.sh             # GPU monitoring script
├── config/                       # Configuration files
│   ├── topics/                  # Topic configuration files
│   │   ├── drugs.json
│   │   ├── god.json
│   │   ├── president.json
│   │   ├── temperature.json
│   │   └── [other topic files]
│   └── requirements.txt         # Python dependencies
├── results/                      # All simulation outputs
│   ├── simulations/             # Simulation output directories
│   │   ├── output5034/
│   │   ├── output7000-7013/
│   │   └── old_output/
│   ├── analysis/                # Analysis results
│   │   ├── argument_survival_results_*/
│   │   └── individual_files/    # Standalone result files
│   └── visualizations/          # All plots and images
│       └── individual_plots/    # Standalone plot files
├── helpers/                      # Core library modules (unchanged)
│   ├── bots.py                  # Agent classes and behaviors
│   ├── conversation.py          # Conversation management
│   ├── model.py                 # LLM interface classes
│   ├── graph.py                 # Graphing utilities
│   ├── data.py                  # Data handling utilities
│   └── prompt_template.py       # Prompt template management
├── logs/                         # Execution logs (unchanged)
├── data/                         # Raw data and temporary files
├── docs/                         # Documentation
│   ├── README.md               # Project documentation
│   ├── CLAUDE.md               # Claude-specific documentation
│   ├── PARALLEL_PREDICTION_ANALYSIS.md # Comprehensive parallel analysis guide
│   ├── PARALLEL_ANALYSIS_USAGE.md      # Quick parallel analysis usage guide
│   └── DIRECTORY_STRUCTURE.md  # This file
├── .env.example                  # Environment variable template
└── .gitignore                    # Git ignore file
```

## 📂 **Key Changes Made**

### ✅ **Organized by Function**
- **Source code** separated by purpose (simulations, analysis, utilities)
- **Scripts** consolidated in one location
- **Results** organized by type (simulations, analysis, visualizations)
- **Configuration** centralized in config directory

### ✅ **Import Compatibility**
- Added `sys.path` modifications to maintain import functionality
- Updated all file path references in code
- Tested import functionality from new locations

### ✅ **Documentation Updated**
- All example commands updated for new structure
- Clear instructions for running simulations and analysis
- Path references corrected throughout

## 🚀 **Usage Examples**

**Run a simulation:**
```bash
python src/simulations/small_sim.py --topic drugs --num_conversations 150
```

**Run analysis:**
```bash
python src/analysis/aggregate_and_plot_stats.py --folder results/simulations/output7003
```

**Submit SLURM simulation job:**
```bash
cd scripts && sbatch run_sim_array.sh
```

**Submit parallel prediction analysis job:**
```bash
cd scripts && sbatch run_analysis_parallel.sh
```

**Aggregate parallel analysis results:**
```bash
python src/simulations/aggregate_analysis_results.py --input-pattern "analysis_split_*.json"
```

## 🔍 **Benefits of New Structure**

1. **Easier Navigation** - Logical grouping of related files
2. **Cleaner Root Directory** - Only essential config files at top level
3. **Better Version Control** - Organized .gitignore patterns
4. **Scalable** - Easy to add new scripts in appropriate folders
5. **Professional** - Follows standard project organization conventions