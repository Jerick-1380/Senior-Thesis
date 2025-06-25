import os
import json
import glob
import argparse
import numpy as np

def aggregate_results(folder):
    pattern = os.path.join(folder, "past_strengths_sim_*.json")
    json_files = glob.glob(pattern)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found with pattern {pattern}")

    means = []
    variances = []

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
        past_strengths = data.get("past_strengths", [])
        if not past_strengths:
            print(f"Warning: No past_strengths found in {file}. Skipping.")
            continue
        last_iteration = past_strengths[-1]
        sim_mean = np.mean(last_iteration)
        sim_var = np.var(last_iteration)
        means.append(sim_mean)
        variances.append(sim_var)

    return means, variances

def main():
    parser = argparse.ArgumentParser(description="Summarize simulation results from multiple folders.")
    parser.add_argument(
        "--folders", nargs="+", required=True,
        help="List of folders to process (each should contain JSON simulation output files)."
    )
    parser.add_argument(
        "--output", type=str, default="summary_results.txt",
        help="Output file to write the summary statistics."
    )
    args = parser.parse_args()

    output_lines = []
    output_lines.append("Summary of Simulation Folder Statistics\n")
    output_lines.append("=======================================\n\n")

    for folder in args.folders:
        try:
            means, variances = aggregate_results(folder)
        except FileNotFoundError as e:
            output_lines.append(f"{folder} - ERROR: {str(e)}\n\n")
            continue

        if not means:
            output_lines.append(f"{folder} - WARNING: No valid simulation data.\n\n")
            continue

        overall_mean_of_means = np.mean(means)
        overall_var_of_means = np.var(means)
        overall_mean_of_vars = np.mean(variances)
        overall_var_of_vars = np.var(variances)

        output_lines.append(f"Folder: {folder}\n")
        output_lines.append(f"  Mean of final means: {overall_mean_of_means:.4f}\n")
        output_lines.append(f"  Variance of final means: {overall_var_of_means:.4f}\n")
        output_lines.append(f"  Mean of final variances: {overall_mean_of_vars:.4f}\n")
        output_lines.append(f"  Variance of final variances: {overall_var_of_vars:.4f}\n\n")

    with open(args.output, "w") as f:
        f.writelines(output_lines)

    print(f"Summary written to {args.output}")

if __name__ == "__main__":
    main()