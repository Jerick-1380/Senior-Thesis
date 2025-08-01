import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate simulation past_strengths JSON files and plot histograms of the last iteration's agent mean strength and variance."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Parent folder containing simulation JSON files (e.g., 'output5033')."
    )
    parser.add_argument(
        "--bins", type=int, default=10,
        help="Number of bins to use in the histograms."
    )
    return parser.parse_args()

def aggregate_results(parent_folder):
    pattern = os.path.join(parent_folder, "past_strengths_sim_*.json")
    json_files = glob.glob(pattern)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found with pattern {pattern}")
    
    means = []
    variances = []
    
    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
        # Assuming "past_strengths" is a list of lists, one per iteration.
        past_strengths = data.get("past_strengths", [])
        if not past_strengths:
            print(f"Warning: No past_strengths found in {file}. Skipping.")
            continue
        
        # Extract last iteration's strengths (assumed to be a list of floats)
        last_iteration = past_strengths[-1]
        # Compute mean and variance for the simulation
        sim_mean = np.mean(last_iteration)
        sim_var = np.var(last_iteration)
        means.append(sim_mean)
        variances.append(sim_var)
        
    return means, variances

def plot_histogram(data, xlabel, ylabel, title, out_file, bins=10):
    plt.figure()
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(out_file)
    plt.close()
    print(f"Saved histogram to {out_file}")

def main():
    args = parse_args()
    means, variances = aggregate_results(args.folder)
    
    if not means or not variances:
        print("No valid simulation data found for aggregation.")
        return
    
    # Plot histogram for mean strengths
    mean_out = os.path.join(args.folder, "hist_mean_strength.png")
    plot_histogram(
        data=means,
        xlabel="Mean Strength",
        ylabel="Frequency",
        title="Histogram of Last Iteration Mean Strength (across simulations)",
        out_file=mean_out,
        bins=args.bins
    )
    
    # Plot histogram for strength variances
    var_out = os.path.join(args.folder, "hist_variance_strength.png")
    plot_histogram(
        data=variances,
        xlabel="Strength Variance",
        ylabel="Frequency",
        title="Histogram of Last Iteration Strength Variance (across simulations)",
        out_file=var_out,
        bins=args.bins
    )
    
    # Also save aggregated data as master JSON
    master_file = os.path.join(args.folder, "master_past_strengths.json")
    all_files = glob.glob(os.path.join(args.folder, "past_strengths_sim_*.json"))
    aggregated_data = {}
    for file in all_files:
        sim_id = os.path.basename(file).split("_")[-1].split(".")[0]
        with open(file, "r") as f:
            data = json.load(f)
        aggregated_data[sim_id] = data.get("past_strengths", [])
    with open(master_file, "w") as f:
        json.dump(aggregated_data, f, indent=4)
    print(f"Aggregated JSON saved to {master_file}")

if __name__ == "__main__":
    main()