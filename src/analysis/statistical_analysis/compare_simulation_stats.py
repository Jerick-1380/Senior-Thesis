import os
import json
import glob
import argparse
import numpy as np
from scipy.stats import ttest_ind, levene

def aggregate_results(folder):
    pattern = os.path.join(folder, "past_strengths_sim_*.json")
    json_files = glob.glob(pattern)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found with pattern {pattern}")

    means = []
    variances = []
    all_last_strengths = []  # <-- store all the raw final strengths

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
        all_last_strengths.extend(last_iteration)  # collect all ending strengths
    return means, variances, all_last_strengths

def main():
    parser = argparse.ArgumentParser(
        description="Compare simulation statistics between two folders by testing means and variances."
    )
    parser.add_argument(
        "--folder1", type=str, required=True,
        help="Path to the first folder containing simulation JSON files (e.g., 'output5033')."
    )
    parser.add_argument(
        "--folder2", type=str, required=True,
        help="Path to the second folder containing simulation JSON files."
    )
    parser.add_argument(
        "--output", type=str, default="comparison_results.txt",
        help="Output text file to save the test results."
    )
    args = parser.parse_args()

    # Aggregate data from each folder
    try:
        means1, vars1, last_strengths1 = aggregate_results(args.folder1)
        means2, vars2, last_strengths2 = aggregate_results(args.folder2)
    except FileNotFoundError as e:
        with open(args.output, "w") as f:
            f.write(str(e) + "\n")
        return

    if not means1 or not means2:
        with open(args.output, "w") as f:
            f.write("Error: One of the folders does not contain valid simulation data.\n")
        return

    # Test for differences in means using a two-sample t-test (Welch's t-test)
    t_stat, p_value = ttest_ind(means1, means2, equal_var=False)
    # Test for differences in variances using Levene's test
    lev_stat, lev_p = levene(last_strengths1, last_strengths2)

    # Prepare output message
    output_lines = []
    output_lines.append("Comparison of Simulation Statistics\n")
    output_lines.append("===================================\n\n")
    
    output_lines.append("T-test for Means:\n")
    output_lines.append("  t-statistic: {:.4f}\n".format(t_stat))
    output_lines.append("  p-value: {:.4f}\n\n".format(p_value))
    
    output_lines.append("Levene's Test for Variances:\n")
    output_lines.append("  Statistic: {:.4f}\n".format(lev_stat))
    output_lines.append("  p-value: {:.4f}\n".format(lev_p))
    
    # Write the output to a text file
    with open(args.output, "w") as f:
        f.writelines(output_lines)
    
    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()