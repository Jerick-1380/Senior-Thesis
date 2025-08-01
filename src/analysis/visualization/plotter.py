import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_simulation_time_series_mean(folder, output_path):
    pattern = os.path.join(folder, "past_strengths_sim_*.json")
    json_files = glob.glob(pattern)

    if not json_files:
        raise FileNotFoundError(f"No JSON files found with pattern {pattern}")

    all_simulations = []
    plt.figure(figsize=(12, 6))

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
        past_strengths = data.get("past_strengths", [])
        if not past_strengths:
            continue

        avg_strengths = [np.mean(round_strengths) for round_strengths in past_strengths]
        all_simulations.append(avg_strengths)
        plt.plot(avg_strengths, alpha=0.4)

    max_len = max(len(sim) for sim in all_simulations)
    padded = np.array([sim + [np.nan] * (max_len - len(sim)) for sim in all_simulations])
    avg_over_time = np.nanmean(padded, axis=0)

    plt.plot(avg_over_time, color='black', linewidth=2.5, label="Mean across simulations")
    plt.title("Average Agent Strength Over Time per Simulation")
    plt.xlabel("Round")
    plt.ylabel("Average Strength")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")
    
def plot_simulation_time_series_var(folder, output_path):
    pattern = os.path.join(folder, "past_strengths_sim_*.json")
    json_files = glob.glob(pattern)

    if not json_files:
        raise FileNotFoundError(f"No JSON files found with pattern {pattern}")

    all_variance_trajectories = []
    plt.figure(figsize=(12, 6))

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
        past_strengths = data.get("past_strengths", [])
        if not past_strengths:
            continue

        var_strengths = [np.var(round_strengths) for round_strengths in past_strengths]
        all_variance_trajectories.append(var_strengths)
        plt.plot(var_strengths, alpha=0.4)

    max_len = max(len(sim) for sim in all_variance_trajectories)
    padded = np.array([sim + [np.nan] * (max_len - len(sim)) for sim in all_variance_trajectories])
    avg_var_over_time = np.nanmean(padded, axis=0)

    plt.plot(avg_var_over_time, color='black', linewidth=2.5, label="Average variance across simulations")
    plt.title("Variance of Agent Strength Over Time per Simulation")
    plt.xlabel("Round")
    plt.ylabel("Variance of Strength")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")
    
def plot_final_means_boxplot(folders, labels, output_path):
    if len(folders) != len(labels):
        raise ValueError("Number of folders must match number of labels.")

    final_means_by_folder = []

    for folder in folders:
        pattern = os.path.join(folder, "past_strengths_sim_*.json")
        json_files = glob.glob(pattern)

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in folder {folder}")

        final_means = []
        for file in json_files:
            with open(file, "r") as f:
                data = json.load(f)
            past_strengths = data.get("past_strengths", [])
            if not past_strengths:
                continue
            final_round = past_strengths[-1]
            final_means.append(np.mean(final_round))
        
        final_means_by_folder.append(final_means)

    plt.figure(figsize=(10, 6))
    plt.boxplot(final_means_by_folder, labels=labels)
    plt.title("Distribution of Final Mean Strengths per Folder")
    plt.xlabel("Simulation Group")
    plt.ylabel("Final Mean Strength")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Boxplot saved to {output_path}")
    
def plot_final_means_subplots(folder_groups, topic_labels, state_labels, output_path):
    """
    folder_groups: List of lists. Each sublist has 3 folders (None, Moderate, Extreme) for one topic.
    topic_labels: List of topic names, e.g., ["God", "Science", "Policy"]
    state_labels: List of state names, e.g., ["None", "Moderate", "Extreme"]
    """
    num_topics = len(folder_groups)
    fig, axes = plt.subplots(1, num_topics, figsize=(5 * num_topics, 6), sharey=True)

    if num_topics == 1:
        axes = [axes]  # Ensure axes is always iterable

    colors = ['gray', 'blue', 'red']

    for idx, (folders, topic, ax) in enumerate(zip(folder_groups, topic_labels, axes)):
        topic_data = []

        for folder in folders:
            pattern = os.path.join(folder, "past_strengths_sim_*.json")
            json_files = glob.glob(pattern)
            final_means = []

            for file in json_files:
                with open(file, "r") as f:
                    data = json.load(f)
                past_strengths = data.get("past_strengths", [])
                if past_strengths:
                    final_round = past_strengths[-1]
                    final_means.append(np.mean(final_round))

            topic_data.append(final_means)

        bplot = ax.boxplot(topic_data, patch_artist=True, tick_labels=state_labels)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(topic, fontsize=14, fontweight='bold', pad=20)  # Push the title higher
        ax.set_xlabel("Initialization State")
        ax.grid(axis='y')
        ax.tick_params(axis='x', rotation=20)

        if idx == 0:
            ax.set_ylabel("Final Mean Opinion Strength")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Subplot figure saved to {output_path}")
    
def plot_final_means_subplots_extra(folder_groups, topic_labels, state_labels, extra_folders_and_labels, output_path):
    """
    folder_groups: List of lists. Each sublist has 3 folders (None, Moderate, Extreme) for one topic.
    topic_labels: List of topic names, e.g., ["God", "Science", "Policy"]
    state_labels: List of state names, e.g., ["None", "Moderate", "Extreme"]
    extra_folders_and_labels: List of tuples: [(folder_path, topic_name), ...] for extra single-state topics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import glob
    import json

    num_main_topics = len(folder_groups)
    num_extra = len(extra_folders_and_labels)
    total_plots = num_main_topics + num_extra

    fig, axes = plt.subplots(1, total_plots, figsize=(5 * total_plots, 6), sharey=True)

    if total_plots == 1:
        axes = [axes]

    colors = ['gray', 'blue', 'red']

    # Plot regular topics
    for idx, (folders, topic, ax) in zip(range(num_main_topics), zip(folder_groups, topic_labels, axes)):
        topic_data = []

        for folder in folders:
            pattern = os.path.join(folder, "past_strengths_sim_*.json")
            json_files = glob.glob(pattern)
            final_means = []

            for file in json_files:
                with open(file, "r") as f:
                    data = json.load(f)
                past_strengths = data.get("past_strengths", [])
                if past_strengths:
                    final_round = past_strengths[-1]
                    final_means.append(np.mean(final_round))

            topic_data.append(final_means)

        bplot = ax.boxplot(topic_data, patch_artist=True, labels=state_labels)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(topic, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Initialization State")
        ax.grid(axis='y')
        ax.tick_params(axis='x', rotation=20)
        if idx == 0:
            ax.set_ylabel("Final Mean Opinion Strength")

    # Plot extra topics
    for i, (folder, topic) in enumerate(extra_folders_and_labels):
        ax = axes[num_main_topics + i]

        pattern = os.path.join(folder, "past_strengths_sim_*.json")
        json_files = glob.glob(pattern)
        final_means = []

        for file in json_files:
            with open(file, "r") as f:
                data = json.load(f)
            past_strengths = data.get("past_strengths", [])
            if past_strengths:
                final_round = past_strengths[-1]
                final_means.append(np.mean(final_round))

        bplot = ax.boxplot([final_means], patch_artist=True, labels=["None"])
        bplot['boxes'][0].set_facecolor('purple')
        bplot['boxes'][0].set_alpha(0.6)

        ax.set_title(topic, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Initialization State")
        ax.grid(axis='y')
        ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Subplot figure saved to {output_path}")
    
    
def plot_final_vars_subplots(folder_groups, topic_labels, state_labels, output_path):
    """
    folder_groups: List of lists. Each sublist has 3 folders (None, Moderate, Extreme) for one topic.
    topic_labels: List of topic names, e.g., ["God", "Science", "Policy"]
    state_labels: List of state names, e.g., ["None", "Moderate", "Extreme"]
    """
    num_topics = len(folder_groups)
    fig, axes = plt.subplots(1, num_topics, figsize=(5 * num_topics, 6), sharey=True)

    if num_topics == 1:
        axes = [axes]  # Ensure axes is always iterable

    colors = ['gray', 'blue', 'red']

    for idx, (folders, topic, ax) in enumerate(zip(folder_groups, topic_labels, axes)):
        topic_data = []

        for folder in folders:
            pattern = os.path.join(folder, "past_strengths_sim_*.json")
            json_files = glob.glob(pattern)
            final_means = []

            for file in json_files:
                with open(file, "r") as f:
                    data = json.load(f)
                past_strengths = data.get("past_strengths", [])
                if past_strengths:
                    final_round = past_strengths[-1]
                    final_means.append(np.var(final_round))

            topic_data.append(final_means)

        bplot = ax.boxplot(topic_data, patch_artist=True, labels=state_labels)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(topic, fontsize=14, fontweight='bold', pad=20)  # Push the title higher
        ax.set_xlabel("Initialization State")
        ax.grid(axis='y')
        ax.tick_params(axis='x', rotation=20)

        if idx == 0:
            ax.set_ylabel("Final Mean Opinion Strength")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Subplot figure saved to {output_path}")

  
def plot_extra_single_boxplots(extra_folders_and_labels, output_path):
    """
    extra_folders_and_labels: List of tuples (folder_path, topic_name)
    output_path: Path to save the new figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import glob
    import json

    num_plots = len(extra_folders_and_labels)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), sharey=True)

    if num_plots == 1:
        axes = [axes]

    for ax, (folder, topic) in zip(axes, extra_folders_and_labels):
        pattern = os.path.join(folder, "past_strengths_sim_*.json")
        json_files = glob.glob(pattern)
        final_means = []

        for file in json_files:
            with open(file, "r") as f:
                data = json.load(f)
            past_strengths = data.get("past_strengths", [])
            if past_strengths:
                final_round = past_strengths[-1]
                final_means.append(np.mean(final_round))

        bplot = ax.boxplot([final_means], patch_artist=True, labels=["Single"])
        bplot['boxes'][0].set_facecolor('purple')
        bplot['boxes'][0].set_alpha(0.6)

        ax.set_title(topic, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Single Init")
        ax.grid(axis='y')
        ax.tick_params(axis='x', rotation=20)

    axes[0].set_ylabel("Final Mean Opinion Strength")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Separate single boxplots figure saved to {output_path}")

if __name__ == "__main__":
    #plot_simulation_time_series_mean("output7003", "lk99_moderate.png")
    #plot_simulation_time_series_mean("output7004", "lk99_none.png")
    
    plot_simulation_time_series_mean("output7007", "lk99_human_filtered.png")
    plot_simulation_time_series_mean("output7008", "lk99_llm_filtered.png")