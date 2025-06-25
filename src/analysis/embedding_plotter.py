import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.decomposition import PCA
import imageio
import math

def load_json_vectors(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def calculate_semantic_convergence(vectors):
    """
    Calculate semantic convergence by measuring pairwise cosine similarities
    between all argument vectors and taking the mean.
    Higher values = more convergence (agents using similar arguments)
    """
    if len(vectors) < 2:
        return 0.0
    
    # Normalize all vectors
    normalized_vecs = np.array([normalize(v) for v in vectors])
    
    # Calculate pairwise cosine similarities
    similarity_matrix = cosine_similarity(normalized_vecs)
    
    # Get upper triangle (excluding diagonal) to avoid double counting
    n = len(vectors)
    upper_triangle_indices = np.triu_indices(n, k=1)
    pairwise_similarities = similarity_matrix[upper_triangle_indices]
    
    # Return mean pairwise similarity
    return np.mean(pairwise_similarities)

def calculate_opinion_convergence(opinion_scores):
    """
    Opinion convergence = 1 - normalized_variance
    Range: 0 to 1, where 1 = perfect convergence (no variance)
    """
    if len(opinion_scores) < 2:
        return 1.0
    
    variance = np.var(opinion_scores)
    
    # Normalize by maximum possible variance for your data range
    # If opinion scores are 0-1: max_var = 0.25 (when half are 0, half are 1)
    # If opinion scores are 1-5: max_var = 4 (when some are 1, some are 5)
    max_var = 0.25  # Adjust based on your actual data range
    
    return 1 - min(variance / max_var, 1)

def plot_semantic_vs_opinion_convergence(embedding_data, opinion_data, save_path, suffix=""):
    plt.figure(figsize=(12, 8))
    
    all_semantic_convergence = []
    all_opinion_convergence = []
    
    for sim_key in sorted(embedding_data.keys()):
        if sim_key not in opinion_data:
            continue
            
        semantic_conv_over_time = []
        opinion_conv_over_time = []
        
        rounds = sorted(embedding_data[sim_key].keys(), key=int)
        
        for round_num in rounds:
            vectors = np.array(embedding_data[sim_key][round_num])
            if vectors.size > 0:
                semantic_conv = calculate_semantic_convergence(vectors)
                semantic_conv_over_time.append(semantic_conv)
            else:
                semantic_conv_over_time.append(0.0)
            
            if round_num in opinion_data[sim_key]:
                opinions = opinion_data[sim_key][round_num]
                opinion_conv = calculate_opinion_convergence(opinions)
                opinion_conv_over_time.append(opinion_conv)
            else:
                opinion_conv_over_time.append(0.0)
        
        all_semantic_convergence.append(semantic_conv_over_time)
        all_opinion_convergence.append(opinion_conv_over_time)
        plt.plot(semantic_conv_over_time, alpha=0.3, color='blue')
        plt.plot(opinion_conv_over_time, alpha=0.3, color='red')
    
    mean_semantic = np.mean(all_semantic_convergence, axis=0)
    mean_opinion = np.mean(all_opinion_convergence, axis=0)
    
    plt.plot(mean_semantic, color='blue', linewidth=3, label='Semantic Convergence')
    plt.plot(mean_opinion, color='red', linewidth=3, label='Opinion Convergence')
    
    plt.xlabel("Round")
    plt.ylabel("Convergence Score")
    plt.title("Semantic vs Opinion Convergence Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"semantic_vs_opinion_convergence{('_' + suffix) if suffix else ''}.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.close()
    
    return mean_semantic, mean_opinion

def plot_pairwise_similarity_heatmap(data, save_path, sim_key="sim_0", rounds_to_plot=[1, 100, 250, 500], suffix=""):
    if sim_key not in data:
        print(f"[ERROR] Simulation key '{sim_key}' not found in data.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, round_num in enumerate(rounds_to_plot[:4]):
        round_str = str(round_num)
        if round_str not in data[sim_key]:
            continue
            
        vectors = np.array(data[sim_key][round_str])
        if vectors.size == 0:
            continue
        
        normalized_vecs = np.array([normalize(v) for v in vectors])
        similarity_matrix = cosine_similarity(normalized_vecs)
        
        im = axes[i].imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(f"Round {round_num} - Pairwise Similarities")
        axes[i].set_xlabel("Argument Index")
        axes[i].set_ylabel("Argument Index")
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    filename = f"{sim_key}_pairwise_similarity_heatmaps{('_' + suffix) if suffix else ''}.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

def analyze_convergence_timing(semantic_convergence, opinion_convergence, threshold=0.1):
    """
    Analyze which type of convergence happens first
    
    Args:
        semantic_convergence: Array of semantic convergence scores over time
        opinion_convergence: Array of opinion convergence scores over time
        threshold: Minimum change to consider as "convergence started"
    
    Returns:
        Dictionary with timing analysis results
    """
    # Find when each type of convergence starts (first significant increase)
    semantic_start = None
    opinion_start = None
    
    for i in range(1, len(semantic_convergence)):
        if semantic_start is None and semantic_convergence[i] - semantic_convergence[0] > threshold:
            semantic_start = i
        if opinion_start is None and opinion_convergence[i] - opinion_convergence[0] > threshold:
            opinion_start = i
    
    results = {
        'semantic_start_round': semantic_start,
        'opinion_start_round': opinion_start,
        'semantic_leads': semantic_start is not None and (opinion_start is None or semantic_start < opinion_start),
        'opinion_leads': opinion_start is not None and (semantic_start is None or opinion_start < semantic_start),
        'final_semantic_convergence': semantic_convergence[-1],
        'final_opinion_convergence': opinion_convergence[-1]
    }
    
    return results

def main(json_file, image_folder, opinion_file=None, image_name_suffix=""):
    embedding_data = load_json_vectors(json_file)
    
    if opinion_file and os.path.exists(opinion_file):
        opinion_data = load_json_vectors(opinion_file)
        
        semantic_conv, opinion_conv = plot_semantic_vs_opinion_convergence(
            embedding_data, opinion_data, image_folder, suffix=image_name_suffix
        )
        
        timing_results = analyze_convergence_timing(semantic_conv, opinion_conv)
        print("Convergence Timing Analysis:")
        print(f"Semantic convergence starts at round: {timing_results['semantic_start_round']}")
        print(f"Opinion convergence starts at round: {timing_results['opinion_start_round']}")
        print(f"Semantic leads: {timing_results['semantic_leads']}")
        print(f"Opinion leads: {timing_results['opinion_leads']}")
    else:
        print("Opinion data not provided - generating semantic convergence analysis only")
        
        # (You can optionally suffix the semantic-only plot too)
        # ...

    # Heatmap with suffix
    plot_pairwise_similarity_heatmap(embedding_data, image_folder, sim_key="sim_0", suffix=image_name_suffix)

if __name__ == "__main__":
    import sys
    DEFAULT_DIR = "/data/user_data/junkais"

    json_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DEFAULT_DIR, "all_simulation_vectors_free.json")
    image_folder = sys.argv[2] if len(sys.argv) > 2 else "images"
    opinion_file = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DEFAULT_DIR, "opinion_scores.json")
    image_name_suffix = sys.argv[4] if len(sys.argv) > 4 else ""

    os.makedirs(image_folder, exist_ok=True)
    main(json_file, image_folder, opinion_file, image_name_suffix)