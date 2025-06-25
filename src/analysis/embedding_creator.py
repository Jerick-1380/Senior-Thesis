import os
import json
import pandas as pd
import ast
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

OUTPUT_DIR = "/data/user_data/junkais"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_opinion_scores(row) -> List[float]:
    try:
        agentA_strength = float(row["agentA_strength"])
        agentB_strength = float(row["agentB_strength"])
        return [agentA_strength, agentB_strength]
    except Exception as e:
        print(f"[WARN] Failed to parse opinion scores: {e}")
        return []

def process_opinion_scores(folder_path: str, output_file: str = "opinion_scores.json"):
    all_results = {}
    sim_folders = sorted([
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("sim_")
    ])

    for sim_folder in tqdm(sim_folders, desc="Processing simulations for opinion scores"):
        sim_id = int(sim_folder.split("_")[1])
        sim_key = f"sim_{sim_id}"
        sim_file = os.path.join(folder_path, sim_key, "simulation_log.csv")

        if not os.path.exists(sim_file):
            print(f"[WARN] File not found: {sim_file}")
            continue

        try:
            df = pd.read_csv(sim_file)
        except Exception as e:
            print(f"[ERROR] Failed to read {sim_file}: {e}")
            continue

        round_numbers = sorted(df["round"].unique())
        rounds = {}

        for round_num in tqdm(round_numbers, desc=f"{sim_key} opinions", leave=False):
            round_df = df[df["round"] == round_num]
            all_opinion_scores = []
            
            for _, row in round_df.iterrows():
                opinion_scores = extract_opinion_scores(row)
                if opinion_scores:
                    all_opinion_scores.extend(opinion_scores)

            rounds[str(round_num)] = all_opinion_scores

        all_results[sim_key] = rounds

    # Save to JSON in new directory
    output_path = os.path.join(OUTPUT_DIR, output_file)
    with open(output_path, "w") as f:
        json.dump(all_results, f)
    print(f"[INFO] Opinion scores saved to {output_path}")

def extract_arguments(row) -> List[str]:
    try:
        args_A = ast.literal_eval(row["agentA_args"])
        args_B = ast.literal_eval(row["agentB_args"])
        return args_A + args_B
    except Exception as e:
        print(f"[WARN] Failed to parse arguments: {e}")
        return []

def clean_input(args: List[str]) -> List[str]:
    return [str(a) for a in args if isinstance(a, str) and a.strip()]

def process_simulations(folder_path: str):
    all_results = {}
    sim_folders = sorted([
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("sim_")
    ])

    for sim_folder in tqdm(sim_folders, desc="Processing simulations"):
        sim_id = int(sim_folder.split("_")[1])
        sim_key = f"sim_{sim_id}"
        sim_file = os.path.join(folder_path, sim_key, "simulation_log.csv")

        if not os.path.exists(sim_file):
            print(f"[WARN] File not found: {sim_file}")
            continue

        try:
            df = pd.read_csv(sim_file)
        except Exception as e:
            print(f"[ERROR] Failed to read {sim_file}: {e}")
            continue

        round_numbers = sorted(df["round"].unique())
        rounds = {}

        for round_num in tqdm(round_numbers, desc=f"{sim_key}", leave=False):
            round_df = df[df["round"] == round_num]
            all_args = []
            for _, row in round_df.iterrows():
                all_args.extend(extract_arguments(row))

            texts = clean_input(all_args)
            if not texts:
                rounds[str(round_num)] = []
                continue

            vectors = model.encode(texts).tolist()
            rounds[str(round_num)] = vectors

            vecs_np = np.array(vectors)
            mean_vec = np.mean(vecs_np, axis=0)
            squared_dists = np.sum((vecs_np - mean_vec) ** 2, axis=1)
            l2_variance = np.mean(squared_dists)

        all_results[sim_key] = rounds

    output_path = os.path.join(OUTPUT_DIR, "all_simulation_vectors_free.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f)
    print(f"[INFO] Simulation vectors saved to {output_path}")

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "simulations_folder"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "opinion_scores.json"
    process_simulations(folder)
    process_opinion_scores(folder, output_file)