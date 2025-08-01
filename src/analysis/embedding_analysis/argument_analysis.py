import pandas as pd
import numpy as np
import os
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime
import argparse


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class ArgumentSurvivalTracker:
    def __init__(self, output_folder="output6000", similarity_threshold=0.8):
        self.output_folder = output_folder
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.results_folder = f"argument_survival_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_folder, exist_ok=True)

    def parse_arguments(self, arg_string):
        try:
            return ast.literal_eval(arg_string)
        except (ValueError, SyntaxError):
            return [arg_string.strip("[]'\"")]

    def argument_still_present(self, original_arg, current_args):
        for current_arg in current_args:
            if original_arg.strip() == current_arg.strip():
                return True, 1.0, current_arg
        return False, 0.0, None

    def load_simulation_data(self, sim_folder):
        csv_path = os.path.join(self.output_folder, sim_folder, "simulation_log.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found")
            return None
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return None

    def extract_arguments_by_round(self, df):
        arguments_by_round = {}
        for round_num in df['round'].unique():
            round_data = df[df['round'] == round_num]
            all_args = []
            for _, row in round_data.iterrows():
                all_args.extend(self.parse_arguments(row['agentA_args']))
                all_args.extend(self.parse_arguments(row['agentB_args']))
            arguments_by_round[round_num] = all_args
        return arguments_by_round

    def track_argument_survival_single_sim(self, sim_folder):
        df = self.load_simulation_data(sim_folder)
        if df is None:
            return None
        arguments_by_round = self.extract_arguments_by_round(df)

        survival_data = {}
        seen_args = set()
        local_id = 0

        for round_num in sorted(arguments_by_round.keys()):
            for original_arg in arguments_by_round[round_num]:
                unique_key = original_arg.strip().lower()
                key = (unique_key, sim_folder)
                if key in seen_args:
                    continue  # deduplicate within simulation
                seen_args.add(key)

                survival_data_key = f"{sim_folder}_arg_{local_id}"
                local_id += 1

                survival_data[survival_data_key] = {
                    'text_key': unique_key,
                    'simulation': sim_folder,
                    'argument_id': local_id,
                    'origin_round': round_num,
                    'original_argument': original_arg,
                    'survival_rounds': 0,
                    'replacement_round': None,
                    'survived_full_experiment': False,
                    'final_similarity': 0.0
                }
                for r in sorted(arguments_by_round.keys()):
                    if r <= round_num:
                        continue
                    current_args = arguments_by_round[r]
                    still_present, sim, _ = self.argument_still_present(original_arg, current_args)
                    if still_present:
                        survival_data[survival_data_key]['survival_rounds'] = r - round_num
                    else:
                        survival_data[survival_data_key]['replacement_round'] = r
                        break
                final_round = max(arguments_by_round.keys())
                final_args = arguments_by_round[final_round]
                still_present, sim, _ = self.argument_still_present(original_arg, final_args)
                survival_data[survival_data_key]['final_similarity'] = sim
                if still_present:
                    survival_data[survival_data_key]['survival_rounds'] = final_round - round_num
                    survival_data[survival_data_key]['survived_full_experiment'] = True

        return survival_data

    def track_all_simulations(self):
        all_survival_data = {}
        sim_folders = sorted([f for f in os.listdir(self.output_folder) if f.startswith('sim_')],
                             key=lambda x: int(x.split('_')[1]))
        for sim_folder in sim_folders:
            sim_data = self.track_argument_survival_single_sim(sim_folder)
            if sim_data:
                all_survival_data.update(sim_data)
        return all_survival_data

    def analyze_survival_statistics(self, survival_data):
        grouped = defaultdict(dict)  # text_key -> sim_id -> survival_rounds

        for data in survival_data.values():
            key = data['text_key']
            sim_id = data['simulation']
            grouped[key][sim_id] = data['survival_rounds']

        avg_survivals = {
            k: np.mean(list(sim_dict.values())) for k, sim_dict in grouped.items()
        }
        sorted_avg = sorted(avg_survivals.items(), key=lambda x: -x[1])

        df = pd.DataFrame([
            {
                'argument': k,
                'average_survival_rounds': round(avg, 2),
                'appearances': len(grouped[k])
            }
            for k, avg in sorted_avg
        ])
        df.to_csv(os.path.join(self.results_folder, 'aggregated_argument_survival.csv'), index=False)

        plt.figure(figsize=(12, 6))
        plt.hist(list(avg_survivals.values()), bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribution of Average Survival Rounds')
        plt.xlabel('Average Survival Rounds')
        plt.ylabel('Number of Unique Arguments')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.results_folder, 'aggregated_survival_distribution.png'), dpi=300)
        plt.close()

        # --- New Survival Trend by Simulation ---
        sim_rounds = defaultdict(list)

        for sim_id in set(data['simulation'] for data in survival_data.values()):
            sim_args = [data for data in survival_data.values() if data['simulation'] == sim_id]
            max_round = max(data['origin_round'] + data['survival_rounds'] for data in sim_args)
            survivors_by_round = []
            original_total = len(sim_args)

            for r in range(1, max_round + 2):
                survivors = sum(1 for d in sim_args if d['origin_round'] + d['survival_rounds'] + 1 >= r)
                survivors_by_round.append(survivors / original_total)

            sim_rounds[sim_id] = survivors_by_round

        max_len = max(len(v) for v in sim_rounds.values())
        all_rounds_matrix = []

        plt.figure(figsize=(12, 6))
        for sim_id, trend in sim_rounds.items():
            padded = trend + [trend[-1]] * (max_len - len(trend))
            all_rounds_matrix.append(padded)
            plt.plot(range(1, len(padded)+1), padded, alpha=0.3)

        avg_curve = np.mean(all_rounds_matrix, axis=0)
        plt.plot(range(1, len(avg_curve)+1), avg_curve, color='black', linewidth=2, label='Average')
        plt.title('Proportion of Original Arguments Surviving by Round (per simulation)')
        plt.xlabel('Round')
        plt.ylabel('Proportion Remaining')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.results_folder, 'per_sim_survival_trend.png'), dpi=300)
        plt.close()
        
        round0_grouped = defaultdict(dict)

        for data in survival_data.values():
            if data['origin_round'] != 0:
                continue
            key = data['text_key']
            sim_id = data['simulation']
            round0_grouped[key][sim_id] = data['survival_rounds']

        round0_avg_survivals = {
            k: np.mean(list(sim_dict.values())) for k, sim_dict in round0_grouped.items()
        }
        round0_sorted_avg = sorted(round0_avg_survivals.items(), key=lambda x: -x[1])

        df_round0 = pd.DataFrame([
            {
                'argument': k,
                'average_survival_rounds': round(avg, 2),
                'appearances': len(round0_grouped[k])
            }
            for k, avg in round0_sorted_avg
        ])
        df_round0.to_csv(os.path.join(self.results_folder, 'aggregated_argument_survival_round0_only.csv'), index=False)

        # Optional: plot for round 0 only
        plt.figure(figsize=(12, 6))
        plt.hist(list(round0_avg_survivals.values()), bins=20, color='salmon', edgecolor='black')
        plt.title('Distribution of Avg Survival Rounds (Round 0 Arguments Only)')
        plt.xlabel('Average Survival Rounds')
        plt.ylabel('Number of Round 0 Arguments')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.results_folder, 'round0_survival_distribution.png'), dpi=300)
        plt.close()

    def save_detailed_results(self, survival_data):
        path = os.path.join(self.results_folder, 'detailed_argument_survival.csv')
        records = [
            {
                'argument_key': k,
                'simulation': v['simulation'],
                'argument_id': v['argument_id'],
                'origin_round': v['origin_round'],
                'original_argument': v['original_argument'],
                'survival_rounds': v['survival_rounds'],
                'replacement_round': v['replacement_round'],
                'survived_full_experiment': v['survived_full_experiment'],
                'final_similarity': v['final_similarity']
            } for k, v in survival_data.items()
        ]
        pd.DataFrame(records).to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Track and analyze argument survival across simulations.")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Name of the output folder containing sim_x subdirectories (e.g., output7000)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Name of the output results folder (e.g., survival_results_drugs_none)"
    )
    args = parser.parse_args()

    tracker = ArgumentSurvivalTracker(output_folder=args.folder, similarity_threshold=0.8)
    tracker.results_folder = args.output_name
    os.makedirs(tracker.results_folder, exist_ok=True)

    survival_data = tracker.track_all_simulations()
    if not survival_data:
        print("No survival data collected.")
        return
    tracker.save_detailed_results(survival_data)
    tracker.analyze_survival_statistics(survival_data)


if __name__ == "__main__":
    main()

