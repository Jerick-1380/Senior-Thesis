import os

# Set environment variables for HF cache locations
os.environ["HF_HOME"] = "/data/user_data/junkais/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/junkais/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data/user_data/junkais/hf_cache/datasets"
os.environ["HF_METRICS_CACHE"] = "/data/user_data/junkais/hf_cache/metrics"

import time
import numpy as np
import random
from random import sample
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports from 'helpers' directory
from helpers.model import Llama, Llama3_1, GPT4o
from helpers.graph import Grapher, Writer
from helpers.bots import UselessAgent, Agent
from helpers.data import ALL_NAMES, PERSONAS
from helpers.conversation import ConversationCreator


def parse_args():
    """
    Parse command-line arguments for customizing the simulation.
    """
    parser = argparse.ArgumentParser(description="Process some files.")

    parser.add_argument('--args_length', type=int, default=4,
                        help="Number of initial argument lines (perspectives) each agent starts with.")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="Temperature setting for the language model.")
    parser.add_argument('--folder', type=str, default="output",
                        help="Directory to save outputs (plots, text, etc.).")
    parser.add_argument('--topic', type=str, default="drugs",
                        help="Topic to discuss.")
    parser.add_argument('--num_conversations', type=int, default=150,
                        help="Number of conversation rounds.")
    parser.add_argument('--init_args', type=int, default=0,
                        help="How many arguments to start with")

    return parser.parse_args()


if __name__ == "__main__":
    # Disable tqdm progress bars for cleaner console output
    tqdm.disable = True

    # Parse arguments
    args = parse_args()
    topic = args.topic
    num_conversations = args.num_conversations
    host_model = "http://babel-0-35:8082/v1"
    args_length = args.args_length
    init_args = args.init_args

    # Start timing the simulation
    start_time = time.time()

    # Initialize a primary language model
    basic_model = Llama(
        dir="/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct",
        api_base=host_model,
        version=31,
        temperature=args.temperature
    )
    basic_model2 = Llama(
        dir="/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct",
        api_base=host_model,
        version=31,
        temperature=0
    )
    
    basic_model3 = GPT4o(temperature = 0)
    
    # Load topic-specific data (claims, introduction, etc.) from JSON
    with open(f'opinions/{topic}.json', 'r') as f:
        data = json.load(f)
    
    # Collect all strength values across conversations
    all_strengths = []
    
    for i in range(num_conversations):
        num_pro = init_args
        num_con = 0
        pros = [item['text'] for item in data['initial_posts'] if item['type'] == 'pro']
        cons = [item['text'] for item in data['initial_posts'] if item['type'] == 'con']

        selected_pros = random.sample(pros, num_pro)
        selected_cons = random.sample(cons, num_con)

        init_args_list = selected_pros + selected_cons
        random.shuffle(init_args_list)
        
        agent = Agent(
            name="Jimmy",
            persona="",
            model=basic_model3,
            topic=data['id'],
            claims=data['claims'],
            init_args=init_args_list,
            memory_length=5,
            args_length=args_length,
            remove_irrelevant=True,
            extra_desc=""
        )
        
        # Get strength value and add to our collection
        strength_value = agent.calculate_strength(data['claims'], agent.args)
        if(strength_value < 0.1):
            print(f"Agent has strength: {strength_value}")
            print(f"with args: {'\n'.join(agent.args)}")
        all_strengths.append(strength_value)  # Append single value instead of extending
        
        # Print progress
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_conversations} conversations")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_strengths, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Argument Strength')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Argument Strengths for Topic: {topic}')
    plt.grid(alpha=0.3)
    
    # Add statistics
    mean_strength = np.mean(all_strengths)
    median_strength = np.median(all_strengths)
    plt.axvline(mean_strength, color='red', linestyle='dashed', linewidth=1, 
                label=f'Mean: {mean_strength:.2f}')
    plt.axvline(median_strength, color='green', linestyle='dashed', linewidth=1, 
                label=f'Median: {median_strength:.2f}')
    plt.legend()
    
    # Save the histogram
    output_dir = args.folder
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{topic}_strength_histogram.png', dpi=300, bbox_inches='tight')
    
    print(f"Histogram saved to {output_dir}/{topic}_strength_histogram.png")
    print(f"Total samples: {len(all_strengths)}")
    print(f"Mean strength: {mean_strength:.2f}")
    print(f"Median strength: {median_strength:.2f}")
    print(f"Min strength: {min(all_strengths):.2f}")
    print(f"Max strength: {max(all_strengths):.2f}")
    
    # Show the plot
    plt.show()