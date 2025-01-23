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

# Local imports from 'helpers' directory
from helpers.model import Llama
from helpers.graph import Grapher, Writer
from helpers.bots import UselessAgent, Agent
from helpers.data import ALL_NAMES, PERSONAS
from helpers.conversation import ConversationCreator


def parse_args():
    """
    Parse command-line arguments for customizing the simulation.
    """
    parser = argparse.ArgumentParser(description="Process some files.")

    parser.add_argument('--args_length', type=int, default=3,
                        help="Number of initial argument lines (perspectives) each agent starts with.")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="Temperature setting for the language model.")
    parser.add_argument('--agents', type=int, default=20,
                        help="Number of agents to simulate.")
    parser.add_argument('--folder', type=str, default="output",
                        help="Directory to save outputs (plots, text, etc.).")
    parser.add_argument('--epsilon', type=float, default=1,
                        help="Epsilon parameter for conversation pairing strategy.")
    parser.add_argument('--topic', type=str, default="drugs",
                        help="Topic to discuss.")
    parser.add_argument('--num_conversations', type=int, default=150,
                        help="Number of conversation rounds.")
    parser.add_argument('--num_pairs', type=int, default=1,
                        help="Number of agent pairs to converse each round.")
    parser.add_argument('--num_useless', type=int, default=0,
                        help="Number of agents to designate as 'UselessAgent' types.")

    return parser.parse_args()


if __name__ == "__main__":
    # Disable tqdm progress bars for cleaner console output
    tqdm.disable = True

    # Parse arguments
    args = parse_args()
    topic = args.topic
    num_conversations = args.num_conversations
    num_pairs = args.num_pairs
    host_model = "http://babel-1-31:8082/v1"
    args_length = args.args_length
    num_agents = args.agents

    # Start timing the simulation
    start_time = time.time()

    # Prepare agent lists
    agents = []
    names = sample(ALL_NAMES, num_agents)
    personas = sample(PERSONAS, num_agents)

    # Initialize a primary language model
    basic_model = Llama(
        dir="meta-llama/Meta-Llama-3-8B-Instruct",
        api_base=host_model,
        version=3,
        temperature=args.temperature
    )

    # Load topic-specific data (claims, introduction, etc.) from JSON
    with open(f'opinions/{topic}.json', 'r') as f:
        data = json.load(f)

    # Create agents, some as UselessAgent if specified
    for idx, name in enumerate(names):
        if idx < args.num_useless:
            # Create a UselessAgent
            agents.append(
                UselessAgent(
                    name=name,
                    persona=personas[idx],
                    model=basic_model,
                    topic=data['id'],
                    claims=data['claims'],
                    init_args=[],
                    memory_length=5,
                    args_length=args_length
                )
            )
        else:
            # Create a regular Agent
            random.seed(time.time() % 1 * 10000)
            num_pro = args_length // 2
            num_con = args_length - num_pro
            pros = [item['text'] for item in data['initial_posts'] if item['type'] == 'pro']
            cons = [item['text'] for item in data['initial_posts'] if item['type'] == 'con']

            selected_pros = random.sample(pros, num_pro)
            selected_cons = random.sample(cons, num_con)

            init_args = selected_pros + selected_cons
            random.shuffle(init_args)

            agents.append(
                Agent(
                    name=name,
                    persona=personas[idx],
                    model=basic_model,
                    topic=data['id'],
                    claims=data['claims'],
                    init_args=init_args,
                    memory_length=5,
                    args_length=args_length
                )
            )

    # Write initial description of the agents
    initial_writer = Writer(agents, args.folder)
    initial_writer.output_desc("init_desc.txt")

    # Create a conversation controller and run the simulation
    conversation_creator = ConversationCreator(agents)
    conversation_creator.Converse(
        num_conversations=num_conversations,
        intro=data['intro'],
        claims=data['claims'],
        shuffle=True,
        num_pairs=num_pairs,
        doesChat=True,
        epsilon=args.epsilon
    )

    # Measure total elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Prepare grapher for stats over time
    strength_grapher = Grapher(conversation_creator.past_strengths, args.folder)
    # Re-use writer with updated matrix (strength over time)
    final_writer = Writer(agents, conversation_creator.past_strengths, args.folder)

    # Generate and save visual outputs
    strength_grapher.create_gif()
    strength_grapher.plot_lines(num=5)
    strength_grapher.plot_mean()
    strength_grapher.plot_variance()

    # Plot offsets over time
    off_grapher = Grapher(conversation_creator.past_offs, args.folder)
    off_grapher.plot_mean(save_as="means_off.png")

    # Write final description after simulation
    final_writer.output_desc()
    final_writer.write_csv_log(
        conversations=conversation_creator.conversation_log, 
    )

    final_writer.write_conversation_summaries_json(
        conversations=conversation_creator.conversation_log,
    )

    # Print elapsed time
    print(f"Elapsed time: {elapsed_time} seconds", flush=True)