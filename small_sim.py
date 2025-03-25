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
from helpers.model import Llama, Llama3_1
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
    parser.add_argument('--init_args', type=int, default=0,
                        help="How many arguments to start with")
    parser.add_argument('--initial_condition', type=str, default="random",
                    help="Type of initial opinion condition: none, random, structured, extreme, moderate")

    return parser.parse_args()


if __name__ == "__main__":
    # Disable tqdm progress bars for cleaner console output
    tqdm.disable = True

    # Parse arguments
    args = parse_args()
    topic = args.topic
    num_conversations = args.num_conversations
    num_pairs = args.num_pairs
    host_model = "http://babel-1-23:8082/v1"
    args_length = args.args_length
    num_agents = args.agents
    init_args = args.init_args
    initial_condition = args.initial_condition

    # Start timing the simulation
    start_time = time.time()

    # Prepare agent lists
    agents = []
    names = sample(ALL_NAMES, num_agents)
    personas = sample(PERSONAS, num_agents)

    # Initialize a primary language model
    basic_model = Llama(
        dir="/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct",
        api_base=host_model,
        version=31,
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
            if initial_condition == "random":
                random.seed(time.time() % 1 * 10000)
                arguments = [item['text'] for item in data['initial_posts']]
                init_args_list = random.sample(arguments, init_args)
                random.shuffle(init_args_list)
                new_agent = Agent(
                    name=name,
                    persona=personas[idx],
                    model=basic_model,
                    topic=data['id'],
                    claims=data['claims'],
                    init_args=init_args_list,
                    memory_length=5,
                    args_length=args_length,
                    remove_irrelevant=True,
                    extra_desc=""
                )
            elif initial_condition == "none":
                new_agent = Agent(
                    name=name,
                    persona=personas[idx],
                    model=basic_model,
                    topic=data['id'],
                    claims=data['claims'],
                    init_args=[],
                    memory_length=5,
                    args_length=args_length,
                    remove_irrelevant=True,
                    extra_desc=""
                )
            elif initial_condition == "moderate":
                # Loop until the created agent has strength in [0.4, 0.6]
                while True:
                    arguments = [item['text'] for item in data['initial_posts']]
                    init_args_list = random.sample(arguments, init_args)
                    random.shuffle(init_args_list)
                    temp_agent = Agent(
                        name=name,
                        persona=personas[idx],
                        model=basic_model,
                        topic=data['id'],
                        claims=data['claims'],
                        init_args=init_args_list,
                        memory_length=5,
                        args_length=args_length,
                        remove_irrelevant=True,
                        extra_desc=""
                    )
                    if 0.4 <= temp_agent.strength <= 0.6:
                        new_agent = temp_agent
                        break
            elif initial_condition == "extreme":
                # Determine half of the agents (based on their index among regular agents)
                num_regular = len(names) - args.num_useless
                if (idx - args.num_useless) < num_regular / 2:
                    # First half: force strength < 0.2 (more negative)
                    while True:
                        arguments = [item['text'] for item in data['initial_posts']]
                        init_args_list = random.sample(arguments, init_args)
                        random.shuffle(init_args_list)
                        temp_agent = Agent(
                            name=name,
                            persona=personas[idx],
                            model=basic_model,
                            topic=data['id'],
                            claims=data['claims'],
                            init_args=init_args_list,
                            memory_length=5,
                            args_length=args_length,
                            remove_irrelevant=True,
                            extra_desc=""
                        )
                        if temp_agent.strength < 0.2:
                            new_agent = temp_agent
                            break
                else:
                    # Second half: force strength > 0.8 (more positive)
                    while True:
                        arguments = [item['text'] for item in data['initial_posts']]
                        init_args_list = random.sample(arguments, init_args)
                        random.shuffle(init_args_list)
                        temp_agent = Agent(
                            name=name,
                            persona=personas[idx],
                            model=basic_model,
                            topic=data['id'],
                            claims=data['claims'],
                            init_args=init_args_list,
                            memory_length=5,
                            args_length=args_length,
                            remove_irrelevant=True,
                            extra_desc=""
                        )
                        if temp_agent.strength > 0.8:
                            new_agent = temp_agent
                            break
            # Append the newly created Agent
            agents.append(new_agent)


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
    
    #print("Running group discussion among all agents...")
    #conversation_creator.GroupConverse(
    #    k = 5,
     #   num_conversations=num_conversations,
     #   init_prompt=data['intro'],
     #   claims=data['claims'],
     #   conversation_length=6  # You can adjust the number of turns here.
    #)

    # Measure total elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Prepare grapher for stats over time
    strength_grapher = Grapher(conversation_creator.past_strengths, args.folder)
    # Re-use writer with updated matrix (strength over time)
    final_writer = Writer(agents, conversation_creator.past_strengths, args.folder)

    # Generate and save visual outputs
    #strength_grapher.create_gif()
    strength_grapher.plot_lines(num=5)
    strength_grapher.plot_mean()
    strength_grapher.plot_mean_with_rolling()
    strength_grapher.plot_variance()
    strength_grapher.plot_convergence()

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
    for agent in agents:
        print(f"{'\n'.join(agent.args)}")
    print(f"Elapsed time: {elapsed_time} seconds", flush=True)