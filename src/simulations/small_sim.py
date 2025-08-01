import os
import sys
import asyncio
import json
import glob

# Add the project root to Python path to find helpers
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for HF cache locations (use env vars with fallbacks)
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/data/user_data/junkais/hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/data/user_data/junkais/hf_cache/transformers")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE", "/data/user_data/junkais/hf_cache/datasets")
os.environ["HF_METRICS_CACHE"] = os.getenv("HF_METRICS_CACHE", "/data/user_data/junkais/hf_cache/metrics")

# OpenAI API Key - MUST be set in environment or .env file
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file or environment.")

import time
import numpy as np
import random
import argparse
from tqdm import tqdm
from random import sample

# Local imports from 'helpers' directory
from src.core.model import Llama, Llama3_1, GPT4o
from src.core.graph import Grapher, Writer
from src.core.bots import UselessAgent, Agent
from src.core.data import ALL_NAMES, PERSONAS
from src.core.conversation import ConversationCreator

def parse_args():
    """
    Parse command-line arguments for customizing the simulation.
    """
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--args_length', type=int, default=3,
                        help="Number of initial argument lines (perspectives) each agent starts with.")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Temperature setting for the language model.")
    parser.add_argument('--agents', type=int, default=20,
                        help="Number of agents to simulate.")
    parser.add_argument('--folder', type=str, default="output",
                        help="Parent directory to save outputs (plots, text, etc.).")
    parser.add_argument('--sim_id', type=int, default=0, help="Simulation ID (e.g., SLURM_ARRAY_TASK_ID)")
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
    parser.add_argument('--num_parallel', type=int, default=1,
                        help="Number of parallel simulation runs")
    parser.add_argument('--target_strength', type=float, default=None,
                        help="Target strength for 'bounded' initial condition (e.g., 0.4)")
    parser.add_argument(
    '--host_model',
    type=str,
    default="http://localhost:8000/v1",
    help="Base URL for the LLM server (e.g. http://localhost:8000/v1)"
)
    

    return parser.parse_args()


async def run_simulation(sim_id, args, parent_folder):
    # Disable tqdm progress bars for cleaner console output
    tqdm.disable = True
    
    # Create a subfolder for this simulation under the parent folder.
    simulation_folder = os.path.join(parent_folder, f"sim_{sim_id}")
    os.makedirs(simulation_folder, exist_ok=True)
    # Update the folder in args for this simulation only.
    args.folder = simulation_folder

    # --- Use the already-parsed args; do not re-call parse_args() here ---
    topic = args.topic
    num_conversations = args.num_conversations
    num_pairs = args.num_pairs
    host_model = args.host_model
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
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key=os.getenv("VLLM_API_KEY", "token-abc123"), 
        base_url=host_model,
        temperature=args.temperature
    )
    
    #basic_model = GPT4o(
    #max_tokens=75,
    #temperature=args.temperature,
    #api_key=os.getenv("OPENAI_API_KEY")
    #)
    
    # Load topic-specific data (claims, introduction, etc.) from JSON
    with open(f'../../config/topics/{topic}.json', 'r') as f:
        data = json.load(f)

    # Create agents (handling various initial conditions)
    for idx, name in enumerate(names):
        if idx < args.num_useless:
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
                    await temp_agent.initialize()
                    if 0.4 <= temp_agent.strength <= 0.6:
                        new_agent = temp_agent
                        break
            elif initial_condition == "extreme":
                num_regular = len(names) - args.num_useless
                if (idx - args.num_useless) < num_regular / 2:
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
                        await temp_agent.initialize()
                        if temp_agent.strength < 0.2:
                            new_agent = temp_agent
                            break
                else:
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
                        await temp_agent.initialize()
                        if temp_agent.strength > 0.8:
                            new_agent = temp_agent
                            break
            elif initial_condition == "bounded":
                assert args.target_strength is not None, "target_strength must be specified for bounded initial condition"
                target = args.target_strength
                lower_bound = max(0.0, target - 0.1)
                upper_bound = min(1.0, target + 0.1)
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
                    await temp_agent.initialize()
                    if lower_bound <= temp_agent.strength <= upper_bound:
                        new_agent = temp_agent
                        break
                
            agents.append(new_agent)
            
    if initial_condition == "none":
        await asyncio.gather(*(agent.initialize() for agent in agents))
        warmup_convo = ConversationCreator(agents)
        await warmup_convo.Converse(
            num_conversations=args_length,
            intro=data['intro'],
            claims=data['claims'],
            shuffle=True,
            num_pairs=args.num_pairs,
            doesChat=True,
            epsilon=args.epsilon
        )

    # Create a conversation controller and run the simulation
    await asyncio.gather(*(agent.initialize() for agent in agents))
    conversation_creator = ConversationCreator(agents)
    await conversation_creator.Converse(
        num_conversations=num_conversations,
        intro=data['intro'],
        claims=data['claims'],
        shuffle=True,
        num_pairs=num_pairs,
        doesChat=True,
        epsilon=args.epsilon
    )

    # Measure total elapsed time
    elapsed_time = time.time() - start_time

    # Prepare grapher for stats over time using the simulation-specific folder.
    strength_grapher = Grapher(conversation_creator.past_strengths, args.folder)
    final_writer = Writer(agents, conversation_creator.past_strengths, args.folder)

    # Generate and save visual outputs into the simulation subfolder.
    strength_grapher.plot_lines(num=5)
    strength_grapher.plot_mean()
    strength_grapher.plot_variance()
    strength_grapher.plot_convergence()

    off_grapher = Grapher(conversation_creator.past_offs, args.folder)
    off_grapher.plot_mean(save_as="means_off.png")

    final_writer.output_desc()
    final_writer.write_csv_log(conversations=conversation_creator.conversation_log)
    final_writer.write_conversation_summaries_json(conversations=conversation_creator.conversation_log)
    
    # Save simulation-specific past strengths to a JSON file in the parent folder.
    simulation_results = {
        "past_strengths": conversation_creator.past_strengths
    }
    results_file = os.path.join(parent_folder, f"past_strengths_sim_{sim_id}.json")
    with open(results_file, "w") as f:
        json.dump(simulation_results, f, indent=4)
    print(f"Simulation {sim_id} results saved to {results_file}")

    print(f"Elapsed time for simulation {sim_id}: {elapsed_time} seconds", flush=True)

def aggregate_simulation_results(parent_folder):
    """
    Aggregates all per-simulation past strengths JSON files into one master JSON file.
    """
    result_files = glob.glob(os.path.join(parent_folder, "past_strengths_sim_*.json"))
    all_results = {}
    for file in result_files:
        basename = os.path.basename(file)
        # Assumes filename is in the format "past_strengths_sim_<sim_id>.json"
        sim_id = basename.split("_")[-1].split(".")[0]
        with open(file, "r") as f:
            data = json.load(f)
        all_results[sim_id] = data["past_strengths"]
    master_file = os.path.join(parent_folder, "master_past_strengths.json")
    with open(master_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Aggregated results saved to {master_file}")
    
async def main():
    args = parse_args()
    parent_folder = args.folder
    os.makedirs(parent_folder, exist_ok=True)
    # Run a single simulation using the provided sim_id
    await run_simulation(args.sim_id, args, parent_folder)

if __name__ == "__main__":
    asyncio.run(main())
