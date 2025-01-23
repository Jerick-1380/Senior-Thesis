import time
import numpy as np
import random
from random import sample
import threading
import argparse
from helpers.model import *
from helpers.graph import *
from helpers.bots import *
from helpers.data import *

def createStrengths(options):
    """
    Given a 2-element list of options, return a dictionary describing 5 
    gradations of belief in those options.
    """
    dic = {}
    dic[1] = f"You strongly believe that {options[0]}. You want to convince everyone that it is true."
    dic[2] = f"You somewhat believe that {options[0]}. You are open to new suggestions if they make sense."
    dic[3] = f"You are completely neutral on whether {options[0]} or {options[1]}."
    dic[4] = f"You somewhat believe that {options[1]}. You are open to new suggestions if they make sense."
    dic[5] = f"You strongly believe that {options[1]}. You want to convince everyone that it is true."
    return dic

def createStrengthsPOV(options):
    """
    Given a 2-element list of options, return a dictionary describing 
    the perspective of an external observer about 5 gradations of belief.
    """
    dic = {}
    dic[1] = f"They strongly believe that {options[0]}."
    dic[2] = f"They somewhat believe that {options[0]}."
    dic[3] = f"They are completely neutral on whether {options[0]} or {options[1]}."
    dic[4] = f"They somewhat believe that {options[1]}."
    dic[5] = f"They strongly believe that {options[1]}."
    return dic

def smart_shuffle(lst):
    """
    Shuffle the list except keep the first element in place.
    """
    if len(lst) <= 1:
        return lst
    first_element = lst[0]
    rest_of_list = lst[1:]
    random.shuffle(rest_of_list)
    return [first_element] + rest_of_list

def agent_conversation(agentA, agentB, topic, edge_occ, analyzer, graph, updateStrength=True, doesChat=False):
    """
    Conduct a conversation between two agents about the provided topic.
    Update the graph and possibly the agent's strengths using the analyzer.
    """
    if doesChat:
        conversation = agentA.ChatWith(agentB, f"What is your opinion on {topic}", conversation_length=5)

    # Record edge occurrences
    if agentA.strength not in edge_occ:
        edge_occ[agentA.strength] = {}
    if agentB.strength not in edge_occ[agentA.strength]:
        edge_occ[agentA.strength][agentB.strength] = 0
    edge_occ[agentA.strength][agentB.strength] += 1

    # Optionally update strengths
    if updateStrength:
        prev_a = agentA.strength
        prev_b = agentB.strength
        analyzer.changeStrength(agentA)
        analyzer.changeStrength(agentB)
        # The following block is commented out but left for reference to show logic hasn't changed
        '''
        if((agentA.strength > prev_a and prev_b == 5) or (agentA.strength < prev_a and prev_b == 1)):
            analyzer.addArgument(agentB)
            agentB.updateArguments()
        if((agentB.strength > prev_b and prev_a == 5) or (agentB.strength < prev_b and prev_a == 1)):
            analyzer.addArgument(agentA)
            agentA.updateArguments()
        '''

    agentA.reset()
    agentB.reset()
    graph.update(agentA, agentB)

def parse_args():
    """
    Parse command-line arguments for customizing the simulation.
    """
    parser = argparse.ArgumentParser(description="Process some files.")

    parser.add_argument('--pickN', type=int, default=1, help='Choices to give Agent, -1 gives all')
    parser.add_argument('--pairs', type=int, default=1, help='Number of pairs to run each time')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num_agents', type=int, default=25, help='Number of Agents')
    parser.add_argument('--memory_length', type=int, default=5, help='Length of Memory')
    parser.add_argument('--updateStrength', action='store_true', help='Whether to update strength')
    parser.add_argument('--Chat', action='store_true', help='Whether to chat')
    parser.add_argument('--exportNetwork', action='store_true', help='Whether to create network gif')
    parser.add_argument('--exportStrengths', action='store_true', help='Whether to create strength gif')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    num_agents = args.num_agents
    topic = "purple or orange napkins"
    num_conversations = args.epochs
    num_pairs = args.pairs
    host_model = "http://babel-8-15:8082/v1"
    strengths = 5
    update_strength_flag = args.updateStrength
    does_chat_flag = args.Chat

    edge_occ = {}
    start_time = time.time()
    images = []
    agents = []

    # Prepare strength dictionary for the given topic
    strengthDic = createStrengths(ALL_TOPICS[topic])

    # Create random distribution of strengths among agents
    result_list = [1, 2, 3, 4, 5] * int(num_agents/5)
    random.shuffle(result_list)

    # Randomly pick agent names and personas
    names = sample(ALL_NAMES, num_agents)
    personas = sample(PERSONAS, num_agents)

    # Create Agents
    for i, name in enumerate(names):
        # model = GPT4o(max_tokens=150, temperature=0.7)
        model = Llama("meta-llama/Meta-Llama-3-8B-Instruct", host_model, version=3)
        # The Agent constructor has slight differences from the snippet in bots.py, 
        # which is OK as long as logic is the same. 
        # We'll keep them consistent with the posted code usage.
        agent = Agent(
            name=name,
            persona=personas[i],
            model=model,
            topic=strengthDic,
            claims=None,        # This snippet might differ from your usage
            init_args=[],
            memory_length=args.memory_length,
            args_length=8
        )
        # Overwrite agent's strength to match the random distribution
        agent.strength = result_list[i]
        agents.append(agent)

    G = Graph(agents)
    E = EdgeDistribution(strengths)
    if args.exportNetwork:
        G.addFrame()

    analyzer = Analyzer(
        Llama("meta-llama/Meta-Llama-3-8B-Instruct", host_model, version=3, temperature=0),
        strengthDic=strengthDic
    )

    for epoch in range(num_conversations):
        print(f"Epoch: {epoch}", flush=True)

        available_agents = agents.copy()
        threads = []
        pairs = []

        for _ in range(num_pairs):
            if len(available_agents) < 2:
                break
            agentA = random.choice(available_agents)
            available_agents.remove(agentA)

            # Use the MatchMaker for picking a second agent
            match_maker = MatchMaker(model=Llama("meta-llama/Meta-Llama-3-8B-Instruct", host_model, version=3, temperature=0))
            match_maker.configure_agents(agentA, available_agents)

            if args.pickN == -1:
                agentB = match_maker.pick_all_agents()
            else:
                agentB = match_maker.pick_n_agents(args.pickN)

            available_agents.remove(agentB)
            pairs.append((agentA, agentB))

        # Run each conversation in a thread
        for agentA, agentB in pairs:
            thread = threading.Thread(
                target=agent_conversation,
                args=(agentA, agentB, topic, edge_occ, analyzer, G, update_strength_flag, does_chat_flag)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Save frames if requested
        if args.exportNetwork:
            G.addFrame()
        if args.exportStrengths:
            E.addFrame(edge_occ)

    if args.exportNetwork:
        G.exportGIF("output/graph_even_all", 5)
    if args.exportStrengths:
        E.exportGIF("output/edges_even_all", 5)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Show any arguments collected by Agents
    for agent in agents:
        if hasattr(agent, 'arguments') and len(agent.arguments) > 0:
            print(agent.name)
            print(agent.arguments, flush=True)
            print(flush=True)

    print(f"Elapsed time: {elapsed_time} seconds", flush=True)