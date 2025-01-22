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
    dic = {}
    dic[1] = f"You strongly believe that {options[0]}. You want to convince everyone that it is true."
    dic[2] = f"You somewhat believe that {options[0]}. You are open to new suggestions if they make sense."
    dic[3] = f"You are completely neutral on whether {options[0]} or {options[1]}."
    dic[4] = f"You somewhat believe that {options[1]}. You are open to new suggestions if they make sense."
    dic[5] = f"You strongly believe that {options[1]}. You want to convince everyone that it is true."
    return dic

def createStrengthsPOV(options):
    dic = {}
    dic[1] = f"They strongly believe that {options[0]}."
    dic[2] = f"They somewhat believe that {options[0]}."
    dic[3] = f"They are completely neutral on whether {options[0]} or {options[1]}."
    dic[4] = f"They somewhat believe that {options[1]}."
    dic[5] = f"They strongly believe that {options[1]}."
    return dic


def smart_shuffle(lst):
    if len(lst) <= 1:
        return lst  # No need to shuffle if list has 1 or fewer elements
    first_element = lst[0]
    rest_of_list = lst[1:]
    random.shuffle(rest_of_list)
    return [first_element] + rest_of_list

def agent_conversation(agentA, agentB, topic, edge_occ, analyzer, graph,updateStrength = True, doesChat = False):
  #  print(f"{agentA.name} is chatting with {agentB.name}\n")
    if(doesChat):
        conversation = agentA.ChatWith(agentB, f"What is your opinion on {topic}", conversation_length=5)
   # print(conversation)
    #print("----")

    if agentA.strength not in edge_occ:
        edge_occ[agentA.strength] = {}
    if agentB.strength not in edge_occ[agentA.strength]:
        edge_occ[agentA.strength][agentB.strength] = 0
    edge_occ[agentA.strength][agentB.strength] += 1
    if(updateStrength):
        prev_a = agentA.strength
        prev_b = agentB.strength
        analyzer.changeStrength(agentA)
        analyzer.changeStrength(agentB)
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
    parser = argparse.ArgumentParser(description="Process some files.")

    parser.add_argument('--pickN', type=int, default = 1, help='Choices to give Agent, -1 gives all')
    parser.add_argument('--pairs', type=int, default = 1, help='Number of pairs to run each time')
    parser.add_argument('--epochs', type=int, default = 100, help='Number of epochs')
    parser.add_argument('--num_agents', type=int, default = 25, help='Numbers of Agents')
    parser.add_argument('--memory_length', type=int, default = 5, help='Length of Memory')
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
    updateStrength = args.updateStrength
    doesChat = args.Chat

    edge_occ = {}

    start_time = time.time()
    images = []
    agents = []
    strengthDic = createStrengths(ALL_TOPICS[topic])

    result_list = [1, 2, 3, 4, 5] * int(num_agents/5)
    random.shuffle(result_list)

    names = sample(ALL_NAMES, num_agents)
    personas = sample(PERSONAS, num_agents)

    for i, name in enumerate(names):
        #model = GPT4o(max_tokens=150, temperature=0.7)
        model = Llama("meta-llama/Meta-Llama-3-8B-Instruct", host_model, version=3)
        agents.append(Agent(name, personas[i], model, strengthDic, strength=result_list[i], memory_length=args.memory_length))

    G = Graph(agents)
    E = EdgeDistribution(strengths)
    if(args.exportNetwork):
        G.addFrame()

    analyzer = Analyzer(Llama("meta-llama/Meta-Llama-3-8B-Instruct", host_model, version=3, temperature=0), strengthDic=strengthDic)

    for i in range(num_conversations):  
        print(f"Epoch: {i}", flush=True)
        available_agents = agents.copy()  
        threads = []
        pairs = []

        for _ in range(num_pairs): 
            if len(available_agents) < 2:
                break  

            agentA = random.choice(available_agents)
            available_agents.remove(agentA)  

            matchMaker = MatchMaker(agentA, available_agents, Llama("meta-llama/Meta-Llama-3-8B-Instruct", host_model, version=3, temperature=0))
            if(args.pickN==-1):
                agentB = matchMaker.pickAll()
            else:
                agentB = matchMaker.pickN(args.pickN)
            available_agents.remove(agentB) 

            pairs.append((agentA, agentB))
        for agentA, agentB in pairs:
            thread = threading.Thread(target=agent_conversation, args=(agentA, agentB, topic, edge_occ, analyzer, G,updateStrength,doesChat))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if(args.exportNetwork):
            G.addFrame()
        if(args.exportStrengths):
            E.addFrame(edge_occ)

    if(args.exportNetwork):
        G.exportGIF("output/graph_even_all", 5)
    if(args.exportStrengths):
        E.exportGIF("output/edges_even_all", 5)

    end_time = time.time()
    elapsed_time = end_time - start_time
    for agent in agents:
        if(len(agent.arguments)>0):
            print(agent.name)
            print(agent.arguments, flush=True)
            print(flush=True)
    print(f"Elapsed time: {elapsed_time} seconds", flush=True)