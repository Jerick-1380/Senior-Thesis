import os

os.environ["HF_HOME"] = "/data/user_data/junkais/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/junkais/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data/user_data/junkais/hf_cache/datasets"
os.environ["HF_METRICS_CACHE"] = "/data/user_data/junkais/hf_cache/metrics"

import time
import numpy as np
from random import sample
import argparse
from helpers.model import *
from helpers.graph import *
from helpers.bots import *
from helpers.data import *
from helpers.conversation import *
from tqdm import tqdm
from vllm import LLM
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Process some files.")

    parser.add_argument('--args_length', type=int, default = 3)
    parser.add_argument('--temperature', type=float, default = 1.0)
    parser.add_argument('--agents', type=int, help='Number of agents', default=20)
    parser.add_argument('--folder', type=str, help='All agents',default="output")
    parser.add_argument('--epsilon', type=float, help='All agents',default=1)
    parser.add_argument('--topic', type=str, help='All agents', default = "drugs")
    parser.add_argument('--num_conversations', type=int, default = 150)
    parser.add_argument('--num_pairs', type=int, default = 1)
    parser.add_argument('--num_useless', type=int, default = 0)

    return parser.parse_args()



if __name__ == "__main__":

    tqdm.disable = True
    args = parse_args()
    topic = args.topic
    num_conversations = args.num_conversations
    num_pairs = args.num_pairs
    host_model = "http://babel-7-17:8082/v1"
    args_length = args.args_length

    start_time = time.time()
    agents = []

    num_agents = args.agents

    names = sample(ALL_NAMES, num_agents)
    personas = sample(PERSONAS, num_agents)

    basic_model = Llama("meta-llama/Meta-Llama-3-8B-Instruct", host_model, version=3, temperature = args.temperature)
    #model_b = Llama("meta-llama/Meta-Llama-3-8B-Instruct", host_model, version=3, temperature=0)


    #model = LLM(
     #   model="hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
      #  tensor_parallel_size=1,
      #  gpu_memory_utilization=0.95,  
      #  max_model_len=8192,
      #  max_logprobs = 20
    #)

    #model_31_converse = Llama3_1(model, temperature=args.temperature/10)

    with open(f'{topic}.json') as f:
        data = json.load(f)

    for i, name in enumerate(names):
        if(i<args.num_useless):
            agents.append(UselessAgent(name, personas[i],  basic_model, data['id'], data['claims'], [], memory_length=5, args_length = args_length))
        else:
            init_args = []
            random.seed(datetime.now().timestamp()%1 * 10000)
            temp_args = np.random.choice(data['initial_posts'], 100)
            num_pro = args_length/2
            num_con = args_length - num_pro
            args_pro = []
            args_con = []
            for arg in temp_args:
                if(arg['type']=='pro' and len(args_pro)<num_pro):
                    args_pro.append(arg['text'])
                if(arg['type']=='con' and len(args_con)<num_con):
                    args_con.append(arg['text'])
                if(len(args_pro)+len(args_con)>=args_length):
                    break
            init_args = args_pro + args_con
            random.shuffle(init_args)
            agents.append(Agent(name, personas[i],  basic_model, data['id'], data['claims'], init_args, memory_length=5, args_length = args_length))

    writer = Writer(agents,args.folder)
    writer.output_desc("init_desc.txt")

    conversationCreator = ConversationCreator(agents)
    conversationCreator.Converse(num_conversations, data['intro'],data['claims'],shuffle = True, num_pairs=num_pairs, updateStrength = True, doesChat = True, epsilon = args.epsilon)
    

    end_time = time.time()
    elapsed_time = end_time - start_time

    grapher = Grapher(conversationCreator.past_strengths, args.folder)
    writer = Writer(agents,conversationCreator.past_strengths, args.folder)
    grapher.create_gif()
    grapher.plot_lines(num=5)
    grapher.plot_mean()
    grapher.plot_variance()

    grapher_off = Grapher(conversationCreator.past_offs, args.folder)
    grapher_off.plot_mean(save_as = "means_off.png")

    writer.output_desc()

    print(f"Elapsed time: {elapsed_time} seconds", flush=True)