import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os
import numpy as np
import random
import threading
import re
from datetime import datetime
        
def add_newline_after_sentence(paragraph):
    pattern = r'([.!?])\s+'
    formatted_paragraph = re.sub(pattern, r'\1\n', paragraph)
    return formatted_paragraph
def closest_strength_agent(agents):
    if not agents or len(agents) < 2:
        raise ValueError("The list must contain at least two agents.")
    
    first_agent_strength = agents[0].strength
    closest_agent = min(
        agents[1:],  # Exclude the first agent
        key=lambda agent: abs(agent.strength - first_agent_strength)
    )
    
    return closest_agent
def closest_strength_agent_bounded(agents, epsilon):
    if not agents or len(agents) < 2:
        raise ValueError("The list must contain at least two agents.")
    
    first_agent_strength = agents[0].strength
    
    # Filter agents with a strength difference less than epsilon
    qualifying_agents = [
        agent for agent in agents[1:] 
        if abs(agent.strength - first_agent_strength) < epsilon
    ]
    
    if not qualifying_agents:
        return None
    
    # Choose a random agent from the qualifying agents
    return random.choice(qualifying_agents)


def variance(agents):
    strengths = []
    for agent in agents:
        strengths.append(agent.strength)
    return np.var(strengths)
def agent_conversation(agentA, agentB, intro, claims,edge_occ, conversations, strengthChanges,updateStrength = True, doesChat = True):
    if doesChat:
        conversation = agentA.ChatWith(agentB, intro, claims, conversation_length=5)
   #print(add_newline_after_sentence(conversation))
    conversations.append(conversation)
    agentA.reset()
    agentB.reset()




class ConversationCreator:
    def __init__(self, agents):
        self.agents = agents
        self.edge_occ = {}
        self.conversations = []
        self.strengthChanges = {}
        self.summaries = []
        self.past_strengths = []
        self.past_offs = []
        self.past_mayor = ""
    def Converse(self, num_conversations, intro, claims, shuffle = True, num_pairs = 1, updateStrength = False, doesChat = False, epsilon = 1):
        random.seed(datetime.now().timestamp()%1 * 10000)
        for i in range(num_conversations):
            strength_list = []
            off_list = []
            for agent in self.agents:
                if(agent.strength!=0.5):
                    strength_list.append(agent.strength)
                    off_list.append(agent.off)
            self.past_strengths.append(strength_list)
            self.past_offs.append(off_list)
            available_agents = self.agents.copy()
            if(shuffle):
                random.shuffle(available_agents)
            pairs = []
            for _ in range(num_pairs): 
                if len(available_agents) < 2:
                    break
                agentA = available_agents[0]
                if(epsilon==1):  
                    agentB = available_agents[1]
                    available_agents = available_agents[2:]
                elif(epsilon==0):
                    agentB = closest_strength_agent(available_agents)
                    available_agents.remove(agentA)
                    available_agents.remove(agentB)
                else:
                    agentB = closest_strength_agent_bounded(available_agents,epsilon)
                    if(agentB is None):
                        continue
                    available_agents.remove(agentA)
                    available_agents.remove(agentB)
                pairs.append((agentA, agentB))
            for agentA, agentB in pairs:
                agent_conversation(agentA, agentB, intro, claims,self.edge_occ, self.conversations,self.strengthChanges,updateStrength,doesChat)
            self.conversations = []
    def InteractionDic(self):
        return self.edge_occ
    def Variances(self):
        return self.variances
    def AllConversations(self):
        return self.conversations
    def AllChanges(self):
        self.strengthChanges = dict(sorted(self.strengthChanges.items()))
        return self.strengthChanges
       