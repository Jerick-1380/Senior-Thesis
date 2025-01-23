import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os
import numpy as np
import random
import threading
import re
from datetime import datetime

def add_newline_after_sentence(paragraph: str) -> str:
    """
    Insert a newline after each sentence-ending punctuation in the paragraph.
    """
    pattern = r'([.!?])\s+'
    formatted_paragraph = re.sub(pattern, r'\1\n', paragraph)
    return formatted_paragraph

def closest_strength_agent(agents):
    """
    Given a list of Agents, return the one whose strength is closest to 
    the first agent's strength (excluding the first agent).
    """
    if not agents or len(agents) < 2:
        raise ValueError("The list must contain at least two agents.")
    
    first_agent_strength = agents[0].strength
    closest_agent = min(
        agents[1:],  
        key=lambda agent: abs(agent.strength - first_agent_strength)
    )
    return closest_agent

def closest_strength_agent_bounded(agents, epsilon):
    """
    Given a list of Agents and a bound epsilon, return one Agent whose 
    strength is within epsilon of the first agent's strength. 
    If no Agent qualifies, return None.
    """
    if not agents or len(agents) < 2:
        raise ValueError("The list must contain at least two agents.")
    
    first_agent_strength = agents[0].strength
    qualifying_agents = [
        agent for agent in agents[1:] 
        if abs(agent.strength - first_agent_strength) < epsilon
    ]
    
    if not qualifying_agents:
        return None
    
    return random.choice(qualifying_agents)

def variance(agents):
    """
    Return the variance of the 'strength' values across a list of agents.
    """
    strengths = [agent.strength for agent in agents]
    return np.var(strengths)

def agent_conversation(
    agentA, 
    agentB, 
    intro, 
    claims,
    edge_occ, 
    conversations, 
    strength_changes,
    update_strength=True, 
    does_chat=True
):
    """
    Perform a conversation between two agents and store the result in the provided lists and dictionaries.
    """
    if does_chat:
        conversation_text = agentA.chat_with(agentB, intro, claims, conversation_length=5)
        conversations.append(conversation_text)

    agentA.reset()
    agentB.reset()

class ConversationCreator:
    """
    A class to manage multiple rounds of conversations among a list of Agents.
    """

    def __init__(self, agents):
        self.agents = agents
        self.edge_occ = {}
        self.conversations = []
        self.strengthChanges = {}
        self.summaries = []
        self.past_strengths = []
        self.past_offs = []
        self.conversation_log = []
        self.past_mayor = ""

    def Converse(
        self, 
        num_conversations, 
        intro, 
        claims, 
        shuffle=True, 
        num_pairs=1, 
        doesChat=False, 
        epsilon=1
    ):
        random.seed(datetime.now().timestamp() % 1 * 10000)
        for round_idx in range(num_conversations):
            strength_list = []
            off_list = []

            for agent in self.agents:
                if agent.strength != 0.5:
                    strength_list.append(agent.strength)
                    off_list.append(agent.off)

            self.past_strengths.append(strength_list)
            self.past_offs.append(off_list)

            available_agents = self.agents.copy()
            if shuffle:
                random.shuffle(available_agents)

            pairs = []
            for _ in range(num_pairs):
                if len(available_agents) < 2:
                    break

                agentA = available_agents[0]
                if epsilon == 1:
                    agentB = available_agents[1]
                    available_agents = available_agents[2:]
                else:
                    # some logic to pick agentB (closest, bounded, etc.)
                    agentB = available_agents[1]
                    available_agents = available_agents[2:]

                pairs.append((agentA, agentB))

            # Conduct each pair's conversation
            for (agentA, agentB) in pairs:
                conversation_text = ""
                if doesChat:
                    # This method should return the conversation text
                    conversation_text = agentA.chat_with(agentB, intro, claims, conversation_length=5)
                    self.conversations.append(conversation_text)

                # Build a record of data from this round
                record = {
                    "round": round_idx + 1,
                    "agentA": {
                        "name": agentA.name,
                        "strength": agentA.strength,
                        "off": agentA.off,
                        "args": agentA.args
                    },
                    "agentB": {
                        "name": agentB.name,
                        "strength": agentB.strength,
                        "off": agentB.off,
                        "args": agentB.args
                    },
                    "prompt": intro,
                    "conversation_text": conversation_text
                }
                self.conversation_log.append(record)

                # Reset or do any post-processing if needed
                agentA.reset()
                agentB.reset()

            # Clear conversations if you only need them ephemeral
            self.conversations = []

    def InteractionDic(self):
        """
        Return the dictionary holding the edge occurrences (who interacts with whom).
        """
        return self.edge_occ

    def Variances(self):
        """
        Return the variance list (not fully implemented in this version).
        """
        return self.variances

    def AllConversations(self):
        """
        Return all stored conversations.
        """
        return self.conversations

    def AllChanges(self):
        """
        Return all stored strength changes, sorted by key.
        """
        self.strengthChanges = dict(sorted(self.strengthChanges.items()))
        return self.strengthChanges