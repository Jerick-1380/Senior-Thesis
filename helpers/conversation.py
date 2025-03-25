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
    
    
def group_conversation(agents, init_prompt, claims, conversation_length=6, num_participants=None):
    """
    Conduct a group conversation among a random subset of agents.
    
    The conversation begins with an initial prompt. The function randomly selects a specified
    number of agents (num_participants) from the provided agent list. Then, in a round-robin
    fashion, each selected agent takes a turn generating a response based on the conversation
    so far. After each turn, every participating agent’s user history is updated with the new
    utterance so that everyone “hears” it. Finally, each agent updates its perspective based
    on the conversation.
    
    Args:
        agents (list): A list of Agent objects.
        init_prompt (str): The conversation starter.
        claims (dict): The claims used for updating strengths.
        conversation_length (int): Total number of turns (utterances) in the discussion.
        num_participants (int, optional): The number of agents to randomly select from the full list.
                                          If None or greater than len(agents), all agents are used.
        
    Returns:
        str: A transcript of the group discussion.
    """
    # Determine which agents will participate
    if num_participants is None or num_participants > len(agents):
        chosen_agents = agents.copy()
    else:
        chosen_agents = random.sample(agents, num_participants)
    
    conversation_text = ""
    num_chosen = len(chosen_agents)

    # Reset history for each participating agent and give everyone the initial prompt
    for agent in chosen_agents:
        agent.reset()
        agent.user_history.append(init_prompt)
    current_prompt = init_prompt

    # Cycle through the chosen agents in round-robin order
    for turn in range(conversation_length):
        current_agent = chosen_agents[turn % num_chosen]
        # Ensure the current agent sees the current prompt (if not already there)
        current_agent.user_history.append(current_prompt)
        response = current_agent.generate()
        current_agent.model_history.append(response)

        # Append the response to the transcript
        conversation_text += f"{current_agent.name}: {response}\n"

        # Update every agent’s history so that everyone “hears” this utterance
        for agent in chosen_agents:
            agent.user_history.append(response)
        current_prompt = response

        # Optionally clear memories to maintain only the most recent context
        for agent in chosen_agents:
            agent.clear_memory()

    # After the discussion, update each agent's perspective based on the conversation.
    for agent in chosen_agents:
        agent.add_perspective()
        agent.update_strength(claims)

    return conversation_text

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

            # Collect current strengths and offsets
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
                elif(epsilon == 0):
                    agentB = closest_strength_agent(available_agents)
                    available_agents.remove(agentB)
                else:
                    agentB = closest_strength_agent_bounded(available_agents,epsilon)
                    available_agents.remove(agentB)

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
            
            
    def GroupConverse(self, k, num_conversations, init_prompt, claims, conversation_length=6):
        """
        Conduct a group discussion among all agents in self.agents.
        The conversation begins with init_prompt and proceeds in a round-robin fashion.
        At each turn, the current agent generates a response which is added to a shared conversation.
        After the discussion, each agent updates its perspective.
        """
        for round_idx in range(num_conversations):
            # Conduct the group conversation
            conversation_text = group_conversation(self.agents, init_prompt, claims, conversation_length, k)
            
            # After the conversation, collect the strengths and offsets just like in Converse
            strength_list = []
            off_list = []
            for agent in self.agents:
                if agent.strength != 0.5:
                    strength_list.append(agent.strength)
                    off_list.append(agent.off)
            
            # Store them so you can track changes over time
            self.past_strengths.append(strength_list)
            self.past_offs.append(off_list)

            # Build a record of the group conversation
            record = {
                "round": len(self.conversation_log) + 1,
                "agents": [agent.name for agent in self.agents],
                "prompt": init_prompt,
                "conversation_text": conversation_text,
                "strength_list": strength_list,
                "off_list": off_list
            }
            self.conversation_log.append(record)

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