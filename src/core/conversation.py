import os
import numpy as np
import random
import threading
import re
from datetime import datetime
from src.core.bots import batch_add_perspectives
import asyncio

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

    async def Converse(
        self, 
        num_conversations, 
        intro, 
        claims, 
        shuffle=True, 
        num_pairs=1, 
        doesChat=False, 
        epsilon=1,
        conversation_length=2  # add conversation_length parameter
    ):
        random.seed(datetime.now().timestamp() % 1 * 10000)
        strength_list = []
        off_list = []
        pre_convo_records = []

        agents_copy = self.agents.copy()
        if len(agents_copy) % 2 == 1:
            agents_copy = agents_copy[:-1]  # Make even number of agents for pairing

        for i in range(0, len(agents_copy), 2):
            agentA = agents_copy[i]
            agentB = agents_copy[i+1]

            pre_convo_records.append({
                "round": 0,
                "agentA": {
                    "name": agentA.name,
                    "strength": agentA.strength,
                    "off": agentA.off,
                    "args": agentA.args.copy()
                },
                "agentB": {
                    "name": agentB.name,
                    "strength": agentB.strength,
                    "off": agentB.off,
                    "args": agentB.args.copy()
                },
                "prompt": None,
                "conversation_text": None
            })

            if agentA.strength != 0.5:
                strength_list.append(agentA.strength)
                off_list.append(agentA.off)
            if agentB.strength != 0.5:
                strength_list.append(agentB.strength)
                off_list.append(agentB.off)

        self.conversation_log.extend(pre_convo_records)
        self.past_strengths.append(strength_list)
        self.past_offs.append(off_list)
        
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
                elif epsilon == 0:
                    agentB = closest_strength_agent(available_agents)
                    available_agents.remove(agentB)
                else:
                    agentB = closest_strength_agent_bounded(available_agents, epsilon)
                    available_agents.remove(agentB)
                pairs.append((agentA, agentB))

            # Initialize conversation transcripts for each pair
            conversation_texts = ["" for _ in range(len(pairs))]
            # Initialize each agent's conversation with the intro (only for the first agent's turn)
            for agentA, agentB in pairs:
                if not agentA.user_history or agentA.user_history[-1] != intro:
                    agentA.user_history.append(intro)
                if not agentB.user_history or agentB.user_history[-1] != intro:
                    agentB.user_history.append(intro)

            # Loop over the number of conversation turns
            for turn in range(conversation_length):
                # Batch generation for Agent A's turn
                batch_data_A = []
                for agentA, agentB in pairs:
                    # Agent A generates a response based on its current history
                    batch_data_A.append((agentA.desc, agentA.user_history, agentA.model_history))
                responses_A = await self.agents[0].model.generate_batch(batch_data_A)
                for idx, (agentA, agentB) in enumerate(pairs):
                    response_A = responses_A[idx]
                    conversation_texts[idx] += f"{agentA.name}: {response_A}\n"
                    # Update histories: Agent A's response becomes part of its model history,
                    # and Agent B hears it (added to its user history)
                    agentA.model_history.append(response_A)
                    agentB.user_history.append(response_A)
                
                # Batch generation for Agent B's turn
                batch_data_B = []
                for agentA, agentB in pairs:
                    batch_data_B.append((agentB.desc, agentB.user_history, agentB.model_history))
                responses_B = await self.agents[0].model.generate_batch(batch_data_B)
                for idx, (agentA, agentB) in enumerate(pairs):
                    response_B = responses_B[idx]
                    conversation_texts[idx] += f"{agentB.name}: {response_B}\n"
                    agentB.model_history.append(response_B)
                    agentA.user_history.append(response_B)

            # After conversation_length turns, update perspectives and strengths
            agents_to_update = []
            for agentA, agentB in pairs:
                agents_to_update.append(agentA)
                agents_to_update.append(agentB)

            # Batch add perspectives for all agents.
            await batch_add_perspectives(agents_to_update)

            # Then update strengths concurrently.
            await asyncio.gather(*[agent.update_strength(claims) for agent in agents_to_update])

            # Record the full conversation transcript along with updated strengths,
            # and then reset the agents' conversation histories.
            for idx, (agentA, agentB) in enumerate(pairs):
                record = {
                    "round": round_idx + 1,
                    "agentA": {
                        "name": agentA.name,
                        "strength": agentA.strength,
                        "off": agentA.off,
                        "args": agentA.args.copy()
                    },
                    "agentB": {
                        "name": agentB.name,
                        "strength": agentB.strength,
                        "off": agentB.off,
                        "args": agentB.args.copy()
                    },
                    "prompt": intro,
                    "conversation_text": conversation_texts[idx]
                }
                self.conversation_log.append(record)
                agentA.reset()
                agentB.reset()
            
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