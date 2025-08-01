import random
from datetime import datetime
import numpy as np
import asyncio
import time
from helpers.advanced_prompts import (
    ConversationPrompts, 
    PerspectivePrompts, 
    ArgumentPrompts, 
    PredictionPrompts,
    StrengthCalculationHelpers
)

AGENT_INSTRUCTIONS = '''
Carry on the conversation given to you.
Speak in 2 sentences or less.
'''

def get_first_digit_in_string(input_string: str) -> int:
    """
    Return the first digit found in the given string.
    If no digit is found, return None.
    """
    for char in input_string:
        if char.isdigit():
            return int(char)
    return None

def get_text_after_last_newline(input_string: str) -> str:
    """
    Return the text after the last newline character in the given string.
    If no newline is found, return the entire string.
    """
    last_newline_index = input_string.rfind('\n')
    if last_newline_index != -1:
        return input_string[last_newline_index + 1:]
    return input_string

async def batch_add_perspectives(agents):
    """
    Batch process add_perspective for a list of agents.
    For each agent:
      - Append a final prompt to its user_history.
      - Use batch generation to obtain new perspectives.
      - Update the agent's arguments with the new perspective.
      - If the number of arguments exceeds the allowed maximum and remove_irrelevant is True,
        batch compute the off values for each candidate removal and remove the one with the highest off.
      - Refresh the agent's description.
    """
    batch_data = []
    final_prompt = "State a new perspective that you believe in in one sentence from our conversation."
    
    # Append the final prompt and prepare batch data for each agent.
    for agent in agents:
        agent.user_history.append(final_prompt)
        batch_data.append((agent.desc, agent.user_history, agent.model_history))
    
    # Assume all agents share the same model (or use one of them) to process the batch.
    responses = await agents[0].model.generate_batch(batch_data)
    
    # Process each agent's response.
    for i, agent in enumerate(agents):
        final_response = responses[i].strip()
        agent.model_history.append(final_response)
        
        # If the response is "0", do nothing.
        if final_response == "0":
            continue
        
        # Otherwise, add the new perspective.
        agent.args.append(final_response)
        
        # If the number of perspectives exceeds the allowed maximum.
        if len(agent.args) > agent.args_length:
            if not agent.remove_irrelevant:
                # Remove the oldest argument.
                agent.past_arg = agent.args[0]
                agent.args = agent.args[-agent.args_length:]
            else:
                # Create candidate argument lists by removing one argument at a time.
                candidate_arg_lists = [agent.args[:j] + agent.args[j+1:] for j in range(len(agent.args))]
                # Batch compute off values for all candidate argument lists.
                off_values = await agent.batch_compute_off_for_args(candidate_arg_lists)
                best_index = off_values.index(max(off_values))
                agent.past_arg = agent.args[best_index]
                agent.args.pop(best_index)
        # Update the agent's description.
        agent.set_desc()
        
class MatchMaker:
    """
    A class that helps in selecting Agents for conversation based on a prompt 
    generated from a model's output.
    """

    def __init__(self, model):
        self.base_description = ""
        self.agents = []
        self.model = model
        self.hallucinates = 0
        self.responses = 0

    def configure_agents(self, agent, all_agents):
        """
        Configure the matchmaker with a base description taken from the 
        agent's strength dictionary and a list of possible agents.
        """
        self.base_description = f"{agent.strengthDic[agent.strength]}"
        self.agents = all_agents

    def generate(self, message):
        """
        Generate a model response based on the base description and a single user message.
        """
        return self.model.generate(self.base_description, [message], [])

    def describe_agents(self, agents):
        """
        Return a concatenated string of agent descriptions, enumerated.
        """
        descriptions = []
        for idx, agent in enumerate(agents):
            description = f"Person {idx+1}: {agent.outside_desc}"
            descriptions.append(description)
        return "\n".join(descriptions)

    def random_agent(self):
        """
        Return a random agent from the agent list using a time-seeded random.
        """
        random.seed(datetime.now().timestamp() % 1 * 10000)
        return random.choice(self.agents)

    def pick_n_agents(self, num_to_pick=5):
        """
        Prompt the model to pick one agent from a randomly sampled subset of n agents.
        Return the chosen agent or a random agent if model output is invalid.
        """
        random.seed(datetime.now().timestamp() % 1 * 10000)
        choices = random.sample(self.agents, num_to_pick)
        agent_descriptions = self.describe_agents(choices)

        prompt = f'''
        Below is a list of people and their descriptions:
        {agent_descriptions}
        Choose one person you want to talk to.
        Return their number and nothing else.
        You should only return one number.
        Ensure formatting is correct.
        '''
        model_response = self.generate(prompt)
        chosen_index = get_first_digit_in_string(model_response)
        self.responses += 1
        if chosen_index is not None and chosen_index < len(choices):
            return choices[int(chosen_index)]
        else:
            self.hallucinates += 1
            return self.random_agent()

    def pick_all_agents(self):
        """
        Prompt the model to pick one agent from the entire agent list.
        Return the chosen agent or a random agent if model output is invalid.
        """
        choices = self.agents
        agent_descriptions = self.describe_agents(choices)

        prompt = f'''
        Below is a list of people and their descriptions:
        {agent_descriptions}
        Choose one person you want to talk to.
        Return their number and nothing else.
        You should only return one number.
        Ensure formatting is correct.
        '''
        model_response = self.generate(prompt)
        chosen_index = get_first_digit_in_string(model_response)
        self.responses += 1
        if chosen_index is not None and chosen_index <= len(choices):
            return choices[int(chosen_index) - 1]
        else:
            self.hallucinates += 1
            return self.random_agent()


def describe_agents(agents):
    """
    Return a simple textual description of each agent,
    enumerating them by index.
    """
    descriptions = []
    for idx, agent in enumerate(agents):
        description = f"Person {idx}: They are {agent.persona} and {agent.desc}"
        descriptions.append(description)
    return "\n".join(descriptions)

class UselessAgent:
    """
    An agent that tries to distract or detract from the conversation.
    """

    def __init__(
        self, 
        name, 
        persona, 
        model, 
        topic, 
        claims, 
        init_args, 
        memory_length=5, 
        args_length=8
    ):
        self.name = name
        self.persona = persona
        self.model = model
        self.topic = topic
        self.memory_length = memory_length
        self.args = init_args
        self.args_length = args_length
        self.desc = "Try to detract the conversation as much as you can." + AGENT_INSTRUCTIONS
        self.user_history = []
        self.model_history = []
        self.strength = 0.5
        self.off = 0
        self.past_arg = ""
        self.update_strength(claims)

    def generate(self):
        """
        Generate a response from the model based on current user history 
        and model history.
        """
        return self.model.generate(self.desc, self.user_history, self.model_history)

    def clear_memory(self):
        """
        Keep only the last memory_length turns of the conversation in memory.
        """
        if len(self.user_history) > self.memory_length:
            self.user_history = self.user_history[-self.memory_length:]
            self.model_history = self.model_history[-self.memory_length:]

    def reset(self):
        """
        Clear the entire conversation history.
        """
        self.user_history = []
        self.model_history = []

    def add_perspective(self):
        """
        Placeholder for adding a new perspective based on conversation.
        """
        return

    def update_strength(self, claims):
        """
        Placeholder for updating the strength of the agent's perspective 
        based on claims.
        """
        return

    def chat_with(self, agent, init_prompt, claims, conversation_length=6):
        """
        Conduct a conversation with another agent for a specified number of turns.
        Returns the text of the conversation.
        """
        conversation_text = ""
        response_other = init_prompt

        for _ in range(conversation_length):
            # This agent's turn
            self.user_history.append(response_other)
            response_self = self.generate()
            self.model_history.append(response_self)

            # Other agent's turn
            agent.user_history.append(response_self)
            response_other = agent.generate()
            agent.model_history.append(response_other)

            # Construct conversation
            conversation_text += (
                f"{self.name}: {response_self}\n"
                f"{agent.name}: {response_other}\n\n"
            )
            self.clear_memory()

        self.add_perspective()
        agent.add_perspective()

        self.update_strength(claims)
        agent.update_strength(claims)
        return conversation_text



class Agent:
    """
    A general Agent that believes in certain perspectives, 
    updates its strength based on claims, and can chat with other agents.
    """

    def __init__(
        self,
        name,
        persona,
        model,
        topic,
        claims,
        init_args,
        memory_length=5,
        args_length=3,
        remove_irrelevant=True,
        extra_desc = ""
    ):
        self.name = name
        self.persona = persona
        self.model = model
        self.topic = topic
        self.claims = claims            # Ensure we store claims on the instance
        self.memory_length = memory_length
        self.args = init_args
        self.args_length = args_length
        self.remove_irrelevant = remove_irrelevant

        if len(self.args) > 0:
            self.desc = (
                f"You are an expert in {self.topic}. Furthermore, you believe in the following perspectives:\n"
                + "\n".join(self.args)
                + "\n" + AGENT_INSTRUCTIONS
                + "\n" + extra_desc
            )
        else:
            self.desc = (
                f"You are an expert in {self.topic}.\n"
                + AGENT_INSTRUCTIONS
                + "\n" + extra_desc
            )
        self.user_history = []
        self.model_history = []
        self.strength = 0
        self.off = 0
        self.past_arg = ""

    async def initialize(self):
        # Call update_strength here after instantiation
        await self.update_strength(self.claims)

    async def _get_mean_probs(self, context, claims):
        """
        Internal helper function that calculates the average probability
        for pro and con claims, returning (pro_prob, con_prob).
        """
        pro_probs = []
        con_probs = []
    
        pro_probs = await asyncio.gather(*[self.model.calculate_probability(context, claim) for claim in claims['pro']])
        con_probs = await asyncio.gather(*[self.model.calculate_probability(context, claim) for claim in claims['con']])

        pro_prob = np.mean(pro_probs) if pro_probs else 0.0
        con_prob = np.mean(con_probs) if con_probs else 0.0
        return pro_prob, con_prob

    async def calculate_strength(self, claims, args):
        """
        Calculate how 'strong' the agent's 'pro' perspective is compared 
        to its 'con' perspective by comparing the average token-level probabilities.
        A higher value indicates a stronger pro perspective.
        """
        context = '\n'.join(self.args) + self.claims['connector']
        pro_prob, con_prob = await self._get_mean_probs(context, claims)
        return pro_prob / (pro_prob + con_prob + 1e-9)

    async def update_strength(self, claims):
        """
        Update the agent's strength value based on average probability over pro/con claims.
        Also update the 'off' metric as 0.5 times the sum of probabilities.
        """
        context = '\n'.join(self.args) + self.claims['connector']
        pro_prob, con_prob = await self._get_mean_probs(context, claims)

        self.strength = pro_prob / (pro_prob + con_prob + 1e-9)
        self.off = 0.5 * (pro_prob + con_prob)

    async def compute_off_for_args(self, claims, arg_list):
        """
        Compute the 'off' value (0.5 * sum of pro/con probabilities) for a hypothetical list of args.
        Used when deciding which argument to remove if remove_irrelevant=True.
        """
        context = '\n'.join(self.args) + self.claims['connector']
        pro_prob, con_prob = await self._get_mean_probs(context, claims)
        return 0.5 * (pro_prob + con_prob)

    async def generate(self):
        """
        Asynchronously generate a response from the model based on the current description 
        and conversation history.
        """
        return await self.model.generate(self.desc, self.user_history, self.model_history)

    def clear_memory(self):
        """
        Keep only the last memory_length turns of the conversation in memory.
        """
        if len(self.user_history) > self.memory_length:
            self.user_history = self.user_history[-self.memory_length:]
            self.model_history = self.model_history[-self.memory_length:]

    def reset(self):
        """
        Clear the entire conversation history.
        """
        self.user_history = []
        self.model_history = []

    def set_desc(self):
        """
        Re-set the agent's self-description to incorporate the latest perspective arguments.
        """
        self.desc = (
            f"You are an expert in {self.topic}.  "
            +"Furthermore, you believe in the following perspectives: " if len(self.args) > 0 else ""
            +f"{chr(10).join(self.args)}"
            + AGENT_INSTRUCTIONS
        )
        
    async def batch_compute_off_for_args(self, candidate_arg_lists):
        """
        Compute the 'off' values for a batch of candidate argument lists for an agent.
        Each candidate's context is built as the join of its arguments plus the claims connector.
        For each candidate, we compute the average probability over all pro and con claims using
        the agent's model.calculate_probability (which itself uses batched get_probabilities).
        
        Returns:
            A list of off values (one per candidate).
        """
        candidate_count = len(candidate_arg_lists)
        pro_claims = self.claims['pro']
        con_claims = self.claims['con']
        
        # Prepare batch data for all candidates.
        batch_data = []
        # We'll also record the candidate index for each call.
        candidate_indices = []
        for i, candidate in enumerate(candidate_arg_lists):
            context = '\n'.join(candidate) + self.claims['connector']
            for claim in pro_claims:
                batch_data.append((context, claim))
                candidate_indices.append(('pro', i))
            for claim in con_claims:
                batch_data.append((context, claim))
                candidate_indices.append(('con', i))
        
        # Launch all calculate_probability calls concurrently.
        tasks = [self.model.calculate_probability(context, claim) for (context, claim) in batch_data]
        results = await asyncio.gather(*tasks)
        
        # Aggregate the results per candidate.
        candidate_offs = [0] * candidate_count
        num_pro = len(pro_claims)
        num_con = len(con_claims)
        calls_per_candidate = num_pro + num_con
        
        for i in range(candidate_count):
            offset = i * calls_per_candidate
            pro_vals = results[offset: offset + num_pro]
            con_vals = results[offset + num_pro: offset + calls_per_candidate]
            pro_prob = sum(pro_vals) / num_pro if num_pro > 0 else 0.0
            con_prob = sum(con_vals) / num_con if num_con > 0 else 0.0
            candidate_offs[i] = 0.5 * (pro_prob + con_prob)
        
        return candidate_offs

    async def add_perspective(self):
        """
        Ask the model to provide a new perspective from the conversation. 
        If the model returns '0', do nothing; otherwise, add the new perspective.

        If self.remove_irrelevant is True, we attempt removing each argument
        and see which removal results in the highest off. We remove that argument.
        Otherwise, we simply remove the oldest argument.
        """
        # Create conversation history from user and model history
        conversation_history = ""
        for i in range(len(self.user_history)):
            conversation_history += f"Agent A: {self.user_history[i]}\n"
            if i < len(self.model_history):
                conversation_history += f"Agent B: {self.model_history[i]}\n"
        
        # Use the improved perspective extraction prompt
        final_prompt = PerspectivePrompts.extract_perspective_prompt(self.topic, conversation_history)
        
        # Generate perspective using the model directly (not through conversation history)
        final_response = await self.model.generate(final_prompt, [], [])
        
        # Clean up the response to extract just the perspective
        if "Perspective:" in final_response:
            final_response = final_response.split("Perspective:")[-1].strip()
        final_response = final_response.strip()

        # If the response is '0', do nothing
        if len(final_response) == 1 and final_response[0] == "0":
            return

        self.args.append(final_response)

        # If we exceed the maximum number of arguments
        if len(self.args) > self.args_length:
            if not self.remove_irrelevant:
                # OLD BEHAVIOR: remove the oldest argument
                self.past_arg = self.args[0]
                self.args = self.args[-self.args_length:]
            else:
                # NEW BEHAVIOR: remove the argument whose removal yields the highest self.off
                candidate_arg_lists = [self.args[:i] + self.args[i+1:] for i in range(len(self.args))]
                off_values = await self.batch_compute_off_for_args(self.claims, candidate_arg_lists)
                best_off = max(off_values)
                best_index = off_values.index(best_off)
                self.past_arg = self.args[best_index]
                self.args.pop(best_index)
        self.set_desc()

    async def generate_arguments(self, question: str, num_arguments: int = 4) -> list:
        """Generate multiple arguments for a question using improved prompting."""
        argument_prompt = ArgumentPrompts.generate_argument_prompt(question)
        
        # Generate arguments in parallel
        tasks = [self.model.generate(argument_prompt, [], []) for _ in range(num_arguments)]
        arguments = await asyncio.gather(*tasks)
        
        # Clean and filter arguments
        clean_arguments = []
        for arg in arguments:
            cleaned = arg.strip()
            if "Argument:" in cleaned:
                cleaned = cleaned.split("Argument:")[-1].strip()
            if cleaned and cleaned != "0":
                clean_arguments.append(cleaned)
        
        return clean_arguments

    async def predict_with_arguments(self, question: str) -> float:
        """Make a prediction using the agent's collected arguments with improved prompting."""
        if not self.args:
            return 0.5
        
        prompt = PredictionPrompts.predict_with_arguments_prompt(question, self.args)
        
        try:
            probs_dict = await self.model.get_probabilities(prompt, "")
            yes_prob, no_prob = StrengthCalculationHelpers.extract_yes_no_probabilities(probs_dict)
            return StrengthCalculationHelpers.calculate_strength_from_probabilities(yes_prob, no_prob)
        except Exception as e:
            print(f"Error in agent {self.name} prediction: {e}")
            return 0.5

    async def start_conversation(self, topic: str) -> str:
        """Start a conversation about a topic using improved prompting."""
        prompt = ConversationPrompts.start_conversation_prompt(topic)
        response = await self.model.generate(prompt, [], [])
        return response.strip()

    async def continue_conversation(self, topic: str, conversation_history: str) -> str:
        """Continue a conversation using improved prompting.""" 
        prompt = ConversationPrompts.continue_conversation_prompt(topic, conversation_history)
        response = await self.model.generate(prompt, [], [])
        return response.strip()

    async def chat_with(self, agent, init_prompt, claims, conversation_length=6):
        conversation_text = ""
        response_other = init_prompt

        for _ in range(conversation_length):
            # This agent's turn
            self.user_history.append(response_other)
            response_self = await self.generate()
            self.model_history.append(response_self)

            # Other agent's turn
            agent.user_history.append(response_self)
            response_other = await agent.generate()
            agent.model_history.append(response_other)

            # Construct conversation text
            conversation_text += (
                f"{self.name}: {response_self}\n"
                f"{agent.name}: {response_other}\n\n"
            )
            self.clear_memory()
        await asyncio.gather(
            self.add_perspective(),
            agent.add_perspective()
        )

        await asyncio.gather(
            self.update_strength(claims),
            agent.update_strength(claims)
        )

        return conversation_text


class Analyzer:
    """
    A class that analyzes agents' outputs, changes their strength classification, 
    and can summarize conversations.
    """

    def __init__(self, model, strengthDic):
        self.model = model
        self.strengthDic = strengthDic

    def change_strength(self, agent):
        """
        Ask the model to classify the agent's outputs into a strength category (1-5).
        """
        prompt = f'''
        You will be given sentences produced by a person:
        {" ".join(agent.model_history)}
        Choose the classification that best fits the person from the options below:
        {'\n'.join(f"{key}: {value}" for key, value in self.strengthDic.items())}
        You should return 1 number between 1 and 5 and nothing else. Ensure the formatting is correct.
        '''
        result = self.model.generate(user_history=[prompt], desc="", model_history=[])
        chosen_strength = get_first_digit_in_string(result)
        agent.updateStrength(int(chosen_strength))

    def summarize(self, text):
        """
        Summarize multiple conversations, focusing only on changes in opinions if any.
        """
        prompt = f'''
        You will be given several conversations from different people:
        {text}\n
        Summarize the conversations as concisely as possible, only focusing on changes of opinions if any.
        '''
        return self.model.generate(user_history=[prompt], desc="You are a concise summarizer", model_history=[])

    def mayor(self, text, past_mayor):
        """
        Summarize the overall trend or changes in conversations from a mayor's point of view.
        If no previous summary, produce a new summary; otherwise, incorporate the old summary.
        """
        if past_mayor == "":
            prompt = f'''
            You are given information of current conversations that are happening:
            {text}\n
            Summarize the overall trend of the conversation.
            '''
            return self.model.generate(
                user_history=[prompt], 
                desc="You are a social scientist pinpointing how conversations are going. Respond in 5 bullet points or less.", 
                model_history=[]
            )
        else:
            prompt = f'''
            You are given information of what people are talking about:
            {text}\n
            Here is your summary last month:
            {past_mayor}\n
            Summarize the overall trend/changes of the conversations.
            '''
            return self.model.generate(
                user_history=[prompt], 
                desc="You are the mayor of a town. Respond in 5 bullet points or less.", 
                model_history=[]
            )

    def add_argument(self, agent):
        """
        Ask the model for the strongest argument from the agent's outputs, and append it 
        to that agent's arguments list.
        """
        prompt = f'''
        You will be given sentences produced by a person.
        Return the strongest argument given by that person and nothing else.
        You should return 1 sentence from below. Ensure the formatting is correct:\n
        '''
        result = self.model.generate(
            user_history=[prompt + " ".join(agent.model_history)], 
            desc="", 
            model_history=[]
        )
        agent.arguments.append(get_text_after_last_newline(result))