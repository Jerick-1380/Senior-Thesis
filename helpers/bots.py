import random
from datetime import datetime
import numpy as np

AGENT_INSTRUCTIONS = '''
Carry on the conversation given to you.
Speak in 3 sentences or less.
Speak like a normal human.
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
        Keep only the last `memory_length` turns of the conversation in memory.
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
        args_length=8
    ):
        self.name = name
        self.persona = persona
        self.model = model
        self.topic = topic
        self.memory_length = memory_length
        self.args = init_args
        self.args_length = args_length
        self.desc = (
            f"You are an expert in {self.topic}. Furthermore, you believe in the following perspectives: "
            f"{chr(10).join(self.args)}" 
            + AGENT_INSTRUCTIONS
        )
        self.user_history = []
        self.model_history = []
        self.strength = 0
        self.off = 0
        self.past_arg = ""
        self.update_strength(claims)

    def generate(self):
        """
        Generate a response from the model based on the current description 
        and conversation history.
        """
        return self.model.generate(self.desc, self.user_history, self.model_history)

    def clear_memory(self):
        """
        Keep only the last `memory_length` turns of the conversation in memory.
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
            f"You are an expert in {self.topic}. Furthermore, you believe in the following perspectives: "
            f"{chr(10).join(self.args)}" 
            + AGENT_INSTRUCTIONS
        )

    def calculate_strength(self, claims, args):
        """
        Calculate how 'strong' the agent's 'pro' perspective vs. 'con' perspective is, 
        by comparing perplexities on context vs. claims.
        """
        context = (
            f"Let's discuss {self.topic}! I believe that {chr(10).join(args)}. Therefore, I conclude that "
        )
        pro_ppls = []
        con_ppls = []

        for claim_pro in claims['pro']:
            pro_ppls.append(self.model.calculate_perplexity(context, claim_pro))
        for claim_con in claims['con']:
            con_ppls.append(self.model.calculate_perplexity(context, claim_con))

        pro_ppl = np.mean(pro_ppls)
        con_ppl = np.mean(con_ppls)
        return con_ppl / (pro_ppl + con_ppl)

    def add_perspective(self):
        """
        Ask the model to provide a new perspective from the conversation. 
        If the model returns '0', do nothing; otherwise, add the new perspective.
        """
        final_prompt = (
            "State a new perspective that you believe in in one sentence from our conversation. "
            "If nothing is stronger than what you believe in, return 0 and nothing else."
        )
        self.user_history.append(final_prompt)
        final_response = self.generate()
        self.model_history.append(final_response)

        # If the response isn't just '0', treat it as a new perspective
        if len(final_response) != 1 or final_response[0] != "0":
            self.args.append(final_response)
            if len(self.args) > self.args_length:
                self.past_arg = self.args[0]
                self.args = self.args[-self.args_length:]
            self.set_desc()

    def update_strength(self, claims):
        """
        Update the agent's strength value based on perplexity over pro/con claims.
        """
        context = (
            f"Let's discuss {self.topic}! I believe that {chr(10).join(self.args)}. Therefore, I conclude that "
        )
        pro_ppls = []
        con_ppls = []

        for claim_pro in claims['pro']:
            pro_ppls.append(self.model.calculate_perplexity(context, claim_pro))
        for claim_con in claims['con']:
            con_ppls.append(self.model.calculate_perplexity(context, claim_con))

        pro_ppl = np.mean(pro_ppls)
        con_ppl = np.mean(con_ppls)
        self.strength = con_ppl / (pro_ppl + con_ppl)
        self.off = 0.5 * (pro_ppl + con_ppl)

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