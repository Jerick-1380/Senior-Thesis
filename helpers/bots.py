import random
from datetime import datetime
import numpy as np

AGENT_INSTRUCTIONS = '''
Carry on the conversation given to you.
Speak in 3 sentences or less.
Speak like a normal human.
'''

def first_digit_in_string(s: str) -> int:
    for char in s:
        if char.isdigit():
            return int(char)
    return None  # Return None if no digit is found

def get_text_after_last_newline(s):
    # Find the last occurrence of \n
    last_newline_index = s.rfind('\n')
    
    # If \n is found, return everything after it
    if last_newline_index != -1:
        return s[last_newline_index + 1:]
    else:
        # If no \n is found, return the whole string
        return s


class MatchMaker:
    def __init__(self, model):
        self.desc = ""
        self.agents = []
        self.model = model
        self.hallucinates = 0
        self.responses = 0

    def setAgents(self, agent, agents):
        self.desc = f"{agent.strengthDic[agent.strength]}"
        self.agents = agents
    def generate(self, message):
        return self.model.generate(self.desc, [message], [])
    
    def describe_agents(self, agents):
        descriptions = []
        for idx, agent in enumerate(agents):
            description = f"Person {idx+1}: {agent.outside_desc}"
            descriptions.append(description)
        return "\n".join(descriptions)
    
    def random(self):
        random.seed(datetime.now().timestamp()%1 * 10000)
        return random.choice(self.agents)
    
    def pickN(self, n=5):
        random.seed(datetime.now().timestamp()%1 * 10000)
        choices = random.sample(self.agents, n)
        agent_desc = self.describe_agents(choices)
        prompt = f'''
        Below is a list of people and their descriptions:
        {agent_desc}
        Choose one person you want to talk to.
        Return their number at nothing else.
        You should only return one number.
        Ensure formatting is correct.
        '''
        res = self.generate(prompt)
        res = first_digit_in_string(res)
        self.responses += 1
        if(res<len(choices)):
            return choices[int(res)]
        else:
            self.hallucinates += 1
            return self.random()
    
    def pickAll(self):
        choices = self.agents
        agent_desc = self.describe_agents(choices)
        prompt = f'''
        Below is a list of people and their descriptions:
        {agent_desc}
        Choose one person you want to talk to.
        Return their number at nothing else.
        You should only return one number.
        Ensure formatting is correct.
        '''
        res = self.generate(prompt)
        res = first_digit_in_string(res)
        self.responses += 1
        if(res<=len(choices)):
            return choices[int(res)-1]
        else:
            self.hallucinates += 1
            return self.random()

def describe_agents(agents):
    descriptions = []
    for idx, agent in enumerate(agents):
        description = f"Person {idx}: They are {agent.persona} and {agent.desc}"
        descriptions.append(description)
    return "\n".join(descriptions)

class UselessAgent:
    def __init__(self, name, persona, model, topic, claims, init_args, memory_length=5, args_length = 8):
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
        self.updateStrength(claims)

    def generate(self):
        return self.model.generate(self.desc, self.user_history, self.model_history)

    def clearMemory(self):
        if(len(self.user_history) > self.memory_length):
            self.user_history = self.user_history[-self.memory_length:]
            self.model_history = self.model_history[-self.memory_length:]

    def reset(self):
        self.user_history = []
        self.model_history = []

    
    def add_perspective(self):
        return

    def updateStrength(self, claims):
        return
    
    def ChatWith(self, agent, init_prompt, claims, conversation_length=6):
        curr_conversation = ""
        response_other = init_prompt

        for _ in range(conversation_length):
            self.user_history.append(response_other)
            response_self = self.generate()
            self.model_history.append(response_self)

            agent.user_history.append(response_self)
            response_other = agent.generate()
            agent.model_history.append(response_other)
            
            curr_conversation += f"{self.name}: " + response_self + "\n" + f"{agent.name}: " + response_other + "\n\n"
            self.clearMemory()
        self.add_perspective()
        agent.add_perspective()

        self.updateStrength(claims)
        agent.updateStrength(claims)
        return curr_conversation
    

class Agent:
    def __init__(self, name, persona, model, topic, claims, init_args, memory_length=5, args_length = 8):
        self.name = name
        self.persona = persona
        self.model = model
        self.topic = topic
        self.memory_length = memory_length
        self.args = init_args
        self.args_length = args_length
        self.desc = f"You are an expert in {self.topic}. Furthermore, you believe in the following perspectives: {'\n'.join(self.args)}" + AGENT_INSTRUCTIONS
        self.user_history = []
        self.model_history = []
        self.strength = 0
        self.off = 0
        self.past_arg = ""
        self.updateStrength(claims)

    def generate(self):
        return self.model.generate(self.desc, self.user_history, self.model_history)

    def clearMemory(self):
        if(len(self.user_history) > self.memory_length):
            self.user_history = self.user_history[-self.memory_length:]
            self.model_history = self.model_history[-self.memory_length:]

    def reset(self):
        self.user_history = []
        self.model_history = []

    def set_desc(self):
        self.desc = f"You are an expert in {self.topic}. Furthermore, you believe in the following perspectives: {'\n'.join(self.args)}" + AGENT_INSTRUCTIONS

    def calculate_strength(self, claims, args):
        context = f"Let's discuss {self.topic}! I believe that {'\n'.join(args)}. Therefore, I conclude that "
        pro_ppls = []
        con_ppls = []
        for claim_pro in claims['pro']:
            pro_ppls.append(self.model.calculate_perplexity(context, claim_pro))
        for claim_con in claims['con']:
            con_ppls.append(self.model.calculate_perplexity(context, claim_con))
        pro_ppl = np.mean(pro_ppls)
        con_ppl = np.mean(con_ppls)
        return con_ppl/(pro_ppl+con_ppl)


    
    def add_perspective(self):
        final_response = f"State a new perspective that you believe in in one sentence from our conversation. If nothing is stronger than what you believe in, return 0 and nothing else."
        self.user_history.append(final_response)
        response_final = self.generate()
        self.model_history.append(response_final)
        if(len(response_final)!=1 or response_final[0]!="0"):
            self.args.append(response_final)
            if(len(self.args)>self.args_length):
                self.past_arg = self.args[0]
                self.args = self.args[-self.args_length:]
            self.set_desc()

    def updateStrength(self, claims):
        context = f"Let's discuss {self.topic}! I believe that {'\n'.join(self.args)}. Therefore, I conclude that "
        pro_ppls = []
        con_ppls = []
        for claim_pro in claims['pro']:
            pro_ppls.append(self.model.calculate_perplexity(context, claim_pro))
        for claim_con in claims['con']:
            con_ppls.append(self.model.calculate_perplexity(context, claim_con))
        pro_ppl = np.mean(pro_ppls)
        con_ppl = np.mean(con_ppls)
        self.strength = con_ppl/(pro_ppl+con_ppl)
        self.off =  0.5 * (pro_ppl+con_ppl)
    
    def ChatWith(self, agent, init_prompt, claims, conversation_length=6):
        curr_conversation = ""
        response_other = init_prompt

        for _ in range(conversation_length):
            self.user_history.append(response_other)
            response_self = self.generate()
            self.model_history.append(response_self)

            agent.user_history.append(response_self)
            response_other = agent.generate()
            agent.model_history.append(response_other)
            
            curr_conversation += f"{self.name}: " + response_self + "\n" + f"{agent.name}: " + response_other + "\n\n"
            self.clearMemory()
        self.add_perspective()
        agent.add_perspective()

        self.updateStrength(claims)
        agent.updateStrength(claims)
        return curr_conversation

class Analyzer:
    def __init__(self, model, strengthDic):
        self.model = model
        self.strengthDic = strengthDic

    def changeStrength(self, agent):
        prompt = f'''
        You will be given sentences produced by a person:
        {" ".join(agent.model_history)}
        Choose the classification that best fits the person from the options below:
        {'\n'.join(f"{key}: {value}" for key, value in self.strengthDic.items())}
        You should return 1 number between 1 and 5 and nothing else. Ensure the formatting is correct.
        '''
        result = self.model.generate(user_history=[prompt], desc = "", model_history = [])
        res = first_digit_in_string(result)
        agent.updateStrength(int(res))

    def summarize(self, text):
        prompt = f'''
        You will be given several conversations from different people:
        {text}\n
        Summarize the conversations as consisely as possible, only focusing on changes of opinions if any.
        '''
        result = self.model.generate(user_history=[prompt], desc = "You are a concise summarizer", model_history = [])
        return result
    
    def mayor(self, text, past_mayor):
        if(past_mayor==""):
            prompt = f'''
            You are given information of current conversatins that are happening
            {text}\n
            Summarize the overall trend of the conversation.
            '''
            result = self.model.generate(user_history=[prompt], desc = "You are a social scientist pinpointing how conversations are going. Respond in 5 bullet points or less.", model_history = [])
            return result
        else:
            prompt = f'''
            You are given information of what people are talking about
            {text}\n
            Here is your summary last month:
            {past_mayor}\n
            Summarize the overall trend/changes of the conversations.
            '''
            result = self.model.generate(user_history=[prompt], desc = "You are the mayor of a town. Respond in 5 bullet points or less.", model_history = [])
            return result

    def addArgument(self, agent):
        prompt = f'''
        You will be given sentences produced by a person.
        Return the strongest argument given by that person and nothing else.
        You should return 1 sentence from below. Ensure the formatting is correct:\n
        '''
        result = self.model.generate(user_history=[prompt + " ".join(agent.model_history)], desc = "", model_history = [])
        agent.arguments.append(get_text_after_last_newline(result))