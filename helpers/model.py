import requests
import openai
from openai import OpenAI
import backoff
from helpers.prompt_template import PromptTemplate as PT
import requests
from vllm import LLM, SamplingParams
import numpy as np

class Llama:
    def __init__(self, dir, api_base, version=2, max_tokens=150, temperature=0.7):
        self.dir = dir
        self.api_base = api_base
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.tags = {
            "system_prefix": " <s>[INST] <<SYS>>",
            "system_suffix": "<</SYS>>",
            "user_prefix": "",
            "user_suffix": "[/INST]",
            "model_prefix": "",
            "model_suffix": "</s>\n <s>[INST]"
        } if version == 2 else {
            "system_prefix": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n",
            "system_suffix": "<|eot_id|>\n",
            "user_prefix": "<|start_header_id|>user<|end_header_id|>\n",
            "user_suffix": "<|eot_id|>\n",
            "model_prefix": "<|start_header_id|>assistant<|end_header_id|>\n",
            "model_suffix": "<|eot_id|>\n"
        }
        self.session = requests.Session() 

    def generate(self, desc, user_history, model_history):
        prompt_object = PT(desc, user_history, model_history, self.tags)
        prompt = prompt_object.build_prompt()
        headers = {
            "Content-Type": "application/json",
            "Authorization": "DUMMY"  # You can set a dummy API key if not needed
        }
        data = {
            "model": self.dir,  # Ensure this matches the model name
            "prompt": prompt+"<|start_header_id|>assistant<|end_header_id|>",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": 1,
            "stop": ["<|eot_id|>"],
        }
        response = self.session.post(f"{self.api_base}/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["text"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
        
    def get_probabilities(self, context, claim):
        # Construct the prompt
        prompt = f"{context}{claim}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "DUMMY"  # You can set a dummy API key if not needed
        }
        data = {
            "model": self.dir,  # Ensure this matches the model name
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": self.temperature,
            "n": 1,
            "logprobs": 20,
            "echo": False
        }
        response = self.session.post(f"{self.api_base}/completions", headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                logprobs = response_json['choices'][0]['logprobs']
                top_logprobs = logprobs['top_logprobs'][0]
                word_probs = {token.strip(): np.exp(logprob) for token, logprob in top_logprobs.items()}
                return word_probs
            else:
                return {}
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}


    def calculate_perplexity(self, context, claim):
        words = claim.split()
        total_log_prob = 0

        for i in range(1,len(words)):
            partial_claim = " ".join(words[:i])
            word_probs = self.get_probabilities(context, partial_claim)
            if not word_probs:
                print("Error in retrieving probabilities.")
                return float("inf")
            current_word = words[i]
            word_prob = word_probs.get(current_word, 1e-10)
            total_log_prob += np.log(word_prob)
        perplexity = np.exp(-total_log_prob / len(words))
        return perplexity
    
        
    
    


class Llama3_1:
    def __init__(self, model, max_tokens=150, temperature=0.7):
        self.temperature = temperature
        self.model = model
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        self.tags = {
            "system_prefix": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n",
            "system_suffix": "<|eot_id|>\n",
            "user_prefix": "<|start_header_id|>user<|end_header_id|>\n",
            "user_suffix": "<|eot_id|>\n",
            "model_prefix": "<|start_header_id|>assistant<|end_header_id|>\n",
            "model_suffix": "<|eot_id|>\n"
        }

    def generate(self, desc, user_history, model_history):
        prompt_object = PT(desc, user_history, model_history, self.tags)
        prompt = prompt_object.build_prompt()
        output = self.model.generate(prompt, self.params, use_tqdm=False)
        return output[0].outputs[0].text

    def get_probabilities(self, context, claim):
        prompt = f"{context}{claim}"
        response = self.model.generate(prompt,SamplingParams(temperature=self.temperature, max_tokens=1,logprobs=20),use_tqdm=False)
        word_probs = {}
        for token_info in response[0].outputs:
            log_probs_dict = token_info.logprobs[0]
            word_probs = {
            logprob.decoded_token.strip(): np.exp(logprob.logprob)
            for _, logprob in log_probs_dict.items()
            }
        return word_probs

    def calculate_perplexity(self, context, claim):
        words = claim.split()
        total_log_prob = 0

        for i in range(len(words)):
            partial_claim = " ".join(words[:i])
            word_probs = self.get_probabilities(context, partial_claim)
            if not word_probs:
                print("Error in retrieving probabilities.")
                return float("inf")
            current_word = words[i]
            word_prob = word_probs.get(current_word, 1e-10)
            total_log_prob += np.log(word_prob)
        perplexity = np.exp(-total_log_prob / len(words))
        return perplexity

class GPT4o:
    def __init__(self, max_tokens=150, temperature=0.7):
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, desc="", user_history=[], model_history=[]):
        messages = [{"role": "system", "content": desc}] if desc else []
        for i in range(len(model_history)):
            messages.append({"role": "user", "content": user_history[i]})
            messages.append({"role": "assistant", "content": model_history[i]})
        messages.append({"role": "user", "content": user_history[-1]})

        OPENAI_API_KEY = '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'
        client = OpenAI(api_key=OPENAI_API_KEY)

        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def completions_with_backoff(**kwargs):
            return client.chat.completions.create(**kwargs)
        #print(messages)
        chat_completion = completions_with_backoff(
            messages=messages,
            model="gpt-4o-mini",
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return chat_completion.choices[0].message.content