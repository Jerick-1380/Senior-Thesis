import requests
import openai
from openai import OpenAI
import backoff
from helpers.prompt_template import PromptTemplate as PT
import requests
from vllm import LLM, SamplingParams
import numpy as np

class Llama:
    """
    A simple wrapper to interact with an HTTP-based API for a Llama model.
    """

    def __init__(self, dir, api_base, version=2, max_tokens=150, temperature=0.7):
        self.dir = dir
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        if version == 2:
            self.tags = {
                "system_prefix": " <s>[INST] <<SYS>>",
                "system_suffix": "<</SYS>>",
                "user_prefix": "",
                "user_suffix": "[/INST]",
                "model_prefix": "",
                "model_suffix": "</s>\n <s>[INST]"
            }
        else:
            self.tags = {
                "system_prefix": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n",
                "system_suffix": "<|eot_id|>\n",
                "user_prefix": "<|start_header_id|>user<|end_header_id|>\n",
                "user_suffix": "<|eot_id|>\n",
                "model_prefix": "<|start_header_id|>assistant<|end_header_id|>\n",
                "model_suffix": "<|eot_id|>\n"
            }
        self.session = requests.Session()

    def generate(self, base_prompt, user_history, model_history):
        """
        Produce a completion from the model, based on a prompt built from the given 
        base_prompt, user_history, and model_history.
        """
        prompt_object = PT(base_prompt, user_history, model_history, self.tags)
        prompt = prompt_object.build_prompt()

        headers = {
            "Content-Type": "application/json",
            "Authorization": "DUMMY"
        }
        data = {
            "model": self.dir,
            "prompt": prompt + "<|start_header_id|>assistant<|end_header_id|>",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": 1,
            "stop": ["<|eot_id|>"]
        }

        response = self.session.post(f"{self.api_base}/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["text"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"

    def get_probabilities(self, context, claim):
        """
        Query the model for token-level probabilities on the next token(s) after context+claim.
        """
        prompt = f"{context}{claim}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "DUMMY"
        }
        data = {
            "model": self.dir,
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
                word_probs = {
                    token.strip(): np.exp(logprob) 
                    for token, logprob in top_logprobs.items()
                }
                return word_probs
            else:
                return {}
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}

    def calculate_perplexity(self, context, claim):
        """
        Calculate perplexity of the claim by multiplying probabilities of each token
        and taking the exponent of the negative average log probability.
        """
        words = claim.split()
        total_log_prob = 0

        for i in range(1, len(words)):
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
    """
    A wrapper for the vLLM-based Llama model interface (version 3.1).
    """

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

    def generate(self, base_prompt, user_history, model_history):
        """
        Produce a completion using vLLM with the given base_prompt and conversation histories.
        """
        prompt_object = PT(base_prompt, user_history, model_history, self.tags)
        full_prompt = prompt_object.build_prompt()
        output = self.model.generate(full_prompt, self.params, use_tqdm=False)
        return output[0].outputs[0].text

    def get_probabilities(self, context, claim):
        """
        Get top token probabilities from vLLM.
        """
        prompt = f"{context}{claim}"
        response = self.model.generate(
            prompt, 
            SamplingParams(temperature=self.temperature, max_tokens=1, logprobs=20),
            use_tqdm=False
        )
        word_probs = {}
        for token_info in response[0].outputs:
            log_probs_dict = token_info.logprobs[0]
            word_probs = {
                logprob.decoded_token.strip(): np.exp(logprob.logprob)
                for _, logprob in log_probs_dict.items()
            }
        return word_probs

    def calculate_perplexity(self, context, claim):
        """
        Calculate perplexity of a claim by querying the model token by token.
        """
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
    """
    A simple wrapper around an OpenAI-like GPT-4 model.
    """

    def __init__(self, max_tokens=150, temperature=0.7):
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, desc="", user_history=None, model_history=None):
        """
        Generate a response using GPT-4 style API from OpenAI.
        """
        if user_history is None:
            user_history = []
        if model_history is None:
            model_history = []

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

        chat_completion = completions_with_backoff(
            messages=messages,
            model="gpt-4o-mini",
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return chat_completion.choices[0].message.content