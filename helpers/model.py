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
        elif version == 31:
            self.tags = {
           "system_prefix": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n",
            "system_suffix": "<|eot_id|>\n",
            "user_prefix": "<|start_header_id|>user<|end_header_id|>\n",
            "user_suffix": "<|eot_id|>\n",
            "model_prefix": "<|start_header_id|>assistant<|end_header_id|>\n",
            "model_suffix": "<|eot_id|>\n"
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
        words = claim.split()
        if not words:
            return float('inf')
        
        total_log_prob = 0.0

        # For each word, compute its probability conditioned on all preceding words
        for i in range(len(words)):
            # partial_claim is empty for i=0, which handles single-word claims
            partial_claim = " ".join(words[:i])
            word_probs = self.get_probabilities(context, partial_claim)

            if not word_probs:
                print("Error in retrieving probabilities.")
                return float("inf")

            current_word = words[i]
            word_prob = word_probs.get(current_word, min(word_probs.values()))  # fallback to avoid log(0)
            total_log_prob += np.log(word_prob)

        perplexity = np.exp(-total_log_prob / len(words))
        return perplexity
    
    def calculate_probability(self, context, claim):
        """
        Compute the arithmetic mean probability of a claim given the context.
        For each word in the claim, we query the model for the token probability 
        (using get_probabilities) and return the average probability.
        """
        words = claim.split()
        if not words:
            return 0.0
        
        total_prob = 0.0
        count = 0
        

        for i in range(len(words)):
            partial_claim = " ".join(words[:i])
            word_probs = self.get_probabilities(context, partial_claim)
            if not word_probs:
                print("Error in retrieving probabilities.")
                return 0.0

            current_word = words[i]
            # Use the probability for the current word; fallback to the minimum value to avoid zero.
            word_prob = sum(prob for word, prob in word_probs.items() if word.lower() == current_word.lower())
            if word_prob == 0 and word_probs:
                word_prob = min(word_probs.values())
            total_prob += word_prob
            count += 1

        mean_prob = total_prob / count if count > 0 else 0.0
        return mean_prob
    
    def get_word_probability(self, context, word):
        # Reuse get_probabilities with an empty claim to get probabilities for the next token.
        word_probs = self.get_probabilities(context, "")
        return word_probs.get(word, 0.0)


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

    def calculate_probability(self, context, claim):
        """
        Compute the arithmetic mean probability of a claim given the context.
        For each word in the claim, we query the model for the token probability 
        (using get_probabilities) and return the average probability.
        """
        words = claim.split()
        if not words:
            return 0.0
        
        total_prob = 0.0
        count = 0

        for i in range(len(words)):
            partial_claim = " ".join(words[:i])
            word_probs = self.get_probabilities(context, partial_claim)

            if not word_probs:
                print("Error in retrieving probabilities.")
                return 0.0

            current_word = words[i]
            # Use the probability for the current word; fallback to the minimum value to avoid zero.
            word_prob = word_probs.get(current_word, min(word_probs.values()))
            total_prob += word_prob
            count += 1

        mean_prob = total_prob / count if count > 0 else 0.0
        return mean_prob
    
    def get_word_probability(self, context, word):
        """
        Given a context and a word, return the probability of the model generating that word next.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": "DUMMY"
        }
        data = {
            "model": self.dir,
            "prompt": context,
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
                return word_probs.get(word, 0.0)  # Return 0 if word not in top 20 predictions
            else:
                return 0.0
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return 0.0

class GPT4o:
    """
    A simple wrapper around an OpenAI-like GPT-4 model.
    """
    def __init__(self, max_tokens=150, temperature=0.7):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = 'sk-Mwxs7Ru4v6BWdcmFszztGpKbfu7_K8Q7Oit9vCysmVT3BlbkFJ8ZH-3G_EbUHIH8csbDwHfp8y5pnE6rq9_qW-8qOdAA'
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, base_prompt, user_history, model_history):
        """
        Generate a chat completion from the GPT-4 model using the new client.
        The prompt is built from base_prompt, user_history, and model_history.
        """
        messages = [{"role": "system", "content": base_prompt}] if base_prompt else []
        for i in range(len(model_history)):
            messages.append({"role": "user", "content": user_history[i]})
            messages.append({"role": "assistant", "content": model_history[i]})
        messages.append({"role": "user", "content": user_history[-1]})

        @backoff.on_exception(backoff.expo, self.client.RateLimitError)
        def completions_with_backoff(**kwargs):
            return self.client.chat.completions.create(**kwargs)

        chat_completion = completions_with_backoff(
            messages=messages,
            model="gpt-4o-mini",
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return chat_completion.choices[0].message.content

    def get_probabilities(self, context, claim):
        """
        Query the GPT model for token-level probabilities on the next token(s)
        after the concatenated context and claim. Uses the text completions endpoint.
        """
        prompt = context + claim
        try:
            response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=self.temperature,
            n=1,
            logprobs=True,
            top_logprobs = 20
            )
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            word_probs = {
                    logprob_entry.token.strip(): np.exp(logprob_entry.logprob)
                    for logprob_entry in top_logprobs  # Iterate over multiple tokens
                }
            return word_probs
        except Exception as e:
            print(f"Error: {e}")
            return {}

    def calculate_perplexity(self, context, claim):
        """
        Calculate the perplexity of the claim given the context using token probabilities.
        """
        words = claim.split()
        if not words:
            return float('inf')
        
        total_log_prob = 0.0
        for i in range(len(words)):
            partial_claim = " ".join(words[:i])
            word_probs = self.get_probabilities(context, partial_claim)
            if not word_probs:
                print("Error in retrieving probabilities.")
                return float("inf")
            current_word = words[i]
            # Fallback to the minimum probability to avoid log(0)
            word_prob = word_probs.get(current_word, min(word_probs.values()))
            total_log_prob += np.log(word_prob)
        perplexity = np.exp(-total_log_prob / len(words))
        return perplexity

    def calculate_probability(self, context, claim):
        """
        Compute the arithmetic mean probability of a claim given the context.
        For each word in the claim, we query the model for its token probability.
        """
        words = claim.split()
        if not words:
            return 0.0
        
        total_prob = 0.0
        count = 0
        for i in range(len(words)):
            partial_claim = " ".join(words[:i])
            word_probs = self.get_probabilities(context, partial_claim)
            print(f"Context: {context}")
            if not word_probs:
                print("Error in retrieving probabilities.")
                return 0.0
            current_word = words[i]
            # Sum probabilities for tokens matching the current word (ignoring case)
            word_prob = sum(prob for token, prob in word_probs.items() if token.lower() == current_word.lower())
            if word_prob == 0 and word_probs:
                word_prob = min(word_probs.values())
            total_prob += word_prob
            count += 1
        mean_prob = total_prob / count if count > 0 else 0.0
        return mean_prob

    def get_word_probability(self, context, word):
        """
        Retrieve the probability of the specified word given the context.
        """
        word_probs = self.get_probabilities(context, "")
        return word_probs.get(word, 0.0)