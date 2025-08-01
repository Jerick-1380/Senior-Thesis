import openai
from openai import OpenAI, RateLimitError, APIError, AsyncOpenAI
import backoff
from src.core.prompt_template import PromptTemplate as PT
from vllm import LLM, SamplingParams
import numpy as np
import aiohttp
import asyncio
import time
import aiolimiter
from aiolimiter import AsyncLimiter


class Llama:
    def __init__(self, base_url: str, api_key: str, model: str,
                 max_tokens: int = 75, temperature: float = 0.7):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate(self, base_prompt: str, user_history: list, model_history: list) -> str:
        messages = []
        if base_prompt:
            messages.append({"role": "system", "content": base_prompt})
        for i in range(len(user_history)):
            messages.append({"role": "user", "content": user_history[i]})
            if i < len(model_history):
                messages.append({"role": "assistant", "content": model_history[i]})

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=1
            )
            # <-- CHANGE HERE: use .content, not ["content"]
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            print("Rate limit exceeded:", e)
            return ""
        except APIError as e:
            print("API error:", e)
            return ""
        except Exception as e:
            print("Unexpected error:", e)
            return ""

    async def generate_batch(self, batch_data: list) -> list:
        """
        Launch each generate() in its own task to keep parallelism.
        """
        tasks = [
            asyncio.create_task(self.generate(bp, uh, mh))
            for (bp, uh, mh) in batch_data
        ]
        replies = await asyncio.gather(*tasks)
        return replies

    async def get_probabilities(self, context: str, claim: str) -> dict:
        prompt_text = context + claim
        try:
            response = await asyncio.to_thread(
                self.client.completions.create,
                model=self.model,
                prompt=prompt_text,
                max_tokens=1,
                temperature=self.temperature,
                logprobs=20,
                echo=False
            )
            top_logprobs = response.choices[0].logprobs.top_logprobs[0]
            return {tok.strip(): float(np.exp(lp)) for tok, lp in top_logprobs.items()}
        except Exception as e:
            print("Error in get_probabilities:", e)
            return {}

    async def get_probabilities_batch(self, batch_data: list) -> list:
        tasks = [
            asyncio.create_task(self.get_probabilities(ctx, clm))
            for (ctx, clm) in batch_data
        ]
        return await asyncio.gather(*tasks)

    async def calculate_perplexity(self, context: str, claim: str) -> float:
        words = claim.split()
        if not words:
            return float("inf")

        total_logprob = 0.0
        for i in range(len(words)):
            partial = " ".join(words[:i])
            probs = await self.get_probabilities(context, partial)
            if not probs:
                return float("inf")
            current_word = words[i]
            word_prob = probs.get(current_word, min(probs.values()))
            total_logprob += np.log(word_prob)
        return float(np.exp(-total_logprob / len(words)))

    async def calculate_probability(self, context: str, claim: str) -> float:
        words = claim.split()
        if not words:
            return 0.0

        batch_data = [(context, " ".join(words[:i])) for i in range(len(words))]
        results = await self.get_probabilities_batch(batch_data)

        total_prob = 0.0
        for i, probs in enumerate(results):
            current_word = words[i]
            word_prob = sum(p for tok, p in probs.items() if tok.lower() == current_word.lower())
            if word_prob == 0 and probs:
                word_prob = min(probs.values())
            total_prob += word_prob
        return total_prob / len(words)

    async def get_word_probability(self, context: str, word: str) -> float:
        probs = await self.get_probabilities(context, "")
        return probs.get(word, 0.0)

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
    A class to interact with OpenAI's GPT-4o model asynchronously.
    """

    def __init__(self, max_tokens=150, temperature=0.7, api_key=None):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def generate(self, base_prompt, user_history, model_history):
        """
        Generate a response based on the conversation history.
        """
        messages = [{"role": "system", "content": base_prompt}] if base_prompt else []
        for user_msg, model_msg in zip(user_history, model_history):
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": model_msg})
        messages.append({"role": "user", "content": user_history[-1]})

        @backoff.on_exception(backoff.expo, Exception)
        async def completions_with_backoff(**kwargs):
            return await self.client.chat.completions.create(**kwargs)

        response: ChatCompletion = await completions_with_backoff(
            messages=messages,
            model="gpt-4o-mini",
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()

    async def generate_batch(self, batch_data):
        """
        Generate responses for a batch of conversation histories.
        """
        tasks = [self.generate(base_prompt, user_history, model_history)
                 for base_prompt, user_history, model_history in batch_data]
        return await asyncio.gather(*tasks)

    async def get_probabilities(self, context, claim):
        """
        Retrieve token probabilities for the next token after the given context and claim.
        """
        prompt = context + claim

        @backoff.on_exception(backoff.expo, Exception)
        async def completions_with_backoff(**kwargs):
            return await self.client.chat.completions.create(**kwargs)

        response: ChatCompletion = await completions_with_backoff(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=20
        )

        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        word_probs = {
            logprob_entry.token.strip(): np.exp(logprob_entry.logprob)
            for logprob_entry in top_logprobs
        }
        return word_probs

    async def get_probabilities_batch(self, batch_data):
        """
        Retrieve token probabilities for a batch of contexts and claims.
        """
        tasks = [self.get_probabilities(context, claim) for context, claim in batch_data]
        return await asyncio.gather(*tasks)

    async def calculate_perplexity(self, context, claim):
        """
        Calculate the perplexity of the claim given the context.
        """
        words = claim.split()
        if not words:
            return float('inf')

        total_log_prob = 0.0
        for i in range(len(words)):
            partial_claim = " ".join(words[:i])
            word_probs = await self.get_probabilities(context, partial_claim)
            if not word_probs:
                print("Error in retrieving probabilities.")
                return float("inf")
            current_word = words[i]
            word_prob = word_probs.get(current_word, min(word_probs.values()))
            total_log_prob += np.log(word_prob)
        perplexity = np.exp(-total_log_prob / len(words))
        return perplexity

    async def calculate_probability(self, context, claim):
        """
        Compute the arithmetic mean probability of a claim given the context.
        """
        words = claim.split()
        if not words:
            return 0.0

        batch_data = [(context, " ".join(words[:i])) for i in range(len(words))]
        results = await self.get_probabilities_batch(batch_data)

        total_prob = 0.0
        count = 0
        for i, word_probs in enumerate(results):
            if not word_probs:
                print("Error in retrieving probabilities.")
                return 0.0
            current_word = words[i]
            word_prob = sum(prob for token, prob in word_probs.items() if token.lower() == current_word.lower())
            if word_prob == 0 and word_probs:
                word_prob = min(word_probs.values())
            total_prob += word_prob
            count += 1
        mean_prob = total_prob / count if count > 0 else 0.0
        return mean_prob

    async def get_word_probability(self, context, word):
        """
        Retrieve the probability of the specified word given the context.
        """
        word_probs = await self.get_probabilities(context, "")
        return word_probs.get(word, 0.0)