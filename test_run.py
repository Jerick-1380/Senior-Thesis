import os

os.environ["HF_HOME"] = "/data/user_data/junkais/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/junkais/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data/user_data/junkais/hf_cache/datasets"
os.environ["HF_METRICS_CACHE"] = "/data/user_data/junkais/hf_cache/metrics"

import time
from vllm import LLM, SamplingParams

# Define the prompt and sampling parameters
prompt = '''
Given the following information, do you think stock prices will go up or down:

Commercial real estate activity in the First District weakened further modestly in recent months. In
the already-weak office market, vacancy rates increased moderately on average, and Providence in
particular saw the exit of a large downtown tenant. Office rents fell noticeably in the Boston area
in recent months but were reportedly stable (if low) elsewhere. Demand for life sciences space in
greater Boston dwindled further to very low levels. In the retail market, rents and vacancy rates
were mostly steady at moderate levels, although lower-end malls continued to see elevated vacancies. Demand for industrial space slowed further at a modest pace, but rents and occupancy rates
were described as mostly stable at healthy levels. Projections for commercial real estate activity in
2024 were mixed but remained pessimistic on balance. Some contacts predicted an increase in
investment activity driven by declining interest rates. Others remained concerned that office buildings would face rising foreclosure rates even with some decline in interest rates. The industrial
property market faced modest downside risks, and the outlook for Boston’s life sciences properties deteriorated in response to weak recent demand.

You should return 1 if you think stock prices will go up, -1 if it goes down, 0 if no change.
Ensure formatting is correct. You should return 1 number and nothing else.
'''
sampling_params = SamplingParams(temperature=0, max_tokens=8192)


# Function to benchmark a given model
def benchmark_model(model_name, batch_size, num_gpus):
    print("Init\n",flush=True)
    # Initialize the model
    model = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
       # pipeline_parallel_size=2,
        gpu_memory_utilization=0.95,  # Adjust this value to use most of the 40GB VRAM
        max_model_len=4096  # Set this lower if memory issues occur, adjust as needed
    )

    # Warm-up
    print("Warm up\n",flush=True)
    for i in range(5):
        outputs = model.generate([prompt] * batch_size, sampling_params)
        for idx, output in enumerate(outputs):
            print(f"Warm-up {i+1}, Response {idx+1}: {output.outputs[0].text}")

    print("Benchmark\n",flush=True)
    # Benchmarking
    start_time = time.time()
    for _ in range(10):
        model.generate([prompt] * batch_size, sampling_params)
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    throughput = (batch_size * 10) / total_time
    print(f"Model: {model_name}, Batch Size: {batch_size}, GPUs: {num_gpus}, Throughput: {throughput:.2f} requests/sec",flush=True)

# Define configurations to test
configurations = [
    # {"model_name": "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", "batch_size": 1, "num_gpus": 1},
    {"model_name": "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", "batch_size": 1, "num_gpus": 1},
    # {"model_name": "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8", "batch_size": 2, "num_gpus": 2},
    # Add more configurations as needed
]

# Run benchmarks
for config in configurations:
    benchmark_model(config["model_name"], config["batch_size"], config["num_gpus"])