pip install --upgrade vllm
pip install --upgrade mistral_common

from vllm import LLM
from vllm.sampling_params import SamplingParams

# Initialize the model
llm = LLM(model="mistralai/Mistral-Small-Instruct-2409")

# Define sampling parameters as needed
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Generate a response
prompt = "Your input prompt here."
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].text)
