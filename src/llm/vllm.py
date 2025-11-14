from typing import List, Optional, Dict

import openai
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .base import BaseLLM

class vLLM(BaseLLM):
    """Class for LLM providers compatible with OpenAI API."""

    def __init__(self, model_path: str="", base_url="http://127.0.0.1:30000/v1", api_key="None", tp=4):
        super().__init__()
        self.model_name = model_path
        self.client = openai.Client(base_url=base_url, api_key=api_key)

        self.llm = LLM(model=model_path, tokenizer=model_path, max_num_seqs=1, max_model_len=12_000, tensor_parallel_size=tp, dtype="bfloat16", gpu_memory_utilization=0.98, trust_remote_code=True)

        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=8192, top_p=0.9)
        self.sampling_params.n = 1
    
    def generate(
            self, prompt, **kwargs
    ) -> str:
        """Generate a completion using the LLM."""
        outputs = self.llm.generate([prompt], self.sampling_params)
        for output in outputs:
            for completion in output.outputs:
                return completion.text
        raise Exception('Nothing to return?')