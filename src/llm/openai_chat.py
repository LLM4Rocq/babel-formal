from typing import List, Dict

import openai

from .base import BaseLLM


class OpenAIChatLLM(BaseLLM):
    """Class for LLM providers compatible with OpenAI Chat Completions API."""

    def __init__(
        self,
        model_name: str = "",
        generation_parameters: dict = {},
        base_url: str = "http://127.0.0.1:30000/v1",
        api_key: str = "None",
    ):
        super().__init__()
        self.model_name = model_name
        self.generation_parameters = generation_parameters
        self.client = openai.Client(base_url=base_url)

    def generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate a chat completion using the LLM."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.generation_parameters,
            **kwargs
        )
        return response.choices[0].message.content
