# Packages
from pydantic import BaseModel
from openai import OpenAI
import time
import logging
import os
import random as r
from dotenv import load_dotenv
from typing import Optional

# Load configs (GPT api key, etc.)
load_dotenv(os.path.join(os.path.dirname(os.getcwd()), "configs/configs.env"))


class GPTConfig(BaseModel):
    model: str = 'gpt-4o-2024-08-06'
    temperature: float = 0
    max_tokens: int = 1300
    top_p: float = 1
    seed: Optional[int] = None
    frequency_penalty: float = 0
    presence_penalty: float = 0


class GPT:
    def __init__(self, api_key: str, organization: str, project: str, config: GPTConfig):
        """
        Initialize GPT instance with client and parameters.
        
        Args:
            organization (str): Organization ID.
            project (str): Project ID.
            project (str, optional): Project ID. Defaults to None.
            config (GPTConfig object): GPT model settings/configurations.
        """
        self.client = OpenAI(api_key=api_key, organization=organization, project=project)
        self.config = config

    @classmethod
    def default(cls):
        """
        Creating a default GPT instance.
        """
        return cls(
            api_key=os.environ.get("API_KEY"),
            organization=os.environ.get("ORG"),
            project=os.environ.get("PROJECT"),
            config=GPTConfig()
        )

    def gpt_request(self, sys: str, user: str, output_structure: object = None, max_retries: int = 3, timeout: int = 30) -> dict:
        """
        GPT request function with retry on timeout
        
        Args:
            sys (str): System prompt.
            user (str): User prompt.
            output_structure (object): The JSON structure of the output. Only used if the model is 'gpt-4o-2024-08-06' or later. Defaults to None.
            max_retries (int): The number of request retries if error or timeout. Defaults to 3.
            timeout (int): The max time to wait for output. Defaults to 30.

        Returns:
            dict: 'response' key corresponding to request output in string format, 'meta' key corresponding to dictionary of meta data.
        """
        # Requests depend on the model being used
        client_types = {
            'gpt-4o-2024-08-06': self.client.beta.chat.completions.parse,
            'gpt-3.5-turbo': self.client.chat.completions.create,
            'gpt-4': self.client.chat.completions.create,
            'gpt-4-32k': self.client.chat.completions.create,
        }
        retries = 0
        while retries < max_retries:
            try:
                start_time = time.time()
                request_params = {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p,
                    "seed": self.config.seed,
                    "frequency_penalty": self.config.frequency_penalty,
                    "presence_penalty": self.config.presence_penalty,
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user}
                    ]
                }

                if self.config.model not in client_types:
                    raise ValueError(f"Unsupported model: {self.config.model}")
                
                # Include `response_format` only if `output_structure` is not None
                if output_structure is not None:
                    request_params["response_format"] = output_structure

                # Make request
                response = client_types[self.config.model](**request_params)

                # Timeout check
                if time.time() - start_time > timeout:
                    raise TimeoutError("Request exceeded timeout.")
                return {
                    'response': response.choices[0].message.content,
                    'meta': response,
                }
            except Exception as e:
                logging.error(f"Error on attempt {retries + 1}: {e}")
                time.sleep(2.0)
                time.sleep(r.uniform(0.5, 2.0))

        raise TimeoutError(f"Request failed after {max_retries} retries.")