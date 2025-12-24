# Copyright (c) Microsoft. All rights reserved.

"""LiteLLM AMD Model implementation for AMD internal API (GPT-5)."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import litellm
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("litellm_amd_model")


@dataclass
class LitellmAMDModelConfig:
    """Configuration for AMD LiteLLM model."""
    model_name: str
    api_key: str = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
    api_base: str = "https://llm-api.amd.com/azure/engines"
    temperature: float = 1.0
    max_tokens: int = 8000
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    top_k:int = 40


class LiteLLMAMDModel:
    """LiteLLM model implementation for AMD internal API.
    
    This model uses AMD's internal LLM API to access models like GPT-5.
    Compatible with the standard LitellmModel interface.
    """

    def __init__(self, **kwargs):
        """Initialize the LiteLLM AMD model.

        Args:
            **kwargs: Configuration parameters including:
                - model_name: The model ID (e.g., "gpt-5")
                - api_key: AMD API subscription key
                - api_base: Base URL for AMD LLM API
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - model_kwargs: Additional model parameters
        """
        self.config = LitellmAMDModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        
        # Determine API base URL based on model type
        self.api_base_full = self._get_api_base_url()
        
        # Register custom model if registry is provided
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(
                json.loads(Path(self.config.litellm_model_registry).read_text())
            )
        
        logger.info(
            f"Initialized LiteLLMAMDModel with model={self.config.model_name}, "
            f"api_base={self.api_base_full}"
        )
    
    def _get_api_base_url(self) -> str:
        """Get the appropriate API base URL for the model.
        
        Returns:
            Full API base URL for the model
        """
        model = self.config.model_name
        
        if model in ["gemini-3-pro", "gemini-2.5-pro", "gemini-2.5-flash"]:
            return f"https://llm-api.amd.com/vertex/gemini/deployments/{model}"
        elif model in ["claude-3.5", "claude-3.7", "claude-4", "claude-4.5", "Claude-Sonnet-4.5"]:
            return f"https://llm-api.amd.com/claude3/{model}"
        elif model in ["DeepSeek-R1", "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-14B"]:
            return f"https://llm-api.amd.com/api/OnPrem/deployments/{model}"
        else:
            # Default: Azure/GPT models
            return f"{self.config.api_base}/{model}"

    def _is_amd_api_model(self, model: str) -> bool:
        """Check if the model uses AMD API."""
        return model in [
            "gpt-5", "claude-4", "o3", "claude-4.5", "gpt-5.1-codex", "o3-mini",
            "gpt-4.1", "gpt-4.1-mini", "o1", "o1-mini", "o1-preview", "o4-mini", "gpt-4o",
            "claude-3.5", "claude-3.7", "gemini-3-pro", "gemini-2.5-pro", "gemini-2.5-flash",
            "DeepSeek-R1", "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-14B",
            "Claude-Sonnet-4.5"
        ]

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=60, max=512),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                requests.exceptions.InvalidURL,
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                litellm.exceptions.BadRequestError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Internal query method with retry logic.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments
            
        Returns:
            Response object (dict for AMD API, litellm response for local models)
        """
        model = self.config.model_name
        
        # Merge default kwargs with call-specific kwargs
        merged_kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        merged_kwargs.update(self.config.model_kwargs)
        merged_kwargs.update(kwargs)

        # print(f"is_amd_api_model={self._is_amd_api_model(model)}")
        # print(f"model is {model}")
        
        if self._is_amd_api_model(model):
            # Use requests directly for AMD API
            return self._query_amd_api(messages, merged_kwargs)
        else:
            # Use litellm for local models
            return self._query_local_model(messages, merged_kwargs)

    def _query_amd_api(self, messages: list[dict[str, str]], merged_kwargs: dict) -> dict:
        """Query AMD API directly using requests.
        
        Args:
            messages: List of message dictionaries
            merged_kwargs: Merged configuration kwargs
            
        Returns:
            Response dictionary with 'content' field
        """
        model = self.config.model_name
        
        # Prepare request body
        body = {
            "messages": messages,
            "temperature": merged_kwargs.get("temperature", self.config.temperature),
            "stream": False,
            "max_completion_tokens": merged_kwargs.get("max_tokens", self.config.max_tokens),
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
        
        # Determine server and adjust body based on model type
        if model in ["DeepSeek-R1", "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-14B"]:
            server = "https://llm-api.amd.com/api/OnPrem/deployments"
            body['max_completion_tokens'] = 32000
        elif model in ["gpt-4.1", "o3-mini", "gpt-4.1-mini", "o1", "o1-mini", "o1-preview", "o3", "o4-mini", "gpt-4o",
                      "gpt-5", "gpt-5.1-codex"]:
            server = "https://llm-api.amd.com/azure/engines"
        elif model in ["gemini-3-pro", "gemini-2.5-pro", "gemini-2.5-flash"]:
            server = "https://llm-api.amd.com/vertex/gemini/deployments"
            body["max_tokens"] = body["max_completion_tokens"]
        elif model in ["claude-3.5", "claude-3.7", "claude-4", "claude-4.5", "Claude-Sonnet-4.5"]:
            server = "https://llm-api.amd.com/claude3"
            body['max_completion_tokens'] = min(body.get('max_completion_tokens', 8192), 8192)
            body["max_tokens"] = body["max_completion_tokens"]
        else:
            raise ValueError(f"Unsupported AMD API model: {model}")
        
        headers = {"Ocp-Apim-Subscription-Key": self.config.api_key}
        
        try:
            response = requests.post(
                url=f"{server}/{model}/chat/completions",
                json=body,
                headers=headers,
                timeout=800
            )
            response.raise_for_status()
            response_json = response.json()
            
            # Extract content based on response format
            if model in ["gemini-3-pro", "gemini-2.5-pro", "gemini-2.5-flash"]:
                content = response_json['candidates'][0]['content']['parts'][0]['text']
            elif model in ["claude-3.5", "claude-3.7", "claude-4", "claude-4.5", "Claude-Sonnet-4.5"]:
                content = response_json['content'][0]['text']
            else:
                content = response_json['choices'][0]['message']['content']
            
            return {"content": content, "response_json": response_json}
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing response: {e}")
            raise
    
    def _query_local_model(self, messages: list[dict[str, str]], merged_kwargs: dict):
        """Query local model using litellm.
        
        Args:
            messages: List of message dictionaries
            merged_kwargs: Merged configuration kwargs
            
        Returns:
            LiteLLM completion response object
        """
        model = self.config.model_name
        
        try:
            response = litellm.completion(
                model=f"hosted_vllm/{model}",
                messages=messages,
                api_base="http://localhost:30001/v1",
                api_key="dummy",
                timeout=8000,
                **merged_kwargs
            )
            return response
            
        except litellm.exceptions.AuthenticationError as e:
            e.message += (
                " Please verify your API configuration. "
            )
            raise e
        except Exception as e:
            logger.error(f"Error querying local model: {e}")
            raise

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the LLM API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - content: The model's response content as a string
                - extra: Additional metadata including the full response
        """
        response = self._query(messages, **kwargs)
        
        # Calculate cost (set to 0 for AMD internal API)
        cost = 0.0
        
        # Update statistics
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        
        # Extract content based on response type
        if isinstance(response, dict):
            # AMD API response (from requests)
            content = response.get("content", "")
            extra_data = {"response": response.get("response_json", {})}
        else:
            # LiteLLM response (from local model)
            content = response.choices[0].message.content or ""
            extra_data = {"response": response.model_dump()}
        
        logger.debug(f"Received response (length: {len(content)})")
        
        return {
            "content": content,
            "extra": extra_data,
        }

    def get_template_vars(self) -> dict[str, Any]:
        """Get template variables for logging and reporting.
        
        Returns:
            Dictionary containing configuration and statistics
        """
        return asdict(self.config) | {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }

    def __repr__(self) -> str:
        return (
            f"LiteLLMAMDModel(model_name={self.config.model_name}, "
            f"api_base={self.api_base_full}, "
            f"calls={self.n_calls}, "
            f"cost=${self.cost:.4f})"
        )
