# Copyright (c) Microsoft. All rights reserved.

"""LiteLLM AMD Model implementation for AMD internal API (GPT-5)."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import litellm
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
        
        # Construct full API base URL
        self.api_base_full = f"{self.config.api_base}/{self.config.model_name}"
        
        # Register custom model if registry is provided
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(
                json.loads(Path(self.config.litellm_model_registry).read_text())
            )
        
        logger.info(
            f"Initialized LiteLLMAMDModel with model={self.config.model_name}, "
            f"api_base={self.api_base_full}"
        )

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Internal query method with retry logic.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments for litellm.completion
            
        Returns:
            LiteLLM completion response object
        """
        # Merge default kwargs with call-specific kwargs
        merged_kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        merged_kwargs.update(self.config.model_kwargs)
        merged_kwargs.update(kwargs)
        
        try:
            if self.config.model_name in ["gpt-5", "claude-4"]:
                response = litellm.completion(
                    model=f"openai/{self.config.model_name}",
                    messages=messages,
                    api_base=self.api_base_full,
                    api_key=self.config.api_key,
                    resoning_effort="high",
                    extra_headers={
                        'Ocp-Apim-Subscription-Key': self.config.api_key
                    },
                    **merged_kwargs
                )
            else:
                response = litellm.completion(
                model=f"openai/{self.config.model_name}",
                messages=messages,
                api_base = "http://localhost:30000/v1",
                api_key="dummy",
                timeout= 8000,
                **(self.config.model_kwargs | kwargs)
            )

            return response
            
        except litellm.exceptions.AuthenticationError as e:
            e.message += (
                " Please verify your AMD API key. "
                "You can set it with the --api-key parameter or "
                "in the model configuration."
            )
            raise e
        except Exception as e:
            logger.error(f"Error querying AMD LLM API: {e}")
            raise

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the AMD LLM API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments for litellm.completion

        Returns:
            Dictionary containing:
                - content: The model's response content as a string
                - extra: Additional metadata including the full response
        """
        response = self._query(messages, **kwargs)
        
        # Calculate cost (set to 0 for AMD internal API)
        try:
            # For AMD internal API, we don't calculate cost
            # If you have a cost model, you can implement it here
            cost = 0.0
            
            # Alternatively, if you want to use litellm's cost calculator:
            # cost = litellm.cost_calculator.completion_cost(response)
        except Exception as e:
            logger.warning(
                f"Could not calculate cost for model {self.config.model_name}: {e}. "
                "Setting cost to 0."
            )
            cost = 0.0
        
        # Update statistics
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        
        # Extract content
        content = response.choices[0].message.content or ""
        
        logger.debug(f"Received response (length: {len(content)})")
        
        return {
            "content": content,
            "extra": {
                "response": response.model_dump(),
            },
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
