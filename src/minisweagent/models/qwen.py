import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import openai
from openai import OpenAI

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("openrouter_model")


@dataclass
class QwenModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class QwenAPIError(Exception):
    """Custom exception for Qwen API errors."""

    pass


class QwenAuthenticationError(Exception):
    """Custom exception for Qwen authentication errors."""

    pass


class QwenRateLimitError(Exception):
    """Custom exception for Qwen rate limit errors."""

    pass


class QwenModel:
    def __init__(self, **kwargs):
        self.config = QwenModelConfig(**kwargs)
        endpoint = self.config.model_kwargs["endpoint"]
        self.cost = 0.0
        self.n_calls = 0
        self._api_url = f"{endpoint}/chat/completions"
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.endpoint = endpoint

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=600),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                QwenAuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )


    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            # 初始化 OpenAI client
            client = OpenAI(
                api_key=self._api_key,
                base_url=self.endpoint  # 如果使用自定义端点
            )
            
            # 合并配置参数
            request_params = {
                "model": self.config.model_name,
                "messages": messages,
                # **(self.config.model_kwargs | kwargs),
            }
            
            # 调用 API
            response = client.chat.completions.create(**request_params)
            
            # 返回格式化的响应
            return {
                "choices": [
                    {
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content,
                        },
                        "finish_reason": response.choices[0].finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "model": response.model,
                "id": response.id,
            }
            
        except openai.AuthenticationError as e:
            error_msg = "Authentication failed. You can permanently set your API key with `mini-extra config set OPENROUTER_API_KEY YOUR_KEY`."
            raise QwenAuthenticationError(error_msg) from e
        except openai.RateLimitError as e:
            raise QwenRateLimitError("Rate limit exceeded") from e
        except openai.APIError as e:
            raise QwenAPIError(f"API Error: {e}") from e
        except Exception as e:
            raise QwenAPIError(f"Request failed: {e}") from e


    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)

        # Extract cost from usage information
        usage = response.get("usage", {})
        cost = usage.get("cost", 0.0)

        # If total_cost is not available, raise an error
        # if cost == 0.0:
        #     raise QwenAPIError(
        #         f"No cost information available from Qwen API for model {self.config.model_name}. "
        #         "Cost tracking is required but not provided by the API response."
        #     )

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        return {
            "content": response["choices"][0]["message"]["content"] or "",
            "extra": {
                "response": response,  # already is json
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
