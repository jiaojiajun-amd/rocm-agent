import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any
from datetime import datetime
from pathlib import Path

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
        self.client = OpenAI(
                api_key=self._api_key,
                base_url=self.endpoint,  # 如果使用自定义端点
                timeout=600,  # 设置为 None 表示无超时限制
                max_retries=0,
            )
        
        self.error_log_dir = Path('./error_logs')
        self.error_log_dir.mkdir(parents=True, exist_ok=True)
        
    def _save_error_messages(
            self, 
            messages: list[dict[str, str]], 
            error: Exception,
            request_params: dict = None
        ) -> str:
        """
        保存错误时的消息到文件
        
        Args:
            messages: 发送的消息列表
            error: 捕获的异常
            request_params: 完整的请求参数
            
        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        error_type = type(error).__name__
        filename = f"error_{error_type}_{timestamp}.json"
        filepath = self.error_log_dir / filename
        
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "messages": messages,
            "request_params": request_params or {},
            "endpoint": self.endpoint,
            "model": self.config.model_name if hasattr(self, 'config') else "unknown",
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Error messages saved to: {filepath}")
            return str(filepath)
        except Exception as save_error:
            logger.error(f"Failed to save error messages: {save_error}")
            return ""

    @retry(
        stop=stop_after_attempt(30),
        wait=wait_exponential(multiplier=1, min=4, max=1025),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                QwenAuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        request_params = None  # 添加这一行：初始化 request_params
        try:
            # 初始化 OpenAI client
            client = self.client
            
            # 合并配置参数
            request_params = {
                "model": self.config.model_name,
                "messages": messages[-5:],
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
            # 添加：保存错误消息
            self._save_error_messages(messages, e, request_params)
            error_msg = "Authentication failed. You can permanently set your API key with `mini-extra config set OPENROUTER_API_KEY YOUR_KEY`."
            raise QwenAuthenticationError(error_msg) from e
        except openai.RateLimitError as e:
            # 添加：保存错误消息
            self._save_error_messages(messages, e, request_params)
            raise QwenRateLimitError("Rate limit exceeded") from e
        except openai.APIError as e:
            # 添加：保存错误消息
            self._save_error_messages(messages, e, request_params)
            raise QwenAPIError(f"API Error: {e}") from e
        except Exception as e:
            # 添加：保存错误消息
            self._save_error_messages(messages, e, request_params)
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
