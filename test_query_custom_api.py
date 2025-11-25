import pytest
import requests
from tenacity import retry, wait_fixed, retry_if_exception_type
import logging

logger = logging.getLogger(__name__)


@retry(wait=wait_fixed(60),
       retry=retry_if_exception_type(Exception),
       before_sleep=lambda _: logger.info("API error detected, retrying in 5 minutes..."))
def query_custom_api(model_str, messages, temp=None, max_length=8000):
    body = {
        "messages": messages,
        "temperature": temp or 1,
        "stream": False,
        "max_completion_tokens": max_length,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "reasoning_Effort": "low"
    }
    SERVER = ''
    if model_str in ["DeepSeek-R1", "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-14B"]:
        SERVER = "https://llm-api.amd.com/api/OnPrem/deployments"
        body['max_completion_tokens'] = 32000
    elif model_str in ["gpt-4.1", "o3-mini", "gpt-4.1-mini", "o1", "o1-mini", "o1-preview", "o3", "o4-mini", "gpt-4o"]:
        SERVER = "https://llm-api.amd.com/azure/engines"
    elif model_str in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro-preview"]:
        SERVER = "https://llm-api.amd.com/vertex/gemini/deployments/"
        body["max_tokens"] = body["max_completion_tokens"]
    elif model_str in ["claude-3.5", "claude-3.7", "claude-4", "Claude-Sonnet-4.5"]:
        SERVER = "https://llm-api.amd.com/AnthropicVertex/deployments"
        body['max_completion_tokens'] = 8192
        body["max_tokens"] = body["max_completion_tokens"]
    else:
        raise Exception("unsupported model")
        
    HEADERS = {"Ocp-Apim-Subscription-Key": "c1f7f3ee59064fc0a5fad8c2586f1bd9"}
    
    try:
        response = requests.post(
            url=f"{SERVER}/{model_str}/chat/completions",
            json=body,
            headers=HEADERS
        )
        response.raise_for_status()
        print(response.json())
        if model_str in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro-preview"]:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        elif model_str in ["claude-3.5", "claude-3.7", "Claude-Sonnet-4.5"]:
            return response.json()['content'][0]['text']
        return response.json()['choices'][0]['message']['content']
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise


messages = [{"role": "user", "content": "你是anthropic 哪个模型"}]

print(query_custom_api("claude-4.5", messages))