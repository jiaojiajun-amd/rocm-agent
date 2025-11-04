from litellm import completion
import litellm 

# # 调用本地 vLLM 服务
# response = completion(
#     model="hosted_vllm/Qwen/Qwen3-4B",  # 使用 openai/ 前缀
#     messages=[{"role": "user", "content": "你好,介绍一下AMD的产品"}],
#     api_base="http://localhost:8000/v1",  # vLLM 服务地址
#     api_key="dummy"  # vLLM 不需要真实 API key,但需要提供一个值
# )

# print(response.choices[0].message.content)

from litellm import completion
import litellm

litellm.set_verbose = True

api_key = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
MODEL_API_URL = "https://llm.amd.com"
model_id = "gpt-5"

# 方式 1: 如果 API 期望的是 /openai/deployments/{model_id}/chat/completions
litellm_params = {
    "model": "openai/gpt-5",
    "api_base": f"{MODEL_API_URL}/openai/deployments/{model_id}/v1",
    "api_key": api_key,
    "extra_headers": {
        'Ocp-Apim-Subscription-Key': api_key
    }
}

messages = [{"role": "user", "content": "你好,介绍一下AMD的产品"}]

try:
    response = completion(**litellm_params, messages=messages)
    print(response.choices[0].message.content)
except Exception as e:
    print(f"错误: {e}")



# print(response.choices[0].message.content)