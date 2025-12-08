from litellm import completion
import litellm

# 启用调试模式查看详细信息
# litellm.set_verbose = True

api_key = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
model_id = "Claude-Sonnet-4.5"

messages = [{"role": "user", "content": "你好,介绍一下AMD的产品"}]

# 方法1: vertex_ai_beta (Anthropic on Vertex AI)
print("方法1: vertex_ai_beta (Anthropic on Vertex AI)")
print("="*80)
try:
    response = completion(
        model=f"vertex_ai_beta/{model_id}",
        messages=messages,
        api_base=f"https://llm-api.amd.com/claude3/{model_id}",
        api_key=api_key,
        extra_headers={'Ocp-Apim-Subscription-Key': api_key},
        temperature=1.0,
        max_tokens=8000
    )
    print("成功!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")

# 方法2: anthropic_vertex
print("\n方法2: anthropic_vertex")
print("="*80)
try:
    response = completion(
        model=f"anthropic_vertex/{model_id}",
        messages=messages,
        api_base=f"https://llm-api.amd.com/claude3/{model_id}",
        api_key=api_key,
        extra_headers={'Ocp-Apim-Subscription-Key': api_key},
        temperature=1.0,
        max_tokens=8000
    )
    print("成功!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")

# 方法3: openai_like (通用OpenAI兼容，但返回是Anthropic格式)
print("\n方法3: openai_like 带完整URL")
print("="*80)
try:
    response = completion(
        model=model_id,
        messages=messages,
        api_base=f"https://llm-api.amd.com/claude3/{model_id}/chat/completions",
        api_key=api_key,
        custom_llm_provider="openai_like",
        extra_headers={'Ocp-Apim-Subscription-Key': api_key},
        temperature=1.0,
        max_tokens=8000
    )
    print("成功!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")
