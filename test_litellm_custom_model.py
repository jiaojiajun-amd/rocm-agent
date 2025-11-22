from litellm import completion
import litellm
import os

# 关闭 litellm 的自动路由
os.environ["LITELLM_LOG"] = "DEBUG"

api_key = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
model_id = "o3-mini"

messages = [{"role": "user", "content": "你好,介绍一下AMD的产品"}]

print("="*80)
print("测试方案: 使用 hosted_vllm 提供商(通用OpenAI兼容)")
print("="*80)

# 使用 hosted_vllm 提供商，这是一个通用的 OpenAI 兼容接口
# 它会直接使用提供的 api_base，不会自动添加额外的路径
try:
    response = completion(
        model="hosted_vllm/o3-mini",  # 使用 hosted_vllm 前缀
        messages=messages,
        api_base=f"https://llm-api.amd.com/azure/engines/{model_id}/chat/completions",
        api_key=api_key,
        custom_llm_provider="openai",  # 指定使用 OpenAI 格式
        extra_headers={
            'Ocp-Apim-Subscription-Key': api_key
        },
        temperature=1.0,
        max_tokens=8000
    )
    
    print("✓ 成功!")
    print(f"\n响应内容:")
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"✗ 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试方案: 直接使用 openai_like 提供商")
print("="*80)

try:
    response = completion(
        model=model_id,
        messages=messages,
        api_base=f"https://llm-api.amd.com/azure/engines/{model_id}/chat/completions",
        api_key=api_key,
        custom_llm_provider="openai_like",
        extra_headers={
            'Ocp-Apim-Subscription-Key': api_key
        },
        temperature=1.0,
        max_tokens=8000
    )
    
    print("✓ 成功!")
    print(f"\n响应内容:")
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"✗ 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

