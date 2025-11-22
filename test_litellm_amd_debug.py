from litellm import completion
import litellm

# 开启 litellm 调试模式
litellm.set_verbose = True

api_key = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
model_id = "o3-mini"

# 根据您的代码，完整 URL 是:
# https://llm-api.amd.com/azure/engines/gpt-5/chat/completions
api_base = f"https://llm-api.amd.com/azure/engines/{model_id}"

messages = [{"role": "user", "content": "你好,介绍一下AMD的产品"}]

print("="*80)
print("测试 litellm 使用 Azure 前缀")
print("="*80)
print(f"api_base: {api_base}")
print(f"model: azure/{model_id}")
print("="*80)

try:
    response = completion(
        model=f"azure/{model_id}",
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        extra_headers={
            'Ocp-Apim-Subscription-Key': api_key
        },
        temperature=1.0,
        max_tokens=8000
    )
    
    print("\n✓ 成功!")
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()

