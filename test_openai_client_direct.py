from openai import OpenAI

api_key = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
model_id = "o3-mini"

# 使用 OpenAI 客户端直接访问
# 这样可以完全控制 URL 构建
client = OpenAI(
    api_key=api_key,
    base_url=f"https://llm-api.amd.com/azure/engines/{model_id}",
    default_headers={
        'Ocp-Apim-Subscription-Key': api_key
    }
)

messages = [{"role": "user", "content": "你好,介绍一下AMD的产品"}]

print("="*80)
print("使用原生 OpenAI 客户端测试")
print("="*80)
print(f"base_url: {client.base_url}")
print(f"model: {model_id}")
print("-"*80)

try:
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=1.0,
        max_tokens=8000
    )
    
    print("\n✓ 成功!")
    print(f"\n响应内容:")
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"\n✗ 失败: {type(e).__name__}")
    print(f"错误信息: {e}")
    import traceback
    traceback.print_exc()

