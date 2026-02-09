from litellm import completion
import litellm

api_key = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
model_id = "o3-mini"

messages = [{"role": "user", "content": "你好,介绍一下AMD的产品"}]

# 测试配置列表
test_configs = [
    {
        "name": "方法1: azure/ + 完整URL",
        "model": f"azure/{model_id}",
        "api_base": f"https://llm-api.amd.com/azure/engines/{model_id}/chat/completions",
        "api_version": None,
    },
    {
        "name": "方法2: openai/ + 完整URL",
        "model": f"openai/{model_id}",
        "api_base": f"https://llm-api.amd.com/azure/engines/{model_id}/chat/completions",
        "api_version": None,
    },
    {
        "name": "方法3: azure/ + 基础URL + api_version",
        "model": f"azure/{model_id}",
        "api_base": f"https://llm-api.amd.com/azure/engines/{model_id}",
        "api_version": "2024-02-15-preview",
    },
    {
        "name": "方法4: 直接使用model_id (不加前缀) + 完整URL",
        "model": model_id,
        "api_base": f"https://llm-api.amd.com/azure/engines/{model_id}/chat/completions",
        "api_version": None,
    },
    {
        "name": "方法5: custom/ + 完整URL",
        "model": f"custom/{model_id}",
        "api_base": f"https://llm-api.amd.com/azure/engines/{model_id}/chat/completions",
        "api_version": None,
    },
]

for config in test_configs:
    print("\n" + "="*80)
    print(f"测试: {config['name']}")
    print("="*80)
    print(f"model: {config['model']}")
    print(f"api_base: {config['api_base']}")
    print(f"api_version: {config['api_version']}")
    print("-"*80)
    
    try:
        kwargs = {
            "model": config["model"],
            "messages": messages,
            "api_base": config["api_base"],
            "api_key": api_key,
            "extra_headers": {
                'Ocp-Apim-Subscription-Key': api_key
            },
            "temperature": 1.0,
            "max_tokens": 8000,
        }
        
        if config["api_version"]:
            kwargs["api_version"] = config["api_version"]
        
        response = completion(**kwargs)
        
        print("✓ 成功!")
        print(f"响应: {response.choices[0].message.content[:200]}...")
        print("\n找到工作的配置!")
        break
        
    except Exception as e:
        print(f"✗ 失败: {type(e).__name__}: {str(e)[:200]}")
        continue

print("\n" + "="*80)
print("测试完成")
print("="*80)

