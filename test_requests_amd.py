import requests
import json

api_key = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
model_id = "o3-mini"

# 尝试不同的端点
endpoints = [
    f"https://llm-api.amd.com/azure/engines/{model_id}/chat/completions",
    f"https://llm-api.amd.com/azure/engines/{model_id}/completions",
    f"https://llm-api.amd.com/azure/deployments/{model_id}/chat/completions",
    f"https://llm-api.amd.com/v1/chat/completions",
]

headers = {
    "Content-Type": "application/json",
    "api-key": api_key,
    "Ocp-Apim-Subscription-Key": api_key,
}

payload = {
    "messages": [
        {"role": "user", "content": "你好,介绍一下AMD的产品"}
    ],
    "temperature": 1.0,
    "max_tokens": 8000,
}

print("测试不同的 API 端点...\n")

for i, url in enumerate(endpoints, 1):
    print(f"尝试端点 {i}: {url}")
    print("-" * 80)
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}\n")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ 成功!")
            print(f"响应内容: {json.dumps(data, indent=2, ensure_ascii=False)}")
            print("\n" + "="*80 + "\n")
            break
        else:
            print(f"✗ 失败")
            print(f"响应内容: {response.text}")
            print("\n" + "="*80 + "\n")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ 请求异常: {e}")
        print("\n" + "="*80 + "\n")

