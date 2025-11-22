import requests
import json

api_key = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
model_id = "o3-mini"

# Azure OpenAI 标准端点格式
url = f"https://llm-api.amd.com/azure/engines/{model_id}/chat/completions"


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

print(f"请求 URL: {url}")
print(f"请求头: {json.dumps(headers, indent=2)}")
print(f"请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")
print("\n" + "="*80 + "\n")

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30
    )
    
    print(f"状态码: {response.status_code}")
    print(f"响应头: {json.dumps(dict(response.headers), indent=2)}")
    print()
    
    if response.status_code == 200:
        data = response.json()
        print("✓ 成功!")
        print(f"\n完整响应:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")
            print(f"\n提取的回复内容:")
            print(content)
    else:
        print("✗ 失败")
        print(f"\n响应内容:")
        print(response.text)
        
except requests.exceptions.RequestException as e:
    print(f"✗ 请求异常: {e}")
    import traceback
    traceback.print_exc()

