from litellm import completion

api_key = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
model_id = "gpt-5"

# 根据您的代码，完整 URL 是:
# https://llm-api.amd.com/azure/engines/gpt-5/chat/completions
api_base = f"https://llm-api.amd.com/azure/engines/{model_id}"

messages = [{"role": "user", "content": "你好,介绍一下AMD的产品"}]

try:
    response = completion(
        model=f"openai/{model_id}",
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        extra_headers={
            'Ocp-Apim-Subscription-Key': api_key
        },
        temperature=1.0,
        max_tokens=8000
    )
    
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
