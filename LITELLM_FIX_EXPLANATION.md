# LiteLLM AMD API 问题分析和解决方案

## 问题根源

### requests 版本 (✓ 工作)
```python
url = "https://llm-api.amd.com/azure/engines/o3-mini/chat/completions"
```
- 直接使用完整的 URL，包含 `/chat/completions` 端点

### litellm 版本 (✗ 原本不工作)
```python
api_base = "https://llm-api.amd.com/azure/engines/o3-mini"
model = "openai/o3-mini"
```

**问题：**
1. `api_base` 缺少 `/chat/completions` 后缀
2. 使用 `openai/` 前缀时，litellm 会自动构建路径
3. 但 litellm 错误地访问了 `/responses` 端点而不是 `/chat/completions`

**错误的 URL：**
```
https://llm-api.amd.com/azure/engines/gpt-5/responses  ❌
```

**正确的 URL 应该是：**
```
https://llm-api.amd.com/azure/engines/gpt-5/chat/completions  ✓
```

## 解决方案

### 方案 1：使用 Azure 前缀（推荐）

```python
response = completion(
    model=f"azure/{model_id}",  # 改用 azure/ 前缀
    messages=messages,
    api_base=api_base,  # 保持不变
    api_key=api_key,
    extra_headers={'Ocp-Apim-Subscription-Key': api_key},
    temperature=1.0,
    max_tokens=8000
)
```

**原理：** litellm 使用 `azure/` 前缀时，会使用 Azure OpenAI 的路径约定，自动添加正确的 `/chat/completions` 端点。

### 方案 2：完整 URL（备选）

```python
api_base = f"https://llm-api.amd.com/azure/engines/{model_id}/chat/completions"

response = completion(
    model=f"openai/{model_id}",
    messages=messages,
    api_base=api_base,  # 使用完整 URL
    api_key=api_key,
    extra_headers={'Ocp-Apim-Subscription-Key': api_key},
    temperature=1.0,
    max_tokens=8000
)
```

## 测试文件

1. **test_requests_amd_simple.py** - 使用 requests 库的工作版本
2. **test_litellm_amd.py** - 使用 litellm 的修复版本
3. **test_litellm_amd_debug.py** - 带调试输出的版本

## 运行测试

```bash
# 测试 requests 版本（已确认工作）
python test_requests_amd_simple.py

# 测试修复后的 litellm 版本
python test_litellm_amd.py

# 测试带调试信息的版本
python test_litellm_amd_debug.py
```

## 总结

关键差异在于 litellm 如何根据模型前缀（`openai/` vs `azure/`）来构建完整的 API 端点路径：

- `openai/` → litellm 添加 OpenAI 风格的路径（可能是 `/responses` 或其他）
- `azure/` → litellm 添加 Azure OpenAI 风格的路径（`/chat/completions`）

由于 AMD 的 API 使用 Azure OpenAI 兼容格式，应该使用 `azure/` 前缀。

