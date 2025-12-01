# Model Special Tags Fix

## 问题描述

某些模型（如 `lmsys/gpt-oss-20b-bf16`）在响应中包含特殊标签：
- `<|channel|>` - 通道标记
- `<|message|>` - 消息开始标记
- `<|end|>` - 消息结束标记

这些标签会导致以下错误：

```
InternalServerError: You have passed a message containing <|channel|> tags 
in the content field. Instead of doing this, you should pass analysis messages 
(the string between '<|message|>' and '<|end|>') in the 'thinking' field, 
and final messages (the string between '<|message|>' and '<|end|>') in the 
'content' field.
```

## 解决方案

在 `LiteLLMAMDModel` 中添加了 `_clean_special_tags()` 方法：

### 1. 清理输入消息

在发送给模型之前，清理所有消息中的特殊标签：

```python
cleaned_messages = []
for msg in messages:
    cleaned_msg = msg.copy()
    if 'content' in cleaned_msg and cleaned_msg['content']:
        cleaned_msg['content'] = self._clean_special_tags(cleaned_msg['content'])
    cleaned_messages.append(cleaned_msg)
```

### 2. 清理输出响应

从模型响应中提取最终内容，移除特殊标签：

```python
raw_content = response.choices[0].message.content or ""
content = self._clean_special_tags(raw_content)
```

## 标签处理逻辑

`_clean_special_tags()` 方法的处理逻辑：

1. **提取最终消息**：如果内容包含 `<|message|>...<|end|>` 块，提取最后一个块
2. **移除通道标记**：删除所有 `<|channel|>` 及其后续内容（直到下一个 `<|message|>`）
3. **清理所有特殊标签**：移除任何剩余的 `<|...|>` 格式标签

## 影响的模型

目前已知受影响的模型：
- `lmsys/gpt-oss-20b-bf16`
- 其他可能使用类似标签格式的模型

对于不使用这些标签的模型（如 `gpt-5`, `Qwen/Qwen3-8B`），此修复不会产生负面影响。

## 测试

修复已应用，现在可以正常使用这些模型：

```bash
# 使用受影响的模型
bash examples/generate_small_dataset.sh  # 已配置使用 lmsys/gpt-oss-20b-bf16
```

## 兼容性

- ✅ 向后兼容所有现有模型
- ✅ 自动处理特殊标签
- ✅ 保留原始响应在 `extra.raw_content` 中
- ✅ 不影响正常模型的行为

---

**修复日期**: 2025-11-22  
**版本**: 1.0




