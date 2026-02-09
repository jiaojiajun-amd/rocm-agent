# Observation Compression Implementation

## 问题描述

生成训练数据时，所有样本都遇到 `ContextWindowExceededError`，原因是observation（命令输出）过长导致超过模型的上下文窗口限制。

## 解决方案

实现了一个智能的observation压缩系统：
- 对于超过阈值（默认1000 tokens）的observation，自动调用模型进行reasoning压缩
- 在上下文中只保留压缩后的reasoning，避免上下文溢出
- 在训练数据中保留完整的observation，确保CPT数据的完整性

## 修改文件

### 1. `src/minisweagent/agents/mini.py` ✨ 核心修改

#### 新增配置参数（AgentConfig）
```python
observation_reasoning_template: str  # 用于reasoning的prompt模板
max_observation_tokens: int = 1000   # 触发压缩的阈值
```

#### 新增方法
- `count_tokens(text) -> int`: 估算token数量（简单实现：4 chars ≈ 1 token）
- `_reason_about_observation(observation) -> str`: 调用模型对长observation进行reasoning
- `get_full_messages() -> list[dict]`: 获取包含完整observation的消息列表（用于保存训练数据）

#### 修改方法
- `add_message()`: 支持 `full_content` 参数，用于存储原始完整内容
- `get_observation()`: 检测observation长度，超过阈值则自动压缩

#### 消息结构
```python
# 压缩后的消息（用于API调用）
{
    "role": "user",
    "content": "<observation_summary>...reasoning...</observation_summary>",
    "full_content": "Observation: ...original long output..."
}

# get_full_messages() 返回（用于训练数据）
{
    "role": "user",
    "content": "Observation: ...original long output..."
}
```

### 2. `src/minisweagent/config/mini.yaml` 📝 配置更新

新增配置项：
```yaml
agent:
  observation_reasoning_template: |
    The following observation from a command execution is very long...
  max_observation_tokens: 1000
```

### 3. `src/agent_v2/generate_training_data.py` 🔧 数据生成修改

**第157行**:
```python
# 修改前
messages=agent.messages.copy(),

# 修改后  
messages=agent.get_full_messages(),
```

确保训练数据包含完整的observation。

### 4. 新增文件

#### `src/minisweagent/OBSERVATION_COMPRESSION.md` 📚
完整的功能文档，包括：
- 问题描述和解决方案
- 架构设计
- 使用方法
- 配置说明
- 示例代码
- 权衡分析

#### `src/minisweagent/agents/test_mini_compression.py` 🧪
单元测试文件，包含6个测试用例：
- `test_short_observation_not_compressed`: 短observation不压缩
- `test_long_observation_compressed`: 长observation触发压缩
- `test_get_full_messages_restores_content`: 恢复完整内容
- `test_get_full_messages_preserves_normal_messages`: 保留普通消息
- `test_count_tokens`: Token计数测试
- `test_mixed_messages_full_messages`: 混合消息类型测试

#### `src/minisweagent/agents/example_compression_usage.py` 📖
使用示例代码，展示：
- 基本使用方法
- 自定义配置
- 压缩效果分析

## 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│ Agent执行命令                                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Get Observation│
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Count Tokens   │
                    └────────┬───────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
        > 1000 tokens              ≤ 1000 tokens
                   │                   │
                   ▼                   ▼
         ┌──────────────────┐   ┌──────────────┐
         │ Call Model for   │   │ Use Original │
         │ Reasoning        │   │ Observation  │
         └────────┬─────────┘   └──────┬───────┘
                  │                     │
                  ▼                     │
         ┌──────────────────┐          │
         │ Store:           │          │
         │ content=summary  │          │
         │ full_content=orig│          │
         └────────┬─────────┘          │
                  │                    │
                  └──────────┬─────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │ Add to Messages  │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │ For API Calls:   │      │ For Training:    │
    │ Use .messages    │      │ Use              │
    │ (compressed)     │      │ .get_full_messages()│
    └──────────────────┘      └──────────────────┘
```

## 使用方法

### 1. 自动压缩（无需修改代码）

```python
# 初始化agent
agent = MiniAgent(model, env, **config)

# 运行agent（自动压缩长observation）
exit_status, result = agent.run(problem_statement)

# Agent内部使用压缩后的消息
# agent.messages 包含压缩的observation
```

### 2. 获取训练数据

```python
# 获取完整消息（包含原始observation）
full_messages = agent.get_full_messages()

# 保存到训练数据
training_example = TrainingExample(
    messages=full_messages,  # 完整内容用于CPT
    ...
)
```

### 3. 自定义配置

在 `mini.yaml` 中调整：
```yaml
agent:
  max_observation_tokens: 500  # 更积极的压缩
```

或在代码中：
```python
agent = MiniAgent(
    model, 
    env, 
    max_observation_tokens=500
)
```

## 优点

✅ **防止上下文溢出**: 自动压缩长observation
✅ **保留完整信息**: 训练数据包含原始内容
✅ **改进reasoning**: Agent获得聚焦的摘要而非原始长输出
✅ **降低成本**: 减少API调用的token数量
✅ **透明性**: 自动工作，无需修改现有代码
✅ **灵活性**: 可配置阈值和reasoning模板

## 权衡

⚠️ **额外调用**: 每次压缩需要一次额外的模型调用
⚠️ **估算精度**: 简单的token计数可能不完全准确
⚠️ **依赖模型**: Reasoning质量取决于模型能力

## 后续改进建议

1. **精确Token计数**: 使用 `tiktoken` 库进行精确的token计数
   ```python
   import tiktoken
   def count_tokens(self, text: str) -> int:
       encoding = tiktoken.encoding_for_model("gpt-4")
       return len(encoding.encode(text))
   ```

2. **缓存机制**: 对相似的observation缓存reasoning结果

3. **自适应压缩**: 根据任务类型调整压缩策略

4. **分层压缩**: 对超长observation进行多级压缩

5. **提取式摘要**: 对于某些类型的输出（如日志），使用提取式而非生成式摘要

## 测试

运行单元测试：
```bash
cd /home/jiajjiao/rocm-agent
python -m pytest src/minisweagent/agents/test_mini_compression.py -v
```

运行示例：
```bash
python src/minisweagent/agents/example_compression_usage.py
```

## 兼容性

✅ 完全向后兼容
✅ 不影响现有代码
✅ 默认配置安全（1000 tokens阈值）
✅ 可选功能（可通过配置禁用）

## 相关Issue

解决了训练数据生成时的 `ContextWindowExceededError` 问题：
- 原因: observation过长（编译输出、测试输出等）
- 影响: 所有32个样本都失败，`model_calls: 0`
- 解决: 自动压缩长observation，保持上下文窗口在限制内

