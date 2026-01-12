# Scripts V3 - SFT Training Data Generation with Auto History Summarization

这个文件夹包含 v3 版本的数据生成脚本，**启用了自动历史总结功能**。

## 与 v2 的主要区别

| 特性 | v2 | v3 |
|------|----|----|
| `max_context_tokens` | 未设置 (0) | 24000 |
| `keep_recent_messages` | 未设置 | 4 |
| 历史总结 | 不触发 | 自动触发 |
| Context Window 错误 | 可能发生 | 自动避免 |

## mini.yaml 配置变更

v3 使用的 `mini.yaml` 新增了以下配置：

```yaml
agent:
  max_observation_tokens: 1000      # 单个观察超过1000 tokens时压缩
  max_context_tokens: 24000         # 总上下文超过24000 tokens时总结历史
  keep_recent_messages: 4           # 总结时保留最近4条消息
```

## 工作流程

### Step 1: 生成数据

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2/scripts_v3
./generate_large_dataset_r1.sh
```

### Step 2: 转换为 LLaMA-Factory 格式

```bash
python ../scripts_v2/convert_to_llamafactory.py \
    /home/jiajjiao/rocm-agent/training_data_v3/sft_large_dataset_r1.json \
    -o /home/jiajjiao/rocm-agent/training_data_v3/llamafactory/rocm_agent_sharegpt.json \
    --format sharegpt \
    --min-reward 0.3 \
    --dataset-info /home/jiajjiao/rocm-agent/training_data_v3/llamafactory/dataset_info.json
```

### Step 3: 合并多轮数据（可选）

```bash
python ../scripts_v2/merge_sft_datasets.py \
    /home/jiajjiao/rocm-agent/training_data_v3/sft_large_dataset_r1.json \
    /home/jiajjiao/rocm-agent/training_data_v3/sft_large_dataset_r2.json \
    -o /home/jiajjiao/rocm-agent/training_data_v3/sft_merged.json
```

## 文件说明

| 脚本 | 输出 | 说明 |
|------|------|------|
| `generate_large_dataset.sh` | `sft_large_dataset.json` | 基础版本 |
| `generate_large_dataset_r1.sh` | `sft_large_dataset_r1.json` | Round 1 |
| `generate_large_dataset_r2.sh` | `sft_large_dataset_r2.json` | Round 2 |
| `generate_large_dataset_r3.sh` | `sft_large_dataset_r3.json` | Round 3 |
| `generate_large_dataset_r4.sh` | `sft_large_dataset_r4.json` | Round 4 |
| `generate_large_dataset_r5.sh` | `sft_large_dataset_r5.json` | Round 5 |

## 输出格式

每个脚本会生成两种输出：

1. **SFT 训练数据** (`training_data_v3/sft_*.json`)
   - 包含真实的模型输入/输出对
   - 每轮对话作为独立样本
   - 可直接用于 SFT 训练

2. **轨迹数据** (`training_data_v3/sft_*_trajectories/`)
   - 完整的对话历史
   - 所有模型调用详情（包括辅助调用）
   - 用于分析和调试

## 自动历史总结机制

当对话上下文超过 `max_context_tokens` (24000 tokens ≈ 96000 字符) 时：

1. 保留 system 消息和最近 4 条消息
2. 中间的历史消息被发送给模型进行总结
3. 总结内容替换原始历史
4. 继续对话

这确保了即使使用 context window 较小的模型（如 Qwen 32K）也不会超出限制。

