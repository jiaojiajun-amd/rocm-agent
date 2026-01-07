# Scripts V2 - SFT Training Data Generation

这个文件夹包含 v2 版本的数据生成脚本，使用新的 `generate_training_data.py`。

## 输出格式

每个脚本会生成两种输出：

1. **SFT 训练数据** (`sft_*.json`)
   - 包含真实的模型输入/输出对
   - 每轮对话作为独立样本
   - 可直接用于 SFT 训练

2. **轨迹数据** (`sft_*_trajectories/`)
   - 完整的对话历史
   - 所有模型调用详情（包括辅助调用）
   - 用于分析和调试

## 文件说明

| 脚本 | 输出 | 说明 |
|------|------|------|
| `generate_large_dataset.sh` | `sft_large_dataset.json` | 基础版本 |
| `generate_large_dataset_r1.sh` | `sft_large_dataset_r1.json` | Round 1 |
| `generate_large_dataset_r2.sh` | `sft_large_dataset_r2.json` | Round 2 |
| `generate_large_dataset_r3.sh` | `sft_large_dataset_r3.json` | Round 3 |
| `generate_large_dataset_r4.sh` | `sft_large_dataset_r4.json` | Round 4 |
| `generate_large_dataset_r5.sh` | `sft_large_dataset_r5.json` | Round 5 |
| `merge_sft_datasets.py` | 合并后的数据 | 合并多个 SFT 数据集 |
| `convert_to_llamafactory.py` | LLaMA-Factory 格式 | 转换为训练框架格式 |

## 完整工作流程

### Step 1: 生成数据

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2/scripts_v2
./generate_large_dataset.sh
```

### Step 2: 转换为 LLaMA-Factory 格式

```bash
# 转换为 ShareGPT 格式（推荐，多轮对话）
python convert_to_llamafactory.py \
    /home/jiajjiao/rocm-agent/training_data_v2/sft_large_dataset.json \
    -o /home/jiajjiao/rocm-agent/training_data_v2/llamafactory/rocm_agent_sharegpt.json \
    --format sharegpt \
    --min-reward 0.3 \
    --dataset-info /home/jiajjiao/rocm-agent/training_data_v2/llamafactory/dataset_info.json

# 或转换为 Alpaca 格式（单轮）
python convert_to_llamafactory.py \
    /home/jiajjiao/rocm-agent/training_data_v2/sft_large_dataset.json \
    -o /home/jiajjiao/rocm-agent/training_data_v2/llamafactory/rocm_agent_alpaca.json \
    --format alpaca
```

### Step 3: 配置 LLaMA-Factory

将生成的 `dataset_info.json` 合并到 LLaMA-Factory 的配置中：

```bash
# 复制数据到 LLaMA-Factory
cp training_data_v2/llamafactory/*.json /path/to/LLaMA-Factory/data/
```

然后在 LLaMA-Factory 中使用数据集名称进行训练。

## LLaMA-Factory 数据格式

### ShareGPT 格式（多轮对话）

```json
[
  {
    "conversations": [
      {"from": "human", "value": "用户输入1..."},
      {"from": "gpt", "value": "模型输出1..."},
      {"from": "human", "value": "用户输入2..."},
      {"from": "gpt", "value": "模型输出2..."}
    ],
    "system": "系统提示...",
    "instance_id": "rocm_001",
    "reward": 1.0
  }
]
```

### Alpaca 格式（单轮指令）

```json
[
  {
    "instruction": "系统提示...",
    "input": "用户输入...",
    "output": "模型输出...",
    "instance_id": "rocm_001",
    "reward": 1.0
  }
]
```

## 原始 SFT 数据格式

```json
{
  "metadata": {...},
  "samples": [
    {
      "instance_id": "rocm_001",
      "repeat_id": 0,
      "sample_idx": 0,
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "response": {"content": "模型输出..."},
      "reward": 1.0,
      "success": true
    }
  ],
  "summary": {...}
}
```

## 与 v1 的区别

| 特性 | v1 | v2 |
|------|----|----|
| 训练数据 | 完整对话历史 | 真实模型输入/输出 |
| 历史压缩 | 不支持 | 支持 |
| 辅助调用 | 不记录 | 单独记录 |
| 输出格式 | 单文件 | SFT + 轨迹分离 |
| LLaMA-Factory | 需手动处理 | 自动转换脚本 |

