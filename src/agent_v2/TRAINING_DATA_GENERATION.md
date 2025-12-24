# Agent Continuous Pretraining Data Generation

本文档说明如何使用 mini agent 生成用于 continuous pretraining 的训练数据。

## 概述

训练数据生成系统包含以下组件：

1. **generate_training_data.py** - 主数据生成脚本
2. **process_training_data.py** - 数据处理和分析工具
3. **generate_training_data.sh** - 便捷执行脚本

## 数据生成

### 单个任务

生成单个任务的训练数据：

```bash
python generate_training_data.py generate_single \
    --instance /path/to/instance.json \
    --api-key "your-api-key" \
    --model "Qwen/Qwen3-8B" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /path/to/output.json \
    --config /path/to/config.yaml
```

### 批量任务（推荐）

使用多线程并行生成多个任务的训练数据：

```bash
python generate_training_data.py generate-multi \
    --dataset /path/to/dataset.json \
    --api-key "your-api-key" \
    --model "Qwen/Qwen3-8B" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /path/to/output.json \
    --config /path/to/config.yaml \
    --workers 4 \
    --temperature 1.0 \
    --max-tokens 8000
```

或者使用提供的 shell 脚本：

```bash
bash generate_training_data.sh
```

### 主要参数

- `--dataset`: 输入数据集文件（JSON格式）
- `--api-key`: API密钥
- `--model`: 模型名称（如 "Qwen/Qwen3-8B"）
- `--docker-server`: Docker服务器地址
- `--eval-server`: 评估服务器地址
- `--output`: 输出文件路径
- `--config`: 配置文件路径（默认使用 mini.yaml）
- `--workers`: 并行工作线程数
- `--temperature`: 采样温度
- `--max-tokens`: 最大token数
- `--max-tasks`: 限制处理的任务数量（用于测试）
- `--log-file`: 日志文件路径

## 生成的数据格式

生成的训练数据包含以下信息：

```json
{
  "metadata": {
    "model_name": "Qwen/Qwen3-8B",
    "temperature": 1.0,
    "max_tokens": 8000,
    "config_file": "/path/to/config.yaml",
    "dataset_file": "/path/to/dataset.json",
    "workers": 4
  },
  "examples": [
    {
      "instance_id": "task_001",
      "problem_statement": "任务描述...",
      "messages": [
        {"role": "system", "content": "系统提示..."},
        {"role": "user", "content": "用户输入..."},
        {"role": "assistant", "content": "助手响应..."},
        ...
      ],
      "git_diff": "代码变更的git diff...",
      "exit_status": "Submitted",
      "reward": 1.0,
      "speedup": 1.5,
      "success": true,
      "model_calls": 15,
      "evaluation_info": {...},
      "error": null,
      "metadata": {...}
    }
  ],
  "summary": {
    "total_examples": 100,
    "successful": 85,
    "failed": 15,
    "average_reward": 0.85,
    "total_model_calls": 1500
  }
}
```

### 关键字段说明

- **messages**: 完整的对话历史，包含所有 system、user、assistant 消息
- **git_diff**: agent 执行后产生的代码变更
- **reward**: 任务评估奖励（0.0-1.0）
- **success**: 任务是否成功完成
- **model_calls**: 模型调用次数
- **evaluation_info**: 详细的评估信息

## 数据处理和分析

### 分析数据

查看训练数据的统计信息：

```bash
python process_training_data.py analyze \
    --input /path/to/training_data.json
```

输出包括：
- 总样本数
- 成功率
- 平均奖励
- 模型调用统计
- 平均轨迹长度
- 奖励分布

### 过滤数据

根据条件过滤训练数据：

```bash
python process_training_data.py filter_data \
    --input /path/to/training_data.json \
    --output /path/to/filtered_data.json \
    --min-reward 0.5 \
    --successful-only
```

参数：
- `--min-reward`: 最小奖励阈值
- `--successful-only`: 只保留成功的样本

### 导出为 SFT 格式

导出为监督微调（Supervised Fine-Tuning）格式的 JSONL：

```bash
python process_training_data.py export_sft \
    --input /path/to/training_data.json \
    --output /path/to/sft_data.jsonl \
    --min-reward 0.5
```

输出格式（每行一个JSON对象）：
```json
{
  "messages": [...],
  "metadata": {
    "instance_id": "task_001",
    "reward": 1.0,
    "speedup": 1.5,
    "model_calls": 15
  }
}
```

### 导出为轨迹学习格式

导出为包含完整轨迹的格式：

```bash
python process_training_data.py export_trajectory \
    --input /path/to/training_data.json \
    --output /path/to/trajectory_data.json \
    --min-reward 0.5
```

### 查看单个样本

查看特定样本的详细信息：

```bash
python process_training_data.py show_example \
    --input /path/to/training_data.json \
    --index 0
```

## 完整工作流程

### 1. 生成训练数据

```bash
# 编辑 generate_training_data.sh 设置参数
# 然后运行
bash generate_training_data.sh
```

### 2. 分析数据质量

```bash
python process_training_data.py analyze \
    --input /path/to/training_data.json
```

### 3. 过滤高质量样本

```bash
python process_training_data.py filter_data \
    --input /path/to/training_data.json \
    --output /path/to/high_quality_data.json \
    --min-reward 0.7 \
    --successful-only
```

### 4. 导出为训练格式

```bash
# 导出为 SFT 格式
python process_training_data.py export_sft \
    --input /path/to/high_quality_data.json \
    --output /path/to/sft_training_data.jsonl \
    --min-reward 0.7

# 或导出为轨迹格式
python process_training_data.py export_trajectory \
    --input /path/to/high_quality_data.json \
    --output /path/to/trajectory_training_data.json \
    --min-reward 0.7
```

## 配置文件

使用 mini agent 配置文件（默认：`minisweagent/config/mini.yaml`）：

```yaml
agent:
  system_template: |
    You are a helpful assistant...
  instance_template: |
    Please solve this issue: {{task}}...
  step_limit: 0
  cost_limit: 3.0

environment:
  cwd: "/app/rocm-libraries/projects/rocprim"
  timeout: 60
  environment_class: docker_remote

model:
  model_kwargs:
    temperature: 0.0
    drop_params: true
```

## 注意事项

1. **并行度**: 根据服务器资源调整 `--workers` 参数
2. **中间保存**: 数据生成过程中会自动保存中间结果
3. **错误处理**: 单个任务失败不会影响其他任务的执行
4. **资源监控**: 建议监控 Docker 服务器的资源使用情况
5. **数据质量**: 建议使用 `--min-reward` 过滤低质量样本

## 数据用途

生成的训练数据可用于：

1. **Supervised Fine-Tuning (SFT)**: 使用成功的对话轨迹进行监督微调
2. **Reinforcement Learning**: 使用奖励信号进行强化学习
3. **Behavior Cloning**: 学习成功agent的行为模式
4. **Trajectory Prediction**: 预测agent的下一步动作
5. **Code Generation**: 学习从问题描述到代码变更的映射

## 示例脚本

### 生成少量测试数据

```bash
python generate_training_data.py generate-multi \
    --dataset /path/to/dataset.json \
    --max-tasks 5 \
    --output /path/to/test_data.json \
    --workers 2
```

### 生成高质量训练数据

```bash
# 1. 生成数据
python generate_training_data.py generate-multi \
    --dataset /path/to/dataset.json \
    --output /path/to/raw_data.json \
    --workers 8

# 2. 过滤高质量样本
python process_training_data.py filter_data \
    --input /path/to/raw_data.json \
    --output /path/to/filtered_data.json \
    --min-reward 0.8

# 3. 导出训练格式
python process_training_data.py export_sft \
    --input /path/to/filtered_data.json \
    --output /path/to/final_training_data.jsonl
```

## 故障排查

### 问题：Docker连接失败
- 检查 Docker 服务器地址和端口
- 确认 Docker 服务正在运行

### 问题：评估服务器无响应
- 检查评估服务器地址和端口
- 确认评估服务正在运行

### 问题：内存不足
- 减少 `--workers` 数量
- 使用 `--max-tasks` 分批处理

### 问题：数据生成缓慢
- 增加 `--workers` 数量
- 检查网络延迟
- 检查 Docker 容器性能

## 相关文件

- `generate_training_data.py` - 数据生成主脚本
- `process_training_data.py` - 数据处理工具
- `generate_training_data.sh` - 执行脚本
- `test_rocm_agent_amd.py` - 原始测试脚本（参考）
- `eval_utils.py` - 评估工具
- `minisweagent/agents/mini.py` - Mini Agent 实现
- `minisweagent/config/mini.yaml` - 配置文件

