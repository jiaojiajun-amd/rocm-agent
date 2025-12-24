# 快速入门指南

使用 mini agent 生成 continuous pretraining 训练数据的快速指南。

## 步骤 1: 测试系统

首先运行测试脚本确保系统正常工作：

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2
bash test_data_generation.sh
```

这将生成3个测试样本并显示统计信息。

## 步骤 2: 生成训练数据

### 选项 A: 使用默认配置

```bash
bash generate_training_data.sh
```

### 选项 B: 自定义参数

编辑 `generate_training_data.sh` 修改参数：

```bash
#!/bin/bash
python generate_training_data.py generate-multi \
    --dataset /path/to/your/dataset.json \
    --api-key "your-api-key" \
    --model "Qwen/Qwen3-8B" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /path/to/output.json \
    --config /path/to/config.yaml \
    --workers 4
```

### 选项 C: 使用示例脚本

```bash
cd examples
bash generate_small_dataset.sh  # 小规模测试
# 或
bash generate_large_dataset.sh  # 大规模生产
```

## 步骤 3: 分析数据

查看生成数据的统计信息：

```bash
python process_training_data.py analyze \
    --input /path/to/training_data.json
```

## 步骤 4: 过滤高质量数据

保留高质量样本：

```bash
python process_training_data.py filter_data \
    --input /path/to/training_data.json \
    --output /path/to/filtered_data.json \
    --min-reward 0.7 \
    --successful-only
```

## 步骤 5: 导出训练格式

### SFT格式（推荐用于监督微调）

```bash
python process_training_data.py export_sft \
    --input /path/to/filtered_data.json \
    --output /path/to/sft_data.jsonl \
    --min-reward 0.7
```

### 轨迹格式（推荐用于强化学习）

```bash
python process_training_data.py export_trajectory \
    --input /path/to/filtered_data.json \
    --output /path/to/trajectory_data.json \
    --min-reward 0.7
```

## 一键处理管道

使用自动化管道完成步骤3-5：

```bash
bash process_pipeline.sh \
    /path/to/training_data.json \
    /path/to/output_dir \
    0.7
```

## 常用命令速查

### 生成数据
```bash
# 小规模测试（3个任务）
bash test_data_generation.sh

# 中等规模（默认配置）
bash generate_training_data.sh

# 大规模生产
bash examples/generate_large_dataset.sh
```

### 分析数据
```bash
# 查看统计
python process_training_data.py analyze --input data.json

# 查看单个样本
python process_training_data.py show_example --input data.json --index 0
```

### 处理数据
```bash
# 过滤
python process_training_data.py filter_data \
    --input raw.json --output filtered.json --min-reward 0.7

# 导出SFT
python process_training_data.py export_sft \
    --input filtered.json --output sft.jsonl

# 导出轨迹
python process_training_data.py export_trajectory \
    --input filtered.json --output trajectory.json
```

## 输出文件说明

生成的文件结构：

```
training_data/
├── mini_agent_training_data.json     # 原始数据
├── test_training_data.json           # 测试数据
└── processed/                        # 处理后的数据
    ├── filtered_data.json           # 过滤后的数据
    ├── sft_training_data.jsonl      # SFT格式
    └── trajectory_training_data.json # 轨迹格式
```

## 参数说明

### 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--workers` | 并行worker数 | 2-8 |
| `--temperature` | 采样温度 | 1.0（标准）<br>1.5（多样化） |
| `--max-tokens` | 最大token数 | 8000 |
| `--min-reward` | 最小奖励阈值 | 0.7-0.8 |
| `--max-tasks` | 任务数限制 | 测试用：3-10<br>生产用：不限制 |

## 下一步

1. 查看详细文档：`TRAINING_DATA_GENERATION.md`
2. 探索示例脚本：`examples/`
3. 自定义配置：编辑 `minisweagent/config/mini.yaml`
4. 使用生成的数据进行模型训练

## 需要帮助？

- 查看完整文档：`TRAINING_DATA_GENERATION.md`
- 查看示例：`examples/README.md`
- 检查日志文件：`training_data/*.log`

