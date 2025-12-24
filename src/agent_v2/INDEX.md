# Agent Training Data Generation - 工具索引

完整的 agent continuous pretraining 数据生成和处理工具集。

## 📚 文档

| 文档 | 说明 |
|------|------|
| [QUICKSTART.md](QUICKSTART.md) | 快速入门指南 - **从这里开始** |
| [TRAINING_DATA_GENERATION.md](TRAINING_DATA_GENERATION.md) | 完整使用文档 |
| [examples/README.md](examples/README.md) | 示例脚本说明 |
| [INDEX.md](INDEX.md) | 本文档 - 工具索引 |

## 🛠️ 核心工具

### 1. 数据生成工具

#### `generate_training_data.py`
**功能**: 使用 mini agent 生成训练数据

**命令**:
```bash
# 单个任务
python generate_training_data.py generate_single --instance task.json --output out.json

# 批量任务（推荐）
python generate_training_data.py generate-multi --dataset dataset.json --output out.json --workers 4
```

**关键特性**:
- 多线程并行处理
- 自动保存中间结果
- 完整的对话轨迹记录
- 代码变更记录（git diff）
- 详细的评估信息

---

### 2. 数据处理工具

#### `process_training_data.py`
**功能**: 分析、过滤和格式化训练数据

**命令**:
```bash
# 分析数据
python process_training_data.py analyze --input data.json

# 过滤数据
python process_training_data.py filter_data --input data.json --output filtered.json --min-reward 0.7

# 导出SFT格式
python process_training_data.py export_sft --input data.json --output sft.jsonl

# 导出轨迹格式
python process_training_data.py export_trajectory --input data.json --output trajectory.json

# 查看单个样本
python process_training_data.py show_example --input data.json --index 0
```

**关键特性**:
- 数据质量分析
- 灵活的过滤条件
- 多种导出格式
- 详细的统计报告

---

### 3. 数据可视化工具

#### `visualize_data.py`
**功能**: 可视化训练数据统计和质量指标

**命令**:
```bash
# 数据概览
python visualize_data.py overview --input data.json

# 比较两个数据集
python visualize_data.py compare --file1 data1.json --file2 data2.json

# 生成质量报告
python visualize_data.py quality_report --input data.json --output report.txt
```

**关键特性**:
- 丰富的可视化图表
- 数据集对比
- 质量评估报告
- 奖励分布直方图

## 🚀 便捷脚本

### 执行脚本

| 脚本 | 用途 | 运行方式 |
|------|------|---------|
| `generate_training_data.sh` | 默认配置生成数据 | `bash generate_training_data.sh` |
| `test_data_generation.sh` | 快速测试（3个任务） | `bash test_data_generation.sh` |
| `process_pipeline.sh` | 完整处理管道 | `bash process_pipeline.sh input.json output_dir 0.7` |

### 示例脚本

| 脚本 | 说明 | 位置 |
|------|------|------|
| `generate_small_dataset.sh` | 小规模测试（10任务，2 workers） | `examples/` |
| `generate_large_dataset.sh` | 大规模生产（全部任务，8 workers） | `examples/` |
| `generate_diverse_dataset.sh` | 多样化数据（温度1.5） | `examples/` |

## 📋 快速参考

### 完整工作流程

```bash
# 1. 测试系统
bash test_data_generation.sh

# 2. 生成数据
bash generate_training_data.sh

# 3. 可视化概览
python visualize_data.py overview --input training_data/mini_agent_training_data.json

# 4. 处理数据
bash process_pipeline.sh \
    training_data/mini_agent_training_data.json \
    training_data/processed \
    0.7

# 5. 查看最终数据
python process_training_data.py analyze --input training_data/processed/filtered_data.json
```

### 常用参数

#### 数据生成参数
```bash
--dataset       # 输入数据集文件
--output        # 输出文件路径
--workers       # 并行worker数（推荐：2-8）
--temperature   # 采样温度（0.7-1.5）
--max-tokens    # 最大token数（默认：8000）
--max-tasks     # 限制任务数（测试用）
--config        # 配置文件（默认：mini.yaml）
--log-file      # 日志文件
```

#### 数据处理参数
```bash
--input         # 输入文件
--output        # 输出文件
--min-reward    # 最小奖励阈值（推荐：0.7）
--successful-only  # 只保留成功样本
--index         # 样本索引（查看用）
```

## 📊 数据格式

### 原始数据格式
```json
{
  "metadata": {
    "model_name": "Qwen/Qwen3-8B",
    "temperature": 1.0,
    "workers": 4
  },
  "examples": [
    {
      "instance_id": "task_001",
      "problem_statement": "...",
      "messages": [...],
      "git_diff": "...",
      "reward": 1.0,
      "success": true,
      "model_calls": 15
    }
  ]
}
```

### SFT格式（JSONL）
```json
{"messages": [...], "metadata": {...}}
{"messages": [...], "metadata": {...}}
```

### 轨迹格式
```json
{
  "trajectories": [
    {
      "task": "...",
      "messages": [...],
      "git_diff": "...",
      "final_reward": 1.0
    }
  ]
}
```

## 🎯 使用场景

### 场景1: 快速测试
```bash
bash test_data_generation.sh
```

### 场景2: 小规模实验
```bash
cd examples
bash generate_small_dataset.sh
```

### 场景3: 生产环境
```bash
# 1. 生成大规模数据
cd examples
bash generate_large_dataset.sh

# 2. 处理数据
cd ..
bash process_pipeline.sh \
    training_data/large_dataset.json \
    training_data/production \
    0.8

# 3. 生成质量报告
python visualize_data.py quality_report \
    --input training_data/production/filtered_data.json \
    --output training_data/quality_report.txt
```

### 场景4: 数据对比
```bash
python visualize_data.py compare \
    --file1 training_data/dataset_v1.json \
    --file2 training_data/dataset_v2.json \
    --label1 "Version 1" \
    --label2 "Version 2"
```

## 🔍 常见任务

### 查看数据统计
```bash
python process_training_data.py analyze --input data.json
python visualize_data.py overview --input data.json
```

### 过滤高质量数据
```bash
python process_training_data.py filter_data \
    --input raw_data.json \
    --output high_quality.json \
    --min-reward 0.8 \
    --successful-only
```

### 准备训练数据
```bash
# SFT训练
python process_training_data.py export_sft \
    --input filtered.json \
    --output training.jsonl \
    --min-reward 0.7

# RL训练
python process_training_data.py export_trajectory \
    --input filtered.json \
    --output trajectories.json \
    --min-reward 0.7
```

### 调试问题
```bash
# 查看失败的样本
python process_training_data.py show_example --input data.json --index 5

# 生成详细报告
python visualize_data.py quality_report \
    --input data.json \
    --output debug_report.txt
```

## ⚙️ 配置

### Mini Agent 配置
位置: `minisweagent/config/mini.yaml`

关键配置项：
- `step_limit`: 最大步数限制
- `cost_limit`: 成本限制
- `timeout`: 执行超时时间

### 服务器配置
- Docker服务器: `--docker-server "IP:PORT"`
- 评估服务器: `--eval-server "IP:PORT"`

## 📈 性能调优

### 提高生成速度
- 增加 `--workers` 数量
- 使用更快的模型
- 优化网络连接

### 提高数据质量
- 调整 `--temperature`（降低温度）
- 优化 prompt 配置
- 增加 `--min-reward` 阈值

### 节省资源
- 减少 `--workers` 数量
- 使用 `--max-tasks` 分批处理
- 启用日志监控

## 🆘 故障排查

| 问题 | 解决方案 |
|------|---------|
| 脚本无法执行 | `chmod +x script.sh` |
| 服务器连接失败 | 检查服务器地址和端口 |
| 内存不足 | 减少 workers 数量 |
| 数据质量低 | 调整温度、优化配置 |
| 生成速度慢 | 增加 workers、检查网络 |

查看日志：
```bash
tail -f training_data/*.log
```

## 📝 文件结构

```
agent_v2/
├── generate_training_data.py      # 数据生成主脚本
├── process_training_data.py       # 数据处理工具
├── visualize_data.py              # 可视化工具
├── generate_training_data.sh      # 执行脚本
├── test_data_generation.sh        # 测试脚本
├── process_pipeline.sh            # 处理管道
├── QUICKSTART.md                  # 快速入门
├── TRAINING_DATA_GENERATION.md    # 完整文档
├── INDEX.md                       # 本文档
└── examples/                      # 示例脚本
    ├── README.md
    ├── generate_small_dataset.sh
    ├── generate_large_dataset.sh
    └── generate_diverse_dataset.sh
```

## 🎓 学习路径

1. **入门**: 阅读 [QUICKSTART.md](QUICKSTART.md)
2. **实践**: 运行 `test_data_generation.sh`
3. **深入**: 阅读 [TRAINING_DATA_GENERATION.md](TRAINING_DATA_GENERATION.md)
4. **定制**: 修改示例脚本
5. **优化**: 调整参数和配置
6. **生产**: 大规模数据生成

## 🔗 相关资源

- Mini Agent 实现: `minisweagent/agents/mini.py`
- 配置文件: `minisweagent/config/mini.yaml`
- 评估工具: `eval_utils.py`
- 测试参考: `test_rocm_agent_amd.py`

---

**提示**: 建议从 [QUICKSTART.md](QUICKSTART.md) 开始，然后运行测试脚本熟悉工具。

