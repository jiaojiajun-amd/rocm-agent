# Agent Continuous Pretraining 数据生成系统

完整的训练数据生成和处理工具集，用于基于 mini agent 的 continuous pretraining。

## 🚀 快速开始

### 30秒快速测试

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2
bash test_data_generation.sh
```

这将生成3个测试样本，帮助你快速了解系统功能。

### 5分钟入门

1. **阅读快速入门指南**
   ```bash
   cat QUICKSTART.md
   ```

2. **运行测试**
   ```bash
   bash test_data_generation.sh
   ```

3. **查看结果**
   ```bash
   python visualize_data.py overview --input training_data/test_training_data.json
   ```

4. **生成实际数据**
   ```bash
   bash generate_training_data.sh
   ```

## 📦 系统组成

### 核心脚本（Python）

| 文件 | 功能 | 行数 |
|------|------|------|
| `generate_training_data.py` | 数据生成引擎 | ~450行 |
| `process_training_data.py` | 数据处理工具 | ~400行 |
| `visualize_data.py` | 数据可视化 | ~350行 |

### 执行脚本（Shell）

| 文件 | 功能 |
|------|------|
| `generate_training_data.sh` | 默认配置数据生成 |
| `test_data_generation.sh` | 快速测试脚本 |
| `process_pipeline.sh` | 完整处理管道 |

### 示例脚本（examples/）

| 文件 | 场景 |
|------|------|
| `generate_small_dataset.sh` | 小规模测试（10任务） |
| `generate_large_dataset.sh` | 大规模生产（全部任务） |
| `generate_diverse_dataset.sh` | 多样化数据（高温度） |

### 文档

| 文件 | 内容 |
|------|------|
| `QUICKSTART.md` | 快速入门指南 ⭐ 从这里开始 |
| `TRAINING_DATA_GENERATION.md` | 完整使用文档 |
| `INDEX.md` | 工具索引和快速参考 |
| `README_TRAINING_DATA.md` | 本文档 - 系统概述 |
| `examples/README.md` | 示例脚本说明 |

## 🎯 核心功能

### 1. 数据生成
- ✅ 使用 mini agent 执行任务
- ✅ 记录完整对话轨迹
- ✅ 捕获代码变更（git diff）
- ✅ 自动评估结果
- ✅ 多线程并行处理
- ✅ 中间结果自动保存

### 2. 数据处理
- ✅ 数据质量分析
- ✅ 灵活过滤条件
- ✅ 多种导出格式（SFT、轨迹）
- ✅ 统计报告生成
- ✅ 单样本详细查看

### 3. 数据可视化
- ✅ 丰富的统计图表
- ✅ 奖励分布直方图
- ✅ 数据集对比
- ✅ 质量评估报告

## 📊 数据格式

### 输入格式
```json
[
  {
    "instance_id": "rocprim_001",
    "problem_statement": "优化这个kernel...",
    "image_name": "rocm-lib",
    "dataset_name": "rocprim_v5",
    "split": "test"
  }
]
```

### 输出格式
```json
{
  "metadata": {
    "model_name": "Qwen/Qwen3-8B",
    "temperature": 1.0,
    "workers": 4
  },
  "examples": [
    {
      "instance_id": "rocprim_001",
      "problem_statement": "...",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "git_diff": "diff --git a/file.cpp...",
      "exit_status": "Submitted",
      "reward": 1.0,
      "speedup": 1.5,
      "success": true,
      "model_calls": 15,
      "evaluation_info": {...}
    }
  ],
  "summary": {
    "total_examples": 100,
    "successful": 85,
    "average_reward": 0.85
  }
}
```

## 🛠️ 典型使用流程

### 流程1: 测试和验证

```bash
# 1. 快速测试
bash test_data_generation.sh

# 2. 查看结果
python visualize_data.py overview \
    --input training_data/test_training_data.json

# 3. 检查样本
python process_training_data.py show_example \
    --input training_data/test_training_data.json \
    --index 0
```

### 流程2: 小规模实验

```bash
# 1. 生成数据（10个任务）
cd examples
bash generate_small_dataset.sh

# 2. 分析数据
cd ..
python process_training_data.py analyze \
    --input training_data/small_dataset.json

# 3. 过滤数据
python process_training_data.py filter_data \
    --input training_data/small_dataset.json \
    --output training_data/small_filtered.json \
    --min-reward 0.7
```

### 流程3: 生产环境

```bash
# 1. 生成大规模数据
cd examples
bash generate_large_dataset.sh

# 2. 完整处理管道
cd ..
bash process_pipeline.sh \
    training_data/large_dataset.json \
    training_data/production \
    0.8

# 3. 生成质量报告
python visualize_data.py quality_report \
    --input training_data/production/filtered_data.json \
    --output training_data/quality_report.txt

# 4. 使用处理后的数据进行训练
# training_data/production/sft_training_data.jsonl
# training_data/production/trajectory_training_data.json
```

## 🎛️ 参数配置

### 关键参数说明

#### 数据生成参数

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| `--workers` | 并行worker数 | 2-8 | 速度 |
| `--temperature` | 采样温度 | 1.0（标准）<br>1.5（多样） | 多样性 |
| `--max-tokens` | 最大token | 8000 | 轨迹长度 |
| `--min-reward` | 奖励阈值 | 0.7-0.8 | 质量 |

#### 质量控制

- **高质量数据**: `--min-reward 0.8` + `--temperature 1.0`
- **多样化数据**: `--min-reward 0.6` + `--temperature 1.5`
- **平衡数据**: `--min-reward 0.7` + `--temperature 1.0`

## 📈 性能指标

### 预期指标

| 指标 | 良好 | 优秀 |
|------|------|------|
| 成功率 | > 70% | > 85% |
| 平均奖励 | > 0.6 | > 0.8 |
| 平均轨迹长度 | 10-20 | 15-25 |
| 高质量样本(≥0.8) | > 40% | > 60% |

### 生成速度

- 单个任务: ~2-5分钟
- 100个任务（4 workers）: ~3-6小时
- 实际速度取决于：
  - 任务复杂度
  - 网络延迟
  - 服务器性能

## 🔧 系统要求

### 环境依赖

- Python 3.10+
- 已安装的包：
  - `typer`, `rich`, `pyyaml`
  - `jinja2`, `asyncio`
  - 项目依赖（`minisweagent`）

### 外部服务

- Docker服务器（运行agent环境）
- 评估服务器（评估任务结果）
- LLM API（模型推理）

### 存储需求

- 原始数据: ~10MB per 100 tasks
- 处理后数据: ~5-15MB per 100 tasks
- 日志文件: ~1-5MB per 100 tasks

## 🐛 故障排查

### 常见问题

#### 1. 连接失败
```
错误: Failed to connect to Docker server
解决: 检查服务器地址和端口，确认服务运行
```

#### 2. 内存不足
```
错误: Out of memory
解决: 减少 --workers 数量，或使用 --max-tasks 分批
```

#### 3. 数据质量低
```
问题: 平均奖励 < 0.5
解决: 
- 降低温度（--temperature 0.8）
- 检查 prompt 配置
- 查看失败样本日志
```

#### 4. 生成速度慢
```
问题: 单任务 > 10分钟
解决:
- 检查网络延迟
- 优化 Docker 镜像
- 增加 --workers
```

### 调试技巧

```bash
# 查看日志
tail -f training_data/*.log

# 查看失败样本
python process_training_data.py show_example \
    --input data.json \
    --index <failed_index>

# 生成诊断报告
python visualize_data.py quality_report \
    --input data.json \
    --output debug.txt
```

## 📚 进阶使用

### 自定义配置

编辑 `minisweagent/config/mini.yaml`:

```yaml
agent:
  step_limit: 50        # 增加步数限制
  cost_limit: 5.0       # 增加成本限制
  timeout: 120          # 增加超时时间

environment:
  timeout: 90           # 调整环境超时
```

### 自定义处理

```python
# 创建自定义过滤器
from process_training_data import load_training_data

data = load_training_data("data.json")
examples = data["examples"]

# 自定义过滤逻辑
high_quality = [
    ex for ex in examples
    if ex["reward"] > 0.8 
    and ex["model_calls"] < 20
    and len(ex["messages"]) > 10
]
```

### 批量处理

```bash
# 处理多个数据集
for dataset in data/*.json; do
    python generate_training_data.py generate-multi \
        --dataset "$dataset" \
        --output "results/$(basename $dataset)" \
        --workers 4
done
```

## 🎓 最佳实践

1. **测试优先**: 总是先运行 `test_data_generation.sh`
2. **逐步扩展**: 从小规模开始，逐步增加任务数
3. **质量优先**: 使用合适的 `--min-reward` 过滤
4. **保存日志**: 使用 `--log-file` 记录详细信息
5. **监控资源**: 观察CPU、内存、网络使用
6. **版本管理**: 为不同配置保存不同版本
7. **定期备份**: 及时备份生成的数据

## 📞 获取帮助

### 查看帮助信息

```bash
# 主命令帮助
python generate_training_data.py --help
python process_training_data.py --help
python visualize_data.py --help

# 子命令帮助
python generate_training_data.py generate_multi --help
python process_training_data.py filter_data --help
```

### 文档索引

- 快速入门: `QUICKSTART.md`
- 完整文档: `TRAINING_DATA_GENERATION.md`
- 工具索引: `INDEX.md`
- 示例说明: `examples/README.md`

## 🎉 总结

这是一套完整的工具集，用于：

✅ **生成** - 使用 mini agent 生成高质量训练数据  
✅ **处理** - 分析、过滤、格式化数据  
✅ **可视化** - 直观了解数据质量  
✅ **自动化** - 一键执行完整流程  

**立即开始**: `bash test_data_generation.sh`

---

**作者**: Agent Training Data Generation Team  
**日期**: 2025-11  
**版本**: 1.0

