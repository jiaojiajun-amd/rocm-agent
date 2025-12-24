# Multiple Samples Per Task Feature

## 概述

新增功能：每个任务可以生成多个样本（默认32个），通过采样获得更多样化的训练数据。

## 功能说明

### 新增参数

`--samples-per-task`: 每个任务生成的样本数量（默认：1，推荐：32）

### 工作原理

1. 对每个任务，agent会运行多次（samples-per-task次）
2. 由于温度采样（temperature > 0），每次运行会产生不同的输出
3. 每个样本都有独立的：
   - `sample_id`: 样本编号（0到samples-per-task-1）
   - 完整对话轨迹
   - Git diff
   - 评估结果
4. 所有样本保存在同一个输出文件中

### 数据格式

```json
{
  "metadata": {
    "model_name": "gpt-5",
    "temperature": 1.0,
    "samples_per_task": 32,
    "total_tasks": 10,
    "total_samples": 320
  },
  "examples": [
    {
      "instance_id": "task_001",
      "sample_id": 0,
      "messages": [...],
      "reward": 0.95,
      ...
    },
    {
      "instance_id": "task_001",
      "sample_id": 1,
      "messages": [...],
      "reward": 0.88,
      ...
    },
    ...
  ]
}
```

## 使用方法

### 命令行

```bash
python generate_training_data.py generate-multi \
    --dataset data.json \
    --output output.json \
    --samples-per-task 32 \
    --temperature 1.0 \
    --workers 4
```

### Shell脚本

所有示例脚本已更新为默认生成32个样本：

```bash
# 小规模测试：10个任务 × 32个样本 = 320个样本
bash examples/generate_small_dataset.sh

# 大规模生产：全部任务 × 32个样本
bash examples/generate_large_dataset.sh

# 多样化数据：高温度 × 32个样本
bash examples/generate_diverse_dataset.sh
```

### 自定义样本数

修改任何脚本中的 `--samples-per-task` 参数：

```bash
# 生成16个样本
--samples-per-task 16

# 生成64个样本
--samples-per-task 64

# 生成1个样本（原始行为）
--samples-per-task 1
```

## 应用场景

### 1. 多样性训练数据

- **目的**: 获得同一任务的多种解决方案
- **设置**: `--samples-per-task 32 --temperature 1.0`
- **适用**: 监督微调、行为克隆

### 2. 质量过滤

- **目的**: 生成多个样本，选择高质量的
- **设置**: `--samples-per-task 32 --temperature 1.2`
- **处理**: 使用 `process_training_data.py filter_data --min-reward 0.8`

### 3. 集成学习

- **目的**: 训练多个模型或使用集成方法
- **设置**: `--samples-per-task 10 --temperature 1.0`
- **适用**: Model ensemble、RLHF

### 4. 轨迹对比学习

- **目的**: 学习成功vs失败的轨迹差异
- **设置**: `--samples-per-task 50 --temperature 1.5`
- **分析**: 对比高reward和低reward样本

## 性能考虑

### 时间成本

- **1个样本**: 基准时间
- **32个样本**: ~32倍时间
- **建议**: 
  - 测试时用 `--max-tasks 3 --samples-per-task 2`
  - 生产时用 `--samples-per-task 32 --workers 8`

### 存储成本

- 每个样本包含完整对话和git diff
- 10个任务 × 32个样本 ≈ 50-100MB
- 建议定期压缩或归档旧数据

### 并行策略

系统自动并行化：
- 所有 (task, sample_id) 对被视为独立任务
- 使用线程池并行执行
- 进度条显示总样本数

例如：
- 10个任务 × 32个样本 = 320个并行任务
- 4个workers → 平均每个worker处理80个任务

## 最佳实践

### 1. 温度设置

- **Temperature 0.7-1.0**: 平衡的多样性
- **Temperature 1.0-1.5**: 高多样性，更多探索
- **Temperature < 0.7**: 多样性较低，不推荐多采样

### 2. 样本数选择

| 样本数 | 适用场景 | 说明 |
|--------|---------|------|
| 1 | 测试、快速迭代 | 原始行为 |
| 5-10 | 小规模实验 | 获得一些多样性 |
| 16-32 | 生产训练 | **推荐**，良好的多样性 |
| 50+ | 研究、质量过滤 | 高多样性，耗时较长 |

### 3. 质量控制

```bash
# 1. 生成大量样本
python generate_training_data.py generate-multi \
    --samples-per-task 50 \
    --temperature 1.2

# 2. 过滤高质量样本
python process_training_data.py filter_data \
    --min-reward 0.8 \
    --successful-only

# 3. 可能从50个样本中获得15-20个高质量样本
```

### 4. 数据分析

```bash
# 查看每个任务的样本分布
python process_training_data.py analyze --input data.json

# 按任务分组查看reward分布
python visualize_data.py overview --input data.json
```

## 示例：完整工作流程

### 步骤1: 生成多样本数据

```bash
bash examples/generate_small_dataset.sh
# 输出：10任务 × 32样本 = 320个样本
```

### 步骤2: 分析数据质量

```bash
python visualize_data.py overview \
    --input training_data/small_dataset.json
```

### 步骤3: 过滤高质量样本

```bash
python process_training_data.py filter_data \
    --input training_data/small_dataset.json \
    --output training_data/high_quality.json \
    --min-reward 0.75
# 可能得到：150-200个高质量样本
```

### 步骤4: 导出训练格式

```bash
python process_training_data.py export_sft \
    --input training_data/high_quality.json \
    --output training_data/sft_data.jsonl
```

## 注意事项

1. **随机性**: 需要temperature > 0才能获得不同样本
2. **评估成本**: 每个样本都会进行独立评估
3. **存储空间**: 确保有足够的磁盘空间
4. **中间保存**: 系统会自动保存中间结果，防止数据丢失
5. **sample_id**: 从0开始编号，用于区分同一任务的不同样本

## 故障排查

### 问题：所有样本都相同

**原因**: Temperature设置过低或为0

**解决**: 
```bash
--temperature 1.0  # 或更高
```

### 问题：生成速度太慢

**原因**: 样本数太多

**解决**:
```bash
# 减少样本数
--samples-per-task 16

# 或增加workers
--workers 8

# 或先测试小规模
--max-tasks 3 --samples-per-task 5
```

### 问题：内存不足

**原因**: 过多并行任务

**解决**:
```bash
# 减少workers
--workers 2

# 或分批处理
--max-tasks 10  # 先处理10个任务
```

## 技术细节

### 实现

- 每个 (task, sample_id) 对作为独立的worker任务
- 使用ThreadPoolExecutor并行执行
- 每个worker创建独立的model实例
- 自动重置环境，确保样本独立性

### ID命名

- 日志中: `task_001_sample0`, `task_001_sample1`, ...
- 数据中: `instance_id + sample_id` 分开存储
- 便于按任务聚合或按样本分析

---

**版本**: 1.0  
**更新日期**: 2025-11-22  
**兼容性**: 向后兼容（默认samples-per-task=1保持原有行为）

