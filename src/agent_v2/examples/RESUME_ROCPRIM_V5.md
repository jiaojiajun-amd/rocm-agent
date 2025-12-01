# RocPrim V5 数据生成断点续传说明

## 问题分析

`generate_large_dataset.sh` 遇到了和 h2 版本同样的问题：
- 使用 4 个并发worker导致Docker服务器返回 500 错误
- 前 9 个任务部分完成（24-30个成功样本）
- 后 20 个任务完全没有开始
- 所有 29 个任务都需要继续完成

## 解决方案

### 1. 实现了断点续传功能

在 `generate_training_data.py` 中添加了 `--resume` 参数：
- **只统计成功的样本**（失败的不算）
- 自动跳过已完成的任务
- 继续未完成的任务
- 保留所有历史数据

### 2. 降低并发数

将并发数从 4 降低到 2，避免服务器过载

### 3. 创建了进度检查工具

`check_progress.py` - 检查任务完成情况并创建resume数据集

## 使用方法

### 步骤 1: 检查当前进度

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2/examples

python3 check_progress.py \
    /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    /home/jiajjiao/rocm-agent/training_data/large_dataset.log \
    29 \
    32
```

这会：
- 从日志分析实际完成情况
- 显示每个任务的进度
- 创建 `data/rocprim_v5_resume.json`（包含需要继续的任务）

### 步骤 2: 运行断点续传脚本

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2/examples
./generate_large_dataset_resume.sh
```

该脚本特点：
- ✓ 使用 2 个worker（更稳定）
- ✓ 使用 `--resume` 参数（断点续传）
- ✓ 只处理前 29 个任务
- ✓ 只统计成功样本
- ✓ 自动跳过已完成任务
- ✓ 输出到新文件 `large_dataset_v2.json`

预计耗时：
- 前9个任务需要补充约 50 个样本
- 后20个任务需要 640 个样本
- 总共约 690 个样本
- 使用 2 worker约需 4-6 小时

### 步骤 3: 监控进度

实时查看日志：
```bash
tail -f /home/jiajjiao/rocm-agent/training_data/large_dataset_v2.log
```

查看统计信息：
```bash
grep "completed with reward" /home/jiajjiao/rocm-agent/training_data/large_dataset_v2.log | wc -l
```

### 步骤 4: 中断后继续

如果脚本中断，直接再次运行即可：
```bash
./generate_large_dataset_resume.sh
```

`--resume` 参数会：
1. 读取 `large_dataset_v2.json`
2. 统计每个任务的成功样本数
3. 跳过已完成的任务
4. 继续未完成的任务

## 断点续传的关键点

### 只统计成功样本
```python
# 代码中的关键逻辑
for ex in existing_examples:
    if ex.get('success', False):  # 只统计success=True的
        success_count[ex['instance_id']] += 1
```

### 跳过完成的任务
```python
if current_success >= samples_per_task:
    logger.info(f"Skipping {instance_id} - already has {current_success} successful samples")
    continue
```

## 当前状态

根据日志分析的进度（截至最近一次运行）：

```
前 9 个任务（部分完成）:
  ✗ block_adjacent_difference  25/32
  ✗ block_discontinuity        25/32
  ✗ block_exchange             28/32
  ✗ block_histogram            26/32
  ✗ block_radix_rank           28/32
  ✗ block_radix_sort           25/32
  ✗ block_reduce               24/32
  ✗ block_run_length_decode    30/32
  ✗ block_scan                 24/32

后 20 个任务（未开始）:
  ✗ block_sort                  0/32
  ✗ device_adjacent_difference  0/32
  ... (其余 18 个)
```

## 与 H2 版本的区别

| 项目 | RocPrim V5 | RocPrim V5 H2 |
|------|-----------|---------------|
| 任务数 | 29 (前一半) | 29 (全部) |
| 数据集 | rocprim_v5.json | rocprim_v5_h2.json |
| 输出 | large_dataset_v2.json | large_dataset_h2_merged.json |
| API Key | c1f7f3ee... | 65481c69... |
| 原始并发 | 4 workers | 8 workers |
| 新并发 | 2 workers | 2 workers |

## 文件清单

新创建的文件：
- `check_progress.py` - 进度检查工具
- `generate_large_dataset_resume.sh` - 断点续传脚本
- `data/rocprim_v5_resume.json` - 需要续传的任务列表
- `training_data/large_dataset_v2.json` - 新的输出文件
- `RESUME_ROCPRIM_V5.md` - 本说明文档

修改的文件：
- `generate_training_data.py` - 添加 `--resume` 参数
- `generate_large_dataset.sh` - 降低并发数到 2

## 故障排除

### 如果仍然有大量失败

1. **进一步降低并发**：修改脚本，将 `--workers 2` 改为 `--workers 1`

2. **检查Docker服务器**：
   ```bash
   curl http://10.67.77.184:9527/health
   ```

3. **清理旧容器**（联系管理员）

4. **分批运行**：手动编辑 `rocprim_v5_resume.json`，每次只运行几个任务

### 如果输出文件损坏

运行 check_progress.py 会尝试从损坏的文件中恢复数据：
```bash
python3 check_progress.py \
    data/rocprim_v5.json \
    training_data/large_dataset_v2.json \
    29 32
```

### 如果想从头开始

删除输出文件，不使用 `--resume`：
```bash
rm training_data/large_dataset_v2.json
# 修改脚本，去掉 --resume 参数
```

## 预期结果

完成后应该有：
- 29 个任务
- 每个任务 32 个成功样本
- 总共 928 个成功样本
- 成功率 > 80%

## 下一步

完成 rocprim_v5 前一半后，可以：
1. 继续后一半任务（任务 30-58）
2. 或者切换到其他数据集
3. 合并多个数据集用于训练

