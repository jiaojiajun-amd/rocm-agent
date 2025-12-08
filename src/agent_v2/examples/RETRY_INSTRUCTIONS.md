# 重试失败任务的说明

## 问题诊断

从日志分析，原始运行有以下问题：
- 使用 8 个并发worker，Docker服务器无法处理这么高的并发
- 导致 467/928 (50.3%) 的样本失败，错误为 `500 Server Error`
- 19/29 个任务没有达到目标的 32 个成功样本

## 解决方案

### 1. 降低并发数
已将原始脚本 `generate_large_dataset_h2.sh` 的并发数从 8 降低到 2

### 2. 创建重试数据集
已自动创建 `data/rocprim_v5_h2_retry.json`，包含 19 个需要重试的任务：
- 11 个任务完全失败 (0/32 成功)
- 8 个任务部分失败 (需要补充样本)

### 3. 运行重试脚本

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2/examples
./retry_failed_h2.sh
```

该脚本将：
- 使用 2 个worker（更稳定）
- 为 19 个任务各生成 32 个样本
- 总共尝试生成 608 个样本
- 输出到 `training_data/large_dataset_h2_retry.json`

### 4. 合并数据集

重试完成后，运行合并脚本：

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2/examples
python3 merge_datasets.py \
    /home/jiajjiao/rocm-agent/training_data/large_dataset_h2.json \
    /home/jiajjiao/rocm-agent/training_data/large_dataset_h2_retry.json \
    /home/jiajjiao/rocm-agent/training_data/large_dataset_h2_merged.json
```

合并脚本会：
- 优先保留成功的样本
- 为每个 (task, sample_id) 对保留最好的结果
- 显示详细的合并统计信息

## 预期结果

如果重试成功，最终应该有：
- 29 个任务，每个任务 32 个成功样本
- 总共 928 个成功样本
- 成功率接近 100%

## 如果仍然失败

如果使用 2 个worker仍然有较多失败，可以考虑：

1. **进一步降低并发**：修改 `retry_failed_h2.sh`，将 `--workers 2` 改为 `--workers 1`

2. **分批运行**：手动分批运行，每次只处理几个任务

3. **检查Docker服务器**：联系管理员检查服务器资源（内存/GPU/磁盘空间）

4. **添加延迟**：在代码中添加请求之间的延迟，避免瞬时负载过高

## 文件清单

生成的文件：
- `data/rocprim_v5_h2_retry.json` - 需要重试的任务列表
- `src/agent_v2/examples/retry_failed_h2.sh` - 重试脚本（2 workers）
- `src/agent_v2/examples/merge_datasets.py` - 数据集合并脚本
- `training_data/large_dataset_h2_retry.json` - 重试结果（运行后生成）
- `training_data/large_dataset_h2_retry.log` - 重试日志（运行后生成）
- `training_data/large_dataset_h2_merged.json` - 合并后的最终数据集（合并后生成）

修改的文件：
- `src/agent_v2/examples/generate_large_dataset_h2.sh` - workers: 8 → 2

