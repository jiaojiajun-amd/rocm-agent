# Training Data Generation Examples

本目录包含不同场景下的训练数据生成示例脚本。

## 可用脚本

### 1. `generate_small_dataset.sh`
**用途**: 快速测试和验证
- 任务数: 10个任务
- 并行度: 2个worker
- 适用场景: 测试系统、验证配置、快速迭代

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2/examples
bash generate_small_dataset.sh
```

### 2. `generate_large_dataset.sh`
**用途**: 生产环境大规模数据生成
- 任务数: 全部任务
- 并行度: 8个worker
- 温度: 1.0（标准采样）
- 适用场景: 生产训练数据、完整数据集

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2/examples
bash generate_large_dataset.sh
```

### 3. `generate_diverse_dataset.sh`
**用途**: 生成多样化的数据
- 任务数: 全部任务
- 并行度: 4个worker
- 温度: 1.5（更高的随机性）
- 适用场景: 探索性训练、增强数据多样性

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2/examples
bash generate_diverse_dataset.sh
```

## 自定义脚本

您可以基于这些示例创建自己的脚本。主要可调参数：

### 模型参数
- `--temperature`: 采样温度（0.0-2.0）
  - 低温（0.7-1.0）：更确定性的输出
  - 高温（1.2-1.5）：更多样化的输出
- `--max-tokens`: 最大生成token数

### 执行参数
- `--workers`: 并行worker数量
  - 建议值：2-8，取决于服务器资源
- `--max-tasks`: 限制任务数量
  - 用于测试或分批处理

### 服务器配置
- `--docker-server`: Docker服务器地址
- `--eval-server`: 评估服务器地址
- `--api-key`: API密钥

### 输出配置
- `--output`: 输出文件路径
- `--log-file`: 日志文件路径

## 示例：自定义脚本

```bash
#!/bin/bash
# 中等规模、平衡配置

python ../generate_training_data.py generate-multi \
    --dataset /path/to/dataset.json \
    --api-key "your-api-key" \
    --model "Qwen/Qwen3-8B" \
    --docker-server "server-ip:port" \
    --eval-server "server-ip:port" \
    --output /path/to/output.json \
    --config /path/to/config.yaml \
    --workers 4 \
    --max-tasks 50 \
    --temperature 1.0 \
    --max-tokens 8000
```

## 处理生成的数据

数据生成后，使用处理管道：

```bash
# 处理生成的数据
bash ../process_pipeline.sh \
    /home/jiajjiao/rocm-agent/training_data/small_dataset.json \
    /home/jiajjiao/rocm-agent/training_data/processed \
    0.7
```

参数说明：
1. 输入文件路径
2. 输出目录
3. 最小奖励阈值

## 最佳实践

1. **测试优先**: 使用 `generate_small_dataset.sh` 先测试
2. **逐步扩展**: 验证成功后再运行大规模生成
3. **监控资源**: 观察服务器CPU、内存、网络使用情况
4. **调整并行度**: 根据资源情况调整 `--workers`
5. **保存日志**: 使用 `--log-file` 记录详细日志
6. **分批处理**: 对于大数据集，使用 `--max-tasks` 分批

## 故障排查

### 脚本无法执行
```bash
chmod +x generate_*.sh
```

### 相对路径问题
确保从正确的目录运行脚本，或使用绝对路径

### 服务器连接失败
检查服务器地址、端口和服务状态

### 内存不足
减少 `--workers` 数量

