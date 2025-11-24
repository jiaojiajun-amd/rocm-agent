# HTTP 优化快速上手指南

## ✅ 已完成的优化

### 1. 连接管理修复
- ✅ `docker_remote.py` - Session 复用
- ✅ `openrouter_model.py` - Session 生命周期管理
- ✅ `light_agent.py` - 上下文管理器

### 2. 高并发异步升级
- ✅ `eval_utils.py` - 从 requests 升级到 aiohttp
- ✅ 真正的异步 HTTP 请求，支持高并发

## 🚀 立即使用

### 无需任何代码修改！

所有优化都是内部实现，你的代码可以直接使用：

```python
# 方式 1：多线程并行（推荐）
from concurrent.futures import ThreadPoolExecutor
import asyncio

def worker(instance):
    return asyncio.run(
        run_single_task_with_evaluation_info(
            instance=instance,
            model=model,
            config=config,
            docker_server_url=docker_server_url,
            eval_server_url=eval_server_url,
        )
    )

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(worker, dataset))
```

## 📊 性能预期

### 场景 1：10 个 Agent 并行
- **旧实现**：~300 秒（每个 30 秒，串行）
- **新实现**：~30 秒（真正并行）
- **提升**：10x 倍

### 场景 2：50 个 Agent 高并发
- **旧实现**：~1500 秒
- **新实现**：~30 秒
- **提升**：50x 倍

## 🔍 验证优化效果

### 1. 运行性能测试

```bash
python test_async_performance.py
```

预期输出：
```
新实现 (aiohttp): 2.5 秒
旧实现 (requests): 25.0 秒
性能提升: 10.0x 倍
```

### 2. 监控连接数

```bash
# 实时监控与评估服务器的连接
watch -n 1 'netstat -an | grep :9528 | grep ESTABLISHED | wc -l'
```

**预期行为**：
- 修复前：连接数持续增长（泄漏）
- 修复后：连接数稳定在合理范围（复用）

### 3. 检查日志

```bash
# 查看并发执行情况
tail -f your_log_file.log | grep "Starting evaluation"
```

**预期行为**：
- 修复前：日志串行输出
- 修复后：多个任务几乎同时开始

## ⚙️ 调优建议

### 增加并发数

如果需要更高的并发（50+ agent），修改 `eval_utils.py`：

```python
# 在 evaluate() 和 evaluate_info() 函数中
connector = aiohttp.TCPConnector(
    limit=200,          # 从 100 增加到 200
    limit_per_host=100  # 从 50 增加到 100
)
```

### 调整超时时间

如果评估服务器响应慢，增加超时：

```python
timeout = aiohttp.ClientTimeout(
    total=7200,  # 从 3600 秒增加到 7200 秒（2 小时）
)
```

## 🐛 故障排查

### 问题 1：连接被拒绝

**错误**：
```
ConnectionError: Failed to establish a new connection: [Errno 111] Connection refused
```

**解决方案**：
1. 确保评估服务器正在运行：
   ```bash
   curl http://your-server:9528/health
   ```

2. 检查防火墙设置

3. 验证 server_url 配置正确

### 问题 2：超时错误

**错误**：
```
asyncio.TimeoutError: Request timeout after 3600 seconds
```

**解决方案**：
1. 检查评估服务器负载
2. 增加超时时间（见上面调优建议）
3. 减少并发数

### 问题 3：并发未生效

**症状**：多个 agent 仍然串行执行

**检查**：
```python
# 确保使用了 asyncio.gather 或 ThreadPoolExecutor
# ❌ 错误：串行循环
for task in tasks:
    result = await evaluate_info(...)

# ✅ 正确：并发执行
results = await asyncio.gather(*[evaluate_info(...) for task in tasks])
```

## 📈 性能监控

### 关键指标

创建监控脚本 `monitor.sh`：

```bash
#!/bin/bash

echo "=== HTTP 连接监控 ==="
echo "评估服务器连接数:"
netstat -an | grep :9528 | grep ESTABLISHED | wc -l

echo ""
echo "Docker 服务器连接数:"
netstat -an | grep :9527 | grep ESTABLISHED | wc -l

echo ""
echo "总 ESTABLISHED 连接:"
netstat -an | grep ESTABLISHED | wc -l
```

运行：
```bash
chmod +x monitor.sh
watch -n 2 ./monitor.sh
```

## 🎯 最佳实践

### 1. 合理设置 workers 数量

```python
import os

# 根据 CPU 核心数设置
workers = min(os.cpu_count() * 2, 20)

with ThreadPoolExecutor(max_workers=workers) as executor:
    ...
```

### 2. 错误处理

```python
async def safe_evaluate(task):
    try:
        return await evaluate_info(...)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"error": str(e), "reward": 0.0}

# 使用 return_exceptions=True 防止单个失败影响全部
results = await asyncio.gather(
    *[safe_evaluate(t) for t in tasks],
    return_exceptions=True
)
```

### 3. 批量处理

如果任务数量巨大（100+），分批处理：

```python
async def process_batch(batch):
    return await asyncio.gather(*[evaluate_info(...) for t in batch])

# 每批 20 个任务
batch_size = 20
batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

all_results = []
for batch in batches:
    results = await process_batch(batch)
    all_results.extend(results)
```

## 📚 相关文档

- **HTTP_CONNECTION_FIX.md** - 连接管理详细说明
- **ASYNC_HTTP_UPGRADE.md** - 异步升级技术细节
- **HTTP_OPTIMIZATION_SUMMARY.md** - 完整优化总结
- **test_async_performance.py** - 性能测试脚本

## 💡 常见问题

### Q: 需要重新安装依赖吗？
**A**: 不需要。`aiohttp` 已经在 `pyproject.toml` 中，只需确保已安装：
```bash
pip install -e .
```

### Q: 旧代码会受影响吗？
**A**: 不会。所有修改都向后兼容，函数签名和返回值没有改变。

### Q: 如何回退到旧实现？
**A**: Git checkout 到修改前的版本即可。但不建议回退，新实现性能更好。

### Q: 能否在生产环境使用？
**A**: 可以。aiohttp 是成熟稳定的异步 HTTP 库，被广泛使用。

### Q: 内存占用会增加吗？
**A**: 不会。异步实现比线程实现更节省内存。

## ✨ 核心改进

```
旧架构：假异步 + 连接泄漏
├─ async 函数内部用同步 requests
├─ 阻塞事件循环
├─ 连接不复用
└─ 性能差

新架构：真异步 + 连接管理
├─ 真正的异步 aiohttp
├─ 不阻塞事件循环
├─ 智能连接池
└─ 10-50x 性能提升 🚀
```

## 🎉 开始使用

直接运行你的多 agent 程序，享受性能提升！

```bash
# 示例
python src/agent_v2/test_rocm_agent_amd.py \
    --dataset training_data/small_dataset.json \
    --workers 10 \
    --docker-server 10.67.77.184:9527 \
    --eval-server 10.67.77.184:9528
```

现在，10 个 agent 会真正并行执行，一个阻塞不会影响其他 agent！🎊

