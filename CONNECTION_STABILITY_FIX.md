# Docker Remote 连接稳定性优化

## 🎯 设计原则

### ❌ 不使用重试机制
**原因**：命令执行可能有副作用，重试会导致：
- 文件被多次修改
- 资源被重复创建
- 状态不一致
- 幂等性无法保证

### ✅ 确保连接稳定性
通过以下方式避免连接断开：
1. 配置连接池
2. 启用 Keep-Alive
3. 合理的超时时间
4. 稳定的 Session 管理

## 🔧 实现细节

### 1. 超时配置

```python
@dataclass
class RemoteDockerEnvironmentConfig:
    timeout: int = 1800  # 默认 1800 秒（30 分钟）
```

**为什么是 1800 秒？**
- 编译大型项目可能需要 10-20 分钟
- 运行测试套件可能需要 15-25 分钟
- 给予足够的缓冲时间避免误判超时

### 2. 请求超时计算

```python
def execute(self, command: str, ..., timeout: int | None = None):
    default_timeout = 1800  # 默认 30 分钟
    command_timeout = timeout or self.config.timeout
    request_timeout = max(default_timeout, command_timeout + 30)
    
    response = self.session.post(..., timeout=request_timeout)
```

**逻辑**：
- 使用 1800 秒作为最小超时
- 如果命令指定了更长的超时，使用命令超时 + 30 秒
- 额外的 30 秒用于网络传输

### 3. 连接池配置

```python
adapter = HTTPAdapter(
    pool_connections=5,   # 缓存 5 个不同 host 的连接池
    pool_maxsize=10,      # 每个 host 最多 10 个连接
    pool_block=False      # 池满时创建新连接而不是阻塞
)
```

### 4. Keep-Alive

```python
self.session.headers.update({
    'Connection': 'keep-alive',
    'Keep-Alive': 'timeout=300, max=100'
})
```

**效果**：
- 连接保持 300 秒不关闭
- 同一连接最多处理 100 个请求
- 减少 TCP 握手开销

## 📊 连接稳定性分析

### 问题：RemoteDisconnected

```
ERROR: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

**原因分析**：

1. **陈旧连接**
   - Session 复用了被服务器关闭的连接
   - ✅ 解决：Keep-Alive + 连接池

2. **服务器超时**
   - 命令执行时间超过服务器限制
   - ✅ 解决：1800 秒超时

3. **网络不稳定**
   - 瞬时网络中断
   - ✅ 解决：长超时 + 稳定连接

4. **并发冲突**
   - 多个请求同时使用同一连接
   - ✅ 解决：连接池（每个实例独立）

## 🎯 最佳实践

### 1. 长时间命令

如果命令需要超过 30 分钟：

```python
# 显式指定更长的超时
result = env.execute(
    "make all && run_tests", 
    timeout=3600  # 1 小时
)
```

### 2. 多个 Agent 并行

每个 agent 有独立的 RemoteDockerEnvironment 实例：

```python
# Agent 1
env1 = RemoteDockerEnvironment(server_url=..., image=...)
env1.execute("command1")  # 使用 env1.session

# Agent 2（并行）
env2 = RemoteDockerEnvironment(server_url=..., image=...)
env2.execute("command2")  # 使用 env2.session

# ✅ 两个 session 完全独立，互不干扰
```

### 3. 服务器端配置

确保服务器支持长时间请求：

```python
# 在 docker_server.py 中
@app.post("/execute")
async def execute_command(request: ExecuteRequest):
    # 设置足够长的超时
    timeout = request.timeout or 1800
    
    # 执行命令
    result = await asyncio.wait_for(
        run_command(...),
        timeout=timeout
    )
    return result
```

### 4. 监控连接健康

```bash
# 查看连接状态
watch -n 2 'netstat -an | grep :9527 | awk "{print \$6}" | sort | uniq -c'

# 预期输出：
#   10 ESTABLISHED  # 正常连接
#    2 TIME_WAIT    # 正常关闭
```

## 🔍 故障排查

### 场景 1：命令超时

**症状**：
```
ERROR: Failed to execute command remotely: HTTPConnectionPool: Read timed out
```

**解决**：
```python
# 增加命令的超时时间
result = env.execute("long_command", timeout=3600)
```

### 场景 2：频繁断开

**症状**：日志中出现大量 RemoteDisconnected

**检查清单**：
```bash
# 1. 服务器负载
ssh server "top -bn1 | head -20"

# 2. 网络稳定性
ping server -c 100 | tail -5

# 3. 服务器日志
ssh server "tail -100 /var/log/docker_server.log"
```

**可能原因**：
- 服务器资源不足
- 网络不稳定
- 防火墙配置问题

### 场景 3：连接池耗尽

**症状**：
```
WARNING: Connection pool is full, discarding connection
```

**解决**：
```python
# 增加连接池大小（在 __init__ 中）
adapter = HTTPAdapter(
    pool_connections=10,   # 增加到 10
    pool_maxsize=20        # 增加到 20
)
```

## 📈 性能指标

### 超时配置对比

| 场景 | 30 秒超时 | 1800 秒超时 |
|------|-----------|-------------|
| 简单命令 | ✅ 成功 | ✅ 成功 |
| 编译项目 | ❌ 超时 | ✅ 成功 |
| 运行测试 | ❌ 超时 | ✅ 成功 |
| 长时间任务 | ❌ 超时 | ✅ 成功 |

### 连接稳定性

| 指标 | 无连接池 | 有连接池 + Keep-Alive |
|------|---------|---------------------|
| 断开频率 | 高 | 低 |
| 重连开销 | 大 | 小 |
| 响应延迟 | 高 | 低 |
| 资源占用 | 中 | 低 |

## 🛡️ 安全考虑

### 为什么不重试命令执行？

```python
# ❌ 危险的重试
def execute_with_retry(command):
    for attempt in range(3):
        try:
            return execute(command)
        except:
            continue
    
# 问题：
# 如果 command = "echo 'data' >> file.txt"
# 重试 3 次会导致 file.txt 中有 3 行数据！
```

```python
# ✅ 安全的设计
def execute(command):
    try:
        return self.session.post(...)
    except Exception as e:
        # 不重试，直接返回错误
        return {"output": f"Error: {e}", "returncode": -1}
    
# 命令只执行一次，不会产生副作用
```

### 幂等性

如果需要重试，命令必须是幂等的：

```python
# ✅ 幂等命令（可以安全重试）
- "cat file.txt"
- "ls -la"
- "git status"
- "docker ps"

# ❌ 非幂等命令（不能重试）
- "echo data >> file.txt"  # 会追加多次
- "mkdir mydir"            # 第二次会失败
- "git commit -m 'msg'"    # 会创建多个 commit
- "docker run ..."         # 会创建多个容器
```

## 🎓 技术原理

### Keep-Alive 机制

```
时间轴：
0s:   客户端 → 服务器: POST /execute (Connection: keep-alive)
      客户端 ← 服务器: 200 OK (Keep-Alive: timeout=300)
      [连接保持打开]

10s:  客户端 → 服务器: POST /execute (复用同一个 TCP 连接)
      客户端 ← 服务器: 200 OK
      [连接继续保持]

20s:  客户端 → 服务器: POST /execute (继续复用)
      客户端 ← 服务器: 200 OK

300s: [服务器关闭空闲连接]
310s: 客户端 → 服务器: POST /execute (建立新连接)
```

### 连接池管理

```
RemoteDockerEnvironment 实例
    └── self.session (requests.Session)
         └── HTTPAdapter
              └── urllib3.PoolManager
                   └── HTTPConnectionPool (host:port)
                        ├── 连接 1 [IDLE]
                        ├── 连接 2 [IN_USE]
                        ├── 连接 3 [IDLE]
                        └── ...

请求 1: 使用连接 2
请求 2: 使用连接 3（连接 2 还在用）
请求 3: 使用连接 1（复用空闲连接）
```

## 📝 代码示例

### 正确使用

```python
# 创建环境（每个 agent 一个）
env = RemoteDockerEnvironment(
    server_url="http://server:9527",
    image="rocm/pytorch:latest",
    timeout=1800  # 可选，默认就是 1800
)

# 执行短命令（使用默认超时）
result = env.execute("ls -la")

# 执行长命令（指定更长超时）
result = env.execute(
    "make all && make test", 
    timeout=3600  # 1 小时
)

# 清理
env.cleanup()
```

### 并发使用

```python
from concurrent.futures import ThreadPoolExecutor

def run_agent(task):
    # 每个线程创建独立的环境
    env = RemoteDockerEnvironment(...)
    try:
        result = env.execute(task['command'])
        return result
    finally:
        env.cleanup()

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(run_agent, tasks))
```

## 🎉 总结

### 核心设计

1. ✅ **不重试命令** - 避免副作用
2. ✅ **1800 秒超时** - 支持长时间任务
3. ✅ **连接池 + Keep-Alive** - 确保连接稳定
4. ✅ **独立 Session** - 每个实例隔离

### 稳定性保证

- 连接不会因为陈旧而断开（Keep-Alive）
- 不会因为超时而过早失败（1800 秒）
- 不会因为并发而冲突（独立 Session）
- 不会因为重试而产生副作用（无重试）

### 监控要点

```bash
# 连接状态
netstat -an | grep :9527 | awk '{print $6}' | sort | uniq -c

# 超时情况
grep "timed out" logs/*.log | wc -l

# 断开情况
grep "RemoteDisconnected" logs/*.log | wc -l
```

现在你的 Docker Remote 执行既**稳定**又**安全**！🚀

