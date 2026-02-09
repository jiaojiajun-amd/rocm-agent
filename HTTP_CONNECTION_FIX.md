# HTTP 连接管理问题修复

## 问题描述

代码中多处使用 `requests.post()` 和 `requests.get()` 但没有正确管理 HTTP 连接，可能导致：
1. 连接泄漏 - 连接没有被及时关闭
2. 连接池耗尽 - 大量未关闭的连接堆积
3. 资源浪费 - TCP 连接没有被复用

## 修复内容

### 1. `src/minisweagent/environments/docker_remote.py`
**问题**：类中多次调用 `requests.post()` 但没有复用连接
**修复**：
- 在 `__init__` 中创建 `self.session = requests.Session()`
- 所有 `requests.post()` 改为 `self.session.post()`
- 在 `cleanup()` 方法中添加 `self.session.close()`
- 在 `__init__` 异常处理中也添加 `self.session.close()`

**优势**：
- 连接复用，提高性能
- 自动管理连接池
- 确保资源正确释放

### 2. `src/minisweagent/models/openrouter_model.py`
**问题**：模型类频繁调用 API 但没有复用连接
**修复**：
- 在 `__init__` 中创建 `self.session = requests.Session()`
- `requests.post()` 改为 `self.session.post()`
- 添加 `__del__` 方法，在对象销毁时关闭 session

**优势**：
- 对于频繁的模型调用，连接复用可显著提高性能
- 避免连接泄漏

### 3. `src/liteagent/light_agent.py`
**问题**：评估函数中的 HTTP 请求没有正确管理
**修复**：
- 使用 `with requests.Session() as session:` 上下文管理器
- 确保所有响应处理代码在 with 块内
- 在异常处理中检查 `resp` 是否存在后再访问

**优势**：
- 自动关闭连接
- 异常安全

### 4. `src/agent_v2/eval_utils.py`
**问题**：两个评估函数都存在相同问题
**修复**：与 light_agent.py 相同的修复方案

## 关于你遇到的错误

你看到的错误：
```
requests.exceptions.ConnectionError: HTTPConnectionPool(host='10.67.77.184', port=9527): 
Max retries exceeded with url: /start (Caused by 
NewConnectionError: Failed to establish a new connection: [Errno 111] Connection refused'))
```

**主要原因**：服务器 `10.67.77.184:9527` 没有运行或无法连接

**代码问题**：虽然连接被拒绝主要是服务器端问题，但代码中确实存在连接管理不当的问题，可能在高负载情况下导致资源耗尽。

## 建议

1. **确保评估服务器正在运行**：检查 `10.67.77.184:9527` 上的服务
2. **检查网络连接**：确认防火墙和网络配置
3. **监控连接数**：使用 `netstat -an | grep 9527` 查看连接状态
4. **配置重试机制**：代码中已有重试逻辑，确保配置合理

## 测试建议

```bash
# 检查服务器是否可达
curl http://10.67.77.184:9527/health

# 或使用 telnet
telnet 10.67.77.184 9527
```

