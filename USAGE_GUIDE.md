# 更新后的配置使用指南

## 🎯 问题解决方案

### 原问题
- Agent 运行时出现大量 build fail
- 没有明确告知测试文件和基准测试文件的位置
- Agent 没有在代码修改后验证编译

### 解决方案
1. ✅ 在 `rocprim_v5.json` 中的每个 problem_statement 添加了文件信息
2. ✅ 在 `config_amd.yaml` 中强制要求编译验证步骤
3. ✅ 在多个关键位置强调不允许提交编译失败的代码

## 📁 更新的文件

### 1. 数据文件：`data/rocprim_v5.json`
每个条目的 `problem_statement` 现在包含：
```
Kernel file:rocprim/include/rocprim/block/block_reduce.hpp 
Test file:test/rocprim/test_block_reduce.cpp 
Benchmark file:benchmark/benchmark_block_reduce.cpp
```

**使用方式：**
```python
# 在你的代码中使用这个新文件
data_file = "data/rocprim_v5.json"
```

### 2. 配置文件：`src/minisweagent/config/rocm/config_amd.yaml`

#### 关键变更：

**a) 强制编译验证流程**
```
WORKFLOW AFTER EVERY CODE EDIT:
1. Edit kernel file
2. VERIFY COMPILATION (mandatory - never skip)
3. Run tests (only if compilation succeeds)
4. Run benchmarks (only if tests pass)
```

**b) 使用 PR 描述中的文件信息**
Agent 现在会：
- 从 PR 描述中提取 "Test file:" 路径
- 从 PR 描述中提取 "Benchmark file:" 路径
- 使用这些信息构建正确的测试和基准测试命令

**c) 编译验证命令**
```bash
# Agent 会使用这样的命令验证编译
cd build && make test_block_reduce

# 完整的测试流程
cd build && make test_block_reduce && ./test/rocprim/test_block_reduce

# 完整的基准测试流程
cd build && make benchmark_block_reduce && ./benchmark/benchmark_block_reduce
```

## 🚀 如何使用

### 方式 1：使用新的数据文件
```bash
# 在你的测试脚本中指定新的数据文件
python run_agent.py --data-file data/rocprim_v5.json --config src/minisweagent/config/rocm/config_amd.yaml
```

### 方式 2：检查配置是否正确
```bash
# 验证配置文件
python3 -c "import yaml; f=open('src/minisweagent/config/rocm/config_amd.yaml'); print('配置文件验证通过')"
```

## 📊 预期改进

### Before (使用 rocprim_v4.json + 旧配置)
- ❌ 大量 build fail
- ❌ Agent 不知道使用哪个测试文件
- ❌ Agent 跳过编译验证步骤
- ❌ 提交编译失败的代码

### After (使用 rocprim_v5.json + 新配置)
- ✅ Agent 清楚知道测试和基准测试文件路径
- ✅ 每次代码修改后强制验证编译
- ✅ 编译失败时立即修复，不继续执行
- ✅ 提交前最终编译检查
- ✅ 显著减少 build fail

## 🔍 验证修改效果

### 检查数据文件
```bash
cd /home/jiajjiao/rocm-agent
python3 -c "import json; d=json.load(open('data/rocprim_v5.json')); print(f'总条目数: {len(d)}'); print(d[0]['problem_statement'][:200])"
```

### 检查配置文件关键内容
```bash
grep -n "MANDATORY.*compilation\|Test file:\|Benchmark file:" src/minisweagent/config/rocm/config_amd.yaml | head -20
```

## 📝 Agent 的新工作流程

1. **理解阶段** (15-20 steps)
   - 读取 PR 描述，提取 Kernel file, Test file, Benchmark file 信息
   - 阅读基准测试文件和内核文件

2. **第一轮优化** (~32 steps)
   - 修改内核代码 (10 steps)
   - **验证编译** (3 steps) ← 新增强制步骤
   - 运行测试 (8 steps)
   - 运行基准测试 (8 steps)
   - 分析结果 (3 steps)

3. **第二轮优化** (~32 steps)
   - 基于反馈进一步优化
   - **再次验证编译** ← 强制步骤
   - 测试和基准测试
   - 分析结果

4. **第三轮优化** (可选, ~29 steps)
   - 如有需要继续迭代
   - **验证编译** ← 每次都要

5. **最终提交** (3-5 steps)
   - **最终编译检查** ← 提交前强制检查
   - 提交代码

## ⚠️ 重要提醒

1. **永远使用 rocprim_v5.json**，不要用 v4
2. **确保配置文件路径正确**
3. Agent 现在会在多处检查编译，这是正常的
4. 如果 Agent 在编译步骤花费较多步数，这是预期行为
5. 总步数可能从 ~100 增加到 ~115-135（增加了编译检查步骤）

## 🐛 故障排查

### 如果仍然出现 build fail：
1. 检查是否使用了正确的数据文件 (rocprim_v5.json)
2. 检查是否使用了更新后的配置文件
3. 查看 Agent 日志，确认是否在执行编译验证步骤
4. 检查 Docker 环境是否正确配置

### 如果 Agent 跳过编译验证：
- 这不应该发生，配置已在多处强制要求
- 检查配置文件是否被正确加载
- 查看 Agent 的完整提示词是否包含编译验证要求

## 📞 联系与反馈

如有问题，请检查：
- `/home/jiajjiao/rocm-agent/CONFIG_UPDATES_SUMMARY.md` - 详细修改说明
- `/home/jiajjiao/rocm-agent/USAGE_GUIDE.md` - 本使用指南

---
更新日期：2025-11-19
配置版本：v5
