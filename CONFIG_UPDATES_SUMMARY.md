# Config AMD YAML 更新摘要

## 更新目标
解决当前 agent 运行时大量 build fail 的问题，通过以下方式：
1. 明确告知 agent PR 描述中包含测试文件和 benchmark 文件信息
2. 强制要求在每次代码修改后验证编译是否成功
3. 禁止提交编译失败的代码

## 主要修改内容

### 1. 系统模板 (system_template) 更新
**新增内容：**
- 强调在每次代码修改后必须验证编译
- 明确工作流程：编辑 → 验证编译 → 测试 → 基准测试
- 使用 PR 描述中的测试文件名进行编译验证

### 2. 实例模板 (instance_template) 更新

#### a) 明确 PR 描述中的关键信息
添加了 "CRITICAL INFORMATION IN PR DESCRIPTION" 部分：
- **Kernel file:** 需要优化的具体文件
- **Test file:** 用于验证正确性的测试文件路径
- **Benchmark file:** 用于测量性能的基准测试文件路径

#### b) 更新工作流程 - 添加编译验证步骤

**Phase 2: 第一轮优化 (25-45 steps)**
- 步骤 4: **CRITICAL: Verify compilation** (新增，强制性步骤)
  - 在任何代码修改后必须验证编译
  - 使用 `cd build && make <test_file_name_from_PR>`
  - 如果编译失败，必须修复后才能继续
  - 不允许在编译失败时运行测试或基准测试

**Phase 3: 第二轮优化 (25-45 steps)**
- 步骤 9: **CRITICAL: Verify compilation again** (强制性)

**Phase 5: 最终验证和提交 (5-10 steps)**
- 步骤 12: **MANDATORY: Final compilation check before submit**
  - 提交前必须确保代码编译成功
  - 如果编译失败，立即修复 - 不允许提交

#### c) 更新命令示例
添加了编译验证示例：
```bash
# 验证编译 - 强制步骤
cd build && make test_block_reduce

# 构建和运行测试
cd build && make test_block_reduce && ./test/rocprim/test_block_reduce

# 运行基准测试
cd build && make benchmark_block_reduce && ./benchmark/benchmark_block_reduce
```

#### d) 更新提交规则

**MANDATORY REQUIREMENT BEFORE SUBMIT:**
- 代码必须编译成功
- 运行 `cd build && make <test_file_name>` 验证成功
- 如果编译失败，不允许提交 - 必须先修复错误

**更新期望的工作模式：**
- Round 1: Optimize (10) → **Compile Check (3)** → Test (8) → Benchmark (8) → Analyze (3) = ~32 steps
- Round 2: Optimize (10) → **Compile Check (3)** → Test (8) → Benchmark (8) → Analyze (3) = ~32 steps
- Round 3: Optimize (10) → **Compile Check (3)** → Test (8) → Benchmark (8) = ~29 steps
- 初始阅读: ~20 steps
- 最终编译验证: ~3 steps
- 总计: ~115-135 steps (包含编译检查的 3 轮迭代)

### 3. 记忆模板 (memory_template) 更新

**新增 "KEY FILES FROM PR DESCRIPTION" 部分：**
- 明确指出 PR 描述中包含三个关键文件路径
- Kernel file, Test file, Benchmark file

**新增 "MANDATORY COMPILATION VERIFICATION" 部分：**
- 在每次代码修改后必须验证编译成功
- 提供具体的编译验证命令示例
- 如果编译失败，立即修复
- 永远不要在编译失败的代码上运行测试或基准测试
- 永远不要提交编译失败的代码

**更新 "WORKFLOW - ALWAYS FOLLOW THIS PATTERN"：**
1. 对内核文件进行优化修改
2. **验证编译**（强制性 - 使用测试文件名的 make 命令）
3. 如果编译成功 → 运行测试
4. 如果测试通过 → 运行基准测试
5. 分析结果并迭代

**更新提交条件：**
- **MANDATORY PRE-SUBMIT CHECK:** 代码必须编译成功
- 提交时机：**编译成功** 且（性能提升 或 完成 2-3 轮迭代 或 使用 170-180 步）

## 关键强调点

### 编译验证的重要性
在配置文件的多个位置反复强调：
1. **MANDATORY STEP - DO NOT SKIP**
2. **NEVER SKIP compilation verification**
3. **NEVER SUBMIT code that doesn't compile**
4. **CRITICAL COMPILATION RULE**

### 工作流程顺序
明确规定顺序：
```
优化 → **编译检查** → 测试 → 基准测试
```

### 使用 PR 描述中的信息
- 明确告知 agent 在 PR 描述中查找 "Test file:" 和 "Benchmark file:" 信息
- 提供具体的提取和使用示例

## 预期效果

通过这些修改，agent 应该能够：
1. ✅ 自动识别 PR 描述中的测试文件和基准测试文件信息
2. ✅ 在每次修改代码后强制验证编译
3. ✅ 在编译失败时立即修复，而不是继续运行测试
4. ✅ 避免提交编译失败的代码
5. ✅ 显著减少 build fail 的情况

## 配置文件位置
`/home/jiajjiao/rocm-agent/src/minisweagent/config/rocm/config_amd.yaml`

## 相关文件
- PR 描述数据文件：`/home/jiajjiao/rocm-agent/data/rocprim_v5.json`
- 该文件已更新，在每个条目的 problem_statement 中包含：
  - Kernel file: <路径>
  - Test file: <路径>
  - Benchmark file: <路径>
