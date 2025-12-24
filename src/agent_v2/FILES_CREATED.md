# 创建的文件清单

本文档列出了为 agent continuous pretraining 数据生成系统创建的所有文件。

## 📁 文件结构

```
agent_v2/
├── 核心脚本 (Python)
│   ├── generate_training_data.py      (17KB) - 数据生成引擎
│   ├── process_training_data.py       (12KB) - 数据处理工具
│   └── visualize_data.py              (12KB) - 数据可视化
│
├── 执行脚本 (Shell)
│   ├── generate_training_data.sh      (589B) - 默认配置执行
│   ├── test_data_generation.sh        (1.3KB) - 快速测试
│   └── process_pipeline.sh            (2.1KB) - 完整处理管道
│
├── 文档
│   ├── README_TRAINING_DATA.md        (9.0KB) - 系统概述 ⭐ 主文档
│   ├── QUICKSTART.md                  (4.0KB) - 快速入门 ⭐ 新手必读
│   ├── TRAINING_DATA_GENERATION.md    (8.3KB) - 完整使用文档
│   ├── INDEX.md                       (8.8KB) - 工具索引
│   └── FILES_CREATED.md               (本文档) - 文件清单
│
└── examples/                          - 示例脚本目录
    ├── README.md                      (3.2KB) - 示例说明
    ├── generate_small_dataset.sh      (663B) - 小规模测试
    ├── generate_large_dataset.sh      (655B) - 大规模生产
    └── generate_diverse_dataset.sh    (654B) - 多样化数据

总计: 10个Python/Shell脚本 + 5个Markdown文档
```

## 📝 文件说明

### Python脚本

#### 1. `generate_training_data.py` (17KB)
**功能**: 使用 mini agent 生成训练数据的主引擎

**主要特性**:
- 单任务生成 (`generate_single`)
- 批量并行生成 (`generate_multi`)
- 完整对话轨迹记录
- Git diff 捕获
- 自动评估
- 多线程支持
- 中间结果保存

**使用**:
```bash
python generate_training_data.py generate_multi \
    --dataset data.json \
    --output out.json \
    --workers 4
```

#### 2. `process_training_data.py` (12KB)
**功能**: 数据处理和分析工具

**主要特性**:
- 数据分析 (`analyze`)
- 数据过滤 (`filter_data`)
- SFT格式导出 (`export_sft`)
- 轨迹格式导出 (`export_trajectory`)
- 单样本查看 (`show_example`)

**使用**:
```bash
python process_training_data.py analyze --input data.json
python process_training_data.py filter_data --input data.json --output filtered.json
```

#### 3. `visualize_data.py` (12KB)
**功能**: 数据可视化和质量报告

**主要特性**:
- 数据概览 (`overview`)
- 数据集对比 (`compare`)
- 质量报告 (`quality_report`)
- 文本直方图
- 统计表格

**使用**:
```bash
python visualize_data.py overview --input data.json
python visualize_data.py compare --file1 d1.json --file2 d2.json
```

### Shell脚本

#### 1. `generate_training_data.sh` (589B)
默认配置的数据生成脚本

**配置**:
- 数据集: rocprim_v5.json
- Workers: 4
- 模型: Qwen/Qwen3-8B
- 温度: 1.0

#### 2. `test_data_generation.sh` (1.3KB)
快速测试脚本（3个任务）

**功能**:
- 生成测试数据
- 自动分析
- 显示示例

#### 3. `process_pipeline.sh` (2.1KB)
完整的数据处理管道

**步骤**:
1. 分析原始数据
2. 过滤高质量样本
3. 导出SFT格式
4. 导出轨迹格式
5. 分析过滤后数据

### 示例脚本 (examples/)

#### 1. `generate_small_dataset.sh` (663B)
- 任务数: 10
- Workers: 2
- 用途: 快速测试

#### 2. `generate_large_dataset.sh` (655B)
- 任务数: 全部
- Workers: 8
- 用途: 生产环境

#### 3. `generate_diverse_dataset.sh` (654B)
- 任务数: 全部
- Workers: 4
- 温度: 1.5
- 用途: 探索性训练

### 文档

#### 1. `README_TRAINING_DATA.md` (9.0KB) ⭐ 主文档
完整的系统概述，包含：
- 快速开始
- 系统组成
- 核心功能
- 使用流程
- 最佳实践

**推荐**: 了解整体系统时阅读

#### 2. `QUICKSTART.md` (4.0KB) ⭐ 新手必读
快速入门指南，包含：
- 5分钟入门
- 常用命令速查
- 参数说明
- 下一步建议

**推荐**: 第一次使用时阅读

#### 3. `TRAINING_DATA_GENERATION.md` (8.3KB)
完整的使用文档，包含：
- 详细的命令说明
- 数据格式规范
- 配置选项
- 故障排查

**推荐**: 需要详细了解功能时查阅

#### 4. `INDEX.md` (8.8KB)
工具索引和快速参考，包含：
- 所有工具列表
- 快速参考表
- 使用场景
- 常见任务

**推荐**: 作为速查手册使用

#### 5. `examples/README.md` (3.2KB)
示例脚本说明文档

## 🚀 快速使用指南

### 第一次使用

1. **阅读快速入门**
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

### 日常使用

```bash
# 生成数据
bash generate_training_data.sh

# 处理数据
bash process_pipeline.sh input.json output_dir 0.7

# 查看质量
python visualize_data.py overview --input data.json
```

## 📊 文件依赖关系

```
generate_training_data.py
    ├── 依赖: minisweagent, eval_utils
    ├── 输入: dataset.json
    └── 输出: training_data.json

process_training_data.py
    ├── 依赖: 无外部依赖
    ├── 输入: training_data.json
    └── 输出: filtered.json, sft.jsonl, trajectory.json

visualize_data.py
    ├── 依赖: 无外部依赖
    ├── 输入: training_data.json
    └── 输出: 终端显示 / report.txt

Shell脚本
    ├── generate_training_data.sh → 调用 generate_training_data.py
    ├── test_data_generation.sh → 调用 generate_training_data.py + process_training_data.py + visualize_data.py
    └── process_pipeline.sh → 调用 process_training_data.py (多次)
```

## 🎯 使用场景映射

| 场景 | 使用的文件 |
|------|-----------|
| 快速测试 | `test_data_generation.sh` |
| 小规模实验 | `examples/generate_small_dataset.sh` |
| 生产环境 | `examples/generate_large_dataset.sh` + `process_pipeline.sh` |
| 数据分析 | `process_training_data.py analyze` |
| 数据可视化 | `visualize_data.py overview` |
| 质量评估 | `visualize_data.py quality_report` |
| 数据对比 | `visualize_data.py compare` |

## 📦 输出文件

系统运行后会在 `training_data/` 目录生成：

```
training_data/
├── test_training_data.json              # 测试数据
├── test_training_data.log               # 测试日志
├── mini_agent_training_data.json        # 默认输出
├── small_dataset.json                   # 小规模数据
├── large_dataset.json                   # 大规模数据
├── diverse_dataset.json                 # 多样化数据
└── processed/                           # 处理后的数据
    ├── filtered_data.json              # 过滤后
    ├── sft_training_data.jsonl         # SFT格式
    └── trajectory_training_data.json   # 轨迹格式
```

## 🔧 修改和定制

### 自定义数据生成

1. 复制示例脚本
   ```bash
   cp examples/generate_small_dataset.sh my_custom.sh
   ```

2. 修改参数
   ```bash
   # 编辑 my_custom.sh
   --workers 6 \
   --temperature 1.2 \
   --max-tasks 50
   ```

3. 运行
   ```bash
   bash my_custom.sh
   ```

### 扩展Python脚本

所有Python脚本都使用 `typer` 和标准Python 3.10+，可以轻松扩展：

```python
# 在 process_training_data.py 中添加新命令
@app.command()
def my_custom_command(
    input_file: Path = typer.Option(..., "--input"),
):
    # 你的自定义逻辑
    pass
```

## ✅ 验证安装

运行以下命令验证所有文件都已正确创建：

```bash
cd /home/jiajjiao/rocm-agent/src/agent_v2

# 检查Python脚本
ls -l generate_training_data.py process_training_data.py visualize_data.py

# 检查Shell脚本
ls -l *.sh

# 检查文档
ls -l *.md

# 检查示例
ls -l examples/

# 检查权限
stat -c '%A %n' *.sh examples/*.sh
```

所有 `.sh` 文件应该有执行权限 (`-rwxrwxr-x`)。

## 📞 下一步

1. **新用户**: 阅读 `QUICKSTART.md`
2. **测试系统**: 运行 `bash test_data_generation.sh`
3. **深入学习**: 阅读 `README_TRAINING_DATA.md`
4. **查看详细**: 阅读 `TRAINING_DATA_GENERATION.md`
5. **快速参考**: 使用 `INDEX.md`

---

**创建日期**: 2025-11-22  
**总文件数**: 15 (10个脚本 + 5个文档)  
**总大小**: ~75KB (脚本 + 文档)

