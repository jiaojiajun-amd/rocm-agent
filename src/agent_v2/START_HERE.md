# 🎯 从这里开始

欢迎使用 Agent Continuous Pretraining 数据生成系统！

## 📖 我应该读哪个文档？

根据你的需求选择：

### 🚀 我想快速开始（5分钟）
**阅读**: [QUICKSTART.md](QUICKSTART.md)
```bash
cat QUICKSTART.md
bash test_data_generation.sh
```

### 📚 我想了解整个系统（15分钟）
**阅读**: [README_TRAINING_DATA.md](README_TRAINING_DATA.md)
```bash
cat README_TRAINING_DATA.md
```

### 🔍 我想查找某个功能（速查）
**阅读**: [INDEX.md](INDEX.md)
```bash
cat INDEX.md
```

### 📖 我想了解所有细节（深入）
**阅读**: [TRAINING_DATA_GENERATION.md](TRAINING_DATA_GENERATION.md)
```bash
cat TRAINING_DATA_GENERATION.md
```

### 📋 我想知道创建了哪些文件
**阅读**: [FILES_CREATED.md](FILES_CREATED.md)
```bash
cat FILES_CREATED.md
```

## ⚡ 最快的开始方式

如果你只想尽快看到效果：

```bash
# 1. 进入目录
cd /home/jiajjiao/rocm-agent/src/agent_v2

# 2. 运行测试（生成3个样本，约3-10分钟）
bash test_data_generation.sh

# 3. 查看结果
python visualize_data.py overview \
    --input training_data/test_training_data.json

# 完成！你已经生成了第一批训练数据
```

## 📊 系统概览

```
┌─────────────────────────────────────────┐
│  Agent Continuous Pretraining          │
│  Training Data Generation System        │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │ 生成   │ │ 处理   │ │ 可视化 │
    └────────┘ └────────┘ └────────┘
        │           │           │
        ▼           ▼           ▼
    原始数据   过滤数据   统计报告
    (JSON)    (JSONL)    (可视化)
```

## 🎯 核心功能

- ✅ 使用 mini agent 执行任务
- ✅ 记录完整对话轨迹
- ✅ 捕获代码变更（git diff）
- ✅ 自动评估和打分
- ✅ 多线程并行处理
- ✅ 数据过滤和格式化
- ✅ 统计分析和可视化

## 🛠️ 核心命令

### 生成数据
```bash
# 测试（推荐第一次运行）
bash test_data_generation.sh

# 小规模（10任务）
bash examples/generate_small_dataset.sh

# 大规模（全部任务）
bash examples/generate_large_dataset.sh

# 自定义
bash generate_training_data.sh
```

### 处理数据
```bash
# 完整管道
bash process_pipeline.sh input.json output_dir 0.7

# 单独分析
python process_training_data.py analyze --input data.json

# 过滤
python process_training_data.py filter_data \
    --input data.json --output filtered.json --min-reward 0.7

# 导出SFT格式
python process_training_data.py export_sft \
    --input filtered.json --output sft.jsonl
```

### 可视化
```bash
# 数据概览
python visualize_data.py overview --input data.json

# 质量报告
python visualize_data.py quality_report \
    --input data.json --output report.txt

# 对比数据集
python visualize_data.py compare \
    --file1 d1.json --file2 d2.json
```

## 📁 重要目录

```
agent_v2/
├── *.py                    # Python脚本
├── *.sh                    # Shell脚本
├── *.md                    # 文档
├── examples/               # 示例脚本
└── training_data/          # 生成的数据（自动创建）
```

## 💡 最佳实践

1. **始终从测试开始**: `bash test_data_generation.sh`
2. **逐步扩展规模**: 测试 → 小规模 → 大规模
3. **使用质量过滤**: `--min-reward 0.7` 或更高
4. **保存日志**: 使用 `--log-file` 参数
5. **定期备份**: 及时保存生成的数据

## 🆘 需要帮助？

| 问题 | 解决方案 |
|------|---------|
| 不知道从哪开始 | 阅读 [QUICKSTART.md](QUICKSTART.md) |
| 想了解某个功能 | 查看 [INDEX.md](INDEX.md) |
| 遇到错误 | 查看日志：`tail -f training_data/*.log` |
| 数据质量低 | 调整参数：降低温度、提高min-reward |
| 速度太慢 | 增加workers、检查网络 |

## 📞 命令帮助

所有工具都支持 `--help`:

```bash
python generate_training_data.py --help
python generate_training_data.py generate_multi --help
python process_training_data.py --help
python visualize_data.py --help
```

## 🎉 准备好了吗？

### 选择你的路径：

**路径A: 快速体验（推荐新手）**
```bash
bash test_data_generation.sh
```

**路径B: 小规模实验**
```bash
bash examples/generate_small_dataset.sh
```

**路径C: 生产环境**
```bash
bash examples/generate_large_dataset.sh
bash process_pipeline.sh training_data/large_dataset.json training_data/prod 0.8
```

---

**提示**: 如果这是你第一次使用，强烈建议先运行 `bash test_data_generation.sh`

**下一步**: 阅读 [QUICKSTART.md](QUICKSTART.md) 获取更多信息

🚀 **现在就开始**: `bash test_data_generation.sh`
