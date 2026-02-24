# 训练输出文件详解

本文档详细介绍 Agent-R1 训练过程中生成的所有输出文件，包括 Checkpoint、日志文件和验证数据的结构与用途。

## 目录

- [输出目录概览](#输出目录概览)
- [Checkpoint 文件结构](#checkpoint-文件结构)
- [日志文件](#日志文件)
- [验证和 Rollout 数据](#验证和-rollout-数据)
- [配置文件保存](#配置文件保存)
- [文件清理策略](#文件清理策略)

---

## 输出目录概览

训练完成后，Agent-R1 会在指定目录下生成以下文件结构：

```
checkpoints/{project_name}/{experiment_name}/
├── global_step_50/                    # 第 50 步的检查点
│   ├── actor/                         # Actor 模型
│   │   ├── model/                     # 模型分片
│   │   │   ├── __0_0.distcp           # FSDP 分布式检查点
│   │   │   ├── __1_0.distcp
│   │   │   └── ...
│   │   ├── optimizer/                 # 优化器状态
│   │   │   └── ...
│   │   └── extra/                     # 额外状态（学习率调度器等）
│   │       └── ...
│   ├── critic/                        # Critic 模型（仅 GAE 算法）
│   │   └── ...
│   └── data.pt                        # DataLoader 状态
├── global_step_100/
│   └── ...
├── global_step_150/
│   └── ...
├── latest_checkpointed_iteration.txt  # 最新检查点编号
└── config.yaml                        # 训练配置（可选）

wandb/                                 # WandB 本地日志
├── run-{timestamp}-{id}/
│   ├── files/
│   │   ├── config.yaml
│   │   ├── wandb-metadata.json
│   │   └── ...
│   └── logs/
│       └── ...
```

### 关键配置项

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `trainer.default_local_dir` | 本地检查点目录 | `checkpoints/{project_name}/{experiment_name}` |
| `trainer.default_hdfs_dir` | HDFS 远程目录（可选） | `null` |
| `trainer.save_freq` | 保存频率（步数） | `-1`（不保存） |
| `trainer.project_name` | 项目名称 | - |
| `trainer.experiment_name` | 实验名称 | - |

---

## Checkpoint 文件结构

### Actor 检查点

Actor 检查点包含策略模型的权重和训练状态。

```
global_step_N/actor/
├── model/                             # 模型权重（FSDP 分片格式）
│   ├── __0_0.distcp                   # GPU 0 的分片
│   ├── __1_0.distcp                   # GPU 1 的分片
│   ├── .metadata                      # 分片元数据
│   └── ...
├── optimizer/                         # 优化器状态
│   ├── __0_0.distcp
│   ├── __1_0.distcp
│   └── ...
└── extra/                             # 额外状态
    ├── lr_scheduler.pt                # 学习率调度器
    └── training_state.pt              # 训练状态（step、epoch 等）
```

#### 模型分片说明

Agent-R1 使用 PyTorch FSDP（Fully Sharded Data Parallel）进行分布式训练。每个 GPU 只保存模型的一部分（分片），文件名格式为：

```
__{rank}_{shard}.distcp
```

- `rank`：GPU 的排名编号
- `shard`：分片编号

> **注意**：FSDP 检查点需要使用相同数量的 GPU 才能直接加载。如果要在不同数量的 GPU 上使用，需要先合并模型（见 [08_model_export_deploy.md](./08_model_export_deploy.md)）。

---

### Critic 检查点（仅 GAE）

仅当使用 GAE 算法（`algorithm.adv_estimator=gae`）时，才会保存 Critic 检查点。

```
global_step_N/critic/
├── model/
│   └── ...
├── optimizer/
│   └── ...
└── extra/
    └── ...
```

结构与 Actor 相同。

---

### DataLoader 状态

```
global_step_N/data.pt
```

保存训练数据加载器的状态，用于恢复训练时从正确的数据位置继续。

包含内容：
- 当前 epoch
- 已处理的样本索引
- 随机数生成器状态

---

### 最新检查点标记

```
latest_checkpointed_iteration.txt
```

记录最新保存的 `global_step` 编号，用于自动恢复训练。

文件内容示例：
```
150
```

这表示最新的检查点是 `global_step_150`。

---

### 检查点保存配置

在训练脚本中配置检查点保存：

```bash
python3 -m agent_r1.src.main_agent \
    trainer.save_freq=50 \                      # 每 50 步保存一次
    trainer.default_local_dir=./checkpoints \   # 本地保存目录
    trainer.max_actor_ckpt_to_keep=3 \          # 保留最近 3 个 Actor 检查点
    trainer.max_critic_ckpt_to_keep=3 \         # 保留最近 3 个 Critic 检查点
    ...
```

### 检查点内容配置

可以选择保存的内容：

```yaml
actor_rollout_ref:
  actor:
    checkpoint:
      contents: ['model', 'optimizer', 'extra']  # 保存模型、优化器和额外状态
      # 可选: 'hf_model' - 同时保存 HuggingFace 格式的完整模型
```

---

## 日志文件

### WandB 日志

#### 在线日志

训练指标会实时上传到 WandB 服务器，可以在以下位置查看：

```
https://wandb.ai/{username}/{project_name}/runs/{run_id}
```

#### 本地日志

WandB 也会在本地保存日志副本：

```
wandb/
├── run-{timestamp}-{id}/
│   ├── files/
│   │   ├── config.yaml              # 运行配置
│   │   ├── requirements.txt         # 依赖版本
│   │   ├── wandb-metadata.json      # 运行元数据
│   │   ├── wandb-summary.json       # 最终指标摘要
│   │   └── output.log               # 控制台输出
│   └── logs/
│       ├── debug.log                # 调试日志
│       └── debug-internal.log       # 内部调试日志
└── latest-run -> run-{timestamp}-{id}/  # 指向最新运行的符号链接
```

### 配置 WandB 日志

```bash
# 设置日志器
trainer.logger="['console','wandb']"

# 设置项目和实验名称
trainer.project_name=my_project
trainer.experiment_name=experiment_01

# 离线模式（不上传到服务器）
export WANDB_MODE=offline
```

### Console 日志

训练过程中的控制台输出包含：

1. **配置信息**：启动时打印完整配置
2. **进度信息**：每个训练步骤的进度
3. **指标摘要**：关键指标的实时值
4. **警告和错误**：训练过程中的问题

示例输出：
```
=== Step 50 ===
actor/pg_loss: -0.0234
actor/pg_clipfrac: 0.156
critic/score/mean: 0.456
perf/throughput: 1234.5 tokens/s/gpu
timing_s/gen: 12.34s
```

---

## 验证和 Rollout 数据

### 验证数据保存

配置验证数据保存：

```bash
trainer.validation_data_dir=./validation_logs
```

保存的文件：
```
validation_logs/
├── step_50/
│   ├── generations.jsonl            # 生成的响应
│   └── scores.json                  # 评分结果
├── step_100/
│   └── ...
```

`generations.jsonl` 格式：
```json
{"prompt": "问题...", "response": "生成的响应...", "score": 0.8, "acc": 1.0, "format": 0.75}
{"prompt": "问题...", "response": "生成的响应...", "score": 0.5, "acc": 0.0, "format": 0.5}
```

### Rollout 数据保存

配置训练 rollout 数据保存：

```bash
trainer.rollout_data_dir=./rollout_logs
```

保存的文件：
```
rollout_logs/
├── step_50/
│   ├── prompts.jsonl                # 输入提示
│   ├── responses.jsonl              # 生成的响应
│   ├── rewards.jsonl                # 奖励信息
│   └── tool_calls.jsonl             # 工具调用记录
└── ...
```

### WandB Table 日志

验证生成样本可以记录到 WandB Table：

```bash
trainer.log_val_generations=10  # 记录 10 个验证样本
```

这会在 WandB 中创建一个 `val/generations` 表格，包含：
- 输入 prompt
- 生成的响应
- 评分

---

## 配置文件保存

### Hydra 配置

Hydra 会自动保存运行配置到：

```
outputs/{date}/{time}/
├── .hydra/
│   ├── config.yaml                  # 合并后的完整配置
│   ├── hydra.yaml                   # Hydra 自身配置
│   └── overrides.yaml               # 命令行覆盖的参数
└── main.log                         # 主日志文件
```

### 手动保存配置

在训练脚本中，可以手动保存配置：

```python
from omegaconf import OmegaConf

# 保存为 YAML
with open("config.yaml", "w") as f:
    f.write(OmegaConf.to_yaml(config))

# 保存为 JSON
import json
with open("config.json", "w") as f:
    json.dump(OmegaConf.to_container(config), f, indent=2)
```

---

## 文件清理策略

### 自动清理

通过以下配置启用自动清理：

```bash
# 只保留最近 N 个检查点
trainer.max_actor_ckpt_to_keep=3
trainer.max_critic_ckpt_to_keep=3
```

这会自动删除旧的检查点，只保留最近的 N 个。

### 远程加载后清理

如果从 HDFS 加载检查点，可以配置加载后删除本地副本：

```bash
trainer.del_local_ckpt_after_load=True
```

### 手动清理

清理旧的检查点目录：

```bash
# 删除特定步骤的检查点
rm -rf checkpoints/my_project/exp01/global_step_50

# 只保留最新检查点
latest=$(cat checkpoints/my_project/exp01/latest_checkpointed_iteration.txt)
find checkpoints/my_project/exp01 -type d -name "global_step_*" ! -name "global_step_$latest" -exec rm -rf {} +
```

清理 WandB 本地缓存：

```bash
# 删除所有本地 wandb 日志
rm -rf wandb/

# 删除特定运行
rm -rf wandb/run-{timestamp}-{id}
```

---

## 文件大小估算

| 模型大小 | Actor 检查点 | Critic 检查点 | 总计（每步） |
|---------|-------------|---------------|-------------|
| 1.5B | ~6 GB | ~6 GB | ~12 GB |
| 7B | ~28 GB | ~28 GB | ~56 GB |
| 14B | ~56 GB | ~56 GB | ~112 GB |

> **提示**：如果只使用 GRPO 算法，不需要 Critic，检查点大小减半。

### 存储规划建议

```
预估存储 = 单个检查点大小 × 保留数量 × 1.2（安全余量）

例如：7B 模型，GRPO 算法，保留 3 个检查点
存储 = 28 GB × 3 × 1.2 ≈ 100 GB
```

---

## 常见问题

### Q: 检查点保存失败？

**检查项**：
1. 磁盘空间是否充足
2. 目录权限是否正确
3. HDFS 连接是否正常（如果使用）

### Q: WandB 上传失败？

**解决方案**：
```bash
# 切换到离线模式
export WANDB_MODE=offline

# 训练完成后手动同步
wandb sync wandb/offline-run-{id}
```

### Q: 如何查看检查点内容？

```python
import torch

# 查看 DataLoader 状态
data_state = torch.load("checkpoints/.../global_step_50/data.pt")
print(data_state)

# 查看 FSDP 检查点元数据
from torch.distributed.checkpoint import FileSystemReader
reader = FileSystemReader("checkpoints/.../global_step_50/actor/model")
metadata = reader.read_metadata()
print(metadata)
```

---

## 下一步

- [04_resume_training.md](./04_resume_training.md) - 学习如何从检查点恢复训练
- [08_model_export_deploy.md](./08_model_export_deploy.md) - 了解如何导出和部署模型
