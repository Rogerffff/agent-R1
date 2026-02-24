# 从 Checkpoint 继续训练

本文档详细介绍如何从保存的 Checkpoint 恢复训练，包括不同恢复模式的使用方法、配置说明和常见问题处理。

## 目录

- [恢复模式概述](#恢复模式概述)
- [自动恢复模式 (auto)](#自动恢复模式-auto)
- [禁用恢复模式 (disable)](#禁用恢复模式-disable)
- [指定路径恢复模式 (resume_path)](#指定路径恢复模式-resume_path)
- [恢复训练的完整示例](#恢复训练的完整示例)
- [断点续训注意事项](#断点续训注意事项)
- [常见问题与解决方案](#常见问题与解决方案)

---

## 恢复模式概述

Agent-R1 支持三种训练恢复模式：

| 模式 | 配置值 | 说明 | 适用场景 |
|------|-------|------|---------|
| 自动恢复 | `auto` | 自动查找最新检查点，如果存在则恢复 | 默认推荐，适合大多数场景 |
| 禁用恢复 | `disable` | 始终从头开始训练 | 需要重新训练时使用 |
| 指定路径 | `resume_path` | 从指定的检查点路径恢复 | 需要从特定检查点恢复时使用 |

### 配置方式

```bash
# 命令行配置
python3 -m agent_r1.src.main_agent \
    trainer.resume_mode=auto \           # 恢复模式
    trainer.resume_from_path=null \      # 指定路径（仅 resume_path 模式需要）
    trainer.default_local_dir=./checkpoints/my_project/exp01 \
    ...
```

---

## 自动恢复模式 (auto)

### 工作原理

1. 检查 `trainer.default_local_dir` 目录是否存在
2. 查找 `latest_checkpointed_iteration.txt` 文件
3. 如果存在，读取最新的 `global_step` 编号
4. 加载对应的检查点并恢复训练
5. 如果不存在，从头开始训练

### 使用方法

```bash
python3 -m agent_r1.src.main_agent \
    trainer.resume_mode=auto \
    trainer.default_local_dir=./checkpoints/hotpotqa/grpo-1.5b \
    ...
```

### 工作流程图

```
开始训练
    │
    ▼
检查 default_local_dir 是否存在
    │
    ├── 不存在 ──► 从头开始训练（global_step = 0）
    │
    └── 存在
         │
         ▼
    检查 latest_checkpointed_iteration.txt
         │
         ├── 不存在 ──► 从头开始训练（global_step = 0）
         │
         └── 存在
              │
              ▼
         读取最新 global_step（例如：150）
              │
              ▼
         加载 global_step_150/ 检查点
              │
              ├── Actor 模型权重
              ├── Critic 模型权重（如果使用 GAE）
              ├── 优化器状态
              └── DataLoader 状态
              │
              ▼
         从 global_step = 150 继续训练
```

### 优点

- **无需手动指定路径**：自动查找最新检查点
- **安全恢复**：如果没有检查点，自动从头开始
- **适合长期训练**：训练中断后重启即可自动恢复

---

## 禁用恢复模式 (disable)

### 使用场景

- 需要重新训练模型
- 想要尝试不同的超参数
- 检查点文件损坏

### 使用方法

```bash
python3 -m agent_r1.src.main_agent \
    trainer.resume_mode=disable \
    trainer.default_local_dir=./checkpoints/hotpotqa/grpo-1.5b \
    ...
```

> **警告**：使用 `disable` 模式会从头开始训练，但**不会删除**现有的检查点文件。如果需要清理旧检查点，请手动删除。

### 清理旧检查点

```bash
# 删除所有检查点
rm -rf ./checkpoints/hotpotqa/grpo-1.5b/global_step_*
rm -f ./checkpoints/hotpotqa/grpo-1.5b/latest_checkpointed_iteration.txt

# 或者使用新的实验名称
trainer.experiment_name=grpo-1.5b-v2
```

---

## 指定路径恢复模式 (resume_path)

### 使用场景

- 从特定的检查点恢复（不是最新的）
- 从其他目录的检查点恢复
- 从 HDFS 路径恢复

### 使用方法

```bash
python3 -m agent_r1.src.main_agent \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=./checkpoints/hotpotqa/grpo-1.5b/global_step_100 \
    ...
```

### 从 HDFS 恢复

```bash
python3 -m agent_r1.src.main_agent \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=hdfs://namenode:8020/checkpoints/hotpotqa/grpo-1.5b/global_step_100 \
    trainer.del_local_ckpt_after_load=True \  # 加载后删除本地副本
    ...
```

### 路径格式说明

```
resume_from_path 应该指向包含 actor/ 子目录的检查点目录：

./checkpoints/hotpotqa/grpo-1.5b/global_step_100/
├── actor/           ← resume_from_path 应该指向父目录
│   ├── model/
│   ├── optimizer/
│   └── extra/
├── critic/          ← 可选（仅 GAE）
└── data.pt
```

---

## 恢复训练的完整示例

### 场景 1：训练中断后恢复

假设你正在进行以下训练，但在第 200 步时训练中断：

```bash
# 初始训练命令
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files="['data/hotpotqa/train.parquet']" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    trainer.project_name=hotpotqa \
    trainer.experiment_name=grpo-1.5b \
    trainer.save_freq=50 \
    trainer.total_training_steps=400 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=./checkpoints/hotpotqa/grpo-1.5b \
    trainer.n_gpus_per_node=2 \
    ...
```

检查点状态：
```
./checkpoints/hotpotqa/grpo-1.5b/
├── global_step_50/
├── global_step_100/
├── global_step_150/
├── global_step_200/
└── latest_checkpointed_iteration.txt  # 内容: 200
```

**恢复训练**：只需要重新运行相同的命令：

```bash
# 使用完全相同的命令重新启动
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files="['data/hotpotqa/train.parquet']" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    trainer.project_name=hotpotqa \
    trainer.experiment_name=grpo-1.5b \
    trainer.save_freq=50 \
    trainer.total_training_steps=400 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=./checkpoints/hotpotqa/grpo-1.5b \
    trainer.n_gpus_per_node=2 \
    ...
```

训练将自动从 `global_step_200` 继续。

---

### 场景 2：从特定检查点继续

假设你想从第 100 步的检查点继续，而不是最新的第 200 步：

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files="['data/hotpotqa/train.parquet']" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    trainer.project_name=hotpotqa \
    trainer.experiment_name=grpo-1.5b-from-100 \  # 使用新的实验名称
    trainer.save_freq=50 \
    trainer.total_training_steps=400 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=./checkpoints/hotpotqa/grpo-1.5b/global_step_100 \
    trainer.default_local_dir=./checkpoints/hotpotqa/grpo-1.5b-from-100 \
    trainer.n_gpus_per_node=2 \
    ...
```

> **建议**：从旧检查点恢复时，使用新的 `experiment_name` 和 `default_local_dir`，避免与原有检查点混淆。

---

### 场景 3：修改超参数后继续训练

如果想从某个检查点继续，但使用不同的学习率：

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files="['data/hotpotqa/train.parquet']" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \           # 修改学习率
    trainer.project_name=hotpotqa \
    trainer.experiment_name=grpo-1.5b-lr5e7 \         # 新实验名称
    trainer.save_freq=50 \
    trainer.total_training_steps=600 \                 # 可以增加总步数
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=./checkpoints/hotpotqa/grpo-1.5b/global_step_200 \
    trainer.default_local_dir=./checkpoints/hotpotqa/grpo-1.5b-lr5e7 \
    trainer.n_gpus_per_node=2 \
    ...
```

> **注意**：修改学习率后，优化器的动量等状态会继续使用，但学习率会立即应用新值。

---

## 断点续训注意事项

### 1. GPU 数量必须一致

FSDP 检查点与 GPU 数量绑定。恢复训练时必须使用相同数量的 GPU。

```bash
# 错误示例：原来用 2 GPU，恢复时用 4 GPU
# 原训练：trainer.n_gpus_per_node=2
# 恢复时：trainer.n_gpus_per_node=4  ❌ 这会失败

# 正确做法：使用相同的 GPU 数量
# 原训练：trainer.n_gpus_per_node=2
# 恢复时：trainer.n_gpus_per_node=2  ✅
```

如果需要改变 GPU 数量，请参考 [08_model_export_deploy.md](./08_model_export_deploy.md) 中的模型合并方法。

### 2. 模型架构必须一致

恢复训练时，模型配置必须与保存时完全一致：

- `actor_rollout_ref.model.path` 必须是相同的模型
- 不能修改模型架构相关的配置

### 3. 数据配置可以修改

以下配置在恢复时**可以**修改：

- `data.train_batch_size`：批次大小
- `data.max_prompt_length` / `data.max_response_length`：序列长度
- `trainer.total_training_steps`：总训练步数

### 4. WandB 日志处理

恢复训练时，WandB 会创建一个新的 run。如果想要在同一个 run 中继续记录：

```bash
# 设置 WandB run ID（需要从之前的 run 获取）
export WANDB_RUN_ID=your_previous_run_id
export WANDB_RESUME=must
```

或者，保持默认行为（创建新 run），在 WandB 界面上通过标签关联。

### 5. 学习率调度器

恢复训练时，学习率调度器的状态也会被恢复。如果原来使用了 warmup：

- 恢复后不会重新进行 warmup
- 学习率会从中断时的位置继续

### 6. 数据迭代位置

通过 `data.pt` 文件，训练会从正确的数据位置继续：

- epoch 计数会继续
- 已见过的样本不会重复（在同一 epoch 内）
- 随机采样状态会恢复

---

## 常见问题与解决方案

### Q1: 恢复训练时报错 "Checkpoint not found"

**原因**：指定的检查点路径不存在或不完整。

**解决方案**：
```bash
# 检查检查点是否存在
ls -la ./checkpoints/hotpotqa/grpo-1.5b/

# 检查 latest_checkpointed_iteration.txt
cat ./checkpoints/hotpotqa/grpo-1.5b/latest_checkpointed_iteration.txt

# 检查具体检查点目录
ls -la ./checkpoints/hotpotqa/grpo-1.5b/global_step_200/
```

### Q2: 恢复后 Loss 突然变化

**原因**：
1. 学习率设置与原来不同
2. batch size 改变导致梯度估计不同
3. 数据顺序问题

**解决方案**：
- 使用完全相同的配置恢复
- 如果需要修改配置，使用新的实验名称

### Q3: 恢复后 GPU 内存不足

**原因**：恢复时加载了额外的状态（如优化器状态）。

**解决方案**：
```bash
# 启用优化器 offload
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# 或者减小 batch size
data.train_batch_size=16
actor_rollout_ref.actor.ppo_mini_batch_size=8
```

### Q4: 从 HDFS 恢复很慢

**原因**：需要下载整个检查点到本地。

**解决方案**：
```bash
# 1. 手动预先下载
hdfs dfs -get hdfs://path/to/checkpoint ./local_checkpoint

# 2. 然后从本地恢复
trainer.resume_mode=resume_path
trainer.resume_from_path=./local_checkpoint
```

### Q5: 如何验证恢复是否成功

检查以下几点：

1. **global_step 是否正确**：
   ```
   日志应该显示：Resuming from step 200
   ```

2. **训练指标是否连续**：
   ```
   WandB 曲线应该从中断点继续，没有明显跳跃
   ```

3. **学习率是否正确**：
   ```
   检查 actor/lr 指标是否与预期一致
   ```

---

## 快速参考

### 恢复训练命令模板

```bash
# 自动恢复（推荐）
python3 -m agent_r1.src.main_agent \
    trainer.resume_mode=auto \
    trainer.default_local_dir=./checkpoints/{project}/{experiment} \
    ... # 其他配置与原训练相同

# 从特定检查点恢复
python3 -m agent_r1.src.main_agent \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=./checkpoints/{project}/{experiment}/global_step_N \
    trainer.experiment_name={experiment}-continued \
    trainer.default_local_dir=./checkpoints/{project}/{experiment}-continued \
    ... # 其他配置

# 从头开始（清理重训）
python3 -m agent_r1.src.main_agent \
    trainer.resume_mode=disable \
    trainer.experiment_name={experiment}-v2 \
    trainer.default_local_dir=./checkpoints/{project}/{experiment}-v2 \
    ... # 其他配置
```

---

## 下一步

- [05_training_configuration.md](./05_training_configuration.md) - 了解完整的训练配置选项
- [03_training_outputs.md](./03_training_outputs.md) - 了解检查点文件结构
- [08_model_export_deploy.md](./08_model_export_deploy.md) - 了解如何合并和导出模型
