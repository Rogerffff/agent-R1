# 配置系统详解

本文档详细说明 Agent-R1 的配置系统，包括启动脚本参数与 YAML 配置文件的关系。

---

## 配置框架概述

Agent-R1 使用 **Hydra + OmegaConf** 作为配置管理框架：

- **Hydra**: Facebook 开源的配置管理框架，支持命令行参数覆盖
- **OmegaConf**: YAML 配置解析库，支持插值语法

### 配置文件位置

| 文件 | 作用 |
|------|------|
| `agent_r1/src/config/agent_trainer.yaml` | 默认配置模板，包含所有参数的默认值 |
| `examples/trainer/run_grpo_hotpotqa.sh` | 启动脚本，通过命令行覆盖默认配置 |

---

## 启动脚本与 YAML 配置的关系

### 参数优先级

```
命令行参数（最高优先级）
        ↓
YAML 配置文件默认值（较低优先级）
```

**核心原则**：命令行参数会覆盖 YAML 中的同名参数。

### 配置加载流程

```
python3 -m agent_r1.src.main_agent key=value ...
                    │
                    ▼
        ┌──────────────────────┐
        │  @hydra.main 装饰器   │
        │  (main_agent.py:151) │
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │ 加载 agent_trainer.yaml │
        │     (默认配置)          │
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │   解析命令行参数        │
        │   覆盖默认值           │
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │ OmegaConf.resolve()   │
        │ 展开 ${} 插值引用      │
        └──────────┬───────────┘
                   │
                   ▼
            最终配置对象
```

---

## 配置语法说明

### 1. 点号语法（嵌套参数）

使用点号 `.` 访问嵌套参数：

```bash
# YAML 中的结构：
# actor_rollout_ref:
#   actor:
#     optim:
#       lr: 1e-6

# 命令行覆盖：
actor_rollout_ref.actor.optim.lr=1e-6
```

### 2. 插值语法（参数引用）

YAML 中使用 `${}` 引用其他参数：

```yaml
# agent_trainer.yaml 中的示例：
actor_rollout_ref:
  rollout:
    prompt_length: ${data.max_prompt_length}  # 引用 data.max_prompt_length
    response_length: ${data.max_response_length_single_turn}

ref:
  log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
```

**作用**：保持参数一致性，修改源参数时引用会自动更新。

### 3. 列表语法

列表参数使用 Python 列表格式：

```bash
# 单个文件
data.train_files=['data/hotpotqa/train.parquet']

# 多个文件
data.train_files=['data/train1.parquet','data/train2.parquet']

# 工具列表
tool.tools=['search']

# 日志器列表
trainer.logger=['console','wandb']
```

### 4. 布尔值和数字

```bash
# 布尔值（True/False）
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.actor.use_kl_loss=True

# 整数
trainer.n_gpus_per_node=4
data.train_batch_size=128

# 浮点数
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
```

---

## 实际示例：run_grpo_hotpotqa.sh 解析

### 完整脚本结构

```bash
#!/bin/bash
export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME=grpo-qwen2.5-1.5b-instruct

python3 -m agent_r1.src.main_agent \
    # === RL 算法选择 ===
    algorithm.adv_estimator=grpo \

    # === 数据配置 ===
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \

    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \

    # === Actor 训练配置 ===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \

    # === Rollout 推理配置 ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \

    # === Reference 模型配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \

    # === 算法参数 ===
    algorithm.kl_ctrl.kl_coef=0.001 \

    # === 训练器配置 ===
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \

    # === 工具配置 ===
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=2048 \

    $@  # 允许额外的命令行参数
```

### 参数分组说明

| 分组 | 前缀 | 作用 |
|------|------|------|
| 数据 | `data.*` | 训练/验证数据、序列长度 |
| 模型 | `actor_rollout_ref.model.*` | 模型路径、优化选项 |
| Actor | `actor_rollout_ref.actor.*` | PPO 训练参数、FSDP 配置 |
| Rollout | `actor_rollout_ref.rollout.*` | vLLM 推理参数、采样配置 |
| Reference | `actor_rollout_ref.ref.*` | 参考模型配置 |
| 算法 | `algorithm.*` | RL 算法选择、KL 控制 |
| 训练器 | `trainer.*` | GPU 数量、日志、保存频率 |
| 工具 | `tool.*` | Agent 工具配置 |

---

## 如何自定义配置

### 方法 1：修改启动脚本

复制现有脚本并修改：

```bash
cp examples/trainer/run_grpo_hotpotqa.sh my_training.sh
vim my_training.sh
# 修改需要的参数
bash my_training.sh
```

### 方法 2：命令行追加参数

利用脚本末尾的 `$@` 追加参数：

```bash
bash examples/trainer/run_grpo_hotpotqa.sh \
    trainer.n_gpus_per_node=4 \
    data.train_batch_size=256
```

**注意**：后面的参数会覆盖前面的参数。

### 方法 3：修改 YAML 默认值

直接编辑配置模板（不推荐，影响所有实验）：

```bash
vim agent_r1/src/config/agent_trainer.yaml
```

---

## 配置调试技巧

### 1. 打印最终配置

训练开始时会自动打印完整配置，检查参数是否正确：

```python
# main_agent.py 中的代码
pprint(OmegaConf.to_container(config, resolve=True))
```

### 2. 验证配置语法

如果配置有误，Hydra 会报错：

```
Error parsing config:
  key 'xxx' not found
```

### 3. 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `Key 'xxx' not found` | 参数名拼写错误 | 检查 agent_trainer.yaml 中的参数名 |
| `cannot convert 'xxx' to int` | 类型不匹配 | 检查参数值类型 |
| `Could not load xxx` | 文件路径错误 | 使用绝对路径或相对于工作目录的路径 |

---

## 常用配置组合

### 快速测试配置

```bash
data.train_batch_size=32 \
trainer.n_gpus_per_node=2 \
trainer.total_epochs=1 \
trainer.test_freq=5
```

### 大批量训练配置

```bash
data.train_batch_size=256 \
trainer.n_gpus_per_node=8 \
trainer.total_epochs=3 \
trainer.save_freq=100
```

### 低显存配置

```bash
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.ref.fsdp_config.param_offload=True
```

---

## 相关文档

- [06_parameter_reference.md](./06_parameter_reference.md) - 完整参数详解
- [07_gpu_config_templates.md](./07_gpu_config_templates.md) - GPU 配置模板
- [04_gpu_selection_guide.md](./04_gpu_selection_guide.md) - GPU 选择与问题排查
