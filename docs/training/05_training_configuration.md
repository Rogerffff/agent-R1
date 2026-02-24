# 训练配置完全指南

本文档详细介绍 Agent-R1 的所有训练配置参数，帮助你理解每个参数的作用和如何根据需求进行配置。

## 目录

- [配置系统概述](#配置系统概述)
- [数据配置 (data)](#数据配置-data)
- [Actor 配置 (actor_rollout_ref)](#actor-配置-actor_rollout_ref)
- [Critic 配置 (critic)](#critic-配置-critic)
- [算法配置 (algorithm)](#算法配置-algorithm)
- [训练器配置 (trainer)](#训练器配置-trainer)
- [工具配置 (tool)](#工具配置-tool)
- [GRPO vs PPO 配置对比](#grpo-vs-ppo-配置对比)
- [配置模板](#配置模板)

---

## 配置系统概述

Agent-R1 使用 [Hydra](https://hydra.cc/) 进行配置管理。主配置文件位于：

```
agent_r1/src/config/agent_trainer.yaml
```

### 配置覆盖方式

```bash
# 命令行覆盖（推荐）
python3 -m agent_r1.src.main_agent \
    data.train_batch_size=32 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    algorithm.adv_estimator=grpo

# 使用配置文件覆盖
python3 -m agent_r1.src.main_agent \
    --config-path=/path/to/configs \
    --config-name=my_config
```

### 配置引用

配置文件支持内部引用：

```yaml
data:
  max_prompt_length: 4096

actor_rollout_ref:
  rollout:
    prompt_length: ${data.max_prompt_length}  # 引用 data.max_prompt_length
```

---

## 数据配置 (data)

控制数据加载和预处理的参数。

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `train_files` | str/list | - | 训练数据文件路径 |
| `val_files` | str/list | - | 验证数据文件路径 |
| `train_batch_size` | int | 1024 | 训练批次大小 |
| `val_batch_size` | int | null | 验证批次大小（null 表示与训练相同） |
| `prompt_key` | str | "prompt" | 数据中提示字段的键名 |
| `reward_fn_key` | str | "data_source" | 用于选择奖励函数的字段 |

### 长度限制参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `max_prompt_length` | int | 512 | 最大提示长度（tokens） |
| `max_response_length` | int | 512 | 最大响应总长度（tokens） |
| `max_response_length_single_turn` | int | 256 | 单轮响应最大长度（tokens） |
| `truncation` | str | "error" | 截断模式：error/left/right |

### 数据处理参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `shuffle` | bool | True | 是否打乱数据 |
| `filter_overlong_prompts` | bool | False | 是否过滤超长提示 |
| `filter_overlong_prompts_workers` | int | 1 | 过滤使用的进程数 |
| `return_raw_chat` | bool | False | 是否返回原始聊天消息 |
| `use_default_tool_template` | bool | True | 是否使用默认工具模板 |
| `use_custom_system_prompt` | bool | False | 是否使用自定义系统提示 |

### 多模态参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `image_key` | str | "images" | 图像数据字段键名 |
| `video_key` | str | "videos" | 视频数据字段键名 |

### 示例

```yaml
data:
  train_files: "['data/hotpotqa/train.parquet']"
  val_files: "['data/hotpotqa/validation.parquet']"
  train_batch_size: 32
  max_prompt_length: 4096
  max_response_length: 4096
  max_response_length_single_turn: 512
  shuffle: True
```

---

## Actor 配置 (actor_rollout_ref)

控制策略模型（Actor）、推理（Rollout）和参考策略（Ref）的参数。

### 模型配置 (actor_rollout_ref.model)

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `path` | str | - | 模型路径（HuggingFace 格式） |
| `enable_gradient_checkpointing` | bool | True | 启用梯度检查点（省内存） |
| `use_remove_padding` | bool | False | 移除填充（提高效率） |
| `use_liger` | bool | False | 使用 Liger 内核优化 |

### Actor 训练配置 (actor_rollout_ref.actor)

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `strategy` | str | "fsdp" | 分布式策略：fsdp/fsdp2 |
| `ppo_mini_batch_size` | int | 256 | PPO 小批次大小 |
| `ppo_micro_batch_size_per_gpu` | int | null | 每 GPU 微批大小 |
| `ppo_max_token_len_per_gpu` | int | 16384 | 每 GPU 最大 token 数 |
| `ppo_epochs` | int | 1 | PPO 训练轮数 |
| `grad_clip` | float | 1.0 | 梯度裁剪阈值 |
| `entropy_coeff` | float | 0 | 熵正则化系数 |
| `use_torch_compile` | bool | True | 使用 torch.compile |

#### PPO Clip 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `clip_ratio` | float | 0.2 | PPO 裁剪范围 |
| `clip_ratio_low` | float | 0.2 | 下界裁剪 |
| `clip_ratio_high` | float | 0.2 | 上界裁剪 |
| `clip_ratio_c` | float | 3.0 | Dual-Clip 下界 |

#### KL 损失参数（GRPO 专用）

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `use_kl_loss` | bool | False | 是否使用 KL 损失 |
| `kl_loss_coef` | float | 0.001 | KL 损失系数 |
| `kl_loss_type` | str | "low_var_kl" | KL 类型：kl/abs/mse/low_var_kl |

#### 优化器参数 (actor_rollout_ref.actor.optim)

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `lr` | float | 1e-6 | 学习率 |
| `weight_decay` | float | 0.01 | 权重衰减 |
| `lr_warmup_steps_ratio` | float | 0 | warmup 比例 |
| `warmup_style` | str | "constant" | warmup 风格：constant/cosine |

#### FSDP 配置 (actor_rollout_ref.actor.fsdp_config)

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `param_offload` | bool | False | 参数卸载到 CPU |
| `optimizer_offload` | bool | False | 优化器卸载到 CPU |
| `fsdp_size` | int | -1 | FSDP 分片大小（-1 自动） |

### Rollout 推理配置 (actor_rollout_ref.rollout)

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `name` | str | "vllm" | 推理引擎：vllm/sglang_async |
| `mode` | str | "sync" | 模式：sync/async |
| `temperature` | float | 1.0 | 采样温度 |
| `top_k` | int | -1 | Top-K 采样（-1 禁用） |
| `top_p` | float | 1.0 | Top-P 采样 |
| `n_repeat` | int | 1 | 每个提示的采样数（GRPO 需要 > 1） |

#### vLLM 特定参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `tensor_model_parallel_size` | int | 2 | 张量并行大小 |
| `gpu_memory_utilization` | float | 0.5 | GPU 内存使用率 |
| `max_num_batched_tokens` | int | 32768 | 最大批处理 token 数 |
| `max_num_seqs` | int | 1024 | 最大序列数 |
| `enable_chunked_prefill` | bool | True | 分块预填充 |
| `enforce_eager` | bool | True | 强制 eager 模式 |

### 参考策略配置 (actor_rollout_ref.ref)

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `strategy` | str | "fsdp" | 分布式策略 |
| `log_prob_micro_batch_size_per_gpu` | int | null | 微批大小 |

---

## Critic 配置 (critic)

仅在使用 GAE 算法时需要配置。GRPO 不使用 Critic。

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `strategy` | str | "fsdp" | 分布式策略 |
| `ppo_mini_batch_size` | int | 同 Actor | 小批次大小 |
| `ppo_max_token_len_per_gpu` | int | 32768 | 每 GPU 最大 token |
| `cliprange_value` | float | 0.5 | 值函数裁剪范围 |
| `grad_clip` | float | 1.0 | 梯度裁剪 |

### 优化器参数 (critic.optim)

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `lr` | float | 1e-5 | 学习率（通常比 Actor 大） |
| `weight_decay` | float | 0.01 | 权重衰减 |

### 模型配置 (critic.model)

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `path` | str | 同 Actor | Critic 模型路径 |
| `tokenizer_path` | str | 同 Actor | Tokenizer 路径 |
| `enable_gradient_checkpointing` | bool | True | 梯度检查点 |

---

## 算法配置 (algorithm)

控制强化学习算法的核心参数。

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `adv_estimator` | str | "gae" | 优势估计器：gae/grpo/reinforce_plus_plus/rloo/remax |
| `gamma` | float | 1.0 | 折扣因子 |
| `lam` | float | 1.0 | GAE λ 参数 |
| `norm_adv_by_std_in_grpo` | bool | True | GRPO 是否按标准差归一化 |

### KL 惩罚配置 (algorithm.kl_ctrl)

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `type` | str | "fixed" | KL 控制类型：fixed/adaptive |
| `kl_coef` | float | 0.001 | KL 系数 |
| `target_kl` | float | 0.1 | 目标 KL（仅 adaptive） |
| `horizon` | int | 10000 | 自适应调整窗口 |

### 奖励配置

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `use_kl_in_reward` | bool | False | 是否在奖励中加入 KL |
| `kl_penalty` | str | "kl" | KL 惩罚类型：kl/abs/mse/low_var_kl |
| `use_process_rewards` | bool | False | 是否使用过程奖励 |

---

## 训练器配置 (trainer)

控制训练流程的参数。

### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `total_epochs` | int | 30 | 总训练轮数 |
| `total_training_steps` | int | null | 总训练步数（优先于 epochs） |
| `project_name` | str | - | 项目名称 |
| `experiment_name` | str | - | 实验名称 |
| `nnodes` | int | 1 | 节点数 |
| `n_gpus_per_node` | int | 8 | 每节点 GPU 数 |

### 日志参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `logger` | list | ["console", "wandb"] | 日志后端 |
| `log_val_generations` | int | 0 | 记录的验证样本数 |
| `rollout_data_dir` | str | null | Rollout 数据保存目录 |
| `validation_data_dir` | str | null | 验证数据保存目录 |

### 保存参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `save_freq` | int | -1 | 保存频率（-1 不保存） |
| `default_local_dir` | str | - | 本地保存目录 |
| `default_hdfs_dir` | str | null | HDFS 保存目录 |
| `max_actor_ckpt_to_keep` | int | null | 保留的 Actor 检查点数 |
| `max_critic_ckpt_to_keep` | int | null | 保留的 Critic 检查点数 |

### 恢复参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `resume_mode` | str | "auto" | 恢复模式：auto/disable/resume_path |
| `resume_from_path` | str | null | 恢复路径 |
| `del_local_ckpt_after_load` | bool | False | 加载后删除本地副本 |

### 验证参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `test_freq` | int | -1 | 验证频率（-1 不验证） |
| `val_before_train` | bool | True | 训练前先验证 |
| `critic_warmup` | int | 0 | Critic 预热步数 |

---

## 工具配置 (tool)

控制工具调用的参数。

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `max_turns` | int | 10 | 最大工具调用轮数 |
| `use_batch_tool_calls` | bool | True | 批量执行工具 |
| `tools` | list | ["search"] | 使用的工具列表 |
| `env` | str | "nous" | 工具环境：nous/mathtir/retool |
| `max_tool_response_length` | int | 512 | 工具响应最大长度 |

### 验证专用工具配置 (tool.val_kwargs)

可以为验证设置不同的工具配置：

```yaml
tool:
  val_kwargs:
    max_turns: ${tool.max_turns}
    tools: ${tool.tools}
    env: ${tool.env}
    max_tool_response_length: ${tool.max_tool_response_length}
```

---

## GRPO vs PPO 配置对比

### GRPO 配置（推荐用于多数场景）

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.n_repeat=5 \
    algorithm.use_kl_in_reward=False \
    ...
```

**特点**：
- 不需要 Critic 模型，节省显存
- 需要多次采样（n_repeat > 1）
- 使用 KL 损失约束策略更新

### PPO (GAE) 配置

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.n_repeat=1 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.type=adaptive \
    trainer.critic_warmup=5 \
    ...
```

**特点**：
- 需要 Critic 模型
- 每个提示只需一次采样
- 优势估计更稳定

### 配置对比表

| 配置项 | GRPO | PPO (GAE) |
|-------|------|-----------|
| `algorithm.adv_estimator` | grpo | gae |
| `actor.use_kl_loss` | True | False |
| `rollout.n_repeat` | 5+ | 1 |
| `use_kl_in_reward` | False | True |
| `kl_ctrl.type` | fixed | adaptive |
| Critic 模型 | 不需要 | 需要 |
| 显存占用 | 较低 | 较高 |

---

## 配置模板

### 模板 1：2x RTX 4090 + 1.5B 模型 + GRPO

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="['data/hotpotqa/train.parquet']" \
    data.val_files="['data/hotpotqa/validation.parquet']" \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.max_response_length_single_turn=512 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n_repeat=5 \
    \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    trainer.logger="['console','wandb']" \
    trainer.project_name=hotpotqa \
    trainer.experiment_name=grpo-1.5b \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_training_steps=400 \
    trainer.resume_mode=auto \
    \
    tool.max_turns=5 \
    tool.tools="['search']" \
    tool.max_tool_response_length=1024
```

### 模板 2：8x A100 + 7B 模型 + PPO

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=gae \
    \
    data.train_files="['data/hotpotqa/train.parquet']" \
    data.val_files="['data/hotpotqa/validation.parquet']" \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n_repeat=1 \
    \
    critic.model.path=Qwen/Qwen2.5-7B-Instruct \
    critic.optim.lr=1e-5 \
    \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.type=adaptive \
    algorithm.kl_ctrl.target_kl=0.05 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.critic_warmup=5 \
    trainer.total_training_steps=1000
```

---

## 下一步

- [06_hyperparameter_tuning.md](./06_hyperparameter_tuning.md) - 学习如何调优超参数
- [07_distributed_training.md](./07_distributed_training.md) - 了解分布式训练配置
