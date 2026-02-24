# 参数详解手册

本文档提供 Agent-R1 所有训练参数的完整说明，按模块分类。

---

## 目录

1. [data - 数据配置](#1-data---数据配置)
2. [actor_rollout_ref.model - 模型配置](#2-actor_rollout_refmodel---模型配置)
3. [actor_rollout_ref.actor - Actor 训练配置](#3-actor_rollout_refactor---actor-训练配置)
4. [actor_rollout_ref.rollout - Rollout 推理配置](#4-actor_rollout_refrollout---rollout-推理配置)
5. [actor_rollout_ref.ref - Reference 模型配置](#5-actor_rollout_refref---reference-模型配置)
6. [algorithm - RL 算法配置](#6-algorithm---rl-算法配置)
7. [trainer - 训练器配置](#7-trainer---训练器配置)
8. [tool - 工具系统配置](#8-tool---工具系统配置)
9. [参数约束关系](#9-参数约束关系)

---

## 1. data - 数据配置

### 数据文件

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_files` | list/str | `~/data/rlhf/gsm8k/train.parquet` | 训练数据文件路径 |
| `val_files` | list/str | `~/data/rlhf/gsm8k/test.parquet` | 验证数据文件路径 |
| `prompt_key` | str | `prompt` | 数据中 prompt 字段的键名 |
| `reward_fn_key` | str | `data_source` | 奖励函数选择的键名 |

### 序列长度

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_prompt_length` | int | `512` | Prompt 最大 token 长度 |
| `max_response_length` | int | `512` | 响应最大 token 长度（多轮总长度） |
| `max_response_length_single_turn` | int | `256` | 单轮响应最大长度（Rollout 使用） |
| `truncation` | str | `error` | 超长处理方式：`left`/`right`/`error` |

**说明**：
- `max_response_length` 用于整个对话的响应长度上限
- `max_response_length_single_turn` 用于每次 Rollout 生成的长度上限
- `truncation=error` 会在序列超长时报错，`left`/`right` 会截断

### 批处理

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_batch_size` | int | `1024` | 训练批大小（全局） |
| `val_batch_size` | int | `null` | 验证批大小（默认与训练相同） |
| `shuffle` | bool | `True` | 是否打乱数据 |

### 数据处理选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `filter_overlong_prompts` | bool | `False` | 过滤超长 Prompt |
| `filter_overlong_prompts_workers` | int | `1` | 过滤使用的进程数 |
| `return_raw_input_ids` | bool | `False` | 保留原始 token IDs |
| `return_raw_chat` | bool | `False` | 保留原始聊天格式 |
| `use_default_tool_template` | bool | `True` | 使用默认工具模板 |
| `use_custom_system_prompt` | bool | `False` | 使用自定义系统提示 |

### 多模态支持

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_key` | str | `images` | 图像数据字段键名 |
| `video_key` | str | `videos` | 视频数据字段键名 |

---

## 2. actor_rollout_ref.model - 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | str | `~/models/deepseek-llm-7b-chat` | 模型路径（HuggingFace 格式） |
| `external_lib` | str | `null` | 外部库路径（自定义模型） |
| `override_config` | dict | `{}` | 覆盖模型配置 |
| `enable_gradient_checkpointing` | bool | `True` | 启用梯度检查点（节省显存） |
| `use_remove_padding` | bool | `False` | 移除填充优化 |
| `use_liger` | bool | `False` | 使用 Liger Kernel 优化 |

### 重要说明

**enable_gradient_checkpointing**:
- **强烈建议启用**
- 以计算换显存，节省约 30-50% 显存
- 训练速度降低约 20%

**use_remove_padding**:
- 移除序列中的填充 token，提高计算效率
- 对变长序列特别有效

---

## 3. actor_rollout_ref.actor - Actor 训练配置

### 批处理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ppo_mini_batch_size` | int | `256` | PPO 更新的迷你批大小 |
| `ppo_micro_batch_size_per_gpu` | int | `null` | 单 GPU 微批大小（梯度累积） |
| `use_dynamic_bsz` | bool | `False` | 动态批大小（基于 token 数） |
| `ppo_max_token_len_per_gpu` | int | `16384` | 动态模式下单 GPU 最大 token 数 |
| `ppo_epochs` | int | `1` | PPO 训练 epoch 数 |
| `shuffle` | bool | `False` | 是否打乱批次 |

**批大小关系**：
```
train_batch_size（数据级别）
    └── ppo_mini_batch_size（PPO 更新级别）
        └── ppo_micro_batch_size_per_gpu（单 GPU 级别，用于梯度累积）
```

### 优化器参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `optim.lr` | float | `1e-6` | 学习率 |
| `optim.weight_decay` | float | `0.01` | L2 正则化系数 |
| `optim.lr_warmup_steps` | int | `-1` | 预热步数（-1 表示使用比例） |
| `optim.lr_warmup_steps_ratio` | float | `0.0` | 预热步数比例 |
| `optim.warmup_style` | str | `constant` | 预热方式：`constant`/`cosine` |
| `optim.min_lr_ratio` | float | `null` | 余弦衰减最小学习率比例 |

### PPO 损失参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `clip_ratio` | float | `0.2` | PPO 裁剪比例 |
| `clip_ratio_low` | float | `0.2` | PPO 裁剪下界 |
| `clip_ratio_high` | float | `0.2` | PPO 裁剪上界 |
| `clip_ratio_c` | float | `3.0` | Dual-clip 下界 |
| `grad_clip` | float | `1.0` | 梯度裁剪上限 |
| `entropy_coeff` | float | `0` | 熵奖励系数（鼓励探索） |
| `loss_agg_mode` | str | `token-mean` | 损失聚合模式 |

**loss_agg_mode 选项**：
- `token-mean`：按 token 平均
- `seq-mean-token-sum`：序列平均后 token 求和
- `seq-mean-token-mean`：序列和 token 都平均

### GRPO 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_kl_loss` | bool | `False` | 启用 KL 散度损失（**GRPO 必须为 True**） |
| `kl_loss_coef` | float | `0.001` | KL 损失系数 |
| `kl_loss_type` | str | `low_var_kl` | KL 计算方式 |

### FSDP 分布式配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | str | `fsdp` | 分布式策略：`fsdp`/`fsdp2` |
| `fsdp_config.param_offload` | bool | `False` | 参数卸载到 CPU |
| `fsdp_config.optimizer_offload` | bool | `False` | 优化器状态卸载到 CPU |
| `fsdp_config.fsdp_size` | int | `-1` | FSDP 分片大小（-1 自动） |
| `fsdp_config.reshard_after_forward` | bool | `True` | 前向后重新分片 |
| `use_torch_compile` | bool | `True` | 使用 torch.compile 优化 |
| `ulysses_sequence_parallel_size` | int | `1` | 序列并行大小 |

**offload 影响**：
- `param_offload=True`：显著降低显存，但训练变慢
- `optimizer_offload=True`：进一步降低显存，训练更慢

### 检查点配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint.contents` | list | `['model', 'optimizer', 'extra']` | 保存内容 |

---

## 4. actor_rollout_ref.rollout - Rollout 推理配置

### 引擎选择

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | `vllm` | 推理引擎：`vllm`/`sglang` |
| `mode` | str | `sync` | 生成模式：`sync`/`async` |

### 并行配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tensor_model_parallel_size` | int | `2` | 张量并行大小 |

**重要**：
- 必须能整除 `n_gpus_per_node`
- 值越大，单卡显存越低，但通信开销增加

### 显存配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gpu_memory_utilization` | float | `0.5` | GPU 显存利用率（0.0-1.0） |
| `dtype` | str | `bfloat16` | 数据类型 |

**gpu_memory_utilization 建议**：
- `0.5`：安全配置，防止 OOM
- `0.6`：默认配置，平衡性能和稳定性
- `0.7+`：激进配置，可能导致 OOM

### 采样配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | `1.0` | 采样温度 |
| `top_k` | int | `-1` | Top-K 采样（-1 禁用） |
| `top_p` | float | `1.0` | Top-P 采样（核采样） |
| `do_sample` | bool | `True` | 使用采样（False 为贪心） |
| `n_repeat` | int | `1` | 重复采样次数（**GRPO > 1**） |
| `use_fire_sampling` | bool | `False` | 使用 FIRE 采样 |

**n_repeat 说明**：
- GRPO 算法需要多次采样，通常设为 4-8
- 实际批大小 = `train_batch_size × n_repeat`

### 序列配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt_length` | int | `${data.max_prompt_length}` | Prompt 长度 |
| `response_length` | int | `${data.max_response_length_single_turn}` | 响应长度 |
| `max_model_len` | int | `null` | 模型最大长度 |
| `max_num_seqs` | int | `1024` | 最大并发序列数 |
| `max_num_batched_tokens` | int | `32768` | 最大批处理 token 数 |

### 停止条件

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `stop_token_ids` | list | `[]` | 停止 token ID 列表 |
| `stop` | list | `[]` | 停止字符串列表 |
| `ignore_eos` | bool | `False` | 忽略 EOS token |

**Qwen2.5 示例**：
```bash
actor_rollout_ref.rollout.stop_token_ids=[151658]  # <|im_end|>
```

### 性能优化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enforce_eager` | bool | `True` | 禁用 CUDA Graph（调试用） |
| `free_cache_engine` | bool | `True` | 释放缓存引擎 |
| `enable_chunked_prefill` | bool | `True` | 分块预填充 |
| `disable_log_stats` | bool | `True` | 禁用日志统计 |
| `load_format` | str | `dummy_dtensor` | 权重加载格式 |

### 日志概率计算

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `log_prob_micro_batch_size_per_gpu` | int | `null` | 日志概率计算微批大小 |
| `log_prob_use_dynamic_bsz` | bool | `${actor.use_dynamic_bsz}` | 动态批大小 |
| `log_prob_max_token_len_per_gpu` | int | `${actor.ppo_max_token_len_per_gpu}` | 最大 token 数 |

### 验证配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `val_kwargs.temperature` | float | `0` | 验证时温度（0 = 贪心） |
| `val_kwargs.top_k` | int | `-1` | 验证时 Top-K |
| `val_kwargs.top_p` | float | `1.0` | 验证时 Top-P |
| `val_kwargs.do_sample` | bool | `False` | 验证时不采样 |

### 多轮对话配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `multi_turn.enable` | bool | `False` | 启用多轮对话 |
| `multi_turn.max_turns` | int | `null` | 最大轮数 |
| `multi_turn.tool_config_path` | str | `null` | 工具配置路径 |
| `multi_turn.format` | str | `chatml` | 对话格式 |

---

## 5. actor_rollout_ref.ref - Reference 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | str | `fsdp` | 分布式策略 |
| `fsdp_config.param_offload` | bool | `False` | 参数卸载到 CPU |
| `fsdp_config.reshard_after_forward` | bool | `True` | 前向后重新分片 |
| `log_prob_micro_batch_size_per_gpu` | int | `null` | 日志概率微批大小 |
| `use_torch_compile` | bool | `${actor.use_torch_compile}` | torch.compile 优化 |
| `ulysses_sequence_parallel_size` | int | `${actor.ulysses_sequence_parallel_size}` | 序列并行大小 |

**建议**：
- Reference 模型不需要训练，可以设置 `param_offload=True` 节省显存

---

## 6. algorithm - RL 算法配置

### 算法选择

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `adv_estimator` | str | `gae` | 优势估计器 |

**adv_estimator 选项**：
| 值 | 算法 | 是否需要 Critic |
|----|------|----------------|
| `gae` | PPO (GAE) | 是 |
| `grpo` | GRPO | 否 |
| `reinforce_plus_plus` | REINFORCE++ | 否 |
| `reinforce_plus_plus_baseline` | REINFORCE++ Baseline | 否 |
| `remax` | ReMax | 否 |
| `rloo` | RLOO | 否 |

### GAE 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gamma` | float | `1.0` | 折扣因子 |
| `lam` | float | `1.0` | GAE lambda |

### KL 控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_kl_in_reward` | bool | `False` | 在奖励中使用 KL |
| `kl_penalty` | str | `kl` | KL 惩罚类型 |
| `kl_ctrl.type` | str | `fixed` | KL 控制类型 |
| `kl_ctrl.kl_coef` | float | `0.001` | KL 系数 |
| `kl_ctrl.horizon` | int | `10000` | KL 控制视野 |
| `kl_ctrl.target_kl` | float | `0.1` | 目标 KL 值 |

### GRPO 特有

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `norm_adv_by_std_in_grpo` | bool | `True` | 标准化优势值 |
| `use_process_rewards` | bool | `False` | 使用过程奖励 |

---

## 7. trainer - 训练器配置

### GPU 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_gpus_per_node` | int | `8` | 每节点 GPU 数 |
| `nnodes` | int | `1` | 节点数 |

### 训练控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `total_epochs` | int | `30` | 总训练 epoch 数 |
| `total_training_steps` | int | `null` | 总训练步数（覆盖 epochs） |
| `critic_warmup` | int | `0` | Critic 预热步数 |
| `val_before_train` | bool | `True` | 训练前验证 |
| `balance_batch` | bool | `True` | 平衡批次 |

### 日志与监控

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `logger` | list | `['console', 'wandb']` | 日志器列表 |
| `project_name` | str | `verl_examples` | WandB 项目名 |
| `experiment_name` | str | `gsm8k` | 实验名称 |
| `log_val_generations` | int | `0` | 记录验证生成数量 |

### 保存与恢复

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_freq` | int | `-1` | 保存频率（-1 不保存） |
| `test_freq` | int | `-1` | 测试频率（-1 不测试） |
| `default_local_dir` | str | `checkpoints/${project_name}/${experiment_name}` | 检查点目录 |
| `resume_mode` | str | `auto` | 恢复模式：`auto`/`disable`/`resume_path` |
| `resume_from_path` | str | `null` | 恢复路径 |
| `max_actor_ckpt_to_keep` | int | `null` | 保留 Actor 检查点数 |
| `max_critic_ckpt_to_keep` | int | `null` | 保留 Critic 检查点数 |

### 数据导出

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rollout_data_dir` | str | `null` | Rollout 数据保存目录 |
| `validation_data_dir` | str | `null` | 验证数据保存目录 |

### Ray 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ray_wait_register_center_timeout` | int | `300` | Ray 注册超时（秒） |

---

## 8. tool - 工具系统配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_turns` | int | `10` | 最大工具调用轮数 |
| `use_batch_tool_calls` | bool | `True` | 批量工具调用 |
| `tools` | list | `['search']` | 可用工具列表 |
| `env` | str | `nous` | 工具环境 |
| `max_tool_response_length` | int | `512` | 工具响应最大长度 |

**tools 可选值**：
- `search`：搜索工具
- `python`：Python 代码执行工具
- `wiki_search`：Wikipedia 搜索工具

### 验证时工具配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `val_kwargs.max_turns` | int | `${tool.max_turns}` | 验证时最大轮数 |
| `val_kwargs.use_batch_tool_calls` | bool | `${tool.use_batch_tool_calls}` | 验证时批量调用 |
| `val_kwargs.tools` | list | `${tool.tools}` | 验证时工具 |
| `val_kwargs.env` | str | `${tool.env}` | 验证时环境 |
| `val_kwargs.max_tool_response_length` | int | `${tool.max_tool_response_length}` | 验证时响应长度 |

---

## 9. 参数约束关系

### 批处理大小约束

```
约束 1: real_train_batch_size % n_gpus == 0
        其中 real_train_batch_size = train_batch_size × n_repeat

约束 2: train_batch_size >= ppo_mini_batch_size

约束 3: ppo_mini_batch_size % ppo_micro_batch_size_per_gpu == 0

约束 4: ppo_micro_batch_size_per_gpu × ulysses_sequence_parallel_size >= n_gpus
```

**示例验证**：
```
配置:
  train_batch_size = 128
  n_repeat = 5
  n_gpus = 4
  ppo_mini_batch_size = 64
  ppo_micro_batch_size_per_gpu = 2
  ulysses_sequence_parallel_size = 1

验证:
  real_train_batch_size = 128 × 5 = 640
  约束1: 640 % 4 == 0 ✓
  约束2: 128 >= 64 ✓
  约束3: 64 % 2 == 0 ✓
  约束4: 2 × 1 = 2 < 4 ✗ 需要增加 micro_batch_size 或 sp_size
```

### tensor_model_parallel_size 约束

```
约束: tensor_model_parallel_size 必须能整除 n_gpus_per_node

示例:
  n_gpus_per_node = 4
  有效值: 1, 2, 4
  无效值: 3, 5, 6, ...
```

### 显存相关约束

```
vLLM 显存 = GPU总显存 × gpu_memory_utilization

约束: gpu_memory_utilization + Actor显存占比 < 1.0

实践建议:
  - gpu_memory_utilization = 0.5-0.6 较安全
  - 大模型或长序列时降低此值
```

### GRPO 必须参数

```
GRPO 算法 (algorithm.adv_estimator=grpo) 必须设置:
  - actor_rollout_ref.actor.use_kl_loss = True
  - actor_rollout_ref.rollout.n_repeat > 1 (通常 4-8)
```

---

## 相关文档

- [05_config_system.md](./05_config_system.md) - 配置系统说明
- [07_gpu_config_templates.md](./07_gpu_config_templates.md) - GPU 配置模板
- [04_gpu_selection_guide.md](./04_gpu_selection_guide.md) - GPU 选择与问题排查
