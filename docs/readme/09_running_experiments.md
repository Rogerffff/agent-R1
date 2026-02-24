# 实验运行指南

本文档详细介绍如何运行 Agent-R1 训练实验，包括训练脚本参数说明、不同 RL 算法的配置、多 GPU 训练和监控。

---

## 训练入口点

Agent-R1 使用统一的训练入口：

```bash
python3 -m agent_r1.src.main_agent [配置参数]
```

训练脚本位于 `examples/trainer/` 目录：

| 脚本 | 算法 | 任务 |
|------|------|------|
| `run_ppo_hotpotqa.sh` | PPO | HotpotQA |
| `run_ppo_multihopqa.sh` | PPO | 多数据集 MultihopQA |
| `run_ppo_retool.sh` | PPO | ReTool (代码执行) |
| `run_grpo_hotpotqa.sh` | GRPO | HotpotQA |
| `run_grpo_multihopqa.sh` | GRPO | 多数据集 MultihopQA |
| `run_grpo_retool.sh` | GRPO | ReTool (代码执行) |
| `run_rpp_hotpotqa.sh` | REINFORCE++ | HotpotQA |

---

## 配置参数详解

Agent-R1 使用 Hydra 风格的配置系统，支持命令行参数覆盖。配置模板位于 `agent_r1/src/config/agent_trainer.yaml`。

### 数据配置 (data)

```yaml
data:
  train_files: ['data/hotpotqa/train.parquet']  # 训练数据文件
  val_files: ['data/hotpotqa/validation.parquet']  # 验证数据文件
  train_batch_size: 128           # 训练批次大小
  max_prompt_length: 8192         # 最大 prompt 长度
  max_response_length: 8192       # 最大响应总长度
  max_response_length_single_turn: 1024  # 单轮最大响应长度
  shuffle: true                   # 是否打乱数据
  use_default_tool_template: True # 使用默认工具模板
```

**参数说明**：

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `train_batch_size` | 每个训练步骤的样本数 | 128 |
| `max_prompt_length` | 输入 prompt 的最大 token 数 | 8192 |
| `max_response_length` | 多轮交互的总响应长度 | 8192 |
| `max_response_length_single_turn` | 每轮生成的最大长度 | 1024 |

### Actor-Rollout-Ref 配置

#### 模型配置 (actor_rollout_ref.model)

```yaml
actor_rollout_ref:
  model:
    path: 'Qwen/Qwen2.5-1.5B-Instruct'  # 模型路径或 HuggingFace ID
    enable_gradient_checkpointing: True  # 梯度检查点（节省显存）
    use_remove_padding: True             # 移除 padding（提高效率）
```

#### Actor 配置 (actor_rollout_ref.actor)

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp                    # 分布式策略
    ppo_mini_batch_size: 64           # PPO mini-batch 大小
    ppo_micro_batch_size_per_gpu: 2   # 每 GPU micro-batch 大小
    grad_clip: 1.0                    # 梯度裁剪
    clip_ratio: 0.2                   # PPO clip 比例
    ppo_epochs: 1                     # 每次更新的 epoch 数
    use_kl_loss: False                # 是否使用 KL 损失（GRPO 需要）
    kl_loss_coef: 0.001               # KL 损失系数
    kl_loss_type: low_var_kl          # KL 损失类型
    optim:
      lr: 1e-6                        # 学习率
      weight_decay: 0.01              # 权重衰减
    fsdp_config:
      param_offload: False            # 参数卸载到 CPU
      optimizer_offload: False        # 优化器卸载到 CPU
```

#### Rollout 配置 (actor_rollout_ref.rollout)

```yaml
actor_rollout_ref:
  rollout:
    name: vllm                        # 推理引擎：vllm 或 sglang
    temperature: 1.0                  # 采样温度
    top_k: -1                         # Top-K 采样（-1 禁用）
    top_p: 1                          # Top-P 采样
    tensor_model_parallel_size: 2     # 张量并行大小
    gpu_memory_utilization: 0.6       # GPU 显存利用率
    stop_token_ids: [151658]          # 停止 token ID
    n_repeat: 1                       # 每个样本的生成次数（GRPO > 1）
```

#### Ref 配置 (actor_rollout_ref.ref)

```yaml
actor_rollout_ref:
  ref:
    strategy: fsdp
    log_prob_micro_batch_size_per_gpu: 2
    fsdp_config:
      param_offload: True             # Reference 模型参数卸载
```

### Critic 配置

```yaml
critic:
  model:
    path: 'Qwen/Qwen2.5-1.5B-Instruct'  # Critic 模型路径
    enable_gradient_checkpointing: True
  optim:
    lr: 1e-5                          # Critic 学习率（通常比 Actor 高）
  ppo_micro_batch_size_per_gpu: 2
  grad_clip: 1.0
  cliprange_value: 0.5                # 值函数 clip 范围
```

### 算法配置 (algorithm)

```yaml
algorithm:
  gamma: 1.0                          # 折扣因子
  lam: 1.0                            # GAE lambda
  adv_estimator: gae                  # 优势估计器：gae/grpo/reinforce_plus_plus
  kl_ctrl:
    type: fixed                       # KL 控制类型
    kl_coef: 0.001                    # KL 惩罚系数
  use_kl_in_reward: False             # 是否在奖励中包含 KL
  use_process_rewards: False          # 是否使用过程奖励
```

### Trainer 配置

```yaml
trainer:
  project_name: hotpotqa              # 项目名称（用于日志）
  experiment_name: ppo-qwen2.5-1.5b   # 实验名称
  logger: ['console', 'wandb']        # 日志器
  nnodes: 1                           # 节点数
  n_gpus_per_node: 4                  # 每节点 GPU 数
  total_epochs: 1                     # 总训练轮数
  save_freq: -1                       # 保存频率（-1 不保存）
  test_freq: 10                       # 测试频率
  val_before_train: True              # 训练前验证
  critic_warmup: 0                    # Critic 预热步数
  resume_mode: auto                   # 恢复模式：auto/disable
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

### 工具配置 (tool)

```yaml
tool:
  max_turns: 5                        # 最大交互轮数
  tools: ['search']                   # 使用的工具列表
  env: nous                           # 环境类型
  max_tool_response_length: 2048      # 工具响应最大长度
```

---

## 不同 RL 算法配置

### PPO (Proximal Policy Optimization)

PPO 使用 Critic 进行优势估计，适合稳定训练。

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=5 \
    # ... 其他参数
```

**PPO 关键配置**：
- `algorithm.adv_estimator=gae`: 使用 GAE 优势估计
- `algorithm.use_kl_in_reward=True`: 在奖励中加入 KL 惩罚
- `trainer.critic_warmup=5`: Critic 预热 5 步

**完整 PPO 训练脚本**：

```bash
export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME=ppo-qwen2.5-1.5b-instruct

python3 -m agent_r1.src.main_agent \
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.model.path=$BASE_MODEL \
    critic.optim.lr=1e-5 \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=5 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=1 \
    tool.max_turns=5 \
    tool.tools=['search']
```

### GRPO (Group Relative Policy Optimization)

GRPO 不需要 Critic，通过组内相对比较计算优势。

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.n_repeat=5 \
    # ... 其他参数
```

**GRPO 关键配置**：
- `algorithm.adv_estimator=grpo`: 使用 GRPO 算法
- `actor_rollout_ref.rollout.n_repeat=5`: 每个样本生成 5 个响应
- `actor_rollout_ref.actor.use_kl_loss=True`: 使用 KL 损失正则化

**完整 GRPO 训练脚本**：

```bash
export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME=grpo-qwen2.5-1.5b-instruct

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=1 \
    tool.max_turns=5 \
    tool.tools=['search']
```

### REINFORCE++

REINFORCE++ 是改进的策略梯度方法，不需要 Critic。

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    # ... 其他参数
```

**REINFORCE++ 关键配置**：
- `algorithm.adv_estimator=reinforce_plus_plus`: 使用 REINFORCE++ 算法
- 无需 Critic warmup

### 算法对比

| 特性 | PPO | GRPO | REINFORCE++ |
|------|-----|------|-------------|
| 需要 Critic | ✓ | ✗ | ✗ |
| 多样本采样 | 可选 | 必需 (n_repeat > 1) | 可选 |
| 显存需求 | 高 | 中 | 低 |
| 训练稳定性 | 高 | 中 | 中 |
| KL 控制方式 | 奖励惩罚 | 损失项 | 奖励惩罚 |

---

## 任务特定配置

### 多跳 QA (HotpotQA)

```bash
# 使用搜索工具
tool.tools=['search']
tool.env=nous
tool.max_turns=5
tool.max_tool_response_length=2048

# 停止 token（Qwen 模型）
actor_rollout_ref.rollout.stop_token_ids=[151658]
```

### 多数据集训练

```bash
# 多个训练文件
data.train_files=['data/hotpotqa/train.parquet','data/2wiki/train.parquet']
data.val_files=['data/hotpotqa/validation.parquet','data/2wiki/validation.parquet','data/musique/validation.parquet']

# 使用 wiki_search 工具（需要启动搜索服务）
tool.tools=['wiki_search']
```

### ReTool (代码执行)

```bash
# 使用 Python 执行工具
tool.tools=['python']
tool.env=retool
data.use_default_tool_template=False  # ReTool 使用自定义模板

# 更长的上下文
data.max_prompt_length=16384
data.max_response_length=16384
data.max_response_length_single_turn=8192
```

---

## 多 GPU 训练配置

### 单节点多 GPU

```bash
trainer.nnodes=1
trainer.n_gpus_per_node=4  # 使用 4 个 GPU
```

### 张量并行

```bash
# 将模型分布到多个 GPU
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

### 显存优化

```bash
# 1. 梯度检查点
actor_rollout_ref.model.enable_gradient_checkpointing=True
critic.model.enable_gradient_checkpointing=True

# 2. 参数卸载（Ref 模型）
actor_rollout_ref.ref.fsdp_config.param_offload=True

# 3. 减小 micro-batch 大小
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
critic.ppo_micro_batch_size_per_gpu=1

# 4. 动态批处理
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
```

### 多节点训练

```bash
# 节点配置
trainer.nnodes=2
trainer.n_gpus_per_node=8

# Ray 初始化
ray_init.num_cpus=64
```

### 显存估算

| 模型大小 | 最小 GPU 显存 | 推荐配置 |
|----------|---------------|----------|
| 1.5B | 24GB x 2 | 2x RTX 3090 |
| 3B | 24GB x 4 | 4x RTX 3090 |
| 7B | 40GB x 4 | 4x A100-40G |
| 14B | 80GB x 4 | 4x A100-80G |

---

## 训练监控与日志

### 日志配置

```bash
# 启用控制台和 Wandb 日志
trainer.logger=['console','wandb']
trainer.project_name=hotpotqa
trainer.experiment_name=ppo-qwen2.5-1.5b
```

### 关键监控指标

#### 奖励指标

| 指标 | 说明 |
|------|------|
| `reward/mean` | 平均奖励 |
| `reward/std` | 奖励标准差 |
| `reward/max` | 最大奖励 |
| `reward/min` | 最小奖励 |
| `train/correct_ratio` | 正确率 |
| `format_correct_ratio` | 格式正确率 |

#### 训练指标

| 指标 | 说明 |
|------|------|
| `actor/loss` | Actor 策略损失 |
| `actor/entropy` | 策略熵 |
| `actor/kl_divergence` | KL 散度 |
| `critic/loss` | Critic 值函数损失 |
| `critic/values_mean` | 值函数均值 |

#### 生成指标

| 指标 | 说明 |
|------|------|
| `generation/response_length` | 响应长度 |
| `generation/num_turns` | 交互轮数 |
| `generation/tool_calls` | 工具调用次数 |

### 验证与检查点

```bash
# 训练前验证
trainer.val_before_train=True

# 测试频率（每 N 步）
trainer.test_freq=10

# 保存检查点频率
trainer.save_freq=100  # -1 表示不保存

# 检查点目录
trainer.default_local_dir=checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

### 恢复训练

```bash
# 自动恢复（如果有检查点）
trainer.resume_mode=auto

# 从指定路径恢复
trainer.resume_mode=resume_path
trainer.resume_from_path=/path/to/checkpoint
```

---

## 快速开始示例

### 示例 1: 在 HotpotQA 上运行 PPO

```bash
# 1. 准备数据
python examples/data_preprocess/hotpotqa.py --local_dir ./data/hotpotqa

# 2. 构建搜索索引
cd scripts/hotpotqa_search && python process_hotpotqa.py && cd ../..

# 3. 运行训练
cp examples/trainer/run_ppo_hotpotqa.sh ./
bash run_ppo_hotpotqa.sh
```

### 示例 2: 在 HotpotQA 上运行 GRPO

```bash
cp examples/trainer/run_grpo_hotpotqa.sh ./
bash run_grpo_hotpotqa.sh
```

### 示例 3: 在 ReTool 上运行 GRPO

```bash
# 1. 准备数据
python examples/data_preprocess/retool.py --local_dir ./data/retool

# 2. 运行训练
cp examples/trainer/run_grpo_retool.sh ./
bash run_grpo_retool.sh
```

### 示例 4: 自定义参数训练

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files=['./my_data/train.parquet'] \
    data.val_files=['./my_data/val.parquet'] \
    data.train_batch_size=64 \
    actor_rollout_ref.model.path='Qwen/Qwen2.5-3B-Instruct' \
    actor_rollout_ref.rollout.n_repeat=3 \
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=5 \
    trainer.save_freq=50 \
    tool.tools=['search'] \
    tool.max_turns=3
```

---

## 常见问题

### Q1: OOM (显存不足)

```bash
# 减小 batch 大小
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
critic.ppo_micro_batch_size_per_gpu=1

# 启用参数卸载
actor_rollout_ref.ref.fsdp_config.param_offload=True

# 降低 GPU 显存利用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

### Q2: 训练不稳定

```bash
# 降低学习率
actor_rollout_ref.actor.optim.lr=5e-7

# 增加 KL 惩罚
algorithm.kl_ctrl.kl_coef=0.01

# 使用梯度裁剪
actor_rollout_ref.actor.grad_clip=1.0
```

### Q3: 奖励不增长

- 检查数据格式是否正确
- 验证奖励函数是否正确计算
- 增加训练轮数或 batch 大小
- 尝试不同的 RL 算法

### Q4: 工具调用失败

- 确保搜索服务正常运行
- 检查 `tool.tools` 配置是否正确
- 验证 `stop_token_ids` 设置

### Q5: Wandb 连接问题

```bash
# 使用离线模式
export WANDB_MODE=offline

# 或只使用控制台日志
trainer.logger=['console']
```

---

## 性能调优建议

### 1. 吞吐量优化

```bash
# 启用 chunked prefill
actor_rollout_ref.rollout.enable_chunked_prefill=True

# 增加 max_num_batched_tokens
actor_rollout_ref.rollout.max_num_batched_tokens=65536

# 使用 torch compile
actor_rollout_ref.actor.use_torch_compile=True
```

### 2. 稳定性优化

```bash
# 使用低方差 KL
actor_rollout_ref.actor.kl_loss_type=low_var_kl

# 适当的 clip 比例
actor_rollout_ref.actor.clip_ratio=0.2
```

### 3. 显存优化

```bash
# 使用 bfloat16
actor_rollout_ref.rollout.dtype=bfloat16

# 动态批处理
actor_rollout_ref.actor.use_dynamic_bsz=True
```

---

## 下一步

- [10_inference.md](./10_inference.md) - 使用训练好的模型进行推理
- [07_reward_system.md](./07_reward_system.md) - 自定义奖励函数
- [08_data_preparation.md](./08_data_preparation.md) - 准备自定义数据集
