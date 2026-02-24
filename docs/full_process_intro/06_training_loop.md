# 06. 训练循环详解

本文档详细讲解 Agent-R1 的核心训练循环，从 `RayAgentTrainer.fit()` 方法开始，深入每一个训练步骤的具体实现。

---

## 目录

1. [训练循环概述](#1-训练循环概述)
2. [训练循环初始化](#2-训练循环初始化)
3. [数据准备阶段](#3-数据准备阶段)
4. [步骤 1: 生成阶段](#4-步骤-1-生成阶段)
5. [步骤 2: 奖励计算](#5-步骤-2-奖励计算)
6. [步骤 3: 对数概率计算](#6-步骤-3-对数概率计算)
7. [步骤 4: 参考策略计算](#7-步骤-4-参考策略计算)
8. [步骤 5: 值函数计算](#8-步骤-5-值函数计算)
9. [步骤 6: 优势计算](#9-步骤-6-优势计算)
10. [步骤 7: 模型更新](#10-步骤-7-模型更新)
11. [步骤 8: 验证与检查点](#11-步骤-8-验证与检查点)
12. [完整数据流示例](#12-完整数据流示例)

---

## 1. 训练循环概述

### 1.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RayAgentTrainer.fit() 主训练循环                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  for epoch in total_epochs:                                                 │
│      for batch in dataloader:                                               │
│                                                                             │
│          ┌───────────────────────────────────────────────────────────────┐  │
│          │ 1. 生成阶段 (gen)                                              │  │
│          │    ToolGenerationManager.run_llm_loop()                       │  │
│          │    → 多轮工具调用生成 + 构建 action_mask                        │  │
│          └───────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│          ┌───────────────────────────────────────────────────────────────┐  │
│          │ 2. 奖励计算 (reward)                                           │  │
│          │    reward_fn() 计算格式奖励 + 答案奖励                          │  │
│          └───────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│          ┌───────────────────────────────────────────────────────────────┐  │
│          │ 3. 对数概率计算 (old_log_prob)                                  │  │
│          │    actor_rollout_wg.compute_log_prob()                        │  │
│          │    → 计算当前策略下生成序列的对数概率                            │  │
│          └───────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│          ┌───────────────────────────────────────────────────────────────┐  │
│          │ 4. 参考策略对数概率 (ref) - 可选                                 │  │
│          │    ref_policy_wg.compute_ref_log_prob()                       │  │
│          │    → 用于计算 KL 散度惩罚                                       │  │
│          └───────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│          ┌───────────────────────────────────────────────────────────────┐  │
│          │ 5. 值函数估计 (values) - 仅 GAE 算法                            │  │
│          │    critic_wg.compute_values()                                 │  │
│          └───────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│          ┌───────────────────────────────────────────────────────────────┐  │
│          │ 6. 优势计算 (adv) - 在 Driver 上执行                            │  │
│          │    apply_kl_penalty() + compute_advantage()                   │  │
│          │    → GRPO/GAE/REINFORCE++ 等算法                               │  │
│          └───────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│          ┌───────────────────────────────────────────────────────────────┐  │
│          │ 7. 模型更新                                                    │  │
│          │    update_critic() + update_actor()                           │  │
│          │    → PPO 目标函数优化                                           │  │
│          └───────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│          ┌───────────────────────────────────────────────────────────────┐  │
│          │ 8. 验证和保存                                                  │  │
│          │    _validate() + _save_checkpoint()                           │  │
│          └───────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 代码位置

训练循环的核心代码位于：

| 函数/方法 | 文件位置 | 行号 |
|----------|---------|------|
| `fit()` | `agent_r1/src/agent_ray_trainer.py` | 1798-2282 |
| `apply_kl_penalty()` | `agent_r1/src/agent_ray_trainer.py` | 433-504 |
| `compute_advantage()` | `agent_r1/src/agent_ray_trainer.py` | 526-645 |
| `_create_action_mask()` | `agent_r1/src/agent_ray_trainer.py` | 2283-2343 |

### 1.3 关键配置

训练相关的配置项：

```yaml
trainer:
  total_epochs: 1              # 总训练轮数
  project_name: "agent_r1"     # wandb 项目名
  experiment_name: "grpo_run"  # 实验名称
  logger: "wandb"              # 日志后端

  # 验证和保存
  test_freq: 10                # 验证频率（每 N 步）
  save_freq: 50                # 保存检查点频率
  val_before_train: true       # 训练前是否验证

  # Critic 热身
  critic_warmup: 0             # Critic 热身步数（之后才更新 Actor）

  # 序列长度平衡
  balance_batch: false         # 是否平衡各 DP rank 的 token 数

algorithm:
  adv_estimator: "grpo"        # 优势估计算法
  gamma: 1.0                   # 折扣因子
  lam: 1.0                     # GAE lambda
  use_kl_in_reward: false      # 是否使用 KL 惩罚
  kl_penalty: "kl"             # KL 惩罚类型

actor_rollout_ref:
  rollout:
    n: 1                       # 每个 prompt 采样次数
    n_repeat: 4                # batch 重复次数（用于 GRPO 分组）
```

---

## 2. 训练循环初始化

### 2.1 日志记录器初始化

```python
# agent_r1/src/agent_ray_trainer.py:1882-1893
from omegaconf import OmegaConf
from verl.utils.tracking import Tracking

# 初始化日志记录器（支持 wandb、tensorboard）
logger = Tracking(
    project_name=self.config.trainer.project_name,      # "agent_r1"
    experiment_name=self.config.trainer.experiment_name, # "grpo_retool"
    default_backend=self.config.trainer.logger,          # "wandb"
    config=OmegaConf.to_container(self.config, resolve=True),  # 完整配置
)

self.global_steps = 0
```

### 2.2 检查点加载

```python
# agent_r1/src/agent_ray_trainer.py:1900
self._load_checkpoint()  # 如果存在检查点则加载
```

### 2.3 工具生成管理器配置

```python
# agent_r1/src/agent_ray_trainer.py:1931-1944
gen_config = ToolGenerationConfig(
    max_turns=self.config.tool.max_turns,                # 最大对话轮数，如 5
    max_prompt_length=self.config.data.max_prompt_length, # 1024
    max_response_length=self.config.data.max_response_length,  # 8192
    max_response_length_single_turn=self.config.data.max_response_length_single_turn,  # 2048
    use_batch_tool_calls=self.config.tool.use_batch_tool_calls,  # True
)

generation_manager = ToolGenerationManager(
    tokenizer=self.tokenizer,
    processor=self.processor,          # 多模态处理器（可选）
    actor_rollout_wg=self.actor_rollout_wg,  # Actor Worker Group
    config=gen_config,
)
```

---

## 3. 数据准备阶段

### 3.1 从 DataLoader 获取批次

```python
# agent_r1/src/agent_ray_trainer.py:1949-1959
for epoch in range(self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
        # batch_dict 来自 ToolRLDataset.__getitem__()
        # 包含: input_ids, attention_mask, position_ids, raw_prompt_ids, reward_model, data_source

        # 转换为 DataProto 对象
        batch: DataProto = DataProto.from_single_dict(batch_dict)
```

### 3.2 分配唯一 ID

为每个样本分配唯一 ID，用于 GRPO/RLOO 算法的分组计算：

```python
# agent_r1/src/agent_ray_trainer.py:1961-1964
batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
    dtype=object
)
```

**示例**（batch_size=4）：

```python
batch.non_tensor_batch["uid"] = [
    "a1b2c3d4-...",   # 样本 0 的 ID
    "e5f6g7h8-...",   # 样本 1 的 ID
    "i9j0k1l2-...",   # 样本 2 的 ID
    "m3n4o5p6-...",   # 样本 3 的 ID
]
```

### 3.3 批次重复（GRPO 关键步骤）

GRPO 算法需要对同一个 prompt 生成多个响应，然后在组内计算相对优势：

```python
# agent_r1/src/agent_ray_trainer.py:1968-1971
batch = batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n_repeat,  # 通常为 4
    interleave=True,  # 交错排列
)
```

**重复前**（batch_size=4）：

```
样本索引: [0,    1,    2,    3   ]
uid:      [uid0, uid1, uid2, uid3]
```

**重复后**（batch_size=16, n_repeat=4, interleave=True）：

```
样本索引: [0,    0,    0,    0,    1,    1,    1,    1,    2,    2,    2,    2,    3,    3,    3,    3   ]
uid:      [uid0, uid0, uid0, uid0, uid1, uid1, uid1, uid1, uid2, uid2, uid2, uid2, uid3, uid3, uid3, uid3]
```

同一 `uid` 的样本属于同一组，将生成不同的响应，然后计算组内相对优势。

### 3.4 字段分离

将生成所需的字段分离出来：

```python
# agent_r1/src/agent_ray_trainer.py:1974-1987
batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

# 处理多模态输入（如果有）
if "multi_modal_inputs" in batch.non_tensor_batch:
    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])

# 处理原始 prompt 文本
if "raw_prompt" in batch.non_tensor_batch:
    non_tensor_batch_keys_to_pop.append("raw_prompt")

# 分离生成所需字段
gen_batch = batch.pop(
    batch_keys=batch_keys_to_pop,
    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
)
```

**分离后的数据结构**：

```python
# gen_batch（传给 ToolGenerationManager）
gen_batch = DataProto(
    batch={
        "input_ids": tensor([16, 1024]),           # prompt token IDs
        "attention_mask": tensor([16, 1024]),      # prompt 注意力掩码
        "position_ids": tensor([16, 1024]),        # 位置编码
    },
    non_tensor_batch={
        "raw_prompt_ids": array([list, ...]),      # 无填充的 prompt IDs
    }
)

# batch（保留奖励计算所需字段）
batch = DataProto(
    batch={},  # 张量字段已被分离
    non_tensor_batch={
        "uid": array(["uuid1", "uuid1", ...]),     # 用于 GRPO 分组
        "reward_model": array([{...}, ...]),       # 奖励元数据
        "data_source": array(["...", ...]),        # 数据来源
    }
)
```

---

## 4. 步骤 1: 生成阶段

### 4.1 调用生成管理器

```python
# agent_r1/src/agent_ray_trainer.py:1997-2001
with _timer("gen", timing_raw):
    gen_batch_output = generation_manager.run_llm_loop(
        gen_batch=gen_batch,
        env=self.env,  # ReToolEnv 实例
    )
```

`run_llm_loop()` 的详细流程参见 [04_generation_loop.md](./04_generation_loop.md)。

### 4.2 生成输出结构

```python
# gen_batch_output 的结构
gen_batch_output = DataProto(
    batch={
        "input_ids": tensor([16, 9216]),       # prompt + responses
        "attention_mask": tensor([16, 9216]),  # 完整序列掩码
        "position_ids": tensor([16, 9216]),    # 完整位置编码
        "prompts": tensor([16, 1024]),         # 原始 prompt
        "responses": tensor([16, 8192]),       # 生成的响应
        "action_mask": tensor([16, 8192]),     # 动作掩码（关键！）
    },
    non_tensor_batch={
        "turns": array([3, 2, 4, ...]),        # 每个样本的对话轮数
    }
)
```

### 4.3 合并生成输出

```python
# agent_r1/src/agent_ray_trainer.py:2029
batch = batch.union(gen_batch_output)
```

**合并后的 batch 结构**：

```python
batch = DataProto(
    batch={
        "input_ids": tensor([16, 9216]),       # prompt + responses
        "attention_mask": tensor([16, 9216]),  # 完整序列掩码
        "position_ids": tensor([16, 9216]),    # 完整位置编码
        "prompts": tensor([16, 1024]),         # 原始 prompt
        "responses": tensor([16, 8192]),       # 生成的响应
        "action_mask": tensor([16, 8192]),     # 动作掩码
    },
    non_tensor_batch={
        "uid": array(["uuid1", ...]),          # GRPO 分组 ID
        "reward_model": array([{...}, ...]),   # 奖励元数据
        "data_source": array([...]),           # 数据来源
        "turns": array([3, 2, ...]),           # 对话轮数
    }
)
```

### 4.4 计算响应掩码

```python
# agent_r1/src/agent_ray_trainer.py:2032
batch.batch["response_mask"] = compute_response_mask(batch)
```

```python
# compute_response_mask 函数
def compute_response_mask(data: DataProto):
    """从 attention_mask 中提取响应部分"""
    responses = data.batch["responses"]
    response_length = responses.size(1)  # 8192
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]  # [batch, 8192]
```

### 4.5 创建 Action Mask

```python
# agent_r1/src/agent_ray_trainer.py:2051
batch, metrics = self._create_action_mask(batch, metrics)
```

`_create_action_mask` 方法详解：

```python
# agent_r1/src/agent_ray_trainer.py:2283-2343
def _create_action_mask(self, batch: DataProto, metrics: dict):
    """
    创建动作掩码，用于区分模型生成和环境反馈。

    Action Mask 示意图：
    ┌─────────────────────────────────────────────────────────┐
    │ [Prompt] [Model Response] [Tool Output] [Model Response] │
    ├─────────────────────────────────────────────────────────┤
    │    0          1               0               1          │
    └─────────────────────────────────────────────────────────┘

    只对 action_mask=1 的 token 计算策略梯度。
    """
    response_length = batch.batch["responses"].shape[-1]  # 8192
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    # 检查是否已有 action_mask（由生成阶段创建）
    if "action_mask" not in batch.batch.keys():
        # 如果没有，使用全 1 掩码
        action_mask = torch.ones_like(response_mask)
        print("[WARNING] No action mask found in batch, using all ones")
    else:
        action_mask = batch.batch["action_mask"]

    # 记录动作相关指标
    action_ratio = action_mask.sum().item() / (response_mask.sum().item() + 1e-8)
    metrics["action/ratio"] = action_ratio  # 动作 token 占比
    metrics["action/length/max"] = action_mask.sum(dim=-1).max().item()
    metrics["action/length/min"] = action_mask.sum(dim=-1).min().item()
    metrics["action/length/mean"] = action_mask.sum(dim=-1).float().mean().item()

    return batch, metrics
```

**Action Mask 示例**：

```
响应序列: "<think>计算...</think><code>x=1+2</code><interpreter>3</interpreter><answer>\\boxed{3}</answer>"

Token 分布:
[<think>] [计] [算] [...] [</think>] [<code>] [x] [=] [1] [+] [2] [</code>] [<interpreter>] [3] [</interpreter>] [<answer>] [\boxed{3}] [</answer>]

Action Mask:
[  1      1    1    1      1          1        1   1   1   1   1      1            0          0       0             1             1            1    ]
|<-------------------------- 模型生成 Turn 1 ----------------------->| |<- 工具响应 ->| |<---------- 模型生成 Turn 2 ---------->|
```

---

## 5. 步骤 2: 奖励计算

### 5.1 调用奖励函数

```python
# agent_r1/src/agent_ray_trainer.py:2056-2070
with _timer("reward", timing_raw):
    # 使用奖励模型计算分数（如果启用）
    if self.use_rm:
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)

    # 计算规则奖励（同步或异步）
    if self.config.reward_model.launch_reward_fn_async:
        # 异步计算（不阻塞后续步骤）
        future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
    else:
        # 同步计算
        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

### 5.2 compute_reward 函数

```python
def compute_reward(batch: DataProto, reward_fn):
    """计算奖励"""
    # reward_fn 是 AgentRewardManager.compute_reward() 方法
    result = reward_fn(batch)

    # result 结构:
    # {
    #     "reward_tensor": tensor([batch_size, response_length]),  # 稀疏奖励
    #     "acc": [1, 0, 1, ...],          # 答案准确率
    #     "format_acc": [1, 1, 0, ...],   # 格式准确率
    #     "score": [1.0, -1.0, ...],      # 最终分数
    # }

    reward_tensor = result["reward_tensor"]
    reward_extra_infos_dict = {k: v for k, v in result.items() if k != "reward_tensor"}

    return reward_tensor, reward_extra_infos_dict
```

### 5.3 奖励张量结构

奖励是 **稀疏张量**，只在最后一个有效 token 位置有非零值：

```
reward_tensor: shape = [batch_size, response_length]

样本 0 的奖励张量（答案正确，格式正确）:
[0, 0, 0, 0, ..., 0, 0, 0, 1.0, 0, 0, ..., 0]
                          ↑
                      最后一个有效 token 位置

样本 1 的奖励张量（答案错误）:
[0, 0, 0, 0, ..., 0, 0, 0, -1.0, 0, 0, ..., 0]
                           ↑
                       格式正确但答案错误得 -1
```

---

## 6. 步骤 3: 对数概率计算

### 6.1 计算当前策略的对数概率

```python
# agent_r1/src/agent_ray_trainer.py:2075-2095
with _timer("old_log_prob", timing_raw):
    # 调用 Actor Worker 计算对数概率
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

    # old_log_prob 包含：
    # - old_log_probs: tensor([batch_size, response_length])
    # - entropys: tensor([batch_size, response_length])
```

### 6.2 计算熵损失

熵损失用于鼓励探索，防止策略过早收敛：

```python
# agent_r1/src/agent_ray_trainer.py:2078-2092
entropys = old_log_prob.batch["entropys"]
action_masks = batch.batch["action_mask"]
loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode

# 计算加权平均熵（只考虑 action_mask=1 的位置）
entropy_loss = agg_loss(
    loss_mat=entropys,
    loss_mask=action_masks,
    loss_agg_mode=loss_agg_mode,  # "token_mean" 或 "seq_mean"
)

old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
metrics.update(old_log_prob_metrics)

# 移除熵并合并对数概率
old_log_prob.batch.pop("entropys")
batch = batch.union(old_log_prob)
```

### 6.3 对数概率张量结构

```python
# old_log_probs 张量
old_log_probs: tensor([batch_size, response_length])

# 示例（batch_size=2, 截取部分）
[
    [-2.3, -1.5, -0.8, ..., -1.2, 0, 0, ..., 0],  # 样本 0
    [-1.9, -2.1, -1.0, ..., -1.8, 0, 0, ..., 0],  # 样本 1
]
# 注：只有有效 token 位置有非零值，padding 位置为 0
```

---

## 7. 步骤 4: 参考策略计算

### 7.1 计算参考策略的对数概率

参考策略用于计算 KL 散度惩罚，防止策略偏离太远：

```python
# agent_r1/src/agent_ray_trainer.py:2100-2105
if self.use_reference_policy:
    with _timer("ref", timing_raw):
        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
```

### 7.2 KL 散度惩罚

当 `use_kl_in_reward=True` 时，KL 惩罚会被加入奖励中：

```python
# agent_r1/src/agent_ray_trainer.py:2139-2145
if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(
        batch,
        kl_ctrl=self.kl_ctrl_in_reward,
        kl_penalty=self.config.algorithm.kl_penalty,  # "kl", "abs", "mse"
    )
    metrics.update(kl_metrics)
```

### 7.3 apply_kl_penalty 函数详解

```python
# agent_r1/src/agent_ray_trainer.py:433-504
def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty="kl", multi_turn=False):
    """
    应用 KL 散度惩罚。

    公式: r_kl = r - β * KL(π || π_ref)

    其中：
    - r: 原始 token 级别奖励
    - β: KL 惩罚系数（自适应调整）
    - KL: 当前策略和参考策略的 KL 散度
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # 选择掩码
    if "action_mask" in data.batch:
        action_mask = data.batch["action_mask"]
    else:
        attention_mask = data.batch["attention_mask"]
        action_mask = attention_mask[:, -response_length:]

    # 计算 KL 散度
    # kl_penalty 可以是 "kl", "abs", "mse"
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"],   # π(a|s)
        data.batch["ref_log_prob"],    # π_ref(a|s)
        kl_penalty=kl_penalty
    )  # shape: [batch_size, response_length]

    kld = kld * action_mask  # 只计算 action token 的 KL
    beta = kl_ctrl.value     # 当前 KL 系数

    # 计算带 KL 惩罚的奖励
    token_level_rewards = token_level_scores - beta * kld

    # 计算当前批次的平均 KL
    current_kl = masked_mean(kld, mask=action_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    # 自适应更新 KL 系数
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)

    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {
        "actor/reward_kl_penalty": current_kl,    # 当前 KL 散度
        "actor/reward_kl_penalty_coeff": beta,    # 当前 KL 系数
    }

    return data, metrics
```

### 7.4 自适应 KL 控制器

```python
# agent_r1/src/core_algos.py:29-44
class AdaptiveKLController:
    """
    自适应 KL 控制器。

    根据当前 KL 散度自动调整 β 系数：
    - 如果 KL > target_kl，增大 β（加大惩罚）
    - 如果 KL < target_kl，减小 β（减少惩罚）
    """
    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef  # 初始 β
        self.target = target_kl     # 目标 KL
        self.horizon = horizon      # 调整速度

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
```

---

## 8. 步骤 5: 值函数计算

### 8.1 计算 Critic 值函数

只有使用 GAE 算法时才需要 Critic：

```python
# agent_r1/src/agent_ray_trainer.py:2110-2113
if self.use_critic:  # 仅 GAE 算法
    with _timer("values", timing_raw):
        values = self.critic_wg.compute_values(batch)
        batch = batch.union(values)
```

### 8.2 值函数张量结构

```python
# values 张量
values: tensor([batch_size, response_length])

# 每个 token 位置的状态价值估计
# V(s_t) = E[Σ γ^k * r_{t+k}]

# 示例
[
    [0.5, 0.6, 0.7, ..., 0.9, 0, 0, ..., 0],  # 样本 0 的值函数
    [0.3, 0.4, 0.5, ..., 0.8, 0, 0, ..., 0],  # 样本 1 的值函数
]
```

---

## 9. 步骤 6: 优势计算

### 9.1 保存 token 级别分数

```python
# agent_r1/src/agent_ray_trainer.py:2127-2136
# 保存原始分数（不含 KL 惩罚）
batch.batch["token_level_scores"] = reward_tensor

# 保存额外奖励信息（如 acc, format_acc）
if reward_extra_infos_dict:
    batch.non_tensor_batch.update({
        k: np.array(v) for k, v in reward_extra_infos_dict.items()
    })
```

### 9.2 设置 token_level_rewards

```python
# agent_r1/src/agent_ray_trainer.py:2146-2149
if not self.config.algorithm.use_kl_in_reward:
    # 如果不使用 KL 惩罚，rewards 等于 scores
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
```

### 9.3 计算优势函数

```python
# agent_r1/src/agent_ray_trainer.py:2151-2164
norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,  # AdvantageEstimator.GRPO
    gamma=self.config.algorithm.gamma,                   # 1.0
    lam=self.config.algorithm.lam,                       # 1.0
    num_repeat=self.config.actor_rollout_ref.rollout.n,  # 1
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,    # True
    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,  # False
)
```

### 9.4 compute_advantage 函数详解

```python
# agent_r1/src/agent_ray_trainer.py:526-645
def compute_advantage(data, adv_estimator, gamma, lam, num_repeat,
                      multi_turn=False, norm_adv_by_std_in_grpo=True):
    """
    根据指定算法计算优势函数。

    支持的算法：
    ┌─────────────────┬─────────┬─────────┬──────────────────┐
    │ 算法            │ 偏差    │ 方差    │ 是否需要 Critic  │
    ├─────────────────┼─────────┼─────────┼──────────────────┤
    │ GAE             │ 低      │ 可调    │ ✓                │
    │ GRPO            │ 低      │ 中      │ ✗                │
    │ REINFORCE++     │ 高      │ 低      │ ✗                │
    │ RLOO            │ 低      │ 中      │ ✗                │
    │ REMAX           │ 中      │ 低      │ ✗                │
    └─────────────────┴─────────┴─────────┴──────────────────┘
    """
    # 计算 response_mask（如果不存在）
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)

    if adv_estimator == AdvantageEstimator.GRPO:
        # GRPO: 组内相对优势
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            action_mask=data.batch["action_mask"],
            index=data.non_tensor_batch["uid"],  # 用于分组
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )

    elif adv_estimator == AdvantageEstimator.GAE:
        # GAE: 使用 Critic 的 TD(λ) 方法
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            action_mask=data.batch["action_mask"],
            gamma=gamma,
            lam=lam,
        )

    # ... 其他算法 ...

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

    return data
```

### 9.5 GRPO 优势计算详解

```python
# agent_r1/src/core_algos.py:172-198（简化版）
def compute_grpo_outcome_advantage(token_level_rewards, action_mask, index,
                                   norm_adv_by_std_in_grpo=True):
    """
    GRPO 优势计算：在同一 prompt 的多个响应中计算相对优势。

    公式: A_i = (R_i - mean(R_group)) / std(R_group)

    其中 R_group 是同一 prompt 生成的所有响应的奖励。
    """
    # 计算每个样本的总奖励
    # 只对 action_mask=1 的 token 求和
    scores = (token_level_rewards * action_mask).sum(dim=-1)  # [batch_size]

    # 按 uid 分组计算均值和标准差
    id2mean = defaultdict(list)
    for i, uid in enumerate(index):
        id2mean[uid].append(scores[i].item())

    for uid in id2mean:
        values = id2mean[uid]
        mean_val = np.mean(values)
        std_val = np.std(values) + 1e-6
        id2mean[uid] = (mean_val, std_val)

    # 计算归一化优势
    advantages = torch.zeros_like(token_level_rewards)
    for i, uid in enumerate(index):
        mean_val, std_val = id2mean[uid]
        if norm_adv_by_std_in_grpo:
            adv = (scores[i] - mean_val) / std_val
        else:
            adv = scores[i] - mean_val  # Dr.GRPO 变体

        # 将标量优势扩展到所有 action token
        advantages[i] = adv * action_mask[i]

    returns = token_level_rewards  # outcome supervision

    return advantages, returns
```

### 9.6 GRPO 优势计算示例

```
假设 batch_size=8, n_repeat=4（即 2 个原始 prompt，每个重复 4 次）

uid:      [uid0, uid0, uid0, uid0, uid1, uid1, uid1, uid1]
scores:   [1.0,  -1.0, 1.0,  1.0,  -1.0, -1.0, 1.0,  -1.0]

组 uid0: scores = [1.0, -1.0, 1.0, 1.0]
         mean = 0.5, std = 0.866
         advantages = [(1.0-0.5)/0.866, (-1.0-0.5)/0.866, (1.0-0.5)/0.866, (1.0-0.5)/0.866]
                    = [0.577, -1.732, 0.577, 0.577]

组 uid1: scores = [-1.0, -1.0, 1.0, -1.0]
         mean = -0.5, std = 0.866
         advantages = [(-1.0-(-0.5))/0.866, (-1.0-(-0.5))/0.866, (1.0-(-0.5))/0.866, (-1.0-(-0.5))/0.866]
                    = [-0.577, -0.577, 1.732, -0.577]

最终 advantages = [0.577, -1.732, 0.577, 0.577, -0.577, -0.577, 1.732, -0.577]
```

---

## 10. 步骤 7: 模型更新

### 10.1 更新 Critic（仅 GAE）

```python
# agent_r1/src/agent_ray_trainer.py:2171-2177
if self.use_critic:
    with _timer("update_critic", timing_raw):
        critic_output = self.critic_wg.update_critic(batch)

    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    metrics.update(critic_output_metrics)
```

### 10.2 Critic 热身

在热身阶段只更新 Critic，让 Critic 先学习值函数估计：

```python
# agent_r1/src/agent_ray_trainer.py:2180
if self.config.trainer.critic_warmup <= self.global_steps:
    # 热身完成，开始更新 Actor
    ...
```

### 10.3 更新 Actor

```python
# agent_r1/src/agent_ray_trainer.py:2181-2190
# 更新 Actor
with _timer("update_actor", timing_raw):
    batch.meta_info["multi_turn"] = (
        self.config.actor_rollout_ref.rollout.multi_turn.enable
    )
    actor_output = self.actor_rollout_wg.update_actor(batch)

actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
metrics.update(actor_output_metrics)
```

### 10.4 PPO 目标函数

Actor 更新使用 PPO 目标函数：

```python
# PPO Clipped Objective
L_CLIP = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

其中：
- r(θ) = π(a|s) / π_old(a|s) = exp(log_prob - old_log_prob)  # 概率比
- A: 优势函数
- ε: clip 参数（通常 0.2）
```

### 10.5 Actor 更新返回的指标

```python
actor_output_metrics = {
    "actor/loss": 0.023,           # 策略损失
    "actor/clip_ratio": 0.15,      # 被裁剪的样本比例
    "actor/entropy": 1.23,         # 策略熵
    "actor/pg_loss": 0.018,        # 策略梯度损失
    "actor/approx_kl": 0.008,      # 近似 KL 散度
}
```

---

## 11. 步骤 8: 验证与检查点

### 11.1 验证

```python
# agent_r1/src/agent_ray_trainer.py:2221-2233
if (
    self.val_reward_fn is not None
    and self.config.trainer.test_freq > 0
    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
):
    with _timer("testing", timing_raw):
        val_metrics: dict = self._validate()
        if is_last_step:
            last_val_metrics = val_metrics
    metrics.update(val_metrics)
```

### 11.2 保存检查点

```python
# agent_r1/src/agent_ray_trainer.py:2236-2241
if self.config.trainer.save_freq > 0 and (
    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
):
    with _timer("save_checkpoint", timing_raw):
        self._save_checkpoint()
```

### 11.3 记录指标

```python
# agent_r1/src/agent_ray_trainer.py:2246-2272
# 基本指标
metrics.update({
    "training/global_step": self.global_steps,
    "training/epoch": epoch,
})

# 数据指标（奖励统计、响应长度等）
metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))

# 计时指标
metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

# 吞吐量指标
n_gpus = self.resource_pool_manager.get_n_gpus()
metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

# 记录到日志系统
logger.log(data=metrics, step=self.global_steps)
```

### 11.4 典型指标示例

```python
metrics = {
    # 训练进度
    "training/global_step": 100,
    "training/epoch": 0,

    # 奖励统计
    "train/reward/mean": 0.45,
    "train/reward/max": 1.0,
    "train/reward/min": -1.0,
    "train/reward/std": 0.8,

    # 响应长度
    "train/response_length/mean": 256,
    "train/response_length/max": 512,

    # 动作统计
    "action/ratio": 0.75,          # 75% 的 token 是模型生成
    "action/length/mean": 192,

    # Actor 指标
    "actor/loss": 0.023,
    "actor/entropy_loss": 1.5,
    "actor/clip_ratio": 0.12,

    # 计时指标
    "time/gen": 15.2,              # 生成耗时（秒）
    "time/reward": 0.5,
    "time/old_log_prob": 2.1,
    "time/adv": 0.1,
    "time/update_actor": 8.3,
    "time/step": 26.2,             # 总步骤时间

    # 吞吐量
    "perf/tokens_per_second": 5000,
    "perf/samples_per_second": 20,
}
```

---

## 12. 完整数据流示例

### 12.1 数据在训练步骤中的变化

以下是一个 batch 从数据加载到模型更新的完整数据流：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据准备阶段                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DataLoader 输出 (batch_size=4):                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ input_ids:      [4, 1024]     # prompt token IDs                    │    │
│  │ attention_mask: [4, 1024]     # attention mask                      │    │
│  │ position_ids:   [4, 1024]     # position IDs                        │    │
│  │ raw_prompt_ids: [list×4]      # 无填充的 prompt                      │    │
│  │ reward_model:   [{...}×4]     # 奖励元数据                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  batch.repeat(n_repeat=4, interleave=True)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 扩展后 batch_size = 4 × 4 = 16                                       │    │
│  │ uid: ["u0","u0","u0","u0","u1","u1","u1","u1","u2",...]             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  batch.pop() 分离生成字段                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ gen_batch:                                                          │    │
│  │   input_ids:      [16, 1024]                                        │    │
│  │   attention_mask: [16, 1024]                                        │    │
│  │   raw_prompt_ids: [list×16]                                         │    │
│  │                                                                     │    │
│  │ batch:                                                              │    │
│  │   uid:          ["u0","u0",...] × 16                                │    │
│  │   reward_model: [{...}×16]                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              生成阶段                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ToolGenerationManager.run_llm_loop(gen_batch, env)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 输出 gen_batch_output:                                               │    │
│  │   input_ids:      [16, 9216]    # prompt(1024) + response(8192)     │    │
│  │   attention_mask: [16, 9216]                                         │    │
│  │   prompts:        [16, 1024]    # 原始 prompt                        │    │
│  │   responses:      [16, 8192]    # 生成的响应                         │    │
│  │   action_mask:    [16, 8192]    # 1=模型生成, 0=工具响应              │    │
│  │   turns:          [3, 2, 4, ...] # 每个样本的轮数                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  batch = batch.union(gen_batch_output)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 合并后 batch:                                                        │    │
│  │   batch 字段:                                                        │    │
│  │     input_ids, attention_mask, position_ids, prompts,               │    │
│  │     responses, action_mask, response_mask                           │    │
│  │   non_tensor_batch 字段:                                             │    │
│  │     uid, reward_model, data_source, turns                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              奖励计算阶段                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  compute_reward(batch, reward_fn)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ reward_tensor: [16, 8192]                                           │    │
│  │   稀疏张量，只在最后一个有效 token 位置有非零值                        │    │
│  │   例如: [..., 0, 0, 1.0, 0, 0, ...] (正确答案)                       │    │
│  │   例如: [..., 0, 0, -1.0, 0, 0, ...] (格式正确但答案错误)            │    │
│  │                                                                     │    │
│  │ reward_extra_infos_dict:                                            │    │
│  │   acc: [1, 0, 1, 1, 0, ...] × 16                                    │    │
│  │   format_acc: [1, 1, 0, 1, ...] × 16                                │    │
│  │   score: [1.0, -1.0, -1.0, 1.0, ...] × 16                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  batch.batch["token_level_scores"] = reward_tensor                          │
│  batch.batch["token_level_rewards"] = reward_tensor  (无 KL 惩罚时)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              对数概率计算阶段                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  actor_rollout_wg.compute_log_prob(batch)                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ old_log_probs: [16, 8192]                                           │    │
│  │   每个 token 的对数概率                                               │    │
│  │   例如: [-2.3, -1.5, -0.8, ..., -1.2, 0, 0, ...]                    │    │
│  │                                                                     │    │
│  │ entropys: [16, 8192]                                                │    │
│  │   每个 token 的熵                                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  batch.batch["old_log_probs"] = old_log_probs                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              优势计算阶段                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  compute_advantage(batch, adv_estimator="grpo", ...)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ GRPO 计算过程:                                                        │    │
│  │                                                                     │    │
│  │ 1. 计算每个样本的总奖励:                                              │    │
│  │    scores = (token_level_rewards * action_mask).sum(-1)             │    │
│  │    scores: [1.0, -1.0, 1.0, 1.0,   # uid0 组                        │    │
│  │             -1.0, -1.0, 1.0, -1.0, # uid1 组                        │    │
│  │             ...]                                                     │    │
│  │                                                                     │    │
│  │ 2. 按 uid 分组计算均值和标准差:                                       │    │
│  │    uid0: mean=0.5, std=0.866                                        │    │
│  │    uid1: mean=-0.5, std=0.866                                       │    │
│  │                                                                     │    │
│  │ 3. 计算归一化优势:                                                    │    │
│  │    A_i = (score_i - mean_group) / std_group                         │    │
│  │    advantages: [0.577, -1.732, 0.577, 0.577,                        │    │
│  │                -0.577, -0.577, 1.732, -0.577, ...]                   │    │
│  │                                                                     │    │
│  │ 4. 扩展到 token 级别:                                                 │    │
│  │    advantages: [16, 8192]                                           │    │
│  │    每个 action_mask=1 的位置都有相同的优势值                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  batch.batch["advantages"] = advantages                                      │
│  batch.batch["returns"] = returns                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              模型更新阶段                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  actor_rollout_wg.update_actor(batch)                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 输入 batch 包含:                                                      │    │
│  │   input_ids:          [16, 9216]                                    │    │
│  │   attention_mask:     [16, 9216]                                    │    │
│  │   responses:          [16, 8192]                                    │    │
│  │   action_mask:        [16, 8192]                                    │    │
│  │   old_log_probs:      [16, 8192]                                    │    │
│  │   advantages:         [16, 8192]                                    │    │
│  │   returns:            [16, 8192]                                    │    │
│  │                                                                     │    │
│  │ PPO 更新:                                                            │    │
│  │   1. 计算新的对数概率 new_log_probs                                   │    │
│  │   2. 计算概率比 ratio = exp(new_log_probs - old_log_probs)          │    │
│  │   3. 计算 PPO 损失:                                                  │    │
│  │      L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)                 │    │
│  │   4. 反向传播更新参数                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  actor_output_metrics:                                                       │
│    actor/loss: 0.023                                                        │
│    actor/clip_ratio: 0.15                                                   │
│    actor/pg_loss: 0.018                                                     │
│    actor/approx_kl: 0.008                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 关键张量形状汇总

| 阶段 | 字段名 | 形状 | 说明 |
|-----|--------|------|------|
| 数据准备 | `input_ids` | `[B, max_prompt_len]` | 左填充的 prompt |
| 数据准备 | `uid` | `[B]` | GRPO 分组 ID |
| 生成后 | `input_ids` | `[B, prompt_len + response_len]` | 完整序列 |
| 生成后 | `prompts` | `[B, max_prompt_len]` | 原始 prompt |
| 生成后 | `responses` | `[B, response_len]` | 生成的响应 |
| 生成后 | `action_mask` | `[B, response_len]` | 动作掩码 |
| 奖励计算 | `token_level_scores` | `[B, response_len]` | 稀疏奖励张量 |
| 奖励计算 | `token_level_rewards` | `[B, response_len]` | 含 KL 惩罚的奖励 |
| 对数概率 | `old_log_probs` | `[B, response_len]` | 当前策略对数概率 |
| 参考策略 | `ref_log_prob` | `[B, response_len]` | 参考策略对数概率 |
| 值函数 | `values` | `[B, response_len]` | Critic 估计（仅 GAE） |
| 优势计算 | `advantages` | `[B, response_len]` | 优势函数 |
| 优势计算 | `returns` | `[B, response_len]` | 回报 |

其中 `B = batch_size × n_repeat`（例如 4 × 4 = 16）

---

## 总结

Agent-R1 的训练循环是一个完整的 PPO/GRPO 强化学习流程：

1. **数据准备**：从数据集加载 prompt，通过 `repeat()` 扩展批次用于 GRPO 分组
2. **生成阶段**：`ToolGenerationManager` 执行多轮工具调用，构建 `action_mask`
3. **奖励计算**：基于格式和答案正确性计算稀疏奖励
4. **对数概率**：计算当前策略下的 token 级别对数概率
5. **参考策略**：计算参考策略对数概率用于 KL 惩罚（可选）
6. **值函数**：Critic 估计状态价值（仅 GAE）
7. **优势计算**：GRPO 使用组内相对优势，GAE 使用 TD(λ)
8. **模型更新**：PPO clipped objective 优化 Actor 参数

关键设计要点：
- **Action Mask** 确保只对模型生成的 token 计算梯度
- **GRPO 分组** 通过 `uid` 实现同 prompt 多响应的相对比较
- **稀疏奖励** 只在最后一个有效 token 位置给予奖励
- **自适应 KL 系数** 自动调整策略约束强度
