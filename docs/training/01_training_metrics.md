# WandB 训练指标详解

本文档详细介绍 Agent-R1 训练过程中记录到 WandB 的所有指标，帮助你理解每个指标的含义、计算方式和正常范围。

## 目录

- [概述](#概述)
- [Actor 相关指标](#actor-相关指标)
- [Critic 相关指标](#critic-相关指标)
- [Reward 相关指标](#reward-相关指标)
- [数据统计指标](#数据统计指标)
- [性能指标](#性能指标)
- [验证指标](#验证指标)
- [WandB 面板配置建议](#wandb-面板配置建议)

---

## 概述

Agent-R1 使用 WandB（Weights & Biases）记录训练过程中的各种指标。这些指标可以帮助你：

1. **监控训练进度**：了解模型是否在持续改进
2. **诊断问题**：发现训练不稳定或过拟合等问题
3. **调优超参数**：根据指标变化调整学习率、batch size 等
4. **比较实验**：对比不同配置的训练效果

### 指标分类

| 类别 | 前缀 | 说明 |
|------|------|------|
| Actor 指标 | `actor/` | 策略网络（Actor）的训练指标 |
| Critic 指标 | `critic/` | 值函数网络（Critic）的训练指标 |
| 奖励指标 | `critic/score`, `critic/rewards` | 奖励信号相关统计 |
| 数据指标 | `response_length/`, `prompt_length/`, `turns/` | 数据统计 |
| 性能指标 | `perf/`, `timing_s/` | 训练性能和时间统计 |
| 验证指标 | `val-core/`, `val-aux/` | 验证集上的评估指标 |

---

## Actor 相关指标

Actor（策略网络）是 Agent-R1 的核心模型，负责生成响应。以下是 Actor 训练过程中记录的关键指标。

### `actor/pg_loss` - 策略梯度损失

**含义**：策略梯度损失（Policy Gradient Loss），是 Actor 优化的主要目标函数。

**计算公式**：
```
L^{PG} = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```
其中：
- `r_t = π(a|s) / π_old(a|s)` 是新旧策略的概率比
- `A_t` 是优势函数估计
- `ε` 是 clip 范围（默认 0.2）

**正常范围**：通常在 -0.5 到 0.5 之间波动

**异常情况**：
- 持续为正值：可能表示策略正在变差
- 突然跳变：可能是学习率过大或数据分布变化

**代码位置**：`agent_r1/src/core_algos.py` 的 `compute_policy_loss()` 函数

---

### `actor/pg_clipfrac` - 策略裁剪比例

**含义**：被 PPO clip 限制的动作比例，表示策略更新的激进程度。

**计算方式**：
```python
pg_clipfrac = mean(ratio > (1 + clip_ratio) or ratio < (1 - clip_ratio))
```

**正常范围**：0.1 - 0.3

**解读**：
- **< 0.1**：策略更新过于保守，可以尝试增大学习率
- **0.1 - 0.3**：健康范围，表示策略在稳定更新
- **> 0.3**：策略更新过于激进，可能导致不稳定

---

### `actor/ppo_kl` - PPO KL 散度

**含义**：当前策略与旧策略之间的 KL 散度，衡量策略变化幅度。

**计算公式**：
```
KL = E[log(π_old(a|s)) - log(π(a|s))]
```

**正常范围**：0.001 - 0.1

**解读**：
- **< 0.001**：策略几乎没有变化，学习率可能过小
- **0.001 - 0.1**：健康范围
- **> 0.1**：策略变化过大，可能导致训练不稳定

---

### `actor/pg_clipfrac_lower` - Dual-Clip 下界裁剪比例

**含义**：Dual-Clip PPO 中被下界裁剪的比例。用于防止策略在负优势情况下过度更新。

**正常范围**：通常很小（< 0.05）

**相关配置**：`actor_rollout_ref.actor.clip_ratio_c`（默认 3.0）

---

### `actor/grad_norm` - 梯度范数

**含义**：Actor 参数梯度的 L2 范数，衡量梯度大小。

**正常范围**：0.1 - 10.0

**异常情况**：
- **< 0.01**：梯度消失，模型可能停止学习
- **> 100**：梯度爆炸，需要减小学习率或检查数据

**相关配置**：`actor_rollout_ref.actor.grad_clip`（默认 1.0）会裁剪超过阈值的梯度

---

### `actor/lr` - 学习率

**含义**：当前 Actor 的学习率。

**配置位置**：`actor_rollout_ref.actor.optim.lr`

**注意**：如果使用 warmup，学习率会从 0 逐渐增加到目标值。

---

### `actor/entropy_loss` - 熵损失

**含义**：策略熵的负值，用于鼓励探索。

**计算公式**：
```
L^{entropy} = -E[H(π(·|s))] = E[Σ π(a|s) * log(π(a|s))]
```

**正常范围**：取决于动作空间大小

**相关配置**：
- `actor_rollout_ref.actor.entropy_coeff`（默认 0）：熵正则化系数
- 设置 > 0 会鼓励更多探索

---

### `actor/kl_loss` - KL 损失（GRPO）

**含义**：GRPO 算法中的 KL 散度损失项，用于约束策略不要偏离参考策略太远。

**出现条件**：仅当 `actor_rollout_ref.actor.use_kl_loss=True` 时出现

**计算方式**：
```python
kl_loss = kl_loss_coef * KL(π || π_ref)
```

**相关配置**：
- `actor_rollout_ref.actor.kl_loss_coef`（默认 0.001）
- `actor_rollout_ref.actor.kl_loss_type`（默认 `low_var_kl`）

**KL 类型说明**：

| 类型 | 计算方式 | 特点 |
|------|---------|------|
| `kl` | `log(π) - log(π_ref)` | 标准 KL |
| `abs` | `|log(π) - log(π_ref)|` | 绝对值 KL |
| `mse` | `0.5 * (log(π) - log(π_ref))²` | 平方 KL |
| `low_var_kl` | 低方差估计 | 推荐，更稳定 |

---

### `actor/kl_coef` - KL 系数

**含义**：当前的 KL 惩罚系数。

**相关配置**：
- `algorithm.kl_ctrl.type`：`fixed`（固定）或 `adaptive`（自适应）
- `algorithm.kl_ctrl.kl_coef`：初始/固定 KL 系数

---

### `actor/reward_kl_penalty` - 奖励中的 KL 惩罚

**含义**：应用于奖励的 KL 惩罚值。

**出现条件**：仅当 `algorithm.use_kl_in_reward=True` 时出现

**计算方式**：
```python
reward = score - kl_coef * KL
```

---

### `perf/mfu/actor` - Actor 模型浮点利用率

**含义**：Model FLOPS Utilization，衡量 GPU 计算效率。

**正常范围**：0.3 - 0.6（取决于硬件和模型大小）

**解读**：
- **< 0.2**：GPU 利用率低，可能存在数据加载瓶颈
- **> 0.5**：GPU 利用率较高

---

## Critic 相关指标

Critic（值函数网络）用于估计状态价值，仅在使用 GAE（Generalized Advantage Estimation）算法时出现。

> **注意**：如果使用 GRPO 算法（`algorithm.adv_estimator=grpo`），以下 Critic 指标不会出现。

### `critic/vf_loss` - 值函数损失

**含义**：Critic 的 MSE 损失，衡量价值预测的准确性。

**计算公式**：
```
L^{VF} = 0.5 * E[max((V - R)², (V_clip - R)²)]
```
其中：
- `V` 是预测的价值
- `R` 是目标回报
- `V_clip` 是裁剪后的预测价值

**正常范围**：0.01 - 1.0（取决于奖励规模）

**异常情况**：
- 持续增大：Critic 无法准确估计价值
- 不收敛：可能需要调整 Critic 学习率

---

### `critic/vf_clipfrac` - 值函数裁剪比例

**含义**：被裁剪的值函数预测比例。

**相关配置**：`critic.cliprange_value`（默认 0.5）

**正常范围**：< 0.3

---

### `critic/vpred_mean` - 预测价值均值

**含义**：Critic 预测价值的平均值。

**解读**：应该与实际奖励规模相近

---

### `critic/vf_explained_var` - 值函数解释方差

**含义**：值函数对回报的解释程度。

**计算公式**：
```
explained_var = 1 - Var(returns - values) / Var(returns)
```

**正常范围**：0.5 - 1.0

**解读**：
- **< 0**：值函数预测比随机猜测还差
- **0 - 0.5**：值函数有一定预测能力，但不够好
- **0.5 - 0.9**：健康范围
- **> 0.9**：值函数预测非常准确

---

### `critic/grad_norm` - Critic 梯度范数

**含义**：Critic 参数梯度的 L2 范数。

**相关配置**：`critic.grad_clip`（默认 1.0）

---

### `critic/lr` - Critic 学习率

**含义**：当前 Critic 的学习率。

**配置位置**：`critic.optim.lr`（默认 1e-5）

---

## Reward 相关指标

### `critic/score/mean|max|min` - 分数统计

**含义**：每个样本获得的原始分数（奖励）统计。

**来源**：奖励函数计算的结果

**正常范围**：取决于任务，通常在 [-1, 1]

---

### `critic/rewards/mean|max|min` - Token 级奖励统计

**含义**：Token 级别的奖励统计（应用 KL 惩罚后）。

**计算方式**：
```python
token_level_rewards = token_level_scores - kl_coef * KL
```

---

### `critic/advantages/mean|max|min` - 优势统计

**含义**：优势函数估计的统计。

**解读**：
- `mean` 应该接近 0（优势会被标准化）
- `max` 和 `min` 表示优势的范围

---

### `critic/returns/mean|max|min` - 回报统计

**含义**：目标回报的统计。

---

### `critic/values/mean|max|min` - 价值统计

**含义**：Critic 预测价值的统计。

**出现条件**：仅使用 GAE 算法时

---

### `critic/process_rewards/mean|max|min|count` - 过程奖励

**含义**：工具调用的过程奖励统计。

**出现条件**：仅当 `algorithm.use_process_rewards=True` 时

**解读**：
- `count`：使用过程奖励的工具调用次数

---

### `critic/acc/mean|max|min` - 准确率

**含义**：答案准确率统计。

**来源**：奖励函数返回的 `acc` 字段

---

### `critic/format/mean|max|min` - 格式奖励

**含义**：格式奖励统计。

**来源**：奖励函数返回的 `format` 字段

---

## 数据统计指标

### `response_length/mean|max|min|clip_ratio` - 响应长度

**含义**：生成响应的 token 长度统计。

**解读**：
- `clip_ratio`：达到最大响应长度的样本比例
- 如果 `clip_ratio` 很高，考虑增加 `data.max_response_length`

---

### `prompt_length/mean|max|min|clip_ratio` - 提示长度

**含义**：输入提示的 token 长度统计。

---

### `turns/mean|max|min` - 对话轮数

**含义**：工具调用的轮数统计。

**相关配置**：`tool.max_turns`

---

### `action/ratio` - 动作比例

**含义**：模型生成的 token 占总 token 的比例。

**解读**：
- 在多轮工具调用中，部分 token 来自环境（工具响应）
- 只有模型生成的 token 才参与梯度计算

---

### `action/length/mean|max|min` - 动作长度

**含义**：每个样本中模型生成的 token 数量统计。

---

## 性能指标

### `timing_s/*` - 各阶段耗时（秒）

| 指标 | 说明 |
|------|------|
| `timing_s/gen` | 生成响应耗时 |
| `timing_s/reward` | 计算奖励耗时 |
| `timing_s/old_log_prob` | 计算旧策略 log prob 耗时 |
| `timing_s/ref` | 计算参考策略 log prob 耗时 |
| `timing_s/values` | 计算价值函数耗时 |
| `timing_s/adv` | 计算优势函数耗时 |
| `timing_s/update_critic` | 更新 Critic 耗时 |
| `timing_s/update_actor` | 更新 Actor 耗时 |
| `timing_s/testing` | 验证耗时 |
| `timing_s/save_checkpoint` | 保存检查点耗时 |

---

### `timing_per_token_ms/*` - 单位 Token 耗时（毫秒）

与 `timing_s/*` 对应，但按 token 数量归一化。

---

### `perf/total_num_tokens` - 总处理 Token 数

**含义**：当前步骤处理的总 token 数量。

---

### `perf/time_per_step` - 每步耗时

**含义**：完成一个训练步骤的总时间（秒）。

---

### `perf/throughput` - 吞吐量

**含义**：每秒每 GPU 处理的 token 数量。

**计算公式**：
```
throughput = total_tokens / time_per_step / num_gpus
```

---

### `perf/max_memory_allocated_gb` - 最大分配内存

**含义**：GPU 上分配的最大内存（GB）。

**用途**：监控内存使用，防止 OOM

---

### `perf/max_memory_reserved_gb` - 最大预留内存

**含义**：GPU 上预留的最大内存（GB）。

---

### `perf/cpu_memory_used_gb` - CPU 内存使用

**含义**：CPU 内存使用量（GB）。

---

## 验证指标

验证指标在 `trainer.test_freq` 步骤后记录，反映模型在验证集上的表现。

### `val-core/{data_source}/{metric}@{n}` - 核心验证指标

**格式**：`val-core/{数据源}/{指标名}@{采样数}`

**示例**：
- `val-core/hotpotqa/score/mean@1`：HotpotQA 数据集上的平均分数
- `val-core/gsm8k/acc/mean@5`：GSM8K 数据集上 5 次采样的平均准确率

**常见指标**：

| 指标 | 说明 |
|------|------|
| `mean@N` | N 次采样的平均分数 |
| `best@N/mean` | N 次采样中最高分的平均值 |
| `worst@N/mean` | N 次采样中最低分的平均值 |
| `maj@N/mean` | N 次采样的多数投票结果 |

---

### `val-aux/{data_source}/{metric}` - 辅助验证指标

**含义**：其他辅助指标，如格式奖励、响应长度等。

---

### `val/generations` - 验证生成样本

**含义**：WandB Table，包含验证时生成的样本。

**配置**：`trainer.log_val_generations`（默认 0，不记录）

---

## WandB 面板配置建议

为了高效监控训练，建议创建以下 WandB 面板分组：

### 1. 核心训练指标面板

```
- actor/pg_loss
- actor/pg_clipfrac
- actor/ppo_kl
- actor/grad_norm
- critic/score/mean
```

### 2. 性能监控面板

```
- perf/throughput
- perf/mfu/actor
- timing_s/gen
- timing_s/update_actor
- perf/max_memory_allocated_gb
```

### 3. 奖励与准确率面板

```
- critic/score/mean
- critic/acc/mean
- critic/format/mean
- val-core/*/acc/mean@*
```

### 4. 稳定性监控面板

```
- actor/grad_norm
- actor/pg_clipfrac
- critic/vf_loss (如果使用 GAE)
- actor/kl_loss (如果使用 GRPO)
```

### 推荐图表配置

1. **使用 Smoothing**：设置 0.8-0.9 的平滑系数，减少噪声
2. **Y 轴范围**：为关键指标设置固定 Y 轴范围，便于比较
3. **分组显示**：按指标类型分组，避免面板过于拥挤

---

## 下一步

- [02_key_metrics_monitoring.md](./02_key_metrics_monitoring.md) - 了解最需要关注的关键指标
- [03_training_outputs.md](./03_training_outputs.md) - 了解训练输出的文件结构
