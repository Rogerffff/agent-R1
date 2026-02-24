# 07. 优势估计算法详解

本文档详细讲解 Agent-R1 支持的各种优势估计算法，包括 GAE、GRPO、REINFORCE++、RLOO 和 REMAX，分析它们的数学原理、实现细节和适用场景。

---

## 目录

1. [优势函数概述](#1-优势函数概述)
2. [GAE (Generalized Advantage Estimation)](#2-gae-generalized-advantage-estimation)
3. [GRPO (Group Relative Policy Optimization)](#3-grpo-group-relative-policy-optimization)
4. [REINFORCE++ 算法](#4-reinforce-算法)
5. [RLOO (Leave-One-Out)](#5-rloo-leave-one-out)
6. [REMAX 算法](#6-remax-算法)
7. [KL 散度惩罚](#7-kl-散度惩罚)
8. [PPO 策略损失](#8-ppo-策略损失)
9. [算法对比与选择指南](#9-算法对比与选择指南)
10. [完整计算示例](#10-完整计算示例)

---

## 1. 优势函数概述

### 1.1 什么是优势函数

优势函数 (Advantage Function) 衡量某个动作相对于平均动作的好坏程度：

```
A(s, a) = Q(s, a) - V(s)

其中：
- Q(s, a): 状态-动作价值函数，在状态 s 采取动作 a 后的期望累积奖励
- V(s): 状态价值函数，状态 s 的期望累积奖励
- A(s, a): 优势函数，动作 a 相对于平均动作的额外价值
```

### 1.2 为什么需要优势函数

策略梯度方法的目标是最大化期望回报：

```
∇_θ J(θ) = E[Σ_t ∇_θ log π_θ(a_t|s_t) * G_t]

其中 G_t 是从时刻 t 开始的累积奖励。
```

直接使用 G_t 会导致高方差。使用优势函数可以：
- **降低方差**：减去基线 V(s) 不改变梯度期望，但显著降低方差
- **加速收敛**：更稳定的梯度估计使训练更快收敛
- **提供信号**：正优势表示好于平均，负优势表示差于平均

### 1.3 Agent-R1 支持的算法

| 算法 | 是否需要 Critic | 偏差 | 方差 | 主要特点 |
|-----|----------------|------|------|----------|
| GAE | ✓ | 低 | 可调 | TD(λ) 方法，通用性强 |
| GRPO | ✗ | 低 | 中 | 组内相对比较，适合 LLM |
| REINFORCE++ | ✗ | 高 | 低 | 基本策略梯度+白化 |
| RLOO | ✗ | 低 | 中 | 留一法计算基线 |
| REMAX | ✗ | 中 | 低 | 使用贪婪采样作为基线 |

### 1.4 代码位置

所有优势估计算法的实现位于：

```
agent_r1/src/core_algos.py
```

---

## 2. GAE (Generalized Advantage Estimation)

### 2.1 数学原理

GAE 使用 TD(λ) 方法估计优势函数，通过参数 λ 在偏差和方差之间权衡：

```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)    # TD 误差

A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}

展开形式：
A_t^{GAE} = δ_t + γλ * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
```

**参数说明**：
- `γ` (gamma): 折扣因子，控制未来奖励的重要性（通常 0.99）
- `λ` (lam): GAE 参数，控制偏差-方差权衡（通常 0.95）
  - λ=0: 相当于 TD(0)，高偏差低方差
  - λ=1: 相当于蒙特卡洛，低偏差高方差

### 2.2 代码实现

```python
# agent_r1/src/core_algos.py:118-169
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,  # [batch_size, response_length]
    values: torch.Tensor,               # [batch_size, response_length]
    action_mask: torch.Tensor,          # [batch_size, response_length]
    gamma: torch.Tensor,                # 折扣因子
    lam: torch.Tensor,                  # GAE lambda
):
    """
    计算 GAE 优势和回报。

    关键步骤：
    1. 使用 action_mask 提取有效 token 的奖励和价值
    2. 从后向前计算 TD 误差和 GAE 优势
    3. 将结果映射回原始位置
    4. 白化优势（归一化）
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []

        # 提取有效 token 的奖励和价值
        extracted_rewards, lengths, indices = extract_and_pad_by_mask(
            token_level_rewards, action_mask
        )
        extracted_values, _, _ = extract_and_pad_by_mask(values, action_mask)

        max_length = max(lengths)

        # 从后向前计算 GAE
        for t in reversed(range(max_length)):
            # 下一步的价值（最后一步为 0）
            nextvalues = extracted_values[:, t + 1] if t < max_length - 1 else 0.0

            # TD 误差: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = extracted_rewards[:, t] + gamma * nextvalues - extracted_values[:, t]

            # GAE: A_t = δ_t + γλ * A_{t+1}
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)

        # 反转得到正序优势
        extracted_advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # 将优势映射回原始位置
        advantages = torch.zeros_like(token_level_rewards)
        for i, length in enumerate(lengths):
            advantages[i, indices[i]] = extracted_advantages[i][:length]

        # 计算回报: returns = advantages + values
        returns = advantages + values

        # 白化优势（归一化）
        advantages = verl_F.masked_whiten(advantages, action_mask)

    return advantages, returns
```

### 2.3 Action Mask 的处理

GAE 需要特殊处理 action_mask，因为工具响应的 token 不应该参与计算：

```python
# extract_and_pad_by_mask 函数
def extract_and_pad_by_mask(tensor, mask, padding_value=0.0):
    """
    根据 mask 提取有效元素并填充到相同长度。

    示例：
    tensor = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    mask   = [[  1,   1,   0,   0,   1]]

    extracted = [[0.1, 0.2, 0.5]]  # 只保留 mask=1 的位置
    """
    extracted_tensors = []
    original_indices = []
    lengths = []

    for i in range(tensor.shape[0]):
        # 找到 mask=1 的位置
        indices = torch.where(mask[i] > 0)[0]
        extracted = tensor[i, indices]
        extracted_tensors.append(extracted)
        original_indices.append(indices)
        lengths.append(len(indices))

    # 填充到相同长度
    padded_tensor = pad_sequence(extracted_tensors, batch_first=True, padding_value=padding_value)

    return padded_tensor, lengths, original_indices
```

### 2.4 GAE 计算示例

```
假设 response_length=10, gamma=0.99, lam=0.95

原始数据：
token_level_rewards = [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0]  # 稀疏奖励
values              = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 0]
action_mask         = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0]

提取有效 token（action_mask=1）：
extracted_rewards = [0, 0, 0, 0, 0, 0, 1.0]
extracted_values  = [0.3, 0.4, 0.5, 0.7, 0.8, 0.95, 1.0]

从后向前计算：
t=6: δ_6 = 1.0 + 0.99*0 - 1.0 = 0.0
     A_6 = 0.0 + 0.99*0.95*0 = 0.0

t=5: δ_5 = 0 + 0.99*1.0 - 0.95 = 0.04
     A_5 = 0.04 + 0.99*0.95*0.0 = 0.04

t=4: δ_4 = 0 + 0.99*0.95 - 0.8 = 0.1405
     A_4 = 0.1405 + 0.99*0.95*0.04 = 0.178

... (继续向前计算)

最终 advantages（白化前）：
[..., 0.178, 0.04, 0.0, ...]
```

---

## 3. GRPO (Group Relative Policy Optimization)

### 3.1 数学原理

GRPO 是专为 LLM 设计的无 Critic 方法，通过在同一 prompt 的多个响应中计算相对优势：

```
A_i = (R_i - μ_group) / σ_group

其中：
- R_i: 样本 i 的总奖励
- μ_group: 同组样本的平均奖励
- σ_group: 同组样本的标准差
```

**关键思想**：
- 不需要训练 Critic 网络
- 利用同一 prompt 的多个采样进行比较
- 通过标准差归一化消除奖励尺度影响

### 3.2 代码实现

```python
# agent_r1/src/core_algos.py:172-227
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # [batch_size, response_length]
    action_mask: torch.Tensor,          # [batch_size, response_length]
    index: np.ndarray,                  # 分组 ID (uid)
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,  # 是否使用标准差归一化
):
    """
    计算 GRPO 优势。

    参数：
        norm_adv_by_std_in_grpo:
            - True: 原始 GRPO，使用标准差归一化
            - False: Dr.GRPO 变体，只减去均值
    """
    response_length = token_level_rewards.shape[-1]

    # 计算每个样本的总奖励（只对 action token 求和）
    scores = (token_level_rewards * action_mask).sum(dim=-1)  # [batch_size]

    # 按 uid 分组
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # 收集每组的分数
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        # 计算每组的均值和标准差
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # 单样本组：不减去均值
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        # 计算归一化优势
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                # GRPO: A = (R - μ) / (σ + ε)
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                # Dr.GRPO: A = R - μ（不除以标准差）
                scores[i] = scores[i] - id2mean[index[i]]

        # 将标量优势扩展到 token 级别
        # 所有 action_mask=1 的位置都有相同的优势值
        scores = scores.unsqueeze(-1).tile([1, response_length]) * action_mask

    return scores, scores  # 返回 (advantages, returns)
```

### 3.3 GRPO 的 batch.repeat 配合

GRPO 需要 `batch.repeat()` 配合使用：

```python
# 在训练循环中
batch = batch.repeat(
    repeat_times=4,      # 每个 prompt 重复 4 次
    interleave=True,     # 交错排列
)

# 重复后的 uid 分布
uid = ["u0", "u0", "u0", "u0", "u1", "u1", "u1", "u1", ...]

# 每组生成 4 个不同的响应，然后计算组内相对优势
```

### 3.4 GRPO 计算示例

```
batch_size = 8 (2 个原始 prompt × 4 次重复)

uid:    ["u0", "u0", "u0", "u0", "u1", "u1", "u1", "u1"]
scores: [1.0, -1.0, 1.0, 0.5, -1.0, 0.5, 1.0, -0.5]

组 u0:
  scores = [1.0, -1.0, 1.0, 0.5]
  mean = 0.375
  std = 0.854

  归一化优势:
  A_0 = (1.0 - 0.375) / 0.854 = 0.732
  A_1 = (-1.0 - 0.375) / 0.854 = -1.610
  A_2 = (1.0 - 0.375) / 0.854 = 0.732
  A_3 = (0.5 - 0.375) / 0.854 = 0.146

组 u1:
  scores = [-1.0, 0.5, 1.0, -0.5]
  mean = 0.0
  std = 0.791

  归一化优势:
  A_4 = (-1.0 - 0.0) / 0.791 = -1.264
  A_5 = (0.5 - 0.0) / 0.791 = 0.632
  A_6 = (1.0 - 0.0) / 0.791 = 1.264
  A_7 = (-0.5 - 0.0) / 0.791 = -0.632

最终 advantages（token 级别）:
对于样本 0，假设 action_mask = [1, 1, 1, 0, 0, 1, 1, 0, ...]
advantages = [0.732, 0.732, 0.732, 0, 0, 0.732, 0.732, 0, ...]
```

### 3.5 Dr.GRPO 变体

Dr.GRPO 是 GRPO 的简化版本，不使用标准差归一化：

```python
# 设置 norm_adv_by_std_in_grpo=False
# 优势计算变为: A = R - μ（不除以 σ）

# 论文: https://arxiv.org/abs/2503.20783
```

**Dr.GRPO 的优点**：
- 计算更简单
- 避免了标准差为 0 时的数值不稳定
- 在某些任务上表现更好

---

## 4. REINFORCE++ 算法

### 4.1 数学原理

REINFORCE++ 是基本策略梯度方法的改进版本，使用折扣累积奖励并进行白化：

```
G_t = Σ_{k=0}^{T-t} γ^k * r_{t+k}    # 折扣累积奖励

A_t = whiten(G_t)                     # 白化（归一化）
```

### 4.2 代码实现

```python
# agent_r1/src/core_algos.py:321-356
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: torch.Tensor,
):
    """
    计算 REINFORCE++ 优势。

    步骤：
    1. 提取有效 token 的奖励
    2. 从后向前计算折扣累积奖励
    3. 白化归一化
    """
    with torch.no_grad():
        running_return = 0

        # 提取有效 token 的奖励
        extracted_rewards, lengths, indices = extract_and_pad_by_mask(
            token_level_rewards, action_mask
        )
        max_length = max(lengths)
        extracted_returns = torch.zeros_like(extracted_rewards)

        # 从后向前计算折扣累积奖励
        for t in reversed(range(max_length)):
            running_return = extracted_rewards[:, t] + gamma * running_return
            extracted_returns[:, t] = running_return

        # 映射回原始位置
        returns = torch.zeros_like(token_level_rewards)
        for i, length in enumerate(lengths):
            returns[i, indices[i]] = extracted_returns[i][:length]

        # 白化优势
        advantages = verl_F.masked_whiten(returns, action_mask)
        advantages = advantages * action_mask

    return advantages, returns
```

### 4.3 REINFORCE++ with Baseline

带基线的 REINFORCE++ 使用组内均值作为基线：

```python
# agent_r1/src/core_algos.py:230-270
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards, action_mask, index, epsilon=1e-6
):
    """
    REINFORCE++ with Baseline。

    优势计算: A_i = R_i - μ_group
    然后进行白化。
    """
    response_length = token_level_rewards.shape[-1]
    scores = (token_level_rewards * action_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # 按组收集分数
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        # 计算组均值
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            else:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))

        # 减去基线
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        # 扩展到 token 级别并白化
        scores = scores.unsqueeze(-1).tile([1, response_length]) * action_mask
        scores = verl_F.masked_whiten(scores, action_mask)

    return scores, scores
```

---

## 5. RLOO (Leave-One-Out)

### 5.1 数学原理

RLOO 使用留一法计算基线：每个样本的基线是组内其他样本的平均奖励。

```
baseline_i = (Σ_{j≠i} R_j) / (n - 1)

A_i = R_i - baseline_i
    = R_i - (Σ_{j≠i} R_j) / (n - 1)
    = R_i * n / (n-1) - μ_group * n / (n-1)
```

**关键优势**：
- 消除了偏差：每个样本的基线不包含自身
- 更准确的优势估计

### 5.2 代码实现

```python
# agent_r1/src/core_algos.py:273-318
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    action_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6
):
    """
    计算 RLOO 优势。

    使用留一法基线：baseline_i = (Σ_{j≠i} R_j) / (n - 1)
    """
    response_length = token_level_rewards.shape[-1]
    scores = (token_level_rewards * action_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # 按组收集分数
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        # 计算组均值
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            else:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))

        # 计算 RLOO 优势
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                # A_i = R_i * n/(n-1) - μ * n/(n-1)
                # 这等价于 A_i = R_i - (Σ_{j≠i} R_j) / (n-1)
                scores[i] = (
                    scores[i] * response_num / (response_num - 1)
                    - id2mean[index[i]] * response_num / (response_num - 1)
                )

        # 扩展到 token 级别
        scores = scores.unsqueeze(-1).tile([1, response_length]) * action_mask

    return scores, scores
```

### 5.3 RLOO 与 GRPO 的对比

```
假设组内有 4 个样本: R = [1.0, -1.0, 0.5, 0.5]
mean = 0.25

GRPO 优势（减去均值）:
A_0 = 1.0 - 0.25 = 0.75
A_1 = -1.0 - 0.25 = -1.25
A_2 = 0.5 - 0.25 = 0.25
A_3 = 0.5 - 0.25 = 0.25

RLOO 优势（留一法）:
baseline_0 = (-1.0 + 0.5 + 0.5) / 3 = 0.0
baseline_1 = (1.0 + 0.5 + 0.5) / 3 = 0.667
baseline_2 = (1.0 + (-1.0) + 0.5) / 3 = 0.167
baseline_3 = (1.0 + (-1.0) + 0.5) / 3 = 0.167

A_0 = 1.0 - 0.0 = 1.0
A_1 = -1.0 - 0.667 = -1.667
A_2 = 0.5 - 0.167 = 0.333
A_3 = 0.5 - 0.167 = 0.333

RLOO 的优势估计更能体现每个样本的相对好坏。
```

---

## 6. REMAX 算法

### 6.1 数学原理

REMAX 使用贪婪采样（argmax）的结果作为基线：

```
A_t = G_t - R_baseline

其中：
- G_t: 从时刻 t 开始的累积奖励
- R_baseline: 使用贪婪采样生成的响应的奖励
```

### 6.2 代码实现

```python
# agent_r1/src/core_algos.py:359-391
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,    # [batch_size] 贪婪采样的奖励
    action_mask: torch.Tensor,
):
    """
    计算 REMAX 优势。

    使用贪婪采样的奖励作为基线。
    """
    response_length = token_level_rewards.shape[-1]
    scores = (token_level_rewards * action_mask).sum(dim=-1)

    with torch.no_grad():
        # 累积奖励（只对 action token）
        masked_rewards = token_level_rewards * action_mask
        returns = masked_rewards.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # 减去贪婪采样的基线
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * action_mask

    return advantages, returns
```

### 6.3 REMAX 的训练流程

REMAX 需要额外的贪婪采样步骤：

```python
# agent_r1/src/agent_ray_trainer.py:2006-2026
if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
    with _timer("gen_max", timing_raw):
        # 复制生成批次
        gen_baseline_batch = deepcopy(gen_batch)

        # 设置贪婪采样（不随机）
        gen_baseline_batch.meta_info["do_sample"] = False

        # 生成贪婪采样响应
        gen_baseline_output = self.actor_rollout_wg.generate_sequences(
            gen_baseline_batch
        )

        # 计算贪婪采样的奖励
        batch = batch.union(gen_baseline_output)
        reward_baseline_tensor = self.reward_fn(batch)
        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)  # [batch_size]

        # 清理并保存基线奖励
        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
        batch.batch["reward_baselines"] = reward_baseline_tensor
```

### 6.4 REMAX 的优缺点

**优点**：
- 提供了一个强基线（最佳响应的估计）
- 不需要训练 Critic

**缺点**：
- 需要额外的生成步骤（计算成本翻倍）
- 贪婪采样可能不代表最优响应

---

## 7. KL 散度惩罚

### 7.1 KL 惩罚的作用

KL 散度惩罚用于约束策略更新幅度，防止模型偏离参考策略太远：

```
r_kl = r - β * KL(π || π_ref)

其中：
- r: 原始奖励
- β: KL 惩罚系数
- KL: 当前策略和参考策略的 KL 散度
```

### 7.2 KL 惩罚类型

```python
# agent_r1/src/core_algos.py:557-589
def kl_penalty(logprob, ref_logprob, kl_penalty):
    """计算 KL 散度惩罚。"""

    if kl_penalty == "kl":
        # 标准 KL: KL(π || π_ref) ≈ log π - log π_ref
        return logprob - ref_logprob

    if kl_penalty == "abs":
        # 绝对值: |log π - log π_ref|
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        # 均方误差: 0.5 * (log π - log π_ref)^2
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty == "low_var_kl":
        # 低方差 KL 估计（Schulman 方法）
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
```

### 7.3 自适应 KL 控制器

```python
# agent_r1/src/core_algos.py:29-44
class AdaptiveKLController:
    """
    自适应 KL 控制器。

    根据当前 KL 动态调整系数 β：
    - KL > target: 增大 β（加强约束）
    - KL < target: 减小 β（放松约束）
    """
    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef  # 初始 β
        self.target = target_kl    # 目标 KL
        self.horizon = horizon     # 调整速度

    def update(self, current_kl, n_steps):
        target = self.target
        # 比例误差，限制在 [-0.2, 0.2]
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        # 指数平滑更新
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
```

### 7.4 配置示例

```yaml
algorithm:
  use_kl_in_reward: true    # 是否在奖励中加入 KL 惩罚
  kl_penalty: "kl"          # KL 惩罚类型

  kl_ctrl:
    type: "adaptive"        # 固定或自适应
    kl_coef: 0.02           # 初始系数
    target_kl: 0.01         # 目标 KL
    horizon: 10000          # 调整速度
```

---

## 8. PPO 策略损失

### 8.1 PPO Clipped Objective

PPO 使用裁剪目标函数防止过大的策略更新：

```
L^{CLIP}(θ) = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

其中：
- r(θ) = π_θ(a|s) / π_old(a|s) = exp(log π_θ - log π_old)
- A: 优势函数
- ε: 裁剪参数（通常 0.2）
```

### 8.2 代码实现

```python
# agent_r1/src/core_algos.py:437-505
def compute_policy_loss(
    old_log_prob,      # π_old 的对数概率
    log_prob,          # π_θ 的对数概率
    advantages,        # 优势函数
    action_mask,       # 动作掩码
    cliprange=None,    # 裁剪范围
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,  # 双裁剪的下界
    loss_agg_mode="token-mean",
):
    """
    计算 PPO 策略损失。

    支持：
    - 标准 PPO 裁剪
    - 非对称裁剪（cliprange_low != cliprange_high）
    - 双裁剪 PPO（处理负优势）
    """
    # 计算概率比
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)

    # 估计 KL 散度
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, action_mask)

    # 未裁剪损失: -ratio * A
    pg_losses1 = -advantages * ratio

    # 裁剪损失: -clip(ratio, 1-ε, 1+ε) * A
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )

    # PPO 目标: max(-ratio*A, -clip(ratio)*A)
    #         = -min(ratio*A, clip(ratio)*A)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    # 裁剪比例统计
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), action_mask
    )

    # 双裁剪 PPO（针对负优势）
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(),
        action_mask
    )

    # 最终损失
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=action_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
```

### 8.3 损失聚合模式

```python
# agent_r1/src/core_algos.py:399-434
def agg_loss(loss_mat, loss_mask, loss_agg_mode):
    """
    聚合损失矩阵为标量。

    支持的模式：
    - "token-mean": 所有有效 token 的平均损失
    - "seq-mean-token-sum": 先序列求和，再批次平均
    - "seq-mean-token-mean": 先序列平均，再批次平均
    - "seq-mean-token-sum-norm": Dr.GRPO 风格的归一化
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)

    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.mean(seq_losses)

    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(seq_losses)

    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]

    return loss
```

---

## 9. 算法对比与选择指南

### 9.1 算法特性对比

| 特性 | GAE | GRPO | REINFORCE++ | RLOO | REMAX |
|-----|-----|------|-------------|------|-------|
| 需要 Critic | ✓ | ✗ | ✗ | ✗ | ✗ |
| 需要参考策略 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 需要额外采样 | ✗ | ✗ | ✗ | ✗ | ✓ |
| 计算复杂度 | 高 | 低 | 低 | 低 | 中 |
| 内存占用 | 高 | 低 | 低 | 低 | 中 |
| 偏差 | 低 | 低 | 高 | 低 | 中 |
| 方差 | 可调 | 中 | 低 | 中 | 低 |

### 9.2 选择建议

**推荐 GRPO 的场景**：
- LLM 对齐和 RLHF
- Agent 工具使用训练
- 资源有限，不想训练 Critic
- 需要快速迭代

**推荐 GAE 的场景**：
- 传统 RL 任务
- 需要精确的优势估计
- 有足够计算资源

**推荐 RLOO 的场景**：
- 对偏差敏感的任务
- 组内样本数较多

**推荐 REMAX 的场景**：
- 有明确的最优响应参考
- 愿意付出额外采样成本

### 9.3 Agent-R1 默认配置

```yaml
# ReTool 默认使用 GRPO
algorithm:
  adv_estimator: "grpo"
  gamma: 1.0                    # 不使用折扣（outcome supervision）
  lam: 1.0                      # GAE 不适用
  norm_adv_by_std_in_grpo: true # 使用标准差归一化

actor_rollout_ref:
  rollout:
    n: 1
    n_repeat: 4  # 每个 prompt 4 个采样，用于 GRPO 分组
```

---

## 10. 完整计算示例

### 10.1 示例设置

```
batch_size = 8 (2 个 prompt × 4 次重复)
response_length = 10
max_turns = 2

uid:    ["u0", "u0", "u0", "u0", "u1", "u1", "u1", "u1"]

响应示例（样本 0）:
"<think>分析...</think><code>print(2+2)</code><interpreter>4</interpreter><answer>\\boxed{4}</answer>"

token_level_rewards (样本 0, 稀疏):
[0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0]  # 最后一个有效位置为 1.0

action_mask (样本 0):
[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
 ↑ ↑ ↑ ↑ ↑ ↑       ↑
 模型生成        工具响应  模型生成
```

### 10.2 各算法的优势计算

**GRPO 计算**：

```python
# 计算每个样本的总奖励
scores = [1.0, -1.0, 1.0, 0.5, -0.5, 0.5, 1.0, -1.0]

# 分组统计
group_u0 = [1.0, -1.0, 1.0, 0.5]  → mean=0.375, std=0.854
group_u1 = [-0.5, 0.5, 1.0, -1.0] → mean=0.0, std=0.707

# 归一化优势
advantages = [
    (1.0 - 0.375) / 0.854,   # 0.732
    (-1.0 - 0.375) / 0.854,  # -1.610
    (1.0 - 0.375) / 0.854,   # 0.732
    (0.5 - 0.375) / 0.854,   # 0.146
    (-0.5 - 0.0) / 0.707,    # -0.707
    (0.5 - 0.0) / 0.707,     # 0.707
    (1.0 - 0.0) / 0.707,     # 1.414
    (-1.0 - 0.0) / 0.707,    # -1.414
]

# 扩展到 token 级别（乘以 action_mask）
# 样本 0 的优势张量:
[0.732, 0.732, 0.732, 0.732, 0.732, 0.732, 0, 0, 0.732, 0]
```

**RLOO 计算**：

```python
# 组 u0 的留一法基线
baseline_0 = (-1.0 + 1.0 + 0.5) / 3 = 0.167
baseline_1 = (1.0 + 1.0 + 0.5) / 3 = 0.833
baseline_2 = (1.0 + (-1.0) + 0.5) / 3 = 0.167
baseline_3 = (1.0 + (-1.0) + 1.0) / 3 = 0.333

# RLOO 优势
A_0 = 1.0 - 0.167 = 0.833
A_1 = -1.0 - 0.833 = -1.833
A_2 = 1.0 - 0.167 = 0.833
A_3 = 0.5 - 0.333 = 0.167
```

**REINFORCE++ 计算**：

```python
# 从后向前计算累积奖励（gamma=1.0）
# 样本 0, 提取 action_mask=1 的位置:
extracted_rewards = [0, 0, 0, 0, 0, 0, 1.0]  # 只有最后一个有奖励

# 累积奖励
G_6 = 1.0
G_5 = 0 + 1.0*1.0 = 1.0
G_4 = 0 + 1.0*1.0 = 1.0
...
G_0 = 1.0

returns = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# 白化后的优势
advantages = whiten(returns)
```

### 10.3 PPO 更新

```python
# 假设以下数据（样本 0）
old_log_prob = [-2.1, -1.8, -2.3, ...]  # 生成时的对数概率
new_log_prob = [-2.0, -1.9, -2.2, ...]  # 更新后的对数概率
advantages = [0.732, 0.732, 0.732, ...]
action_mask = [1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
cliprange = 0.2

# 概率比
ratio = exp(new_log_prob - old_log_prob)
     = [1.105, 0.905, 1.105, ...]

# 未裁剪损失
pg_loss1 = -ratio * A
         = [-0.809, -0.662, -0.809, ...]

# 裁剪损失
clipped_ratio = clip(ratio, 0.8, 1.2)
              = [1.105, 0.905, 1.105, ...]  # 未超出范围

pg_loss2 = -clipped_ratio * A
         = [-0.809, -0.662, -0.809, ...]

# 最终损失（取较大者）
pg_loss = max(pg_loss1, pg_loss2) = [-0.809, -0.662, ...]

# 聚合
total_loss = mean(pg_loss * action_mask) / sum(action_mask)
```

---

## 总结

Agent-R1 提供了丰富的优势估计算法选择：

1. **GRPO**（推荐）：专为 LLM 设计，无需 Critic，计算简单高效
2. **GAE**：经典方法，需要 Critic，偏差-方差可调
3. **REINFORCE++**：基本策略梯度改进，简单直接
4. **RLOO**：留一法基线，更准确的优势估计
5. **REMAX**：使用贪婪采样作为基线

关键设计要点：
- **Action Mask** 确保只对模型生成的 token 计算优势
- **分组机制** 通过 `uid` 实现 GRPO/RLOO 的组内比较
- **白化** 归一化优势，稳定训练
- **PPO 裁剪** 限制策略更新幅度
- **自适应 KL** 动态调整约束强度

根据任务需求选择合适的算法，可以显著影响训练效率和最终性能。
