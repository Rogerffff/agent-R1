# 09. 数据形状参考手册

本文档是 Agent-R1 训练流程中所有关键数据结构和张量形状的完整参考手册，帮助开发者理解数据在各个阶段的变化。

---

## 目录

1. [符号约定](#1-符号约定)
2. [数据准备阶段](#2-数据准备阶段)
3. [生成阶段](#3-生成阶段)
4. [奖励计算阶段](#4-奖励计算阶段)
5. [对数概率计算阶段](#5-对数概率计算阶段)
6. [优势计算阶段](#6-优势计算阶段)
7. [模型更新阶段](#7-模型更新阶段)
8. [DataProto 完整结构](#8-dataproto-完整结构)
9. [常用配置值](#9-常用配置值)
10. [调试技巧](#10-调试技巧)

---

## 1. 符号约定

### 1.1 常用符号

| 符号 | 含义 | 典型值 |
|-----|------|--------|
| `B` | 原始 batch_size | 4 |
| `B'` | 重复后的 batch_size (B × n_repeat) | 16 |
| `P` | max_prompt_length | 1024 |
| `R` | max_response_length | 8192 |
| `R_s` | max_response_length_single_turn | 2048 |
| `L` | 完整序列长度 (P + R) | 9216 |
| `V` | vocab_size | 32000 |
| `n_repeat` | 每个 prompt 的重复次数 | 4 |
| `T` | max_turns | 5 |

### 1.2 数据类型

| 类型 | 说明 |
|-----|------|
| `tensor` | PyTorch 张量 |
| `array` | NumPy 数组 |
| `list` | Python 列表 |
| `dict` | Python 字典 |
| `int/float` | 标量 |

---

## 2. 数据准备阶段

### 2.1 DataLoader 输出

```python
# 从 ToolRLDataset.__getitem__() 返回
batch_dict = {
    # 张量字段（torch.Tensor）
    "input_ids":      [B, P],      # prompt token IDs，左填充
    "attention_mask": [B, P],      # 注意力掩码，1=有效，0=填充
    "position_ids":   [B, P],      # 位置编码

    # 非张量字段（Python 对象）
    "raw_prompt_ids": [list] × B,  # 无填充的原始 prompt IDs
    "reward_model":   [dict] × B,  # 奖励元数据
    "data_source":    [str] × B,   # 数据来源
}
```

### 2.2 转换为 DataProto

```python
batch = DataProto.from_single_dict(batch_dict)

# 结构
batch = DataProto(
    batch={
        "input_ids":      tensor([B, P]),
        "attention_mask": tensor([B, P]),
        "position_ids":   tensor([B, P]),
    },
    non_tensor_batch={
        "raw_prompt_ids": array([list, ...], dtype=object),
        "reward_model":   array([dict, ...], dtype=object),
        "data_source":    array([str, ...], dtype=object),
    },
    meta_info={}
)
```

### 2.3 添加 UID

```python
batch.non_tensor_batch["uid"] = array([str] × B)

# 示例
uid = ["a1b2c3d4-...", "e5f6g7h8-...", ...]
```

### 2.4 batch.repeat() 后

```python
batch = batch.repeat(repeat_times=n_repeat, interleave=True)

# 形状变化: B → B' = B × n_repeat
# 假设 B=4, n_repeat=4 → B'=16

batch = DataProto(
    batch={
        "input_ids":      tensor([B', P]),   # [16, 1024]
        "attention_mask": tensor([B', P]),   # [16, 1024]
        "position_ids":   tensor([B', P]),   # [16, 1024]
    },
    non_tensor_batch={
        "uid":            array([str] × B'), # 每个原始样本的 uid 重复 4 次
        "raw_prompt_ids": array([list] × B'),
        "reward_model":   array([dict] × B'),
        "data_source":    array([str] × B'),
    }
)

# uid 分布示例（interleave=True）
uid = ["u0", "u0", "u0", "u0", "u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2", "u3", "u3", "u3", "u3"]
```

### 2.5 batch.pop() 分离后

```python
gen_batch = batch.pop(
    batch_keys=["input_ids", "attention_mask", "position_ids"],
    non_tensor_batch_keys=["raw_prompt_ids"]
)

# gen_batch（传给生成管理器）
gen_batch = DataProto(
    batch={
        "input_ids":      tensor([B', P]),
        "attention_mask": tensor([B', P]),
        "position_ids":   tensor([B', P]),
    },
    non_tensor_batch={
        "raw_prompt_ids": array([list] × B'),
    }
)

# batch（保留奖励计算字段）
batch = DataProto(
    batch={},
    non_tensor_batch={
        "uid":          array([str] × B'),
        "reward_model": array([dict] × B'),
        "data_source":  array([str] × B'),
    }
)
```

---

## 3. 生成阶段

### 3.1 ToolGenerationManager 输入

```python
gen_batch = DataProto(
    batch={
        "input_ids":      tensor([B', P]),    # [16, 1024]
        "attention_mask": tensor([B', P]),    # [16, 1024]
        "position_ids":   tensor([B', P]),    # [16, 1024]
    },
    non_tensor_batch={
        "raw_prompt_ids": array([list] × B'),
    },
    meta_info={
        "do_sample":    True,
        "temperature":  0.7,
        "top_p":        0.9,
        "top_k":        50,
    }
)
```

### 3.2 单轮生成输出

```python
# actor_rollout_wg.generate_sequences() 返回
gen_output = DataProto(
    batch={
        "input_ids":      tensor([B', P + R_s]),  # prompt + 单轮响应
        "attention_mask": tensor([B', P + R_s]),
        "position_ids":   tensor([B', P + R_s]),
        "responses":      tensor([B', R_s]),      # 仅响应部分
    }
)
```

### 3.3 多轮生成最终输出

```python
# run_llm_loop() 返回的 gen_batch_output
gen_batch_output = DataProto(
    batch={
        "input_ids":      tensor([B', L]),    # [16, 9216] 完整序列
        "attention_mask": tensor([B', L]),    # [16, 9216]
        "position_ids":   tensor([B', L]),    # [16, 9216]
        "prompts":        tensor([B', P]),    # [16, 1024] 原始 prompt
        "responses":      tensor([B', R]),    # [16, 8192] 完整响应
        "action_mask":    tensor([B', R]),    # [16, 8192] 动作掩码
    },
    non_tensor_batch={
        "turns":          array([int] × B'),  # 每个样本的实际轮数
    }
)
```

### 3.4 Action Mask 详解

```
action_mask 形状: [B', R] = [16, 8192]

含义:
- 1: 模型生成的 token（需要计算梯度）
- 0: 环境反馈的 token（不计算梯度）

示例（某个样本的响应）:
响应: "<think>分析</think><code>print(1+1)</code><interpreter>2</interpreter><answer>\\boxed{2}</answer>"

Token 分解:
[<think>] [分] [析] [</think>] [<code>] [print] [...] [</code>] [<interpreter>] [2] [</interpreter>] [<answer>] [\\boxed{2}] [</answer>]

Action Mask:
[   1       1    1      1         1       1      1      1           0          0        0            1           1           1    ]
|<------------------- 模型生成 Turn 1 ----------------------->| |<----- 工具响应 ----->| |<----- 模型生成 Turn 2 ----->|
```

### 3.5 合并后的完整 batch

```python
batch = batch.union(gen_batch_output)

batch = DataProto(
    batch={
        "input_ids":      tensor([B', L]),    # [16, 9216]
        "attention_mask": tensor([B', L]),    # [16, 9216]
        "position_ids":   tensor([B', L]),    # [16, 9216]
        "prompts":        tensor([B', P]),    # [16, 1024]
        "responses":      tensor([B', R]),    # [16, 8192]
        "action_mask":    tensor([B', R]),    # [16, 8192]
        "response_mask":  tensor([B', R]),    # [16, 8192] 响应有效位置
    },
    non_tensor_batch={
        "uid":          array([str] × B'),
        "reward_model": array([dict] × B'),
        "data_source":  array([str] × B'),
        "turns":        array([int] × B'),
    }
)
```

---

## 4. 奖励计算阶段

### 4.1 奖励函数输出

```python
# compute_reward() 返回
reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)

reward_tensor: tensor([B', R])  # [16, 8192] 稀疏奖励张量
reward_extra_infos_dict: {
    "acc":        [int/float] × B',  # 答案准确率 (0 或 1)
    "format_acc": [int/float] × B',  # 格式准确率 (0 或 1)
    "score":      [float] × B',      # 最终分数
}
```

### 4.2 reward_tensor 结构

```
reward_tensor 是稀疏张量，只在最后一个有效 token 位置有非零值。

示例（某个样本）:
[0, 0, 0, ..., 0, 0, 0, 1.0, 0, 0, ..., 0]
                       ↑
              最后一个有效 token 位置

典型值:
-  1.0: 格式正确 + 答案正确
- -1.0: 格式正确 + 答案错误
- -1.0: 格式不正确
```

### 4.3 奖励计算后的 batch

```python
batch.batch["token_level_scores"] = reward_tensor   # [B', R]
batch.batch["token_level_rewards"] = reward_tensor  # [B', R] (无 KL 惩罚时)

batch.non_tensor_batch.update({
    "acc":        array([...]),
    "format_acc": array([...]),
    "score":      array([...]),
})
```

---

## 5. 对数概率计算阶段

### 5.1 compute_log_prob 输出

```python
# actor_rollout_wg.compute_log_prob() 返回
old_log_prob = DataProto(
    batch={
        "old_log_probs": tensor([B', R]),  # [16, 8192] 对数概率
        "entropys":      tensor([B', R]),  # [16, 8192] 熵
    }
)
```

### 5.2 对数概率张量结构

```
old_log_probs 形状: [B', R] = [16, 8192]

含义: 每个 token 的对数概率 log π(a_t | s_t)

示例（某个样本，截取部分）:
[-2.3, -1.5, -0.8, -1.2, ..., -1.8, 0, 0, ..., 0]
  ↑     ↑     ↑     ↑          ↑    ↑
有效 token 的对数概率      填充位置为 0
```

### 5.3 参考策略对数概率

```python
# ref_policy_wg.compute_ref_log_prob() 返回
ref_log_prob = DataProto(
    batch={
        "ref_log_prob": tensor([B', R]),  # [16, 8192]
    }
)

# 合并后
batch.batch["old_log_probs"] = old_log_probs  # [B', R]
batch.batch["ref_log_prob"] = ref_log_prob    # [B', R] (如果启用)
```

---

## 6. 优势计算阶段

### 6.1 KL 惩罚后的奖励

```python
# 如果 use_kl_in_reward=True
batch.batch["token_level_rewards"] = token_level_scores - β * KL
# 形状: [B', R]

# 否则
batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
```

### 6.2 compute_advantage 输出

```python
# 不同算法返回相同形状
batch.batch["advantages"] = tensor([B', R])  # [16, 8192]
batch.batch["returns"] = tensor([B', R])     # [16, 8192]
```

### 6.3 各算法的优势结构

**GRPO**:
```
advantages 是标量优势扩展到 token 级别

示例（某个样本，优势值为 0.732）:
[0.732, 0.732, 0.732, 0, 0, 0.732, 0.732, ...]
   ↑                   ↑
action_mask=1 的位置  action_mask=0 的位置为 0
```

**GAE**:
```
advantages 是每个 token 的独立优势估计

示例（某个样本）:
[0.5, 0.4, 0.3, 0, 0, 0.6, 0.8, ...]
 ↑    ↑    ↑        ↑    ↑
每个位置有不同的优势值
```

### 6.4 值函数（仅 GAE）

```python
# critic_wg.compute_values() 返回
values = DataProto(
    batch={
        "values": tensor([B', R]),  # [16, 8192]
    }
)

batch.batch["values"] = values  # 状态价值估计
```

---

## 7. 模型更新阶段

### 7.1 Actor 更新输入

```python
# update_actor() 接收的 batch
batch = DataProto(
    batch={
        "input_ids":           tensor([B', L]),   # [16, 9216]
        "attention_mask":      tensor([B', L]),   # [16, 9216]
        "position_ids":        tensor([B', L]),   # [16, 9216]
        "responses":           tensor([B', R]),   # [16, 8192]
        "action_mask":         tensor([B', R]),   # [16, 8192]
        "old_log_probs":       tensor([B', R]),   # [16, 8192]
        "advantages":          tensor([B', R]),   # [16, 8192]
        "returns":             tensor([B', R]),   # [16, 8192]
        "token_level_scores":  tensor([B', R]),   # [16, 8192]
        "token_level_rewards": tensor([B', R]),   # [16, 8192]
    },
    meta_info={
        "multi_turn": True,
    }
)
```

### 7.2 Critic 更新输入（仅 GAE）

```python
# update_critic() 额外需要的字段
batch.batch["values"] = tensor([B', R])   # 旧的值函数估计
batch.batch["returns"] = tensor([B', R])  # 目标回报
```

### 7.3 更新输出

```python
# update_actor() 返回
actor_output = DataProto(
    batch={},
    meta_info={
        "metrics": {
            "actor/loss": float,           # 策略损失
            "actor/pg_loss": float,        # 策略梯度损失
            "actor/clip_ratio": float,     # 裁剪比例
            "actor/approx_kl": float,      # 近似 KL
            "actor/entropy": float,        # 策略熵
        }
    }
)

# update_critic() 返回
critic_output = DataProto(
    batch={},
    meta_info={
        "metrics": {
            "critic/loss": float,          # 值函数损失
            "critic/clip_ratio": float,    # 裁剪比例
        }
    }
)
```

---

## 8. DataProto 完整结构

### 8.1 训练循环中的完整 batch

在训练步骤结束时，batch 包含以下所有字段：

```python
batch = DataProto(
    # ==================== 张量字段 ====================
    batch={
        # --- 序列数据 ---
        "input_ids":           tensor([B', L]),   # 完整序列 token IDs
        "attention_mask":      tensor([B', L]),   # 注意力掩码
        "position_ids":        tensor([B', L]),   # 位置编码
        "prompts":             tensor([B', P]),   # 原始 prompt
        "responses":           tensor([B', R]),   # 生成的响应

        # --- 掩码 ---
        "action_mask":         tensor([B', R]),   # 动作掩码 (1=模型, 0=环境)
        "response_mask":       tensor([B', R]),   # 响应有效位置掩码

        # --- 奖励 ---
        "token_level_scores":  tensor([B', R]),   # 原始 token 级别分数
        "token_level_rewards": tensor([B', R]),   # 含 KL 惩罚的奖励

        # --- 对数概率 ---
        "old_log_probs":       tensor([B', R]),   # 当前策略对数概率
        "ref_log_prob":        tensor([B', R]),   # 参考策略对数概率（可选）

        # --- 优势和回报 ---
        "advantages":          tensor([B', R]),   # 优势函数
        "returns":             tensor([B', R]),   # 回报

        # --- 值函数（仅 GAE）---
        "values":              tensor([B', R]),   # Critic 值估计

        # --- REMAX 专用 ---
        "reward_baselines":    tensor([B']),      # 贪婪采样奖励（可选）
    },

    # ==================== 非张量字段 ====================
    non_tensor_batch={
        # --- 标识 ---
        "uid":           array([str] × B'),      # GRPO 分组 ID
        "data_source":   array([str] × B'),      # 数据来源

        # --- 奖励元数据 ---
        "reward_model":  array([dict] × B'),     # 奖励配置
        "acc":           array([float] × B'),    # 答案准确率
        "format_acc":    array([float] × B'),    # 格式准确率
        "score":         array([float] × B'),    # 最终分数

        # --- 生成信息 ---
        "turns":         array([int] × B'),      # 实际对话轮数
        "raw_prompt_ids": array([list] × B'),    # 无填充 prompt
    },

    # ==================== 元信息 ====================
    meta_info={
        "multi_turn":         bool,              # 是否多轮对话
        "global_token_num":   list,              # 每个样本的有效 token 数
        "do_sample":          bool,              # 是否随机采样
        "temperature":        float,             # 温度参数
        "top_p":              float,             # top-p 参数
    }
)
```

### 8.2 形状速查表

| 字段名 | 形状 | 说明 |
|-------|------|------|
| `input_ids` | `[B', P+R]` | 完整序列 |
| `attention_mask` | `[B', P+R]` | 注意力掩码 |
| `position_ids` | `[B', P+R]` | 位置编码 |
| `prompts` | `[B', P]` | 原始 prompt |
| `responses` | `[B', R]` | 生成响应 |
| `action_mask` | `[B', R]` | 动作掩码 |
| `response_mask` | `[B', R]` | 响应掩码 |
| `token_level_scores` | `[B', R]` | 原始奖励 |
| `token_level_rewards` | `[B', R]` | 最终奖励 |
| `old_log_probs` | `[B', R]` | 当前策略 log prob |
| `ref_log_prob` | `[B', R]` | 参考策略 log prob |
| `advantages` | `[B', R]` | 优势函数 |
| `returns` | `[B', R]` | 回报 |
| `values` | `[B', R]` | 值函数（GAE） |
| `reward_baselines` | `[B']` | REMAX 基线 |

---

## 9. 常用配置值

### 9.1 典型训练配置

```yaml
# 数据配置
data:
  max_prompt_length: 1024        # P
  max_response_length: 8192      # R
  max_response_length_single_turn: 2048  # R_s
  train_batch_size: 4            # B

# 工具配置
tool:
  max_turns: 5                   # T
  use_batch_tool_calls: true

# 生成配置
actor_rollout_ref:
  rollout:
    n: 1
    n_repeat: 4                  # n_repeat
    temperature: 0.7
    top_p: 0.9
    top_k: 50

# 算法配置
algorithm:
  adv_estimator: "grpo"
  gamma: 1.0
  lam: 1.0
  use_kl_in_reward: false
```

### 9.2 不同算法的形状差异

| 算法 | 额外字段 | 形状 |
|-----|---------|------|
| GAE | `values` | `[B', R]` |
| REMAX | `reward_baselines` | `[B']` |
| GRPO/RLOO | 无 | - |

### 9.3 维度计算

```python
# 常用维度计算
B' = B × n_repeat           # 重复后的 batch_size
L = P + R                   # 完整序列长度
effective_tokens = B' × avg_response_length  # 有效 token 数

# 示例
B = 4, n_repeat = 4 → B' = 16
P = 1024, R = 8192 → L = 9216
avg_response_length = 500 → effective_tokens ≈ 8000
```

---

## 10. 调试技巧

### 10.1 检查张量形状

```python
def print_batch_shapes(batch: DataProto):
    """打印 batch 中所有张量的形状"""
    print("=== Tensor Batch ===")
    for key, value in batch.batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {list(value.shape)}")

    print("=== Non-Tensor Batch ===")
    for key, value in batch.non_tensor_batch.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'N/A'}")

    print("=== Meta Info ===")
    for key, value in batch.meta_info.items():
        print(f"  {key}: {value}")
```

### 10.2 验证 action_mask

```python
def validate_action_mask(batch: DataProto):
    """验证 action_mask 的正确性"""
    action_mask = batch.batch["action_mask"]
    response_mask = batch.batch["response_mask"]

    # action_mask 应该是 response_mask 的子集
    assert (action_mask <= response_mask).all(), "action_mask should be subset of response_mask"

    # 统计
    total_tokens = response_mask.sum().item()
    action_tokens = action_mask.sum().item()
    ratio = action_tokens / (total_tokens + 1e-8)

    print(f"Total response tokens: {total_tokens}")
    print(f"Action tokens: {action_tokens}")
    print(f"Action ratio: {ratio:.2%}")
```

### 10.3 检查奖励分布

```python
def analyze_rewards(batch: DataProto):
    """分析奖励分布"""
    scores = batch.batch["token_level_scores"]
    action_mask = batch.batch["action_mask"]

    # 计算每个样本的总奖励
    sample_rewards = (scores * action_mask).sum(dim=-1)

    print(f"Reward statistics:")
    print(f"  Mean: {sample_rewards.mean():.4f}")
    print(f"  Std: {sample_rewards.std():.4f}")
    print(f"  Min: {sample_rewards.min():.4f}")
    print(f"  Max: {sample_rewards.max():.4f}")

    # 正负奖励比例
    positive_ratio = (sample_rewards > 0).float().mean().item()
    print(f"  Positive ratio: {positive_ratio:.2%}")
```

### 10.4 检查 GRPO 分组

```python
def analyze_grpo_groups(batch: DataProto):
    """分析 GRPO 分组情况"""
    uids = batch.non_tensor_batch["uid"]
    scores = (batch.batch["token_level_scores"] * batch.batch["action_mask"]).sum(dim=-1)

    from collections import defaultdict
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(scores[i].item())

    print(f"Number of groups: {len(groups)}")
    print(f"Samples per group: {len(list(groups.values())[0])}")

    for uid, group_scores in list(groups.items())[:3]:
        print(f"\nGroup {uid[:8]}...")
        print(f"  Scores: {group_scores}")
        print(f"  Mean: {np.mean(group_scores):.4f}")
        print(f"  Std: {np.std(group_scores):.4f}")
```

### 10.5 常见问题排查

**问题 1: action_mask 全为 1**
```python
# 检查生成配置
if batch.batch.get("action_mask") is None:
    print("WARNING: action_mask not found, multi-turn may not be enabled")

# 检查环境响应是否正确标记
if batch.batch["action_mask"].sum() == batch.batch["response_mask"].sum():
    print("WARNING: No tool responses detected, check tool environment")
```

**问题 2: 优势全为 0**
```python
# GRPO 单样本组问题
uids = batch.non_tensor_batch["uid"]
unique_uids = len(set(uids))
if unique_uids == len(uids):
    print("WARNING: Each sample is in its own group, GRPO advantages will be 0")
    print("Check n_repeat configuration")
```

**问题 3: 奖励稀疏位置错误**
```python
# 检查奖励位置
scores = batch.batch["token_level_scores"]
nonzero_positions = (scores != 0).nonzero()
print(f"Reward positions: {nonzero_positions}")

# 应该只有最后一个有效 token 有奖励
response_mask = batch.batch["response_mask"]
for i in range(scores.shape[0]):
    valid_length = response_mask[i].sum().item()
    reward_pos = (scores[i] != 0).nonzero()
    if len(reward_pos) > 0:
        if reward_pos[0, 0].item() != valid_length - 1:
            print(f"WARNING: Sample {i} has reward at wrong position")
```

---

## 总结

本参考手册涵盖了 Agent-R1 训练流程中所有关键数据结构的形状和含义：

1. **数据准备**: 从 DataLoader 到 DataProto 的转换，batch.repeat() 的影响
2. **生成阶段**: 多轮生成的输出，action_mask 的创建
3. **奖励计算**: 稀疏奖励张量的结构
4. **对数概率**: 当前策略和参考策略的对数概率
5. **优势计算**: 不同算法的优势结构差异
6. **模型更新**: 完整的训练输入输出

核心要点：
- `B' = B × n_repeat` 是理解形状变化的关键
- `action_mask` 区分模型生成和环境反馈
- `uid` 用于 GRPO/RLOO 的组内比较
- 奖励是稀疏张量，只在最后一个有效 token 位置有值
