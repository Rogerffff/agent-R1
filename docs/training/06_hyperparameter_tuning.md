# 超参数调优指南

本文档介绍 Agent-R1 训练中关键超参数的调优方法，帮助你根据训练曲线和任务特点找到最佳配置。

## 目录

- [超参数概览](#超参数概览)
- [学习率调优](#学习率调优)
- [Batch Size 选择](#batch-size-选择)
- [KL 系数调整](#kl-系数调整)
- [Clip Ratio 设置](#clip-ratio-设置)
- [采样数 (n_repeat) 调优](#采样数-n_repeat-调优)
- [任务类型推荐配置](#任务类型推荐配置)
- [调优流程](#调优流程)

---

## 超参数概览

### 影响程度分级

| 超参数 | 影响程度 | 主要影响 |
|-------|---------|---------|
| 学习率 (lr) | ⭐⭐⭐⭐⭐ | 训练稳定性和收敛速度 |
| Batch Size | ⭐⭐⭐⭐ | 训练稳定性和显存占用 |
| KL 系数 | ⭐⭐⭐⭐ | 探索-利用平衡 |
| n_repeat | ⭐⭐⭐ | 优势估计质量（GRPO） |
| Clip Ratio | ⭐⭐⭐ | 策略更新幅度 |
| 温度 (temperature) | ⭐⭐ | 生成多样性 |

### 推荐调优顺序

1. **先固定其他参数，调优学习率**
2. **确定合适的 Batch Size（受显存限制）**
3. **调整 KL 系数**
4. **根据任务调整 n_repeat**
5. **微调 Clip Ratio 和其他参数**

---

## 学习率调优

### 基本原则

```
学习率过大 → 训练不稳定，loss 波动大，可能发散
学习率过小 → 收敛太慢，需要更多训练步数
```

### 推荐值

| 模型大小 | 推荐学习率范围 | 起始值 |
|---------|---------------|-------|
| < 3B | 5e-7 ~ 5e-6 | 1e-6 |
| 3B - 7B | 1e-7 ~ 1e-6 | 5e-7 |
| 7B - 14B | 5e-8 ~ 5e-7 | 1e-7 |
| > 14B | 1e-8 ~ 1e-7 | 5e-8 |

### 判断标准

**学习率过大的信号**：
```
- actor/grad_norm 持续接近 grad_clip 值
- actor/pg_clipfrac > 0.5
- critic/score/mean 大幅波动
- loss 出现 NaN
```

**学习率过小的信号**：
```
- actor/pg_clipfrac < 0.05
- critic/score/mean 长时间不变
- actor/ppo_kl < 0.001
```

### 调优策略

```python
# 策略 1: 二分搜索
# 如果训练不稳定，学习率减半
# 如果训练太慢，学习率加倍

# 起始：1e-6
# 如果不稳定 → 5e-7 → 2.5e-7 → ...
# 如果太慢 → 2e-6 → 4e-6 → ...

# 策略 2: 预热 (Warmup)
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1  # 10% 预热
actor_rollout_ref.actor.optim.warmup_style=constant      # 或 cosine
```

### 配置示例

```bash
# 保守配置（新任务推荐）
actor_rollout_ref.actor.optim.lr=5e-7
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1

# 激进配置（已验证稳定的任务）
actor_rollout_ref.actor.optim.lr=2e-6
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0
```

---

## Batch Size 选择

### 基本原则

```
Batch Size 大 → 梯度估计更准确，训练更稳定，但显存占用高
Batch Size 小 → 更多随机性，可能帮助探索，但训练可能不稳定
```

### 相关参数

| 参数 | 说明 |
|-----|------|
| `data.train_batch_size` | 每步处理的样本数 |
| `actor.ppo_mini_batch_size` | PPO 更新的小批次 |
| `actor.ppo_micro_batch_size_per_gpu` | 每 GPU 的微批次 |

### 关系

```
train_batch_size = 每步采集的样本数
ppo_mini_batch_size = PPO 更新时使用的批次
ppo_micro_batch_size_per_gpu = 每个 GPU 一次前向的样本数

约束：
- train_batch_size 应该能被 ppo_mini_batch_size 整除
- ppo_mini_batch_size 应该能被 (n_gpus * ppo_micro_batch_size_per_gpu) 整除
```

### 显存占用估算

```
显存 ≈ 模型参数 + 优化器状态 + 激活值
激活值 ≈ micro_batch_size × max_length × hidden_size × layers

减少显存：
1. 减小 ppo_micro_batch_size_per_gpu
2. 启用 gradient_checkpointing
3. 启用 param_offload / optimizer_offload
```

### 推荐配置

| GPU 配置 | train_batch_size | ppo_mini_batch_size | micro_batch_size |
|---------|-----------------|--------------------|--------------------|
| 2x 4090 (24GB) | 16-32 | 8-16 | 1-2 |
| 4x 4090 (24GB) | 32-64 | 16-32 | 1-2 |
| 8x A100 (40GB) | 64-128 | 32-64 | 2-4 |
| 8x A100 (80GB) | 128-256 | 64-128 | 4-8 |

### 配置示例

```bash
# 2x 4090 配置
data.train_batch_size=32
actor_rollout_ref.actor.ppo_mini_batch_size=16
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2

# 内存不足时的优化
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

---

## KL 系数调整

### 作用

KL 系数控制策略更新与参考策略的偏离程度：

```
KL 系数大 → 策略更新保守，不容易偏离太远，但学习慢
KL 系数小 → 策略更新激进，学习快，但可能不稳定
```

### 两种 KL 约束方式

**方式 1：KL 损失（GRPO 推荐）**
```bash
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

**方式 2：KL 惩罚奖励（PPO 推荐）**
```bash
algorithm.use_kl_in_reward=True
algorithm.kl_penalty=kl
algorithm.kl_ctrl.type=adaptive
algorithm.kl_ctrl.kl_coef=0.001
algorithm.kl_ctrl.target_kl=0.05
```

### 推荐值

| 场景 | kl_loss_coef | 说明 |
|-----|-------------|------|
| 默认值 | 0.001 | 适合大多数任务 |
| 训练不稳定 | 0.01 | 更强的约束 |
| 需要更多探索 | 0.0001 | 更弱的约束 |

### 判断标准

**KL 约束过强的信号**：
```
- actor/ppo_kl 持续 < 0.001
- critic/score/mean 上升很慢
- actor/pg_clipfrac 很低
```

**KL 约束过弱的信号**：
```
- actor/ppo_kl > 0.1
- 策略震荡，分数波动大
- 验证准确率下降（过拟合）
```

### 自适应 KL 控制器

```bash
# 自动调整 KL 系数，保持实际 KL 接近目标值
algorithm.kl_ctrl.type=adaptive
algorithm.kl_ctrl.target_kl=0.05
algorithm.kl_ctrl.horizon=10000
```

工作原理：
- 如果实际 KL > target_kl，增大 kl_coef
- 如果实际 KL < target_kl，减小 kl_coef

---

## Clip Ratio 设置

### 作用

Clip Ratio 限制策略更新的幅度：

```python
ratio = π_new(a|s) / π_old(a|s)
clipped_ratio = clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
```

### 推荐值

| 参数 | 默认值 | 推荐范围 |
|-----|-------|---------|
| `clip_ratio` | 0.2 | 0.1 - 0.3 |
| `clip_ratio_low` | 0.2 | 0.1 - 0.3 |
| `clip_ratio_high` | 0.2 | 0.1 - 0.3 |
| `clip_ratio_c` | 3.0 | 2.0 - 5.0 |

### 调优建议

```
训练不稳定 → 减小 clip_ratio（更保守）
训练太慢 → 增大 clip_ratio（更激进）
```

### Dual-Clip PPO

```bash
# 同时限制上下界
actor_rollout_ref.actor.clip_ratio_low=0.2   # 下界
actor_rollout_ref.actor.clip_ratio_high=0.2  # 上界
actor_rollout_ref.actor.clip_ratio_c=3.0     # Dual-clip 参数
```

---

## 采样数 (n_repeat) 调优

### 作用

n_repeat 控制每个提示采样多少个响应（仅 GRPO 相关）：

```
n_repeat 大 → 优势估计更准确，但计算成本高
n_repeat 小 → 计算快，但优势估计方差大
```

### 推荐值

| 场景 | n_repeat | 说明 |
|-----|----------|------|
| 快速实验 | 3 | 最低限度 |
| 标准训练 | 5 | 默认推荐 |
| 高质量训练 | 8 | 更稳定的优势估计 |
| 极限场景 | 16+ | 非常稳定，但很慢 |

### 与 Batch Size 的关系

```
实际每步处理的响应数 = train_batch_size × n_repeat

例如：
train_batch_size=32, n_repeat=5 → 每步 160 个响应
```

### 配置示例

```bash
# 快速实验
actor_rollout_ref.rollout.n_repeat=3
data.train_batch_size=32

# 标准训练
actor_rollout_ref.rollout.n_repeat=5
data.train_batch_size=32

# 高质量训练
actor_rollout_ref.rollout.n_repeat=8
data.train_batch_size=16  # 减小 batch_size 保持总量
```

---

## 任务类型推荐配置

### 问答任务（HotpotQA、MuSiQue）

```bash
# 特点：需要多轮工具调用，答案相对明确
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.rollout.n_repeat=5
tool.max_turns=5
data.max_response_length=4096
```

### 数学任务（GSM8K、MATH）

```bash
# 特点：需要严格的推理，代码执行
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.optim.lr=5e-7  # 较小的学习率
actor_rollout_ref.actor.kl_loss_coef=0.0005  # 更宽松的 KL
actor_rollout_ref.rollout.n_repeat=8  # 更多采样
tool.env=retool
```

### 代码任务

```bash
# 特点：需要代码生成和执行
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.optim.lr=5e-7
actor_rollout_ref.rollout.temperature=0.8  # 稍低温度
tool.env=retool
data.max_response_length_single_turn=1024  # 更长的代码块
```

---

## 调优流程

### 第一阶段：快速验证

目标：确认训练流程正常运行

```bash
# 小数据集 + 短训练
data.train_batch_size=16
trainer.total_training_steps=50
trainer.test_freq=10
actor_rollout_ref.rollout.n_repeat=3
```

检查项：
- [ ] 训练能正常启动
- [ ] 没有 OOM
- [ ] WandB 正常记录
- [ ] 指标在合理范围

### 第二阶段：学习率搜索

```bash
# 尝试不同学习率
for lr in 1e-7 5e-7 1e-6 2e-6; do
    python3 -m agent_r1.src.main_agent \
        actor_rollout_ref.actor.optim.lr=$lr \
        trainer.experiment_name=lr_search_$lr \
        trainer.total_training_steps=100 \
        ...
done
```

选择标准：
- 分数持续上升
- 梯度范数稳定
- clipfrac 在 0.1-0.3

### 第三阶段：KL 系数调优

```bash
# 固定最佳学习率，搜索 KL 系数
for kl in 0.0001 0.001 0.01; do
    python3 -m agent_r1.src.main_agent \
        actor_rollout_ref.actor.kl_loss_coef=$kl \
        trainer.experiment_name=kl_search_$kl \
        ...
done
```

### 第四阶段：完整训练

使用最佳超参数进行完整训练：

```bash
python3 -m agent_r1.src.main_agent \
    actor_rollout_ref.actor.optim.lr=<最佳学习率> \
    actor_rollout_ref.actor.kl_loss_coef=<最佳KL系数> \
    trainer.total_training_steps=<完整步数> \
    trainer.save_freq=50 \
    ...
```

---

## 常见问题

### Q: 训练开始很好，后期变差？

**可能原因**：
- 学习率过大，后期不稳定
- 过拟合到训练数据

**解决方案**：
```bash
# 使用学习率衰减
actor_rollout_ref.actor.optim.warmup_style=cosine
actor_rollout_ref.actor.optim.min_lr_ratio=0.1

# 或增强 KL 约束
actor_rollout_ref.actor.kl_loss_coef=0.01
```

### Q: 分数一直不上升？

**可能原因**：
- 学习率太小
- KL 约束太强
- 奖励函数问题

**解决方案**：
```bash
# 增大学习率
actor_rollout_ref.actor.optim.lr=2e-6

# 减小 KL 约束
actor_rollout_ref.actor.kl_loss_coef=0.0001

# 检查奖励函数
trainer.log_val_generations=10  # 查看生成样本
```

### Q: 显存不够怎么办？

**解决方案优先级**：
1. 减小 `ppo_micro_batch_size_per_gpu`
2. 启用 `gradient_checkpointing`
3. 减小 `max_response_length`
4. 启用 `optimizer_offload`
5. 使用更小的模型

---

## 下一步

- [07_distributed_training.md](./07_distributed_training.md) - 分布式训练配置
- [02_key_metrics_monitoring.md](./02_key_metrics_monitoring.md) - 根据指标调优
