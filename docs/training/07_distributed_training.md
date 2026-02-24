# 多 GPU / 多节点训练配置

本文档详细介绍如何在不同硬件配置下进行 Agent-R1 的分布式训练，包括单机多卡和多节点集群配置。

## 目录

- [分布式训练概述](#分布式训练概述)
- [单机多卡配置](#单机多卡配置)
- [多节点分布式训练](#多节点分布式训练)
- [内存优化策略](#内存优化策略)
- [性能调优](#性能调优)
- [常见问题排查](#常见问题排查)

---

## 分布式训练概述

### Agent-R1 分布式架构

Agent-R1 使用以下技术实现分布式训练：

```
┌─────────────────────────────────────────────────────────────┐
│                        Ray Cluster                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   Ray Head Node                         ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     ││
│  │  │ TaskRunner  │  │   Logger    │  │  Scheduler  │     ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘     ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌───────────────────────┐  ┌───────────────────────┐      │
│  │    Worker Group 1     │  │    Worker Group 2     │      │
│  │  ┌─────────────────┐ │  │  ┌─────────────────┐  │      │
│  │  │ ActorRollout    │ │  │  │ Critic Worker   │  │      │
│  │  │ ┌────┐ ┌────┐  │ │  │  │ ┌────┐ ┌────┐  │  │      │
│  │  │ │GPU0│ │GPU1│  │ │  │  │ │GPU0│ │GPU1│  │  │      │
│  │  │ └────┘ └────┘  │ │  │  │ └────┘ └────┘  │  │      │
│  │  └─────────────────┘ │  │  └─────────────────┘  │      │
│  └───────────────────────┘  └───────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 说明 |
|-----|------|
| **Ray** | 分布式计算框架，管理资源和任务调度 |
| **FSDP** | PyTorch 全分片数据并行，用于模型分片 |
| **vLLM** | 推理引擎，支持张量并行 |

### 关键配置参数

| 参数 | 说明 |
|-----|------|
| `trainer.nnodes` | 节点数 |
| `trainer.n_gpus_per_node` | 每节点 GPU 数 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 推理时的张量并行大小 |
| `actor_rollout_ref.actor.strategy` | 分布式策略：fsdp/fsdp2 |

---

## 单机多卡配置

### 2x GPU 配置

适用于：2x RTX 4090 / 2x A100 等

```bash
#!/bin/bash

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # GPU 配置
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    \
    # 推理并行（使用所有 GPU）
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    \
    # 批次配置（根据显存调整）
    data.train_batch_size=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    \
    # 显存优化
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # 其他配置...
    "$@"
```

#### 2x RTX 4090 (24GB) 推荐配置

```bash
# 1.5B 模型
data.train_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.rollout.n_repeat=5

# 3B 模型
data.train_batch_size=16
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.45
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

### 4x GPU 配置

适用于：4x RTX 4090 / 4x A100 等

```bash
#!/bin/bash

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # GPU 配置
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    \
    # 推理并行（使用 2 或 4 GPU）
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    \
    # 批次配置
    data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    \
    "$@"
```

### 8x GPU 配置

适用于：8x A100 / 8x H100 等

```bash
#!/bin/bash

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # GPU 配置
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    \
    # 大模型使用 4-way 张量并行
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    \
    # 批次配置
    data.train_batch_size=128 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    \
    # 7B+ 模型配置
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    \
    "$@"
```

#### 张量并行大小选择

| 模型大小 | 推荐 tensor_model_parallel_size |
|---------|-------------------------------|
| < 3B | 1 或 2 |
| 3B - 7B | 2 |
| 7B - 14B | 2 或 4 |
| 14B - 34B | 4 或 8 |
| > 34B | 8 |

---

## 多节点分布式训练

### Ray 集群配置

#### 步骤 1：启动 Ray Head 节点

在主节点上运行：

```bash
# 启动 Ray head
ray start --head --port=6379

# 查看集群状态
ray status
```

#### 步骤 2：启动 Worker 节点

在每个 worker 节点上运行：

```bash
# 连接到 head 节点
ray start --address='<head_node_ip>:6379'
```

#### 步骤 3：运行训练

在 head 节点上运行训练脚本：

```bash
python3 -m agent_r1.src.main_agent \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    ...
```

### SLURM 集群配置

```bash
#!/bin/bash
#SBATCH --job-name=agent-r1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00

# 获取节点列表
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 在 head 节点启动 Ray
if [[ "$SLURMD_NODENAME" == "$head_node" ]]; then
    ray start --head --port=6379 --num-cpus=$SLURM_CPUS_PER_TASK
else
    ray start --address="$head_node_ip:6379" --num-cpus=$SLURM_CPUS_PER_TASK
fi

# 等待集群就绪
sleep 10

# 运行训练
python3 -m agent_r1.src.main_agent \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    ray_init.num_cpus=$SLURM_CPUS_PER_TASK \
    ...
```

### 多节点配置示例（2 节点 x 8 GPU）

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # 节点配置
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    \
    # 总共 16 GPU
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    \
    # 更大的批次
    data.train_batch_size=256 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    \
    # Ray 配置
    ray_init.num_cpus=64 \
    \
    "$@"
```

---

## 内存优化策略

### 策略 1：梯度检查点

用计算换内存，减少激活值存储：

```bash
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

**效果**：减少约 30-50% 显存占用
**代价**：训练速度降低约 20%

### 策略 2：参数卸载

将部分参数卸载到 CPU：

```bash
# 参考策略参数卸载
actor_rollout_ref.ref.fsdp_config.param_offload=True

# Actor 参数卸载（影响训练速度）
actor_rollout_ref.actor.fsdp_config.param_offload=True
```

### 策略 3：优化器卸载

将优化器状态卸载到 CPU：

```bash
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

**效果**：节省大量显存（优化器状态通常占模型参数的 2-3 倍）
**代价**：训练速度降低约 30-50%

### 策略 4：GPU 内存利用率

控制 vLLM 推理时的显存占用：

```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.5  # 默认 50%
```

**建议**：
- 训练稳定后可逐步提高到 0.6-0.7
- 如果 OOM，降低到 0.4-0.5

### 策略 5：移除填充

减少无效计算：

```bash
actor_rollout_ref.model.use_remove_padding=True
```

### 策略组合建议

| GPU 显存 | 模型大小 | 推荐策略 |
|---------|---------|---------|
| 24GB | 1.5B | gradient_checkpointing |
| 24GB | 3B | gradient_checkpointing + gpu_memory_utilization=0.45 |
| 24GB | 7B | gradient_checkpointing + param_offload + optimizer_offload |
| 40GB | 7B | gradient_checkpointing |
| 40GB | 14B | gradient_checkpointing + param_offload |
| 80GB | 14B | gradient_checkpointing |
| 80GB | 34B+ | gradient_checkpointing + param_offload |

---

## 性能调优

### 吞吐量优化

#### 1. 增大批次大小

```bash
# 在显存允许的情况下
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=32
```

#### 2. 启用 torch.compile

```bash
actor_rollout_ref.actor.use_torch_compile=True
```

#### 3. 分块预填充

```bash
actor_rollout_ref.rollout.enable_chunked_prefill=True
actor_rollout_ref.rollout.max_num_batched_tokens=32768
```

### 减少空闲时间

#### 1. 平衡批次

确保每个 GPU 的工作量均匀：

```bash
trainer.balance_batch=True
```

#### 2. 异步模式

```bash
actor_rollout_ref.rollout.mode=async
```

### 监控性能指标

关注以下 WandB 指标：

```
perf/throughput          # tokens/s/GPU
perf/mfu/actor           # 模型浮点利用率
timing_s/gen             # 生成耗时
timing_s/update_actor    # 更新耗时
```

---

## 常见问题排查

### 问题 1：Ray 连接超时

**错误信息**：
```
RayConnectionError: Failed to connect to Ray cluster
```

**解决方案**：
```bash
# 检查 Ray 状态
ray status

# 重启 Ray
ray stop
ray start --head --port=6379

# 增加超时时间
trainer.ray_wait_register_center_timeout=600
```

### 问题 2：NCCL 通信错误

**错误信息**：
```
RuntimeError: NCCL error
```

**解决方案**：
```bash
# 设置 NCCL 环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand（如果没有）
export NCCL_P2P_DISABLE=1  # 禁用 P2P（某些硬件需要）
```

### 问题 3：GPU 内存不均衡

**现象**：部分 GPU 显存占满，部分空闲

**解决方案**：
```bash
# 确保张量并行大小正确
actor_rollout_ref.rollout.tensor_model_parallel_size=<GPU数>

# 启用批次平衡
trainer.balance_batch=True
```

### 问题 4：多节点训练速度慢

**可能原因**：网络带宽不足

**解决方案**：
```bash
# 减少跨节点通信
# 使用更大的本地批次
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4

# 如果有 InfiniBand，确保启用
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0  # InfiniBand 接口名
```

### 问题 5：训练卡住不动

**诊断步骤**：
```bash
# 检查 Ray 日志
cat /tmp/ray/session_latest/logs/*.log | tail -100

# 检查 GPU 状态
nvidia-smi

# 检查进程状态
ps aux | grep python
```

**常见原因**：
1. 数据加载阻塞 → 增加 `filter_overlong_prompts_workers`
2. 检查点保存阻塞 → 检查磁盘空间
3. WandB 上传阻塞 → 切换到离线模式

---

## 配置速查表

### 显存占用估算

```
显存 = 模型参数 × 参数类型大小 × (1 + 优化器倍数) + 激活值

模型参数显存（BF16）:
- 1.5B → ~3 GB
- 7B → ~14 GB
- 14B → ~28 GB
- 34B → ~68 GB

优化器倍数（AdamW）: 2-3倍

激活值: 取决于批次大小和序列长度
```

### 推荐配置模板

| GPU 配置 | 模型 | train_batch_size | micro_batch | tp_size |
|---------|------|------------------|-------------|---------|
| 2x4090 | 1.5B | 32 | 2 | 2 |
| 2x4090 | 3B | 16 | 1 | 2 |
| 4x4090 | 1.5B | 64 | 2 | 2 |
| 4x4090 | 7B | 32 | 1 | 4 |
| 8xA100-40G | 7B | 128 | 4 | 2 |
| 8xA100-80G | 14B | 128 | 4 | 4 |
| 2x8xA100 | 34B | 256 | 2 | 8 |

---

## 下一步

- [08_model_export_deploy.md](./08_model_export_deploy.md) - 模型导出和部署
- [06_hyperparameter_tuning.md](./06_hyperparameter_tuning.md) - 超参数调优
