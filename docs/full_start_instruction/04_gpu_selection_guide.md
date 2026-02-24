# GPU 选择指南与问题排查

本文档帮助你选择合适的 GPU 配置，并提供常见问题的解决方案。

---

## GPU 显存需求速查表

### 按模型大小分类

| 模型大小 | RL 算法 | 最小显存 | 推荐配置 | Vast.ai 参考价格 |
|----------|---------|----------|----------|------------------|
| **1.5B** | GRPO | 48GB | 2x RTX 4090 (24GB) | ~$0.8/hr |
| **1.5B** | PPO | 60GB | 4x RTX 4090 (24GB) | ~$1.5/hr |
| **3B** | GRPO | 72GB | 4x RTX 4090 (24GB) | ~$1.5/hr |
| **3B** | PPO | 96GB | 4x RTX 4090 (24GB) | ~$1.5/hr |
| **7B** | GRPO | 120GB | 4x RTX 4090 或 2x A100-40GB | ~$3/hr |
| **7B** | PPO | 160GB | 4x A100-40GB | ~$5/hr |
| **14B** | GRPO | 240GB | 4x A100-80GB | ~$8/hr |
| **14B** | PPO | 320GB | 8x A100-40GB 或 4x H100 | ~$12/hr |

> **注意**：以上为估算值，实际显存使用受批次大小、序列长度等参数影响。

### 为什么 GRPO 显存需求更低？

| 因素 | PPO | GRPO |
|------|-----|------|
| Actor 模型 | ✓ | ✓ |
| Reference 模型 | ✓ | ✓ |
| **Critic 模型** | ✓ 需要 | ✗ 不需要 |
| 显存占用 | 高 | 较低 |

GRPO 不需要 Critic 模型，因此节省约 30% 显存。

---

## 影响显存的关键参数

### 参数优先级排序（从影响最大到最小）

```
影响程度:  ████████████████████  模型大小（无法改变）
           ████████████████      tensor_model_parallel_size
           ██████████████        ppo_micro_batch_size_per_gpu
           ████████████          gpu_memory_utilization
           ██████████            enable_gradient_checkpointing
           ████████              param_offload
           ██████                optimizer_offload
           ████                  max_response_length
```

### 参数说明与调整建议

#### 1. `tensor_model_parallel_size` - 模型并行度

**作用**：将模型分布到多个 GPU 上

**调整规则**：
- 必须能整除 `n_gpus_per_node`
- 值越大，单卡显存越低，但通信开销增加

```bash
# 2 GPU 配置
actor_rollout_ref.rollout.tensor_model_parallel_size=2

# 4 GPU 配置（7B+ 模型推荐）
actor_rollout_ref.rollout.tensor_model_parallel_size=4

# 8 GPU 配置（14B+ 模型）
actor_rollout_ref.rollout.tensor_model_parallel_size=8
```

#### 2. `ppo_micro_batch_size_per_gpu` - 单卡批次大小

**作用**：每个 GPU 每次处理的样本数

**调整规则**：
- 显存不足时首先降低此值
- 最小可设为 1
- 值越小，训练越慢但显存越低

```bash
# 显存充足
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4

# 显存紧张（默认）
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2

# 显存严重不足
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
```

#### 3. `gpu_memory_utilization` - vLLM 显存利用率

**作用**：控制 vLLM 推理引擎可使用的显存比例

**调整规则**：
- 默认 0.6，安全范围 0.5-0.7
- 太高可能导致 OOM
- 太低会降低推理吞吐

```bash
# 安全配置
actor_rollout_ref.rollout.gpu_memory_utilization=0.5

# 默认配置
actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# 激进配置（不推荐）
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
```

#### 4. `enable_gradient_checkpointing` - 梯度检查点

**作用**：以计算换显存，减少激活值存储

**调整规则**：
- **始终启用**，除非显存非常充足
- 大约节省 30-50% 显存
- 训练速度降低约 20%

```bash
# 强烈推荐始终开启
actor_rollout_ref.model.enable_gradient_checkpointing=True
critic.model.enable_gradient_checkpointing=True
```

#### 5. `param_offload` - 参数卸载到 CPU

**作用**：将模型参数从 GPU 卸载到 CPU

**调整规则**：
- Reference 模型建议开启
- Actor/Critic 模型开启会显著降低速度

```bash
# Reference 模型卸载（推荐）
actor_rollout_ref.ref.fsdp_config.param_offload=True

# Actor 模型卸载（显存极度紧张时）
actor_rollout_ref.actor.fsdp_config.param_offload=True
```

#### 6. `optimizer_offload` - 优化器卸载到 CPU

**作用**：将优化器状态从 GPU 卸载到 CPU

**调整规则**：
- **最后手段**，会大幅降低训练速度
- 仅在其他方法都无效时使用

```bash
# 不推荐，除非必要
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

---

## 显存优化流程图

当遇到 OOM (Out of Memory) 时，按以下顺序调整：

```
OOM 发生
    │
    ▼
Step 1: enable_gradient_checkpointing=True ──→ 仍然 OOM
    │                                              │
    ▼                                              ▼
Step 2: ppo_micro_batch_size_per_gpu=1 ──────→ 仍然 OOM
    │                                              │
    ▼                                              ▼
Step 3: gpu_memory_utilization=0.5 ──────────→ 仍然 OOM
    │                                              │
    ▼                                              ▼
Step 4: ref.param_offload=True ──────────────→ 仍然 OOM
    │                                              │
    ▼                                              ▼
Step 5: 增加 tensor_model_parallel_size ─────→ 仍然 OOM
    │                                              │
    ▼                                              ▼
Step 6: actor.param_offload=True ────────────→ 仍然 OOM
    │                                              │
    ▼                                              ▼
问题解决                                     需要更多 GPU 或更大显存
```

---

## 配置模板

### 配置 A：2x RTX 4090 - 1.5B GRPO（入门配置）

```bash
trainer.n_gpus_per_node=2
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.ref.fsdp_config.param_offload=True
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=32
```

### 配置 B：4x RTX 4090 - 1.5B/3B GRPO（标准配置）

```bash
trainer.n_gpus_per_node=4
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.ref.fsdp_config.param_offload=True
data.train_batch_size=128
actor_rollout_ref.actor.ppo_mini_batch_size=64
```

### 配置 C：4x A100-40GB - 7B GRPO

```bash
trainer.n_gpus_per_node=4
actor_rollout_ref.rollout.tensor_model_parallel_size=4
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.ref.fsdp_config.param_offload=True
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=32
```

### 配置 D：8x A100-80GB - 14B GRPO

```bash
trainer.n_gpus_per_node=8
actor_rollout_ref.rollout.tensor_model_parallel_size=8
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.ref.fsdp_config.param_offload=True
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=32
```

---

## Vast.ai GPU 选择建议

### 推荐 GPU 型号

| GPU | 显存 | 性价比 | 适合场景 |
|-----|------|--------|----------|
| RTX 4090 | 24GB | ⭐⭐⭐⭐⭐ | 1.5B-3B 模型，最佳性价比 |
| RTX 3090 | 24GB | ⭐⭐⭐⭐ | 预算有限的入门选择 |
| A100-40GB | 40GB | ⭐⭐⭐ | 7B 模型 |
| A100-80GB | 80GB | ⭐⭐⭐ | 14B+ 模型 |
| H100 | 80GB | ⭐⭐ | 追求最高性能（价格较高） |

### Vast.ai 搜索技巧

1. **筛选 GPU 型号**：在搜索框输入 `4090` 或 `A100`
2. **筛选显存**：设置 VRAM >= 24GB
3. **筛选卡数**：设置 GPU Count >= 2
4. **按性价比排序**：点击 `$/hr` 列排序

---

## 常见问题与排错

### 问题 1：CUDA Out of Memory

**现象**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate xxx MiB
```

**解决步骤**：

1. 检查 GPU 显存使用：
   ```bash
   nvidia-smi
   ```

2. 按显存优化流程图依次调整参数

3. 如果仍然 OOM，考虑：
   - 使用更大显存的 GPU
   - 增加 GPU 数量

### 问题 2：CUDA/Driver 版本不匹配

**现象**：
```
RuntimeError: CUDA error: no kernel image is available for execution
```
或
```
CUDA driver version is insufficient for CUDA runtime version
```

**解决方案**：

1. 检查版本：
   ```bash
   nvidia-smi  # 查看 Driver 和 CUDA 版本
   python -c "import torch; print(torch.version.cuda)"
   ```

2. 确保版本匹配：
   | PyTorch CUDA | 需要 Driver 版本 |
   |--------------|------------------|
   | CUDA 12.1 | >= 525.60.13 |
   | CUDA 12.4 | >= 535.xx |
   | CUDA 11.8 | >= 450.80.02 |

3. 如果不匹配，选择其他 Vast.ai 实例或更新镜像

### 问题 3：vLLM 初始化失败

**现象**：
```
RuntimeError: Failed to initialize NCCL
```
或
```
vllm.engine.llm_engine: ERROR - Error initializing engine
```

**解决方案**：

1. 检查 GPU 通信：
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```

2. 降低 gpu_memory_utilization：
   ```bash
   actor_rollout_ref.rollout.gpu_memory_utilization=0.5
   ```

3. 检查 tensor_model_parallel_size 是否正确：
   ```bash
   # 必须能整除 GPU 数量
   # 例如 4 GPU 可以用 1, 2, 4
   ```

### 问题 4：HuggingFace 下载失败

**现象**：
```
ConnectionError: HTTPSConnectionPool(host='huggingface.co', port=443)
```

**解决方案**：

1. 设置镜像（中国用户）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. 手动下载模型：
   ```bash
   pip install huggingface_hub
   huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./model
   ```

3. 使用本地模型路径：
   ```bash
   actor_rollout_ref.model.path=./model
   ```

### 问题 5：Ray 集群问题

**现象**：
```
ray.exceptions.RaySystemError: System error
```
或训练卡住不动

**解决方案**：

1. 停止现有 Ray 进程：
   ```bash
   ray stop --force
   ```

2. 清理临时文件：
   ```bash
   rm -rf /tmp/ray/*
   ```

3. 重新启动训练

### 问题 6：WandB 连接问题

**现象**：
```
wandb: ERROR Error communicating with wandb process
```

**解决方案**：

1. 使用离线模式：
   ```bash
   export WANDB_MODE=offline
   ```

2. 或仅使用控制台日志：
   ```bash
   trainer.logger=['console']
   ```

3. 训练完成后同步：
   ```bash
   wandb sync wandb/run-xxx/
   ```

### 问题 7：训练速度异常缓慢

**可能原因及解决**：

| 原因 | 诊断方法 | 解决方案 |
|------|----------|----------|
| CPU 瓶颈 | `htop` 看 CPU 使用 | 增加数据加载 workers |
| IO 瓶颈 | `iostat` | 使用 SSD 存储 |
| 网络瓶颈 | 下载速度测试 | 预下载所有数据 |
| 参数卸载 | 检查配置 | 关闭不必要的 offload |
| 批次太小 | 检查配置 | 在显存允许范围内增大批次 |

### 问题 8：pip 安装依赖失败

**现象**：编译错误或依赖冲突

**解决方案**：

1. flash-attn 编译失败：
   ```bash
   # 确保有 build-essential
   apt-get install build-essential

   # 尝试指定版本
   pip install flash-attn==2.5.8 --no-build-isolation
   ```

2. 版本冲突：
   ```bash
   # 创建新环境
   conda create -n verl python=3.10 -y
   conda activate verl

   # 重新安装
   pip install -e . --force-reinstall
   ```

---

## 性能调优建议

### 1. 提高训练吞吐量

```bash
# 启用 chunked prefill
actor_rollout_ref.rollout.enable_chunked_prefill=True

# 增加 batched tokens（在显存允许范围内）
actor_rollout_ref.rollout.max_num_batched_tokens=65536
```

### 2. 提高训练稳定性

```bash
# 使用低方差 KL
actor_rollout_ref.actor.kl_loss_type=low_var_kl

# 适当的梯度裁剪
actor_rollout_ref.actor.grad_clip=1.0
```

### 3. 使用混合精度

```bash
# 默认使用 bfloat16
actor_rollout_ref.rollout.dtype=bfloat16
```

---

## 总结

### 新手推荐配置

| 目标 | 推荐配置 | 预计成本 |
|------|----------|----------|
| 学习测试 | 2x RTX 4090 + 1.5B GRPO | ~$1/hr |
| 正式训练 | 4x RTX 4090 + 1.5B-3B GRPO | ~$1.5/hr |
| 大模型 | 4x A100-40GB + 7B GRPO | ~$3/hr |

### 关键记忆点

1. **GRPO 比 PPO 省显存** - 不需要 Critic
2. **遇到 OOM 先降 micro_batch_size**
3. **始终开启 gradient_checkpointing**
4. **Reference 模型可以 offload**
5. **gpu_memory_utilization 保持 0.5-0.6**

---

## 相关文档

- [01_vast_template_setup.md](./01_vast_template_setup.md) - 模板创建
- [02_provisioning_script.md](./02_provisioning_script.md) - 环境配置脚本
- [03_run_training_workflow.md](./03_run_training_workflow.md) - 训练流程
- [docs/readme/09_running_experiments.md](../readme/09_running_experiments.md) - 详细参数说明
