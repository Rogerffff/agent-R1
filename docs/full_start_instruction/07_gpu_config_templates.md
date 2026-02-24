# GPU 配置模板

本文档提供不同 GPU 配置下的 GRPO 训练完整脚本模板。

---

## 目录

1. [GPU 显存需求速查表](#1-gpu-显存需求速查表)
2. [RTX 4090 24GB 配置](#2-rtx-4090-24gb-配置)
3. [RTX 4090 48GB 魔改版配置](#3-rtx-4090-48gb-魔改版配置)
4. [A100/H100 80GB 配置](#4-a100h100-80gb-配置)
5. [参数调整指南](#5-参数调整指南)
6. [批处理大小计算器](#6-批处理大小计算器)

---

## 1. GPU 显存需求速查表

### GRPO 算法显存估算

| 模型大小 | 2x GPU | 4x GPU | 8x GPU |
|----------|--------|--------|--------|
| **1.5B** | 24GB ✓ | 24GB ✓ | 24GB ✓ |
| **3B** | 40GB+ | 24GB ✓ | 24GB ✓ |
| **7B** | 80GB | 48GB | 24GB ✓ |

> **说明**：表中显示每张 GPU 所需的最小显存。✓ 表示该配置可行。

### 推荐配置总结

| 模型 | 推荐 GPU 配置 | 预计成本 |
|------|--------------|----------|
| 1.5B | 2x RTX 4090 (24GB) | ~$0.8/hr |
| 1.5B/3B | 4x RTX 4090 (24GB) | ~$1.5/hr |
| 3B/7B | 4x RTX 4090-48GB 或 4x A100 | ~$3/hr |
| 7B | 8x RTX 4090 (24GB) 或 4x A100-80GB | ~$5/hr |

---

## 2. RTX 4090 24GB 配置

### 2.1 配置 A：2x RTX 4090 + 1.5B 模型

**适用场景**：入门测试、快速验证

```bash
#!/bin/bash
export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-1.5b-2gpu'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # === 数据配置 ===
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.max_response_length_single_turn=512 \
    \
    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # === Actor 配置 ===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    # === Rollout 配置 ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    \
    # === Reference 配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    # === 算法配置 ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # === 训练器配置 ===
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    \
    # === 工具配置 ===
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=1024 \
    \
    $@
```

**关键参数说明**：
- `tensor_model_parallel_size=2`：模型分布到 2 张 GPU
- `train_batch_size=64`：较小批次适合 2 GPU
- `ref.fsdp_config.param_offload=True`：Reference 模型卸载到 CPU 节省显存

---

### 2.2 配置 B：4x RTX 4090 + 1.5B 模型

**适用场景**：标准训练配置

```bash
#!/bin/bash
export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-1.5b-4gpu'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # === 数据配置 ===
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    \
    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # === Actor 配置 ===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    # === Rollout 配置 ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    \
    # === Reference 配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    # === 算法配置 ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # === 训练器配置 ===
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    \
    # === 工具配置 ===
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=2048 \
    \
    $@
```

---

### 2.3 配置 C：4x RTX 4090 + 3B 模型

**适用场景**：中等规模模型训练

```bash
#!/bin/bash
export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-3b-4gpu'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # === 数据配置 ===
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.max_response_length_single_turn=512 \
    \
    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # === Actor 配置（降低批大小节省显存）===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    # === Rollout 配置 ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n_repeat=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    \
    # === Reference 配置（卸载到 CPU）===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    # === 算法配置 ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # === 训练器配置 ===
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    \
    # === 工具配置 ===
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=1024 \
    \
    $@
```

**关键调整**：
- `tensor_model_parallel_size=4`：使用全部 4 张 GPU 进行模型并行
- `ppo_micro_batch_size_per_gpu=1`：降低单卡批大小
- `gpu_memory_utilization=0.5`：降低 vLLM 显存占用

---

### 2.4 配置 D：8x RTX 4090 + 7B 模型

**适用场景**：大规模模型训练（需要显存优化）

```bash
#!/bin/bash
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-7b-8gpu'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # === 数据配置 ===
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.max_response_length_single_turn=512 \
    \
    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # === Actor 配置（最小化显存）===
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    # === Rollout 配置 ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.n_repeat=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    \
    # === Reference 配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    # === 算法配置 ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # === 训练器配置 ===
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    \
    # === 工具配置 ===
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=512 \
    \
    $@
```

**关键调整**：
- `tensor_model_parallel_size=8`：全部 8 张 GPU 模型并行
- `actor.fsdp_config.param_offload=True`：Actor 参数也卸载
- `gpu_memory_utilization=0.45`：进一步降低 vLLM 显存
- `max_prompt_length=2048`：缩短序列长度

---

## 3. RTX 4090 48GB 魔改版配置

### 3.1 配置 E：2x RTX 4090-48GB + 3B 模型

```bash
#!/bin/bash
export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-3b-2gpu-48g'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # === 数据配置 ===
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    \
    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # === Actor 配置 ===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    # === Rollout 配置 ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    \
    # === Reference 配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    # === 算法配置 ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # === 训练器配置 ===
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    \
    # === 工具配置 ===
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=2048 \
    \
    $@
```

**48GB 显存优势**：
- `ppo_micro_batch_size_per_gpu=4`：可以使用更大的微批
- `gpu_memory_utilization=0.7`：可以提高 vLLM 显存利用率
- 无需 `param_offload`

---

### 3.2 配置 F：4x RTX 4090-48GB + 7B 模型

```bash
#!/bin/bash
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-7b-4gpu-48g'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # === 数据配置 ===
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    \
    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # === Actor 配置 ===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    # === Rollout 配置 ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    \
    # === Reference 配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    # === 算法配置 ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # === 训练器配置 ===
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    \
    # === 工具配置 ===
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=2048 \
    \
    $@
```

---

## 4. A100/H100 80GB 配置

### 4.1 配置 G：2x A100-80GB + 7B 模型

```bash
#!/bin/bash
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-7b-2xa100'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # === 数据配置 ===
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    \
    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # === Actor 配置（A100 可以使用更大批次）===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    # === Rollout 配置 ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    \
    # === Reference 配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    # === 算法配置 ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # === 训练器配置 ===
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    \
    # === 工具配置 ===
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=2048 \
    \
    $@
```

---

### 4.2 配置 H：4x A100-80GB + 7B 模型

```bash
#!/bin/bash
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-7b-4xa100'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # === 数据配置 ===
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    \
    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # === Actor 配置 ===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    # === Rollout 配置 ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    \
    # === Reference 配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    # === 算法配置 ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # === 训练器配置 ===
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    \
    # === 工具配置 ===
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=2048 \
    \
    $@
```

---

### 4.3 配置 I：8x A100-80GB + 7B 模型（高吞吐）

```bash
#!/bin/bash
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-7b-8xa100'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # === 数据配置 ===
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=512 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    \
    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # === Actor 配置（最大吞吐）===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    # === Rollout 配置 ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    \
    # === Reference 配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    # === 算法配置 ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # === 训练器配置 ===
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    \
    # === 工具配置 ===
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=2048 \
    \
    $@
```

---

## 5. 参数调整指南

### 关键参数与显存关系

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

### OOM 时的调整流程

```
OOM 发生
    │
    ▼
Step 1: enable_gradient_checkpointing=True
    │
    ├── 仍然 OOM ──▶ Step 2
    │
    ▼
Step 2: ppo_micro_batch_size_per_gpu 减半（如 2→1）
    │
    ├── 仍然 OOM ──▶ Step 3
    │
    ▼
Step 3: gpu_memory_utilization 降低 0.1（如 0.6→0.5）
    │
    ├── 仍然 OOM ──▶ Step 4
    │
    ▼
Step 4: ref.fsdp_config.param_offload=True
    │
    ├── 仍然 OOM ──▶ Step 5
    │
    ▼
Step 5: tensor_model_parallel_size 增加（如 2→4）
    │
    ├── 仍然 OOM ──▶ Step 6
    │
    ▼
Step 6: actor.fsdp_config.param_offload=True
    │
    ├── 仍然 OOM ──▶ 需要更多 GPU 或更大显存
    │
    ▼
问题解决
```

### 参数调整速查表

| 目标 | 调整参数 | 方向 |
|------|----------|------|
| 减少显存 | `ppo_micro_batch_size_per_gpu` | 降低 |
| 减少显存 | `gpu_memory_utilization` | 降低 |
| 减少显存 | `tensor_model_parallel_size` | 增加 |
| 减少显存 | `param_offload` | 设为 True |
| 提高吞吐 | `ppo_micro_batch_size_per_gpu` | 增加 |
| 提高吞吐 | `gpu_memory_utilization` | 增加 |
| 提高吞吐 | `train_batch_size` | 增加 |

---

## 6. 批处理大小计算器

### 计算公式

```
real_train_batch_size = train_batch_size × n_repeat

约束条件:
1. real_train_batch_size % n_gpus == 0
2. train_batch_size >= ppo_mini_batch_size
3. ppo_mini_batch_size % ppo_micro_batch_size_per_gpu == 0
```

### 示例计算

**配置**:
```
train_batch_size = 128
n_repeat = 5
n_gpus = 4
ppo_mini_batch_size = 64
ppo_micro_batch_size_per_gpu = 2
```

**验证**:
```
real_train_batch_size = 128 × 5 = 640

约束1: 640 % 4 = 0 ✓
约束2: 128 >= 64 ✓
约束3: 64 % 2 = 0 ✓
```

### 快速配置参考

| GPU 数 | train_batch_size | n_repeat | ppo_mini_batch_size | ppo_micro_batch_size_per_gpu |
|--------|------------------|----------|---------------------|------------------------------|
| 2 | 64 | 5 | 32 | 2 |
| 4 | 128 | 5 | 64 | 2 |
| 8 | 256 | 5 | 128 | 2 |
| 8 | 512 | 5 | 256 | 4 |

---

## 7. 快速验证脚本

在正式训练之前，建议先用小数据集快速验证实验流程是否能跑通。

### 7.1 配置 J：2x RTX 4090-48GB 快速验证（3B 模型）

**用途**：3-10 分钟内验证完整训练流程

```bash
#!/bin/bash
# ============================================================
# 快速验证脚本 - 2x RTX 4090-48GB + 3B 模型
# 预计运行时间：3-10 分钟
# 用途：验证训练流程能否跑通
# ============================================================

export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-3b-quick-test'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    # === 极简数据配置 ===
    data.train_files=['data/hotpotqa_mini/train.parquet'] \
    data.val_files=['data/hotpotqa_mini/validation.parquet'] \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.max_response_length_single_turn=256 \
    \
    # === 模型配置 ===
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # === Actor 配置 ===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    # === Rollout 配置（减少采样次数）===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n_repeat=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    \
    # === Reference 配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    # === 算法配置 ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # === 训练器配置（关闭验证和保存）===
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    \
    # === 工具配置（减少对话轮数）===
    tool.max_turns=2 \
    tool.tools=['search'] \
    tool.max_tool_response_length=512 \
    \
    $@
```

**与正式配置的关键差异**：

| 参数 | 正式配置 | 快速验证 | 说明 |
|------|----------|----------|------|
| `train_batch_size` | 128 | 16 | 减少 8 倍 |
| `n_repeat` | 5 | 1 | 不做多次采样 |
| `max_prompt_length` | 8192 | 1024 | 缩短序列 |
| `max_response_length` | 8192 | 1024 | 缩短序列 |
| `max_turns` | 5 | 2 | 减少工具调用 |
| `test_freq` | 10 | -1 | 关闭验证 |
| `val_before_train` | True | False | 跳过初始验证 |
| `logger` | console,wandb | console | 只用控制台 |

---

### 7.2 生成快速验证用小数据集

在运行快速验证脚本之前，需要先生成小数据集：

```bash
#!/bin/bash
# ============================================================
# 生成快速验证用小数据集
# 只保留 32 条训练数据和 8 条验证数据
# ============================================================

# 创建小数据集目录
mkdir -p data/hotpotqa_mini

# 运行数据预处理，限制样本数量
python examples/data_preprocess/hotpotqa.py \
    --local_dir data/hotpotqa_mini \
    --train_size 32 \
    --val_size 8

echo "✅ 小数据集生成完成！"
echo "   - 训练集: data/hotpotqa_mini/train.parquet (32 条)"
echo "   - 验证集: data/hotpotqa_mini/validation.parquet (8 条)"
echo ""
echo "现在可以运行快速验证脚本了。"
```

---

### 7.3 完整快速验证流程

```bash
# Step 1: 生成小数据集
bash scripts/generate_mini_dataset.sh

# Step 2: 运行快速验证
bash scripts/quick_test_2x4090_48g.sh

# Step 3: 检查输出
# 看到以下输出说明流程跑通：
# [Step 1] Rollout completed
# [Step 1] Computing old log probs...
# [Step 1] Computing ref log probs...
# [Step 1] Computing rewards...
# [Step 1] Updating actor...
# [Step 1] actor/loss: xxx, actor/lr: xxx
```

---

### 7.4 验证成功后的下一步

确认快速验证通过后，使用完整配置进行正式训练：

```bash
# 1. 生成完整数据集（如果还没有）
python examples/data_preprocess/hotpotqa.py \
    --local_dir data/hotpotqa \
    --train_size 25600 \
    --val_size 128

# 2. 使用配置 E 进行正式训练
# 参考本文档 3.1 节的完整配置
```

---

## 相关文档

- [05_config_system.md](./05_config_system.md) - 配置系统说明
- [06_parameter_reference.md](./06_parameter_reference.md) - 参数详解手册
- [04_gpu_selection_guide.md](./04_gpu_selection_guide.md) - GPU 选择与问题排查
