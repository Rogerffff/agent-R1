# 常见问题排查

本文档汇总了 Agent-R1 训练过程中常见的问题和解决方案，帮助你快速定位和解决问题。

## 目录

- [训练不收敛](#训练不收敛)
- [OOM（内存溢出）问题](#oom内存溢出问题)
- [Loss 发散或 NaN](#loss-发散或-nan)
- [工具调用问题](#工具调用问题)
- [Ray 和分布式问题](#ray-和分布式问题)
- [WandB 问题](#wandb-问题)
- [常见错误信息](#常见错误信息)
- [调试技巧](#调试技巧)

---

## 训练不收敛

### 问题描述

训练指标（`critic/score/mean`）长时间没有提升，或者波动剧烈。

### 诊断步骤

```
1. 检查 WandB 曲线
   - critic/score/mean 是否有上升趋势？
   - actor/pg_clipfrac 是否在合理范围（0.1-0.3）？

2. 检查学习率
   - actor/lr 是否正确？
   - 是否太大或太小？

3. 检查奖励函数
   - critic/acc/mean 是否有变化？
   - 验证生成样本是否正确
```

### 解决方案

#### 情况 1：分数完全不动

```bash
# 可能原因：学习率过小或 KL 约束过强

# 增大学习率
actor_rollout_ref.actor.optim.lr=2e-6

# 减小 KL 约束
actor_rollout_ref.actor.kl_loss_coef=0.0001

# 检查数据
trainer.log_val_generations=10  # 查看生成样本
```

#### 情况 2：分数波动剧烈

```bash
# 可能原因：学习率过大或 batch size 过小

# 减小学习率
actor_rollout_ref.actor.optim.lr=5e-7

# 增大 batch size
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=32

# 增强 KL 约束
actor_rollout_ref.actor.kl_loss_coef=0.01
```

#### 情况 3：先上升后下降

```bash
# 可能原因：过拟合

# 增强正则化
actor_rollout_ref.actor.kl_loss_coef=0.01

# 减少训练步数
trainer.total_training_steps=200

# 使用学习率衰减
actor_rollout_ref.actor.optim.warmup_style=cosine
actor_rollout_ref.actor.optim.min_lr_ratio=0.1
```

---

## OOM（内存溢出）问题

### 问题描述

训练过程中出现 CUDA Out of Memory 错误。

### 错误信息示例

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

### 解决方案

#### 方案 1：减小批次大小

```bash
# 首先尝试减小微批大小
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1

# 然后减小小批次大小
actor_rollout_ref.actor.ppo_mini_batch_size=8

# 最后减小总批次大小
data.train_batch_size=16
```

#### 方案 2：启用内存优化

```bash
# 梯度检查点（必选）
actor_rollout_ref.model.enable_gradient_checkpointing=True

# 参数卸载
actor_rollout_ref.ref.fsdp_config.param_offload=True

# 优化器卸载（影响速度）
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

#### 方案 3：减少 vLLM 内存占用

```bash
# 降低 GPU 内存使用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.4

# 减少最大序列数
actor_rollout_ref.rollout.max_num_seqs=256
```

#### 方案 4：减少序列长度

```bash
data.max_prompt_length=2048
data.max_response_length=2048
data.max_response_length_single_turn=256
```

### 显存占用估算

```python
# 粗略估算
模型显存 = 模型参数 × 2 (BF16) × (1 + 优化器倍数)
激活显存 = batch_size × seq_len × hidden_size × layers × 2

# 例如 7B 模型
模型显存 ≈ 7B × 2 × 3 = 42 GB
激活显存 ≈ batch_size × 4096 × 4096 × 32 × 2 / 1024^3 GB
```

---

## Loss 发散或 NaN

### 问题描述

训练过程中 loss 变成 NaN 或无穷大。

### 错误信息示例

```
actor/pg_loss: nan
actor/grad_norm: inf
```

### 诊断步骤

```
1. 检查梯度范数
   - actor/grad_norm 是否突然增大？
   - 是否接近或超过 grad_clip？

2. 检查数据
   - 是否有异常样本？
   - 奖励值是否异常？

3. 检查模型权重
   - 是否有 NaN 权重？
```

### 解决方案

#### 方案 1：调整学习率

```bash
# 减小学习率
actor_rollout_ref.actor.optim.lr=1e-7

# 添加 warmup
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
```

#### 方案 2：增强梯度裁剪

```bash
# 减小梯度裁剪阈值
actor_rollout_ref.actor.grad_clip=0.5
critic.grad_clip=0.5
```

#### 方案 3：检查数据

```python
# 检查数据中的异常值
import pandas as pd

df = pd.read_parquet("data/train.parquet")
print(df.describe())

# 检查是否有空值
print(df.isnull().sum())
```

#### 方案 4：使用混合精度

```bash
# 确保使用 BF16（比 FP16 更稳定）
actor_rollout_ref.rollout.dtype=bfloat16
```

---

## 工具调用问题

### 问题 1：工具调用失败

**错误信息**：
```
Tool execution failed: search
```

**解决方案**：

```bash
# 检查工具服务是否启动
# 对于 search 工具，需要启动搜索服务
bash scripts/hotpotqa_search/start_server.sh

# 检查工具配置
tool.tools="['search']"
tool.env=nous
```

### 问题 2：工具响应过长

**现象**：响应被截断，模型无法获取完整信息

**解决方案**：
```bash
# 增加工具响应长度
tool.max_tool_response_length=2048
```

### 问题 3：工具格式错误

**现象**：模型生成的工具调用格式不正确

**解决方案**：
```bash
# 使用默认工具模板
data.use_default_tool_template=True

# 检查奖励函数的格式奖励
# 确保 format reward 正常工作
```

### 问题 4：工具调用轮数过多

**现象**：模型一直调用工具，不给出最终答案

**解决方案**：
```bash
# 减少最大轮数
tool.max_turns=3

# 增加对格式的奖励
# 在奖励函数中增加对最终答案的权重
```

---

## Ray 和分布式问题

### 问题 1：Ray 连接失败

**错误信息**：
```
RayConnectionError: Failed to connect to Ray cluster
```

**解决方案**：
```bash
# 重启 Ray
ray stop
ray start --head --port=6379

# 检查状态
ray status

# 如果有多个节点，确保所有节点已连接
```

### 问题 2：Worker 超时

**错误信息**：
```
RayTimeoutError: Timed out waiting for worker registration
```

**解决方案**：
```bash
# 增加超时时间
trainer.ray_wait_register_center_timeout=600

# 检查网络连接
ping <other_node_ip>
```

### 问题 3：NCCL 错误

**错误信息**：
```
RuntimeError: NCCL error in: ...
```

**解决方案**：
```bash
# 设置 NCCL 环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果没有 InfiniBand
export NCCL_P2P_DISABLE=1  # 某些硬件需要

# 检查 GPU 通信
nvidia-smi topo -m
```

### 问题 4：GPU 不均衡

**现象**：部分 GPU 满载，部分空闲

**解决方案**：
```bash
# 启用批次平衡
trainer.balance_batch=True

# 确保张量并行正确
actor_rollout_ref.rollout.tensor_model_parallel_size=<GPU数>
```

---

## WandB 问题

### 问题 1：WandB 连接失败

**错误信息**：
```
wandb: ERROR Failed to connect to wandb
```

**解决方案**：
```bash
# 检查 API key
echo $WANDB_API_KEY

# 离线模式
export WANDB_MODE=offline

# 训练完成后手动同步
wandb sync wandb/offline-run-*
```

### 问题 2：WandB 上传慢

**解决方案**：
```bash
# 减少上传频率
# WandB 默认每步上传，可以改为每 N 步

# 或者使用离线模式
export WANDB_MODE=offline
```

### 问题 3：WandB API Key 问题

**解决方案**：
```bash
# 设置环境变量
export WANDB_API_KEY=your_api_key

# 或者登录
wandb login

# 检查 .netrc 文件
cat ~/.netrc | grep wandb
```

---

## 常见错误信息

### 错误 1：模型加载失败

```
OSError: Can't load tokenizer for 'model_path'
```

**解决方案**：
```bash
# 检查模型路径
ls -la model_path/

# 确保包含必要文件
# config.json, tokenizer.json, model.safetensors 等
```

### 错误 2：数据格式错误

```
KeyError: 'prompt'
```

**解决方案**：
```bash
# 检查数据键名
data.prompt_key=prompt

# 检查数据格式
python -c "import pandas as pd; print(pd.read_parquet('data/train.parquet').columns)"
```

### 错误 3：配置解析错误

```
omegaconf.errors.ConfigAttributeError
```

**解决方案**：
```bash
# 检查配置语法
# 列表需要用引号
data.train_files="['file1.parquet', 'file2.parquet']"

# 布尔值
actor_rollout_ref.actor.use_kl_loss=True
```

### 错误 4：张量形状不匹配

```
RuntimeError: shape mismatch
```

**解决方案**：
```bash
# 检查模型和检查点是否匹配
# 确保 GPU 数量一致（FSDP 检查点与 GPU 数量绑定）
```

---

## 调试技巧

### 1. 打印详细日志

```bash
# 设置日志级别
export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=INFO

# Ray 日志
cat /tmp/ray/session_latest/logs/*.log
```

### 2. 小规模测试

```bash
# 使用小数据集快速验证
data.train_batch_size=8
trainer.total_training_steps=10
actor_rollout_ref.rollout.n_repeat=2
```

### 3. 检查中间结果

```python
# 在训练脚本中添加断点
import pdb; pdb.set_trace()

# 打印张量信息
print(f"Shape: {tensor.shape}, Min: {tensor.min()}, Max: {tensor.max()}")
```

### 4. 保存调试信息

```bash
# 保存验证生成样本
trainer.log_val_generations=10
trainer.validation_data_dir=./debug_logs
```

### 5. 分步排查

```
1. 先验证数据加载正确
2. 再验证模型能正常推理
3. 然后验证奖励计算正确
4. 最后验证训练循环正常
```

---

## 快速排查清单

当训练出现问题时，按以下顺序检查：

### 基础检查
- [ ] GPU 是否可用？（`nvidia-smi`）
- [ ] Ray 是否正常？（`ray status`）
- [ ] 数据文件是否存在？
- [ ] 模型路径是否正确？

### 配置检查
- [ ] GPU 数量配置正确？
- [ ] 批次大小配置合理？
- [ ] 序列长度配置正确？

### 资源检查
- [ ] 显存是否足够？
- [ ] CPU 内存是否足够？
- [ ] 磁盘空间是否足够？

### 网络检查（多节点）
- [ ] 节点间网络通畅？
- [ ] NCCL 配置正确？
- [ ] 端口未被占用？

---

## 获取帮助

如果以上方法都无法解决问题：

1. **检查 GitHub Issues**：https://github.com/0russwest0/Agent-R1/issues
2. **提交新 Issue**：附上完整的错误日志和配置
3. **社区讨论**：参与项目讨论

提交 Issue 时请包含：
- 完整的错误信息
- 训练配置（脚本或 config.yaml）
- 环境信息（GPU 型号、CUDA 版本、PyTorch 版本）
- 复现步骤
