# Agent-R1 训练流程概览与架构

本文档是 Agent-R1 Python 工具训练流程详解系列的第一篇，提供整体流程概览和架构说明。

## 目录

1. [训练脚本入口](#1-训练脚本入口)
2. [主入口函数详解](#2-主入口函数详解)
3. [TaskRunner 初始化流程](#3-taskrunner-初始化流程)
4. [组件初始化顺序](#4-组件初始化顺序)
5. [整体数据流图](#5-整体数据流图)
6. [架构图](#6-架构图)
7. [关键文件路径索引](#7-关键文件路径索引)

---

## 1. 训练脚本入口

### 1.1 运行命令

当我们运行 ReTool 训练时，入口是 shell 脚本：

```bash
bash examples/trainer/run_grpo_retool.sh
```

### 1.2 脚本内容解析

**文件**: `examples/trainer/run_grpo_retool.sh`

```bash
# 第 1-3 行：设置环境变量
export BASE_MODEL='russwest404/Qwen3-4B-ReTool-SFT'  # 基础模型路径（HuggingFace）
export PROJECT_NAME='retool'                          # 项目名称（用于日志）
export EXPERIMENT_NAME=grpo-qwen3-4b-sft             # 实验名称（用于日志）

# 第 5-47 行：调用主模块，传入配置参数
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \                    # 使用 GRPO 优势估计算法
    data.train_files=['data/retool/train.parquet'] \  # 训练数据路径
    data.val_files=['data/retool/test.parquet'] \     # 验证数据路径
    data.train_batch_size=128 \                       # 训练批次大小
    data.max_prompt_length=16384 \                    # 最大提示长度
    data.max_response_length=16384 \                  # 最大响应长度
    data.max_response_length_single_turn=8192 \       # 单轮最大响应长度
    data.use_default_tool_template=False \            # 不使用默认工具模板
    actor_rollout_ref.model.path=$BASE_MODEL \        # Actor 模型路径
    actor_rollout_ref.actor.optim.lr=1e-6 \           # Actor 学习率
    actor_rollout_ref.model.use_remove_padding=True \ # 移除填充优化
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \  # PPO mini batch 大小
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \  # 每 GPU micro batch
    actor_rollout_ref.actor.use_kl_loss=True \        # 使用 KL 损失
    actor_rollout_ref.actor.kl_loss_coef=0.001 \      # KL 损失系数
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \ # KL 损失类型
    actor_rollout_ref.model.enable_gradient_checkpointing=True \  # 梯度检查点
    actor_rollout_ref.actor.fsdp_config.param_offload=False \     # 不卸载参数到 CPU
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \ # 不卸载优化器
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \  # 张量并行大小
    actor_rollout_ref.rollout.name=vllm \             # 使用 vLLM 进行推理
    actor_rollout_ref.rollout.stop_token_ids=[] \     # 停止 token（在 yaml 中配置）
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \  # GPU 显存利用率
    actor_rollout_ref.rollout.n_repeat=5 \            # 每个 prompt 重复生成次数
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \  # 参考模型卸载到 CPU
    algorithm.kl_ctrl.kl_coef=0.001 \                 # KL 控制器系数
    trainer.logger=['console','wandb'] \              # 日志记录器
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \                       # 每节点 GPU 数
    trainer.nnodes=1 \                                # 节点数
    trainer.save_freq=-1 \                            # 保存频率（-1 表示不保存）
    trainer.test_freq=10 \                            # 验证频率
    trainer.total_epochs=2 \                          # 总训练轮数
    trainer.val_before_train=True \                   # 训练前验证
    trainer.log_val_generations=0 \                   # 记录验证生成数
    tool.max_turns=5 \                                # 最大工具调用轮数
    tool.tools=['python'] \                           # 使用的工具列表
    tool.env=retool \                                 # 工具环境名称
    tool.max_tool_response_length=512 $@              # 工具响应最大长度
```

### 1.3 关键参数说明

| 参数组 | 参数名 | 说明 | 示例值 |
|--------|--------|------|--------|
| **算法** | `algorithm.adv_estimator` | 优势估计算法 | `grpo` |
| **数据** | `data.train_batch_size` | 每批次原始样本数 | `128` |
| | `data.max_prompt_length` | 最大 prompt token 数 | `16384` |
| | `data.max_response_length` | 最大响应 token 数 | `16384` |
| **模型** | `actor_rollout_ref.model.path` | 预训练模型路径 | HuggingFace ID |
| | `actor_rollout_ref.rollout.n_repeat` | 每 prompt 重复采样数 | `5` |
| **训练** | `trainer.n_gpus_per_node` | 每节点 GPU 数 | `8` |
| | `trainer.total_epochs` | 训练轮数 | `2` |
| **工具** | `tool.max_turns` | 最大工具调用轮数 | `5` |
| | `tool.tools` | 工具列表 | `['python']` |
| | `tool.env` | 环境名称 | `retool` |

---

## 2. 主入口函数详解

### 2.1 Hydra 配置加载

**文件**: `agent_r1/src/main_agent.py`

```python
# 第 151-174 行：Hydra 装饰的主入口
@hydra.main(config_path="config", config_name="agent_trainer", version_base=None)
def main(config):
    """
    Hydra 自动完成以下工作：
    1. 加载 config/agent_trainer.yaml 作为基础配置
    2. 解析命令行参数（如 algorithm.adv_estimator=grpo）并覆盖配置
    3. 将合并后的配置传递给此函数
    """
    run_agent(config)
```

### 2.2 配置文件基础结构

**文件**: `agent_r1/src/config/agent_trainer.yaml`

配置文件包含以下主要部分：

```yaml
# 数据配置
data:
  train_files: []
  val_files: []
  train_batch_size: 256
  max_prompt_length: 1024
  max_response_length: 1024
  # ...

# Actor/Rollout/Reference 配置
actor_rollout_ref:
  hybrid_engine: True  # Actor 和 Rollout 共享同一模型
  model:
    path: null  # 模型路径
    # ...
  actor:
    strategy: fsdp  # 分布式策略
    # ...
  rollout:
    name: vllm  # 推理引擎
    stop: ["</code>"]  # 停止标记（ReTool 特有）
    # ...

# 算法配置
algorithm:
  adv_estimator: grpo  # 优势估计器
  gamma: 1.0           # 折扣因子
  lam: 1.0             # GAE lambda
  # ...

# 工具配置
tool:
  env: retool          # 工具环境
  tools: ['python']    # 工具列表
  max_turns: 5         # 最大轮数
  # ...

# 训练器配置
trainer:
  n_gpus_per_node: 8
  nnodes: 1
  total_epochs: 1
  # ...
```

### 2.3 Ray 集群初始化

**文件**: `agent_r1/src/main_agent.py`，第 182-224 行

```python
def run_agent(config) -> None:
    """
    初始化 Ray 分布式集群并启动训练任务。
    """
    if not ray.is_initialized():
        # 第 207-216 行：初始化本地 Ray 集群
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",  # 启用 tokenizer 并行
                    "NCCL_DEBUG": "WARN",              # 减少 NCCL 日志
                    "VLLM_LOGGING_LEVEL": "WARN",      # 减少 vLLM 日志
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    # 第 220-224 行：创建远程 Actor 执行训练
    runner = TaskRunner.remote()  # 在 Ray 集群中创建 TaskRunner
    ray.get(runner.run.remote(config))  # 等待训练完成
```

**为什么使用 TaskRunner？**

1. **资源隔离**：避免在 Ray head 节点运行繁重任务
2. **分布式调度**：确保训练任务被调度到有 GPU 的节点
3. **内存管理**：独立的进程空间，避免内存泄漏影响主进程

---

## 3. TaskRunner 初始化流程

### 3.1 TaskRunner 定义

**文件**: `agent_r1/src/main_agent.py`，第 232-515 行

```python
@ray.remote(num_cpus=1)  # 请求 1 个 CPU，确保不在 head 节点运行
class TaskRunner:
    """
    Ray 远程 Actor，负责执行主要的训练设置和启动逻辑。
    """

    def run(self, config):
        """执行完整的训练初始化和启动流程。"""
        # 详细步骤见下文
```

### 3.2 初始化步骤详解

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TaskRunner.run() 初始化流程                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  步骤 1: 打印配置                                                    │
│  └── OmegaConf.to_container(config, resolve=True)                  │
│                              │                                      │
│                              ▼                                      │
│  步骤 2: 下载/加载模型                                               │
│  └── copy_to_local(config.actor_rollout_ref.model.path)            │
│                              │                                      │
│                              ▼                                      │
│  步骤 3: 初始化 tokenizer 和 processor                              │
│  ├── tokenizer = hf_tokenizer(local_path, trust_remote_code)       │
│  └── processor = hf_processor(local_path, use_fast=True)           │
│                              │                                      │
│                              ▼                                      │
│  步骤 4: 选择 Worker 类（FSDP 或 Megatron）                          │
│  ├── ActorRolloutRefWorker                                         │
│  └── CriticWorker                                                  │
│                              │                                      │
│                              ▼                                      │
│  步骤 5: 配置资源池和角色映射                                         │
│  ├── role_worker_mapping: {Role.ActorRollout: Worker, ...}         │
│  └── resource_pool_spec: {"global_pool": [8] * nnodes}             │
│                              │                                      │
│                              ▼                                      │
│  步骤 6: 加载奖励函数                                                │
│  ├── reward_fn = load_reward_manager(config, tokenizer)            │
│  └── val_reward_fn = load_reward_manager(config, tokenizer)        │
│                              │                                      │
│                              ▼                                      │
│  步骤 7: 加载工具和环境                                              │
│  ├── tools = [_default_tool(name) for name in config.tool.tools]   │
│  └── env = _default_env(config.tool.env)(tools, max_length)        │
│                              │                                      │
│                              ▼                                      │
│  步骤 8: 创建数据集                                                  │
│  ├── train_dataset = create_rl_dataset(...)                        │
│  └── val_dataset = create_rl_dataset(...)                          │
│                              │                                      │
│                              ▼                                      │
│  步骤 9: 实例化训练器并开始训练                                       │
│  ├── trainer = RayAgentTrainer(...)                                │
│  ├── trainer.init_workers()                                        │
│  └── trainer.fit()                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 关键步骤代码

#### 步骤 4：选择 Worker 类

```python
# 第 310-345 行
if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
    # FSDP 策略：PyTorch 原生分布式
    from .fsdp_workers import ActorRolloutRefWorker, CriticWorker
    from verl.single_controller.ray import RayWorkerGroup

    actor_rollout_cls = (
        AsyncActorRolloutRefWorker
        if config.actor_rollout_ref.rollout.mode == "async"
        else ActorRolloutRefWorker
    )
    ray_worker_group_cls = RayWorkerGroup

elif config.actor_rollout_ref.actor.strategy == "megatron":
    # Megatron 策略：NVIDIA 大模型并行
    from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
    from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

    actor_rollout_cls = ActorRolloutRefWorker
    ray_worker_group_cls = NVMegatronRayWorkerGroup
```

#### 步骤 7：加载工具和环境

```python
# 第 437-458 行
# 加载工具（如 PythonTool）
tools = [_default_tool(name) for name in config.tool.tools]
# config.tool.tools = ['python']
# tools = [PythonTool()]

# 创建工具环境（如 ReToolEnv）
env = _default_env(config.tool.env)(
    tools=tools,
    max_tool_response_length=config.tool.max_tool_response_length
)
# config.tool.env = 'retool'
# env = ReToolEnv(tools=[PythonTool()], max_tool_response_length=512)
```

---

## 4. 组件初始化顺序

### 4.1 完整初始化顺序图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        组件初始化顺序                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ① 配置加载 (Hydra)                                                 │
│     └── agent_trainer.yaml + 命令行参数 → OmegaConf 配置对象        │
│                              │                                      │
│                              ▼                                      │
│  ② 模型下载 (HuggingFace)                                           │
│     └── russwest404/Qwen3-4B-ReTool-SFT → 本地路径                  │
│                              │                                      │
│                              ▼                                      │
│  ③ Tokenizer/Processor                                             │
│     ├── Tokenizer: 文本 ↔ Token IDs                                │
│     └── Processor: 多模态输入处理（可选）                            │
│                              │                                      │
│                              ▼                                      │
│  ④ 工具系统 (Tool System)                                           │
│     ├── PythonTool: 代码执行                                        │
│     └── ReToolEnv: 环境编排                                         │
│                              │                                      │
│                              ▼                                      │
│  ⑤ 奖励函数 (Reward Function)                                       │
│     └── AgentRewardManager: 规则/模型奖励计算                        │
│                              │                                      │
│                              ▼                                      │
│  ⑥ 数据集 (Dataset)                                                 │
│     ├── ToolRLDataset: 训练集                                       │
│     └── ToolRLDataset: 验证集                                       │
│                              │                                      │
│                              ▼                                      │
│  ⑦ 训练器 (RayAgentTrainer)                                         │
│     ├── 配置验证                                                    │
│     └── DataLoader 创建                                             │
│                              │                                      │
│                              ▼                                      │
│  ⑧ Worker 初始化 (init_workers)                                     │
│     ├── 创建 GPU 资源池                                             │
│     ├── 注册 Worker 类                                              │
│     ├── 创建共置 Worker                                             │
│     └── 初始化模型（Critic → RefPolicy → ActorRollout）             │
│                              │                                      │
│                              ▼                                      │
│  ⑨ 训练开始 (fit)                                                   │
│     └── 进入主训练循环                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Worker 初始化顺序的重要性

**为什么 ActorRollout 最后初始化？**

```python
# agent_r1/src/agent_ray_trainer.py，第 1480-1498 行

# 按顺序初始化模型：Critic -> RefPolicy -> ActorRollout
# ActorRollout 最后初始化，让 vLLM 更好地估计 KV cache 可用内存

if self.use_critic:
    self.critic_wg = all_wg["critic"]
    self.critic_wg.init_model()  # 先初始化 Critic

if self.use_reference_policy:
    self.ref_policy_wg = all_wg["ref"]
    self.ref_policy_wg.init_model()  # 再初始化参考策略

# ActorRollout 最后初始化
self.actor_rollout_wg = all_wg["actor_rollout"]
self.actor_rollout_wg.init_model()  # 最后初始化 ActorRollout
```

**原因**：
1. vLLM 在初始化时会探测可用 GPU 内存来分配 KV cache
2. 如果先初始化 ActorRollout（包含 vLLM），它会占用过多内存
3. 后初始化的 Critic 和 RefPolicy 可能因内存不足而失败
4. 反过来，先初始化 Critic 和 RefPolicy，vLLM 就能正确估计剩余内存

---

## 5. 整体数据流图

### 5.1 训练循环数据流

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              训练循环数据流                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  DataLoader                                                                     │
│     │                                                                           │
│     │ batch_dict: {                                                             │
│     │   "input_ids": tensor[B, L],                                              │
│     │   "attention_mask": tensor[B, L],                                         │
│     │   "raw_prompt_ids": list[list],                                           │
│     │   "reward_model": dict,                                                   │
│     │   ...                                                                     │
│     │ }                                                                         │
│     ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 1. DataProto.from_single_dict() → 转换为 DataProto                       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     │ DataProto: {                                                              │
│     │   batch: TensorDict{...},                                                 │
│     │   non_tensor_batch: {...},                                                │
│     │ }                                                                         │
│     ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 2. batch.repeat(n_repeat=5, interleave=True) → 重复采样                  │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     │ B → B * 5 (每个 prompt 生成 5 个响应)                                     │
│     ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 3. generation_manager.run_llm_loop() → 多轮生成                          │   │
│  │    ├── actor_rollout_wg.generate_sequences()                             │   │
│  │    ├── env.batch_step() → 执行工具                                       │   │
│  │    └── 更新滚动状态                                                       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     │ 新增字段: responses, action_mask, turns                                   │
│     ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 4. compute_reward() → 计算奖励                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     │ 新增字段: token_level_scores                                              │
│     ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 5. actor_rollout_wg.compute_log_prob() → 计算对数概率                    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     │ 新增字段: old_log_probs, entropys                                         │
│     ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 6. ref_policy_wg.compute_ref_log_prob() → 参考策略对数概率（可选）       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     │ 新增字段: ref_log_prob                                                    │
│     ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 7. critic_wg.compute_values() → 值函数估计（仅 GAE）                     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     │ 新增字段: values                                                          │
│     ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 8. compute_advantage() → 计算优势函数                                    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     │ 新增字段: advantages, returns                                             │
│     ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 9. actor_rollout_wg.update_actor() → 更新策略网络                        │   │
│  │    critic_wg.update_critic() → 更新值函数（仅 GAE）                      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     │ 输出: actor_output, critic_output (包含 loss 等指标)                      │
│     ▼                                                                           │
│  Logger (wandb/tensorboard) ← 记录指标                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 数据字段变化表

| 阶段 | 新增字段 | 形状 | 说明 |
|------|---------|------|------|
| **DataLoader** | `input_ids` | `(B, L)` | prompt token IDs |
| | `attention_mask` | `(B, L)` | 有效 token 掩码 |
| | `position_ids` | `(B, L)` | 位置编码 |
| | `raw_prompt_ids` | `list[list]` | 无填充的原始 IDs |
| | `reward_model` | `dict` | 奖励配置 |
| **repeat** | - | `(B*n, L)` | batch 扩展 n 倍 |
| **生成** | `responses` | `(B*n, R)` | 生成的响应 |
| | `action_mask` | `(B*n, R)` | 1=模型生成, 0=环境 |
| | `turns` | `(B*n,)` | 每样本轮数 |
| | `prompts` | `(B*n, L)` | 去填充的 prompt |
| **奖励** | `token_level_scores` | `(B*n, R)` | token 级别奖励 |
| **log_prob** | `old_log_probs` | `(B*n, R)` | 当前策略对数概率 |
| | `entropys` | `(B*n, R)` | 熵 |
| **ref** | `ref_log_prob` | `(B*n, R)` | 参考策略对数概率 |
| **values** | `values` | `(B*n, R)` | 值函数估计 |
| **advantage** | `advantages` | `(B*n, R)` | 优势函数 |
| | `returns` | `(B*n, R)` | 回报 |

其中：
- `B` = `train_batch_size` (例如 128)
- `n` = `n_repeat` (例如 5)
- `L` = `max_prompt_length` (例如 16384)
- `R` = `max_response_length` (例如 16384)

---

## 6. 架构图

### 6.1 分布式训练架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Agent-R1 分布式训练架构                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Driver Process (CPU)                              │   │
│  │                                                                          │   │
│  │  RayAgentTrainer                                                         │   │
│  │  ├── fit()              ← 主训练循环                                     │   │
│  │  ├── _validate()        ← 验证循环                                       │   │
│  │  ├── compute_advantage() ← 优势计算（轻量级）                            │   │
│  │  └── logger             ← 指标记录                                       │   │
│  │                                                                          │   │
│  └───────────────────────────────┬──────────────────────────────────────────┘   │
│                                  │                                              │
│                    ┌─────────────┼─────────────┐                                │
│                    │  RPC 调用   │  RPC 调用   │                                │
│                    ▼             ▼             ▼                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        GPU Worker Pool                                    │  │
│  │                                                                           │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    ActorRollout Worker (GPU 0-7)                     │ │  │
│  │  │                                                                      │ │  │
│  │  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │ │  │
│  │  │  │   Actor (FSDP)  │    │  Rollout (vLLM) │    │ RefPolicy (FSDP)│  │ │  │
│  │  │  │                 │    │                 │    │    (可选)        │  │ │  │
│  │  │  │ • update_actor  │    │ • generate      │    │ • compute_ref   │  │ │  │
│  │  │  │ • compute_log_p │    │ • sample        │    │   _log_prob     │  │ │  │
│  │  │  └─────────────────┘    └─────────────────┘    └─────────────────┘  │ │  │
│  │  │                                                                      │ │  │
│  │  └─────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                           │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Critic Worker (GPU 0-7) - 仅 GAE                  │ │  │
│  │  │                                                                      │ │  │
│  │  │  ┌─────────────────┐                                                │ │  │
│  │  │  │  Critic (FSDP)  │                                                │ │  │
│  │  │  │                 │                                                │ │  │
│  │  │  │ • update_critic │                                                │ │  │
│  │  │  │ • compute_values│                                                │ │  │
│  │  │  └─────────────────┘                                                │ │  │
│  │  │                                                                      │ │  │
│  │  └─────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                        External Services                                   │  │
│  │                                                                            │  │
│  │  ┌─────────────────────┐                                                  │  │
│  │  │  SandboxFusion      │  ← PythonTool 代码执行服务                        │  │
│  │  │  (localhost:8080)   │                                                  │  │
│  │  └─────────────────────┘                                                  │  │
│  │                                                                            │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 工具调用流程

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           工具调用流程（单样本示例）                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  User Prompt: "计算 1+1 的结果"                                                  │
│                                                                                 │
│  Turn 1:                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  LLM 生成:                                                               │   │
│  │  "<think>让我用 Python 计算这个</think>                                   │   │
│  │   <code>                                                                 │   │
│  │   ```python                                                              │   │
│  │   print(1+1)                                                             │   │
│  │   ```                                                                    │   │
│  │   </code>"                                                               │   │
│  │                                                                          │   │
│  │  action_mask: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]                  │   │
│  │               全部为 1（LLM 生成）                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                              │
│                                  │ env.step()                                   │
│                                  ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ReToolEnv 处理:                                                         │   │
│  │  1. extract_tool_calls() → 提取代码: "print(1+1)"                        │   │
│  │  2. PythonTool.execute() → 发送到 SandboxFusion                          │   │
│  │  3. 获取结果: "2"                                                        │   │
│  │  4. format_tool_response() → "<interpreter>\n2\n</interpreter>\n"       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                              │
│                                  │ 追加到序列                                   │
│                                  ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  工具响应:                                                                │   │
│  │  "<interpreter>                                                          │   │
│  │   2                                                                      │   │
│  │   </interpreter>"                                                        │   │
│  │                                                                          │   │
│  │  action_mask: [0,0,0,0,0,0,0,0,0,0,0,0,0,0]                              │   │
│  │               全部为 0（环境响应）                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                              │
│                                  │ 继续生成                                     │
│                                  ▼                                              │
│  Turn 2:                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  LLM 生成:                                                               │   │
│  │  "<think>计算结果是 2</think>                                            │   │
│  │   <answer>\boxed{2}</answer>"                                            │   │
│  │                                                                          │   │
│  │  action_mask: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]                      │   │
│  │               全部为 1（LLM 生成）                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  最终 action_mask: [1...1, 0...0, 1...1]                                        │
│                    Turn1  工具响应 Turn2                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键文件路径索引

### 7.1 入口与配置

| 文件 | 说明 |
|------|------|
| `examples/trainer/run_grpo_retool.sh` | 训练脚本入口 |
| `agent_r1/src/main_agent.py` | Python 主入口 |
| `agent_r1/src/config/agent_trainer.yaml` | 配置模板 |

### 7.2 训练核心

| 文件 | 说明 |
|------|------|
| `agent_r1/src/agent_ray_trainer.py` | 分布式训练器 |
| `agent_r1/src/fsdp_workers.py` | FSDP Worker 实现 |
| `agent_r1/src/core_algos.py` | 核心 RL 算法 |

### 7.3 生成与工具

| 文件 | 说明 |
|------|------|
| `agent_r1/llm_agent/generation.py` | 生成管理器 |
| `agent_r1/tool/base.py` | 工具基类 |
| `agent_r1/tool/envs/retool.py` | ReTool 环境 |
| `agent_r1/tool/tools/python_tool.py` | Python 工具 |

### 7.4 数据与奖励

| 文件 | 说明 |
|------|------|
| `agent_r1/src/agent_rl_dataset.py` | RL 数据集 |
| `agent_r1/src/reward.py` | 奖励函数加载 |
| `agent_r1/src/agent_reward_manager.py` | 奖励管理器 |
| `agent_r1/src/reward_score/retool.py` | ReTool 奖励 |

### 7.5 verl 框架

| 文件 | 说明 |
|------|------|
| `verl/verl/protocol.py` | DataProto 定义 |
| `verl/verl/single_controller/ray/` | Ray 分布式 |
| `verl/verl/utils/dataset/rl_dataset.py` | 基础数据集 |

---

## 下一步

继续阅读 [02_data_preparation.md](./02_data_preparation.md) 了解数据准备与加载的详细流程。
