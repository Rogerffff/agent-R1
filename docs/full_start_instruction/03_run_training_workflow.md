# 从租机到训练：端到端工作流程

本文档提供从 Vast.ai 租用实例到完成 RL 训练的完整步骤，每一步都包含验证方法。

---

## 流程总览

```
┌─────────────────┐
│  1. 租用实例     │
└────────┬────────┘
         ↓
┌─────────────────┐
│  2. 连接实例     │
└────────┬────────┘
         ↓
┌─────────────────┐
│  3. 验证环境     │
└────────┬────────┘
         ↓
┌─────────────────┐
│  4. 准备数据     │
└────────┬────────┘
         ↓
┌─────────────────┐
│  5. 配置 WandB   │
└────────┬────────┘
         ↓
┌─────────────────┐
│  6. 运行训练     │
└────────┬────────┘
         ↓
┌─────────────────┐
│  7. 监控与导出   │
└─────────────────┘
```

---

## 步骤 1：租用 Vast.ai 实例

### 1.1 选择模板

1. 登录 [cloud.vast.ai](https://cloud.vast.ai)
2. 点击 **Templates** → 选择你创建的 `agent-r1-training` 模板
3. 点击 **Use Template** 或 **Search Offers**

### 1.2 选择 GPU 实例

根据你的模型大小选择（详见 [04_gpu_selection_guide.md](./04_gpu_selection_guide.md)）：

| 模型 | 推荐配置 | Vast.ai 搜索条件 |
|------|----------|------------------|
| 1.5B (测试) | 2x RTX 4090 | GPU: 2x, VRAM: >=24GB |
| 1.5B-3B (训练) | 4x RTX 4090 | GPU: 4x, VRAM: >=24GB |
| 7B | 4x A100-40GB | GPU: 4x, VRAM: >=40GB |

### 1.3 确认配置

在租用前检查：

- [ ] GPU 数量和显存符合需求
- [ ] 磁盘空间 >= 200GB
- [ ] 网络速度 >= 200Mbps（下载模型需要）
- [ ] 价格在预算内

### 1.4 租用实例

点击 **Rent** 开始租用。实例启动需要 1-5 分钟。

**验证成功**：实例状态变为 `Running`

---

## 步骤 2：连接实例

### 方式 A：SSH 连接（推荐）

1. 在 Vast.ai 控制台找到实例的 SSH 命令：
   ```
   ssh -p <port> root@<ip_address>
   ```

2. 执行连接（可能需要确认主机指纹）

**验证成功**：看到命令行提示符

### 方式 B：Jupyter 连接

1. 点击实例的 **Open** 按钮
2. 在浏览器中打开 Jupyter 界面
3. 打开 Terminal

**验证成功**：Jupyter 界面正常显示

---

## 步骤 3：验证环境

### 3.1 检查 Provisioning 状态

```bash
# 查看 provisioning 日志
cat /var/log/provisioning.log 2>/dev/null || echo "日志文件不存在"

# 检查代码是否已克隆
ls -la /workspace/Agent-R1/
```

**预期输出**：看到 Agent-R1 目录结构

### 3.2 如果 Provisioning 未执行

手动执行环境配置：

```bash
# 克隆代码
cd /workspace
git clone https://github.com/0russwest0/Agent-R1.git
cd Agent-R1
git submodule update --init --recursive

# 安装依赖
cd verl && pip install -e . && cd ..
pip install vllm
pip install flash-attn --no-build-isolation
pip install FlagEmbedding faiss-cpu
```

### 3.3 验证 GPU 和环境

```bash
# 检查 GPU
nvidia-smi
```

**预期输出**：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx   CUDA Version: 12.x     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:xx:xx.x Off |                  N/A |
|   ...                                                                        |
+-----------------------------------------------------------------------------+
```

```bash
# 检查 Python 环境
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import vllm; print('vLLM OK')"
python -c "import ray; print('Ray OK')"
```

**预期输出**：
```
PyTorch: 2.x.x
CUDA: True
vLLM OK
Ray OK
```

**验证成功**：所有检查通过，无报错

---

## 步骤 4：准备训练数据

### 4.1 进入项目目录

```bash
cd /workspace/Agent-R1
```

### Quick Start: Try Default Search Tool on HotpotQA
#### 1. Install `FlagEmbedding` and `faiss`
```bash
pip3 install FlagEmbedding
pip3 install faiss-cpu

### 4.2 下载并处理 HotpotQA 数据集

```bash
# 创建数据目录
mkdir -p data/hotpotqa

# 运行预处理脚本
python examples/data_preprocess/hotpotqa.py \
    --local_dir ./data/hotpotqa \
    --download_method direct \
    --train_size 25600 \
    --val_size 128
```

**预期输出**：
```
Downloading HotpotQA dataset directly from source...
Downloading training data to ./data/hotpotqa/hotpot_train_v1.1.json...
100%|██████████| xxx/xxx [xx:xx<xx:xx, xxxiB/s]
...
```

**验证成功**：
```bash
ls -la data/hotpotqa/
ls -la data/hotpotqa_mini
# 应看到：train.parquet, validation.parquet
```

### 4.3 构建搜索索引

```bash
# 创建语料库目录
mkdir -p data/corpus/hotpotqa

# 下载语料库
wget https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus.jsonl.gz \
    -O data/corpus/hotpotqa/corpus.jsonl.gz

# 解压
gunzip -c data/corpus/hotpotqa/corpus.jsonl.gz > data/corpus/hotpotqa/hpqa_corpus.jsonl

# 构建 FAISS 索引（需要几分钟）
cd scripts/hotpotqa_search
python process_hotpotqa.py
cd ../..
```

**验证成功**：
```bash
ls -la data/corpus/hotpotqa/
# 应看到：hpqa_corpus.npy, index.bin 等文件
```

---

## 步骤 5：配置 WandB 监控

### 5.1 获取 WandB API Key

1. 访问 [wandb.ai](https://wandb.ai) 并登录
2. 点击右上角头像 → **Settings**
3. 找到 **API Keys** 部分，复制 key

### 5.2 登录 WandB

```bash
# 方式 1：交互式登录
wandb login

# 方式 2：使用环境变量
export WANDB_API_KEY=your_api_key_here
```

### 5.3 验证登录

```bash
python -c "import wandb; wandb.login(); print('WandB OK')"
```

**验证成功**：显示 `WandB OK` 且无报错

### 5.4 可选：离线模式

如果网络不稳定，可以使用离线模式：

```bash
export WANDB_MODE=offline
```

训练完成后可以手动同步：
```bash
wandb sync /path/to/wandb/run-xxx
```

---

## 步骤 6：运行 RL 训练

### 6.1 复制训练脚本

```bash
cd /workspace/Agent-R1

# GRPO 训练（推荐，显存需求较低）
cp examples/trainer/run_grpo_hotpotqa.sh ./

# 或 PPO 训练
# cp examples/trainer/run_ppo_hotpotqa.sh ./
```

### 6.2 修改训练参数（根据 GPU 数量）

编辑脚本以匹配你的 GPU 配置：

```bash
vim run_grpo_hotpotqa.sh
```

**2 GPU 配置**（测试用）：
```bash
trainer.n_gpus_per_node=2 \
actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
data.train_batch_size=64 \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
```

**4 GPU 配置**（默认）：
```bash
trainer.n_gpus_per_node=4 \
actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
data.train_batch_size=128 \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
```


### 6.3 启动训练

**使用 tmux 运行（推荐，防止 SSH 断开）**：

```bash
# 创建 tmux 会话
tmux new -s train

# 在 tmux 中运行
bash run_grpo_hotpotqa.sh

# 分离会话：Ctrl+B, 然后按 D
# 重新连接：tmux attach -t train
```

**直接运行**：

pip install cachetools

```bash
bash run_grpo_hotpotqa.sh
```

### 6.4 完整 GRPO 训练命令参考

```bash
export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME=grpo-qwen2.5-1.5b-instruct

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=2048
```

### 6.5 PPO 训练命令参考

PPO 需要额外的 Critic 模型配置：

```bash
export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME=ppo-qwen2.5-1.5b-instruct

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=gae \
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.stop=[] \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=5 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    tool.max_turns=5 \
    tool.tools=['search'] \
    tool.max_tool_response_length=2048
```

**验证训练启动成功**：
- 控制台显示初始化信息
- Ray 集群正常启动
- WandB 创建新的 run

---

## 步骤 7：监控训练与导出结果

### 7.1 查看 WandB 监控

1. 访问 [wandb.ai](https://wandb.ai) → 你的项目
2. 查看实时训练曲线

**关键指标说明**：

| 指标 | 含义 | 期望趋势 |
|------|------|----------|
| `reward/mean` | 平均奖励 | 上升 |
| `train/correct_ratio` | 训练集正确率 | 上升 |
| `actor/loss` | 策略损失 | 先升后降 |
| `actor/entropy` | 策略熵 | 缓慢下降 |
| `generation/num_turns` | 工具调用轮数 | 根据任务而定 |

### 7.2 控制台日志解读

正常训练日志示例：

```
[2024-01-15 10:00:00] Step 0/200 | reward: 0.12 | correct: 8.5% | loss: 2.34
[2024-01-15 10:05:00] Step 10/200 | reward: 0.25 | correct: 15.2% | loss: 1.89
[2024-01-15 10:10:00] Step 20/200 | reward: 0.38 | correct: 22.1% | loss: 1.56
...
```

### 7.3 保存检查点

修改训练参数以保存模型：

```bash
trainer.save_freq=50 \                    # 每 50 步保存一次
trainer.default_local_dir=checkpoints \   # 保存目录
```

检查点位置：
```
checkpoints/
└── hotpotqa/
    └── grpo-qwen2.5-1.5b-instruct/
        ├── global_step_50/
        ├── global_step_100/
        └── ...
```

### 7.4 导出模型

训练完成后，合并模型：

```bash
# 设置环境变量
export CHECKPOINT_DIR=checkpoints/hotpotqa/grpo-qwen2.5-1.5b-instruct/global_step_100
export HF_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct  # 原始模型路径
export TARGET_DIR=./merged_model

# 运行合并脚本
python3 verl/scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path $HF_MODEL_PATH \
    --local_dir $CHECKPOINT_DIR \
    --target_dir $TARGET_DIR
```

### 7.5 从 Vast.ai 下载

**方式 A：使用 scp**

```bash
# 在本地执行
scp -P <port> -r root@<ip>:/workspace/Agent-R1/merged_model ./
```

**方式 B：打包后下载**

```bash
# 在服务器上
cd /workspace/Agent-R1
tar -czvf model.tar.gz merged_model/

# 然后通过 Jupyter 界面下载
```

---

## 常用命令速查

### 环境管理

```bash
# 查看 GPU 使用
nvidia-smi
watch -n 1 nvidia-smi  # 实时监控

# 查看磁盘空间
df -h

# 查看内存使用
free -h
htop
```

### 训练管理

```bash
# 后台运行（tmux）
tmux new -s train
tmux attach -t train
tmux kill-session -t train

# 杀掉训练进程
pkill -f "python3 -m agent_r1"

# 清理 Ray
ray stop --force
```

### 数据管理

```bash
# 查看数据大小
du -sh data/*

# 清理缓存
rm -rf /root/.cache/pip
rm -rf /workspace/cache/huggingface/hub/*  # 谨慎：会删除模型
```

---

## 下一步

如果遇到问题或需要优化配置，请参考：

→ [04_gpu_selection_guide.md](./04_gpu_selection_guide.md) - GPU 选择与问题排查
