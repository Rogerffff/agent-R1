# ReTool GRPO 训练（Vast.ai 实例）

这份文档基于 `docs/tutorial/retool.md`，补充了 Vast.ai 实例配置、SandboxFusion 启动、数据准备与 GRPO 实验参数建议，适用于使用 `verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3` 镜像进行 ReTool 训练。

## 0. 前置准备

- HuggingFace 账号与访问令牌（用于拉取 `russwest404/Qwen3-4B-ReTool-SFT` 和 `haizhongzheng/DAPO-Math-17K-cleaned`）。
- 如需记录到 W&B，准备 `WANDB_API_KEY`；否则可改为仅控制台日志。
- 确保实例磁盘空间 >= 150GB（模型缓存 + 数据集 + 日志）。

## 1. Vast.ai 实例与模板配置

### 1.1 实例选择建议

- **首选**：2 x 48GB 4090（或 4 x 24GB 4090）。  
- **备选**：2 x 80GB A100 / 1 x 80GB A100（更稳但贵）。
- 只是实验可降低 step/epoch，优先选便宜机型。

### 1.2 Template Settings 推荐值

参考 Vast.ai 的 Advanced Setup / Template Settings 文档，在模板里填入以下关键项：

**Docker Image**
```
verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3
```

**Environment Variables（示例）**
```
HF_HOME=/workspace/.cache/huggingface
HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
HUGGING_FACE_HUB_TOKEN=你的HF_TOKEN
WANDB_API_KEY=你的WANDB_KEY   # 没有就删掉
WANDB_MODE=online             # 或 offline
PYTHONUNBUFFERED=1
```

**PROVISIONING_SCRIPT（示例）**
用于实例启动时初始化代码仓库与子模块。不要在这里跑训练，只做准备工作。

**方式 A：使用 URL（推荐）**
```
https://gist.github.com/Rogerffff/7ad6d857c693dc4f7aa3be967a51828c/raw/agentR1-retool.sh
```

**方式 B：直接粘贴脚本内容**
```bash
#!/usr/bin/env bash
set -euo pipefail

export WORKSPACE=/workspace
mkdir -p "$WORKSPACE"
mkdir -p "$WORKSPACE/.cache/huggingface/hub"
mkdir -p "$WORKSPACE/.cache/huggingface/datasets"
cd "$WORKSPACE"

if [ ! -d Agent-R1 ]; then
  git clone https://github.com/0russwest0/Agent-R1.git
fi

cd Agent-R1
git submodule update --init --recursive
mkdir -p data/retool

# SandboxFusion 克隆到 /workspace 下（与 Agent-R1 同级）
cd "$WORKSPACE"
if [ ! -d SandboxFusion ]; then
  git clone https://github.com/bytedance/SandboxFusion.git
fi
```

> **提示**：使用 Gist URL 时，Vast.ai 会自动下载并执行脚本。如果你 fork 了自己的版本，记得更新 URL 或脚本中的仓库地址。

##### 可选：验证 verl 安装

```bash
python3 - <<'PY'
import importlib.util
print("verl installed:", importlib.util.find_spec("verl") is not None)
PY
```

如果上面显示未安装，再执行：
```bash
pip3 install -e verl
```

##### 安装 flash-attn

flash-attn 用于加速注意力计算，显著提升训练速度：

```bash
pip3 install flash-attn --no-build-isolation
```

> **注意**：flash-attn 编译需要较长时间（10-30分钟），且需要 CUDA 环境。如果镜像已预装可跳过。

##### 验证 flash-attn 安装

```bash
python3 - <<'PY'
try:
    import flash_attn
    print(f"flash-attn installed: True, version: {flash_attn.__version__}")
except ImportError:
    print("flash-attn installed: False")
PY
```

##### 综合环境检查

运行以下脚本一次性检查所有关键依赖：

```bash
python3 - <<'PY'
import sys
print("=" * 50)
print("Agent-R1 环境检查")
print("=" * 50)

# Python 版本
print(f"\n[Python] {sys.version}")

# CUDA 和 GPU
try:
    import torch
    print(f"\n[PyTorch] {torch.__version__}")
    print(f"[CUDA available] {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[CUDA version] {torch.version.cuda}")
        print(f"[GPU count] {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
except ImportError:
    print("[PyTorch] NOT INSTALLED")

# vLLM
try:
    import vllm
    print(f"\n[vLLM] {vllm.__version__}")
except ImportError:
    print("\n[vLLM] NOT INSTALLED")

# Ray
try:
    import ray
    print(f"[Ray] {ray.__version__}")
except ImportError:
    print("[Ray] NOT INSTALLED")

# transformers
try:
    import transformers
    print(f"[transformers] {transformers.__version__}")
except ImportError:
    print("[transformers] NOT INSTALLED")

# verl
try:
    import verl
    print(f"[verl] installed")
except ImportError:
    print("[verl] NOT INSTALLED")

# flash-attn
try:
    import flash_attn
    print(f"[flash-attn] {flash_attn.__version__}")
except ImportError:
    print("[flash-attn] NOT INSTALLED")

# HuggingFace token
import os
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
print(f"\n[HF Token] {'已设置' if hf_token else '未设置'}")

# Wandb
wandb_key = os.environ.get("WANDB_API_KEY")
print(f"[Wandb Key] {'已设置' if wandb_key else '未设置'}")

print("\n" + "=" * 50)
PY
```
python3 - <<'PY'
try:
    import verl
    print(f"[verl] installed")
except ImportError:
    print("[verl] NOT INSTALLED")
PY


如果有任何依赖显示 `NOT INSTALLED`，请先安装再继续。

## 2. SandboxFusion 启动与验证

ReTool 的 PythonTool 需要本地 SandboxFusion 服务，默认请求 `http://localhost:8080`，因此沙箱必须运行在同一实例并监听 8080。

### 2.1 安装并启动 SandboxFusion

如果镜像内已有 SandboxFusion 依赖，可跳过安装步骤，只需启动服务。以下是通用流程：

```bash
cd /workspace/SandboxFusion

# 方式 A：conda（如果镜像自带）
conda create -n sandbox-runtime python==3.11
conda activate sandbox-runtime
pip install -r runtime/python/requirement.txt
pip install poetry
poetry install

# 启动服务
mkdir -p docs/build
make run-online
```

如果镜像没有 conda，可以改用 venv：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r runtime/python/requirement.txt
pip install poetry
poetry install

mkdir -p docs/build
make run-online
```

### 2.2 健康检查（确认 8080 可用）

```bash
python - <<'PY'
from sandbox_fusion import set_sandbox_endpoint, run_code, RunCodeRequest
set_sandbox_endpoint("http://localhost:8080")
res = run_code(request=RunCodeRequest(run_timeout=5, code="print(1+1)", language="python"), max_attempts=1)
print(res.status, res.run_result.stdout if res.run_result else res.message)
PY
```

如果输出含 `2`，说明沙箱可用。否则先检查 SandboxFusion 是否在运行、端口是否被占用。

## 3. 数据与模型准备

### 3.1 生成 ReTool 数据

```bash
cd /workspace/Agent-R1
python examples/data_preprocess/retool.py --local_dir ./data/retool
```

该脚本会自动下载 DAPO-Math-17K 数据集并生成：
- `data/retool/train.parquet`
- `data/retool/test.parquet`

### 3.2 下载 SFT 模型（可选预下载）

训练会自动从 HuggingFace 拉取 `russwest404/Qwen3-4B-ReTool-SFT`。如果想提前缓存：

```bash
huggingface-cli download russwest404/Qwen3-4B-ReTool-SFT \
  --local-dir /workspace/.cache/huggingface/hub/russwest404-Qwen3-4B-ReTool-SFT
```

## 4. GRPO 训练配置与启动

建议先复制脚本再改：

```bash
cp examples/trainer/run_grpo_retool.sh ./run_grpo_retool.sh
```

### 4.1 必改项：`</code>` 停止符

ReTool 需要在生成 `</code>` 时停止并触发工具调用。两种方式任选其一：

**方式 A：命令行覆盖（推荐）**
在启动脚本时追加：
```
actor_rollout_ref.rollout.stop='["</code>"]'
```

**方式 B：修改配置文件**
在 `agent_r1/src/config/agent_trainer.yaml` 的 `actor_rollout_ref.rollout` 增加：
```yaml
stop: ["</code>"]
```

### 4.2 2 x 48GB 4090 建议配置（实验版）

```bash
bash run_grpo_retool.sh \
  trainer.n_gpus_per_node=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  data.train_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.max_response_length_single_turn=1024 \
  actor_rollout_ref.rollout.n_repeat=2 \
  tool.max_turns=3 \
  trainer.total_epochs=1 \
  trainer.test_freq=50 \
  actor_rollout_ref.rollout.stop='["</code>"]'
```

### 4.3 4 x 24GB 4090 建议配置（实验版）

```bash
bash run_grpo_retool.sh \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  data.train_batch_size=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.max_response_length_single_turn=1024 \
  actor_rollout_ref.rollout.n_repeat=3 \
  tool.max_turns=3 \
  trainer.total_epochs=1 \
  trainer.test_freq=50 \
  actor_rollout_ref.rollout.stop='["</code>"]'
```

### 4.4 训练日志与 Wandb

如果不想用 Wandb，可在启动时加：
```
trainer.logger=['console']
```
或设置环境变量 `WANDB_MODE=offline`。

## 5. 常见问题排查

1. **模型没有触发工具调用**  
   确认 `actor_rollout_ref.rollout.stop` 已包含 `</code>`。

2. **工具执行失败 / 超时**  
   检查 SandboxFusion 是否运行在 `localhost:8080`，以及 `python_tool.py` 是否可访问。

3. **显存不足**  
   降低 `data.max_prompt_length` / `data.max_response_length`，或减小 `data.train_batch_size`、`actor_rollout_ref.rollout.n_repeat`，必要时开启 `actor_rollout_ref.actor.fsdp_config.param_offload=True`。

4. **HF 下载失败**  
   检查 `HUGGING_FACE_HUB_TOKEN` 是否设置，或手动 `huggingface-cli login`。
