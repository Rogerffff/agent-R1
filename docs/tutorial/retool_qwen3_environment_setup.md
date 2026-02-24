# ReTool 实验环境配置与训练指南

本文档针对使用 Qwen3-4B-ReTool-SFT 模型运行 ReTool 实验时的环境配置问题，提供完整的解决方案。

## 1. 问题分析

### 1.1 核心兼容性问题

| 组件 | 当前版本 | 所需版本 | 状态 |
|------|---------|---------|------|
| verl (Agent-R1 submodule) | 0.2.0.dev | - | 基础版本 |
| vLLM (推荐镜像中) | 0.6.3 | >= 0.8.4 | **不兼容** |
| Qwen3 模型要求 | - | vLLM >= 0.8.4（本文档使用 0.8.5） | 需升级 vLLM |
| transformers | - | >= 4.51.0 | Qwen3 需要 |

**关键发现**：
- 镜像 `verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3` 中的 vLLM 0.6.3 **不支持 Qwen3**
- verl 0.2.0 的代码实际支持 vLLM 0.7.0+（见 `verl/verl/third_party/vllm/__init__.py` 第43-48行）
- **注意**：setup.py 的 `extras_require["vllm"]` 中限制 `vllm<=0.8.3`，使用 `pip install -e verl[vllm]` 会强制降级 vLLM，因此**必须**使用 `pip install -e verl --no-deps` 避免此问题

### 1.2 可选方案对比

| 方案 | 描述 | 优点 | 缺点 | 推荐度 |
|------|------|------|------|--------|
| **A: 手动配置环境 + vLLM 0.8.5** | 不用预装镜像，手动安装 vLLM 0.8.5 | 保持 verl 0.2.0，风险可控 | 需要手动配置 | **推荐** |
| B: 升级 verl 到最新版 | 使用 verl 0.7.0 + vLLM 0.12.0 | 官方支持，功能最全 | 可能需改 Agent-R1 代码 | 中等 |
| C: 降级模型到 Qwen2.5 | 使用 Qwen2.5 + vLLM 0.6.3 | 环境稳定，无风险 | 模型较旧 | 保守选择 |

---

## 2. 推荐方案：手动配置环境 + vLLM 0.8.5

### 2.1 环境要求总结

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Python | **3.10 或 3.11** | vLLM 0.8 对 3.12 支持不稳定 |
| CUDA | >= 12.1 (推荐 12.4) | vLLM 0.8.x 需要 |
| PyTorch | 2.4.0 - 2.6.x | 与 vLLM 兼容，注意 CXX11 ABI 设置 |
| vLLM | **0.8.5** | Qwen3 需要 >= 0.8.4（verl 文档验证版本为 0.8.3） |
| flash-attn | 2.8.x | **必须从源码编译**，详见 3.5 节 |
| transformers | >= 4.51.0 | Qwen3 支持 |
| ray | >= 2.10 | 分布式训练 |
| opentelemetry | >= 1.39.0 | Ray 2.53+ 需要，vLLM 的版本警告可忽略 |

### 2.2 Vast.ai 推荐镜像

**推荐基础镜像**（预装 CUDA + PyTorch，手动安装其余）：

| 镜像 | CUDA | PyTorch | 说明 |
|------|------|---------|------|
| `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` | 12.4 | 2.4.0 | **首选**，版本匹配 |
| `nvcr.io/nvidia/pytorch:24.08-py3` | 12.6 | 2.4.0 | NGC 官方，稳定 |
| `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | 12.6 | 2.6.0 | 社区镜像，vLLM 已装 |

**注意**：如果使用 `hiyouga/verl` 镜像，需将 vLLM 从 0.8.3 升级到 0.8.5。

---

## 3. 完整环境配置步骤

### 3.1 Vast.ai Template Settings

**Docker Image**:
```
pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
```

**Environment Variables**:
```
HF_HOME=/workspace/.cache/huggingface
HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
HUGGING_FACE_HUB_TOKEN=<你的HF_TOKEN>
WANDB_API_KEY=<你的WANDB_KEY>
WANDB_MODE=online
PYTHONUNBUFFERED=1
# VLLM_USE_V1=1  # 可选：启用 vLLM V1 引擎（实验性，可能提升性能）
```

**Disk Space**: >= 150GB

**GPU 配置**:
- 首选：2 x 48GB 4090 或 4 x 24GB 4090
- 备选：2 x 80GB A100

### 3.2 Provisioning Script（实例启动时执行）

```bash
#!/usr/bin/env bash
set -euo pipefail

export WORKSPACE=/workspace
mkdir -p "$WORKSPACE"
mkdir -p "$WORKSPACE/.cache/huggingface/hub"
mkdir -p "$WORKSPACE/.cache/huggingface/datasets"
cd "$WORKSPACE"

# 克隆 Agent-R1
if [ ! -d Agent-R1 ]; then
  git clone https://github.com/0russwest0/Agent-R1.git
fi

cd Agent-R1
git submodule update --init --recursive
mkdir -p data/retool

# 克隆 SandboxFusion
cd "$WORKSPACE"
if [ ! -d SandboxFusion ]; then
  git clone https://github.com/bytedance/SandboxFusion.git
fi

echo "=== Provisioning complete ==="
```

### 3.3 手动环境配置（SSH 进入实例后执行）

#### 依赖冲突分析

| verl 0.2.0 依赖 | vLLM 0.8.5 要求 | 冲突风险 | 解决方案 |
|----------------|----------------|---------|---------|
| `tensordict<=0.6.2` | 无直接要求 | **低** | 安装 tensordict==0.6.2（避免 0.7.x 的 ForkingPickler 错误） |
| `transformers`（无版本） | vLLM 自动安装兼容版本 | **无** | 0.8.5 通常为 4.51.3 |
| `ray[default]>=2.10` | 无冲突 | **无** | 兼容 |

**关键发现**：
- [ForkingPickler 错误](https://github.com/volcengine/verl/issues/700) 发生在 tensordict 0.7.x，需要使用 0.6.2
- verl 的 `tensordict<=0.6.2` 约束实际上是正确的，可避免此问题
- 按正确顺序安装可避免版本冲突

#### 推荐安装顺序（避免冲突）

```bash
cd /workspace/Agent-R1

# ===== 1. 先安装 vLLM 0.8.5（会自动安装 transformers 4.51.3）=====
pip3 install vllm==0.8.5

# ===== 2. 强制安装 tensordict 0.6.2（避免 ForkingPickler 错误）=====
pip3 install tensordict==0.6.2

# ===== 3. 安装 verl（使用 --no-deps 避免覆盖已安装的正确版本）=====
pip3 install -e verl --no-deps

# ===== 4. 安装 verl 的其他依赖（排除已安装的）=====
pip3 install accelerate codetiming datasets dill hydra-core numpy pandas peft \
  "pyarrow>=15.0.0" pybind11 pylatexenc "ray[default]>=2.10" torchdata wandb

# ===== 5. 安装 flash-attn（重要：必须从源码编译）=====
# 详见下方 "3.5 flash-attn 安装指南" 部分
# 简化命令（适用于大多数情况）：
cd /tmp && rm -rf flash-attention && \
git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git && \
cd flash-attention && \
FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=4 \
pip install . --no-build-isolation --no-cache-dir
# 注意：编译需要 15-30 分钟；如遇问题请参考 3.5 节详细说明

# ===== 6. 安装 Agent-R1 特有依赖 =====
# 方式 A：从已克隆的 SandboxFusion 源码安装（更稳）
#pip3 install -e /workspace/SandboxFusion 不应该在主环境中安装
# 从本地安装轻量级客户端
pip install /workspace/SandboxFusion/scripts/client/
# 方式 B：如果 PyPI 可用
# pip3 install sandbox-fusion

# ===== 7. 验证安装 =====
python3 - <<'PY'
import sys
print("=" * 50)
print("环境检查")
print("=" * 50)

# PyTorch
import torch
print(f"[PyTorch] {torch.__version__}")
print(f"[CUDA available] {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[CUDA version] {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

# vLLM
import vllm
print(f"\n[vLLM] {vllm.__version__}")

# transformers
import transformers
print(f"[transformers] {transformers.__version__}")

# tensordict（关键：必须是 0.6.x）
import tensordict
print(f"[tensordict] {tensordict.__version__}")
if tensordict.__version__.startswith("0.7"):
    print("  ⚠️ 警告: tensordict 0.7.x 可能导致 ForkingPickler 错误!")
    print("  运行: pip install tensordict==0.6.2")

# verl
import verl
print(f"[verl] installed")

# flash-attn
try:
    import flash_attn
    print(f"[flash-attn] {flash_attn.__version__}")
except ImportError:
    print("[flash-attn] NOT INSTALLED")

# ray
import ray
print(f"[ray] {ray.__version__}")

# 测试 verl 的 vLLM 集成
print("\n--- vLLM 集成测试 ---")
try:
    from verl.third_party.vllm import LLM, parallel_state
    print("[verl vLLM 集成] OK")
except ImportError as e:
    print(f"[verl vLLM 集成] 错误: {e}")

print("\n" + "=" * 50)
PY
```

#### 可选：检查依赖版本兼容性

```bash
# 检查是否有版本冲突
pip check

# 查看关键包版本
pip show vllm transformers tensordict ray | grep -E "^(Name|Version):"
```

### 3.4 SandboxFusion 配置

ReTool 的 Python 工具需要 SandboxFusion 服务运行在 `localhost:8080`。

```bash
cd /workspace/SandboxFusion


# 创建名为 "sandbox-runtime" 的 conda 环境
conda create -n sandbox-runtime python==3.11
conda activate sandbox-runtime

# 安装依赖
pip install -r runtime/python/requirement.txt
pip install poetry
poetry install

# 准备并运行沙箱
mkdir -p docs/build
make run-online
```

**验证 SandboxFusion**:
```bash
python3 - <<'PY'
from sandbox_fusion import set_sandbox_endpoint, run_code, RunCodeRequest
set_sandbox_endpoint("http://localhost:8080")
res = run_code(request=RunCodeRequest(run_timeout=5, code="print(1+1)", language="python"), max_attempts=1)
print(res.status, res.run_result.stdout if res.run_result else res.message)
PY
```

### 3.5 flash-attn 安装指南（重要）

flash-attn 是一个 CUDA 扩展，必须与 PyTorch 的 C++ ABI 完全匹配才能正常工作。

#### 3.5.1 问题背景

| 问题 | 原因 | 表现 |
|------|------|------|
| **ABI 不兼容** | PyTorch 使用旧 ABI（`CXX11_ABI=0`），但 flash-attn 使用新 ABI | `undefined symbol: _ZN3c105Error...cxx11...` |
| **预编译 wheel 有问题** | GitHub Release 的 `cxx11abiFALSE` wheel 实际包含 CXX11 符号 | 即使下载"正确"的 wheel 也会报错 |
| **pip 缓存问题** | pip 会缓存之前编译的 wheel 并重复使用 | 重装后问题仍然存在 |

#### 3.5.2 检查 PyTorch ABI 设置

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CXX11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"
```

- 如果输出 `CXX11 ABI: False`，说明 PyTorch 使用**旧 ABI**
- 如果输出 `CXX11 ABI: True`，说明 PyTorch 使用**新 ABI（CXX11）**

**大多数 PyTorch 官方发布版本使用旧 ABI（False）**，包括 PyTorch 2.4.0、2.5.x、2.6.0。

#### 3.5.3 正确的安装方法：从源码编译

**关键环境变量**：
- `FLASH_ATTENTION_FORCE_BUILD=TRUE` - **必须**，强制从源码编译，禁止下载预编译 wheel
- `TORCH_CUDA_ARCH_LIST` - GPU 架构，4090 用 `8.9`，A100 用 `8.0`
- `MAX_JOBS` - 并行编译任务数，建议设为 CPU 核心数的一半

**完整安装命令**：

```bash
# 1. 清理旧安装和缓存
pip uninstall -y flash-attn
pip cache purge
rm -rf /root/.cache/pip/wheels/*

# 2. 克隆源码
cd /tmp
rm -rf flash-attention
git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git

# 3. 从源码编译（15-30 分钟）
cd flash-attention
FLASH_ATTENTION_FORCE_BUILD=TRUE \
TORCH_CUDA_ARCH_LIST="8.9" \
MAX_JOBS=4 \
pip install . --no-build-isolation --no-cache-dir

# 4. 验证安装
python -c "from flash_attn.flash_attn_interface import flash_attn_func; print('flash-attn 安装成功！')"
```

#### 3.5.4 验证 ABI 兼容性

安装后检查 `.so` 文件的符号：

```bash
# 检查 CXX11 符号数量
nm -D /opt/conda/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so 2>/dev/null | grep -c "cxx11"

# 如果 PyTorch CXX11 ABI = False，上述命令应该输出 0
# 如果输出 > 0，说明 ABI 不匹配，需要重新编译
```

#### 3.5.5 常见 GPU 架构列表

| GPU | CUDA 架构 | `TORCH_CUDA_ARCH_LIST` |
|-----|----------|------------------------|
| RTX 4090 / 4080 / 4070 | Ada Lovelace | `8.9` |
| RTX 3090 / 3080 / 3070 | Ampere | `8.6` |
| A100 / A30 | Ampere | `8.0` |
| H100 | Hopper | `9.0` |
| 多 GPU 类型 | - | `8.0;8.6;8.9` |

#### 3.5.6 监控编译进度

编译过程中可以使用以下命令监控：

```bash
# 如果编译在后台运行并输出到日志文件
tail -f /tmp/flash_build.log

# 查看当前进度（格式如 [45/73]）
grep -o '\[[0-9]*/73\]' /tmp/flash_build.log | tail -1
```

---

## 4. 数据准备与训练

### 4.1 生成 ReTool 数据

```bash
cd /workspace/Agent-R1
python examples/data_preprocess/retool.py --local_dir ./data/retool
```

### 4.2 下载 SFT 模型（可选预下载）

```bash
huggingface-cli download russwest404/Qwen3-4B-ReTool-SFT \
  --local-dir /workspace/.cache/huggingface/hub/russwest404-Qwen3-4B-ReTool-SFT
```

```bash
#这是 Agent-R1 的一个漏掉的依赖，用于 ReTool 的数学答案评分。
pip install mathruler
```

### 4.3 训练配置

复制并使用训练脚本：
```bash
cp examples/trainer/run_grpo_retool_2x48g_4090.sh ./run_grpo_retool.sh
```

**关键配置说明**（脚本已包含）：
- `actor_rollout_ref.rollout.stop='["</code>"]'` - ReTool 停止符
- `tool.env=retool` - 使用 ReTool 环境
- `tool.tools=['python']` - 启用 Python 工具

**配置约束**（参考 `verl/docs/README_vllm0.8.md`）：
- 启用 CUDA Graph：`enforce_eager=False` + `free_cache_engine=False`
- 释放 KV Cache：`enforce_eager=True` + `free_cache_engine=True`
- **禁止组合**：`enforce_eager=False` + `free_cache_engine=True`（会触发断言错误）

### 4.4 启动训练

**2 x 48GB 4090 配置**：


实际训练命令
```bash
bash run_grpo_retool_2x48g_4090.sh \
  trainer.n_gpus_per_node=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  data.train_batch_size=16 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True
```

**4 x 24GB 4090 配置**：
```bash
bash run_grpo_retool.sh \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False
```

> **重要**：`enforce_eager=False`（启用 CUDA Graph）时，**必须**同时设置 `free_cache_engine=False`，否则会触发 `vllm_rollout_spmd.py` 中的断言错误。如需释放 KV Cache，请改用 `enforce_eager=True` + `free_cache_engine=True`。

---

## 5. 常见问题排查

### 5.1 依赖相关问题

| 问题 | 错误信息 | 解决方案 |
|------|---------|---------|
| **ForkingPickler 错误** | `ImportError: cannot import name 'ForkingPickler' from 'torch.multiprocessing.reductions'` | `pip install tensordict==0.6.2` |
| **vLLM 不支持 Qwen3** | `ValueError: ... model not supported` | 确认 vLLM >= 0.8.4: `pip show vllm` |
| **transformers 版本过低** | `AttributeError: module 'transformers' has no attribute ...` | `pip install transformers>=4.51.0` |
| **依赖版本冲突** | `pip check` 显示冲突 | 按推荐顺序重新安装，使用 `--no-deps` |
| **flash-attn ABI 不兼容** | `undefined symbol: _ZN3c105Error...cxx11...` | 必须从源码编译，详见 3.5 节 |
| **Ray Dashboard 启动失败** | `TypeError: Meter.create_histogram() got an unexpected keyword argument` | 升级 opentelemetry: `pip install "opentelemetry-api>=1.39.0" "opentelemetry-sdk>=1.39.0"` |
| **opentelemetry 版本冲突警告** | `vllm requires opentelemetry-api<1.27.0` | 可忽略，不影响训练核心功能 |

### 5.2 训练相关问题

| 问题 | 解决方案 |
|------|---------|
| 模型没有触发工具调用 | 确认 `actor_rollout_ref.rollout.stop='["</code>"]'` |
| 工具执行失败/超时 | 检查 SandboxFusion 是否在 `localhost:8080` 运行 |
| 显存不足 | 降低 `data.max_prompt_length`、`data.train_batch_size`，开启 `fsdp_config.param_offload=True` |
| HF 下载失败 | 检查 `HUGGING_FACE_HUB_TOKEN` 环境变量 |

### 5.3 依赖冲突修复脚本

如果遇到严重依赖冲突，可以运行以下脚本重置环境：

```bash
# 卸载冲突包
pip uninstall -y vllm transformers tensordict

# 按正确顺序重新安装
pip install vllm==0.8.5
pip install tensordict==0.6.2

# 验证
pip check
python -c "from verl.third_party.vllm import LLM; print('OK')"
```

---

## 6. 备选方案

### 6.1 方案 B：升级 verl 到最新版

如果遇到 verl 0.2.0 + vLLM 0.8.5 的兼容性问题，可考虑升级 verl：

```bash
# 不使用 submodule，直接安装最新 verl
pip install verl
# 或指定版本
pip install verl==0.7.0
```

**注意**：这可能需要修改 Agent-R1 的代码以适配新版 verl API。

### 6.2 方案 C：降级到 Qwen2.5

如果追求稳定性，可使用 Qwen2.5 模型 + 原有镜像：

1. 使用镜像：`verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3`
2. 基座模型改为：`Qwen/Qwen2.5-7B-Instruct` 或类似 Qwen2.5 模型
3. 可能需要自行进行 SFT 来获得工具调用能力

---

## 7. 关键文件路径

| 文件 | 说明 |
|------|------|
| `examples/trainer/run_grpo_retool_2x48g_4090.sh` | 训练脚本 |
| `agent_r1/src/config/agent_trainer.yaml` | 配置模板 |
| `agent_r1/tool/envs/retool.py` | ReTool 环境实现 |
| `agent_r1/tool/tools/python_tool.py` | Python 工具（调用 SandboxFusion） |
| `verl/verl/third_party/vllm/__init__.py` | vLLM 版本兼容层 |

---

## 8. 参考资料

### 8.1 官方文档
- [vLLM Qwen3 Usage Guide](https://github.com/vllm-project/vllm/issues/17327)
- [verl vLLM 0.8+ 升级指南](https://verl.readthedocs.io/en/latest/README_vllm0.8.html)
- [verl 安装文档](https://verl.readthedocs.io/en/latest/start/install.html)
- [vLLM GPU 安装文档](https://docs.vllm.ai/en/v0.8.0/getting_started/installation/gpu.html)
- [Qwen verl 训练指南](https://qwen.readthedocs.io/en/latest/training/verl.html)

### 8.2 依赖冲突相关
- [ForkingPickler 错误 (verl #700)](https://github.com/volcengine/verl/issues/700) - tensordict 0.7.x 与 PyTorch 2.6+ 冲突
- [verl vLLM 0.8+ README](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) - 官方升级指南
- [vLLM 0.8.5 Release Notes](https://github.com/vllm-project/vllm/releases/tag/v0.8.5) - transformers 4.51.3 依赖

### 8.3 版本兼容性总结

```
verl 0.2.0 + vLLM 0.8.5 兼容性矩阵:

┌─────────────────┬──────────────┬─────────────────────────────────────────┐
│ 包              │ 推荐版本      │ 注意事项                                 │
├─────────────────┼──────────────┼─────────────────────────────────────────┤
│ Python          │ 3.10/3.11    │ 3.12 支持不稳定，不推荐                   │
│ PyTorch         │ 2.4.0-2.6.x  │ 镜像预装，注意 CXX11 ABI 设置            │
│ CUDA            │ 12.4         │ 镜像预装                                 │
│ vLLM            │ 0.8.5        │ Qwen3 需要 >= 0.8.4                     │
│ transformers    │ 4.51.3       │ vLLM 自动安装                            │
│ tensordict      │ 0.6.2        │ 必须! 避免 0.7.x                        │
│ ray             │ >= 2.10      │ verl 要求                                │
│ flash-attn      │ 2.8.x        │ 必须从源码编译! 见 3.5 节                │
│ opentelemetry   │ >= 1.39.0    │ Ray 2.53+ 需要，vLLM 警告可忽略          │
└─────────────────┴──────────────┴─────────────────────────────────────────┘

配置约束:
- enforce_eager=False 时，free_cache_engine 必须为 False
- 使用 pip install -e verl --no-deps，不要用 verl[vllm] extra
- flash-attn 必须用 FLASH_ATTENTION_FORCE_BUILD=TRUE 从源码编译
```
