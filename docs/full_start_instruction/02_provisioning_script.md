# Provisioning Script 编写指南

本文档介绍如何编写 Vast.ai 的 Provisioning Script，实现实例启动时自动配置环境。

---

## 什么是 Provisioning Script？

Provisioning Script 是一个 Shell 脚本，在 Vast.ai 实例**首次启动时自动执行**。它可以：

- 安装系统依赖
- 配置 Python 环境
- 克隆代码仓库
- 下载预训练模型
- 执行环境自检

---

## 脚本托管方式

Vast.ai 通过 URL 获取脚本，你需要将脚本托管到可公开访问的位置：

### 方式 1：GitHub Gist（推荐）

1. 访问 [gist.github.com](https://gist.github.com)
2. 创建新 Gist，粘贴脚本内容
3. 保存后点击 **Raw** 获取原始链接

**示例 URL**：
```
https://gist.githubusercontent.com/YOUR_USERNAME/GIST_ID/raw/provision_agent_r1.sh
```

### 方式 2：GitHub 仓库文件

将脚本放入仓库，使用 raw 链接：
```
https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/scripts/provision.sh
```

### 方式 3：其他文件托管服务

任何返回纯文本的 URL 都可以使用。

---

## 完整 Provisioning Script 示例

以下是一个完整的、可直接使用的脚本：

```bash
#!/bin/bash
# =============================================================================
# Agent-R1 Vast.ai Provisioning Script
#
# 🔄 适用镜像: verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3
# 
# 功能：自动配置 Agent-R1 强化学习训练环境
# 使用：将此脚本托管到 GitHub Gist，然后在 Vast.ai 模板中设置：
#       PROVISIONING_SCRIPT=<your_gist_raw_url>
# =============================================================================

set -eo pipefail  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# 1. 系统依赖安装（精简版 - 镜像已有大部分工具）
# =============================================================================
log_info "安装系统依赖..."
apt-get update && apt-get install -y \
    vim \
    htop \
    tmux
# ⬆️ 修改：移除 git wget curl unzip build-essential（镜像已有）

# =============================================================================
# 2. 配置缓存目录
# =============================================================================
log_info "配置缓存目录..."

export HF_HOME=${HF_HOME:-/workspace/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/workspace/cache/torch}
export PIP_CACHE_DIR=/workspace/cache/pip

mkdir -p $HF_HOME $TORCH_HOME $PIP_CACHE_DIR

# 持久化环境变量
cat >> /etc/bash.bashrc << 'EOF'
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export PIP_CACHE_DIR=/workspace/cache/pip
EOF

# =============================================================================
# 3. 克隆 Agent-R1 代码
# =============================================================================
log_info "克隆 Agent-R1 代码库..."

cd /workspace

if [ -d "Agent-R1" ]; then
    log_warn "Agent-R1 目录已存在，更新代码..."
    cd Agent-R1
    git pull origin main
else
    git clone https://github.com/0russwest0/Agent-R1.git
    cd Agent-R1
fi

# 初始化子模块 (verl)
log_info "初始化 verl 子模块..."
git submodule update --init --recursive

# =============================================================================
# 4. 安装 Python 依赖（针对官方镜像优化）
# =============================================================================
log_info "安装项目 verl 依赖..."

cd /workspace/Agent-R1/verl
pip install -e . --cache-dir $PIP_CACHE_DIR

# ⬆️ 保留：安装项目中的 verl 子模块版本

# ❌ 删除以下行（镜像已包含 vLLM 0.6.3）
# log_info "安装 vLLM..."
# pip install vllm --cache-dir $PIP_CACHE_DIR

# ⬇️ 修改：检查 flash-attn，如果没有才安装
log_info "检查 flash-attn..."
if python -c "import flash_attn" 2>/dev/null; then
    log_info "flash-attn 已安装，跳过"
else
    log_info "安装 flash-attn（可能需要几分钟）..."
    pip install flash-attn --no-build-isolation --cache-dir $PIP_CACHE_DIR
fi

log_info "安装 HotpotQA 搜索依赖..."
pip install FlagEmbedding faiss-cpu --cache-dir $PIP_CACHE_DIR

# =============================================================================
# 5. 配置 WandB（可选）
# =============================================================================
if [ -n "$WANDB_API_KEY" ]; then
    log_info "配置 WandB..."
    pip install wandb --cache-dir $PIP_CACHE_DIR
    wandb login $WANDB_API_KEY
else
    log_warn "WANDB_API_KEY 未设置，跳过 WandB 配置"
    log_warn "你可以稍后运行: wandb login <your_api_key>"
fi

# =============================================================================
# 6. 环境自检
# =============================================================================
log_info "========== 环境自检 =========="

# 检查 GPU
log_info "检查 GPU..."
nvidia-smi || { log_error "nvidia-smi 失败！"; exit 1; }

# 检查 Python 版本
log_info "检查 Python..."
python --version

# 检查关键库
log_info "检查 PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

log_info "检查 vLLM..."
python -c "import vllm; print(f'vLLM: {vllm.__version__}')" || log_warn "vLLM 导入失败，可能需要手动检查"

log_info "检查 Ray..."
python -c "import ray; print(f'Ray: {ray.__version__}')"

log_info "检查 transformers..."
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 检查 flash-attn
log_info "检查 flash-attn..."
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" || log_warn "flash-attn 导入失败，部分功能可能受限"

# =============================================================================
# 7. 完成提示
# =============================================================================
log_info "=========================================="
log_info "环境配置完成！"
log_info "=========================================="
log_info ""
log_info "使用镜像: verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3"
log_info "代码位置: /workspace/Agent-R1"
log_info "下一步："
log_info "  1. cd /workspace/Agent-R1"
log_info "  2. 准备数据: python examples/data_preprocess/hotpotqa.py --local_dir ./data/hotpotqa"
log_info "  3. 运行训练: bash run_grpo_hotpotqa.sh"
log_info ""
log_info "详细文档请参考: docs/full_start_instruction/"

```

---

## 脚本各部分说明

### 1. 系统依赖

```bash
apt-get install -y git wget curl vim htop tmux unzip build-essential
```

| 包 | 用途 |
|-----|------|
| git | 克隆代码 |
| wget/curl | 下载文件 |
| build-essential | 编译 flash-attn 需要 |
| htop | 监控资源使用 |
| tmux | 后台运行训练 |

### 2. 缓存目录

```bash
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
```

**为什么要配置缓存目录？**
- `/workspace` 是 Vast.ai 的持久化目录
- 模型下载到此目录后，实例重启不会丢失
- 避免重复下载大模型

### 3. Python 依赖安装顺序

安装顺序很重要：

1. **verl** - 核心框架
2. **vLLM** - 推理引擎
3. **flash-attn** - 注意力加速（需要 verl 先装好）
4. **FlagEmbedding + faiss** - 搜索工具依赖

### 4. 环境自检

自检脚本验证以下内容：

| 检查项 | 成功标志 |
|--------|----------|
| nvidia-smi | 显示 GPU 信息 |
| torch.cuda.is_available() | 返回 True |
| import vllm | 无报错 |
| import ray | 无报错 |
| import flash_attn | 无报错（可选） |

---

## 依赖版本对照表

根据 `verl/setup.py`，以下是主要依赖：

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.10 | 推荐版本 |
| PyTorch | 2.x | CUDA 12.x 兼容 |
| vLLM | <=0.8.3 | 推理引擎 |
| Ray | >=2.10 | 分布式计算 |
| transformers | latest | HuggingFace 模型 |
| tensordict | <=0.6.2 | 数据结构 |
| flash-attn | latest | 注意力加速 |
| pyarrow | >=15.0.0 | Parquet 文件读写 |

---

## 在 Vast.ai 中使用脚本

### 步骤 1：托管脚本

将上面的脚本保存为 Gist 或仓库文件，获取 raw URL。

### 步骤 2：设置环境变量

在 Vast.ai 模板的 Environment Variables 中添加：

```
PROVISIONING_SCRIPT=https://gist.githubusercontent.com/your_username/gist_id/raw/provision_agent_r1.sh
```

### 步骤 3：验证执行

实例启动后，检查日志确认脚本执行成功：

```bash
# 查看 provisioning 日志
cat /var/log/provisioning.log

# 或检查标志文件
ls /workspace/Agent-R1/
```

---

## 自定义脚本

### 添加 HuggingFace 镜像（中国用户）

在脚本开头添加：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 预下载模型

在脚本末尾添加：

```bash
# 预下载 Qwen2.5-1.5B-Instruct
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', torch_dtype='auto')"
```

### 跳过耗时步骤

如果 flash-attn 安装太慢，可以注释掉：

```bash
# pip install flash-attn --no-build-isolation  # 跳过 flash-attn
```

训练仍可运行，但速度会略慢。

---

## 常见问题

### Q1: 脚本执行失败

**检查方法**：
```bash
cat /var/log/provisioning.log
```

**常见原因**：
- 脚本 URL 无法访问
- 脚本语法错误（先在本地测试）
- 网络问题导致 pip 安装失败

### Q2: flash-attn 安装失败

**现象**：编译错误或 CUDA 版本不匹配

**解决方法**：
```bash
# 确保 CUDA 环境正确
nvcc --version

# 尝试指定版本安装
pip install flash-attn==2.5.8 --no-build-isolation
```

### Q3: 环境变量未生效

**检查方法**：
```bash
echo $HF_HOME
```

**解决方法**：重新加载环境
```bash
source /etc/bash.bashrc
```

---

## 下一步

脚本配置完成后，请阅读下一篇文档：

→ [03_run_training_workflow.md](./03_run_training_workflow.md) - 从租机到训练的完整流程
