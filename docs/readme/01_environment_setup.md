# 环境配置指南

本文档详细介绍如何配置 Agent-R1 的开发环境。

---

## 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GPU (16GB+ 显存) | 多卡 A100/H100 (80GB) |
| 内存 | 32GB | 128GB+ |
| 存储 | 100GB SSD | 500GB+ NVMe SSD |

### 软件要求

| 软件 | 版本 |
|------|------|
| Python | 3.10 |
| CUDA | 11.8+ / 12.x |
| Git | 2.x |
| Conda/Miniconda | 最新版 |

---

## 安装步骤

### 步骤 1: 克隆仓库

```bash
git clone https://github.com/0russwest0/Agent-R1.git
cd Agent-R1
```

### 步骤 2: 创建 Conda 环境

```bash
# 创建 Python 3.10 虚拟环境
conda create -n verl python==3.10
conda activate verl
```

### 步骤 3: 安装 verl 框架

Agent-R1 基于 [verl](https://github.com/volcengine/verl) 框架构建，需要先安装它：

```bash
# 初始化 git submodule
git submodule update --init --recursive

# 进入 verl 目录并安装
cd verl
pip3 install -e .
cd ..
```

### 步骤 4: 安装 vLLM

vLLM 用于高效的 LLM 推理：

```bash
# 安装最新稳定版
pip3 install vllm
```

### 步骤 5: 安装 Flash Attention

Flash Attention 用于加速注意力计算：

```bash
pip3 install flash-attn --no-build-isolation
```

### 步骤 6: 安装其他依赖

```bash
# 安装检索相关依赖（用于多跳 QA 任务）
pip3 install FlagEmbedding
pip3 install faiss-cpu  # 或 faiss-gpu 如果有 GPU
```

---

## 搜索服务配置

Agent-R1 的多跳 QA 任务需要配置搜索服务。这里提供两种方案：

### 方案 A: HotpotQA 搜索索引（推荐新手）

适用于 HotpotQA 数据集的轻量级搜索：

```bash
# 1. 创建语料目录
mkdir -p data/corpus/hotpotqa

# 2. 下载 HotpotQA 语料
wget https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus.jsonl.gz \
    -O data/corpus/hotpotqa/corpus.jsonl.gz

# 3. 解压语料文件
gunzip -c data/corpus/hotpotqa/corpus.jsonl.gz > data/corpus/hotpotqa/hpqa_corpus.jsonl

# 4. 构建搜索索引
cd scripts/hotpotqa_search
python process_hotpotqa.py
cd ../../
```

该脚本会：
- 加载语料数据
- 使用 `BAAI/bge-large-en-v1.5` 模型生成 embeddings
- 构建 FAISS 索引用于高效相似度搜索
- 将 embeddings 和索引文件保存到 `data/corpus/hotpotqa` 目录

### 方案 B: KILT Wikipedia 搜索服务（推荐进阶）

适用于更大规模的多跳 QA 任务：

```bash
# 1. 进入搜索服务目录
cd scripts/kilt_search_server

# 2. 安装依赖
pip3 install -r requirements.txt

# 3. 下载并处理 KILT 语料（约 36M 段落）
python process_kilt.py

# 4. 启动搜索 API 服务
bash run_search_api.sh
```

启动后，搜索服务默认监听 `http://localhost:8000`。

### 方案 C: Wikipedia 搜索服务

另一种 Wikipedia 搜索服务实现：

```bash
cd scripts/wiki_search_server

# 处理 Wikipedia 数据
python process_wiki.py

# 启动搜索服务
bash run_search_api.sh
```

---

## 验证安装

### 检查基本依赖

```python
# 在 Python 中执行
import torch
import vllm
import ray

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"vLLM version: {vllm.__version__}")
```

### 检查 verl 安装

```python
import verl
print(f"verl imported successfully")
```

### 检查 Agent-R1 模块

```python
from agent_r1.tool.base import BaseTool, BaseToolEnv
from agent_r1.src.agent_ray_trainer import RayAgentTrainer
print("Agent-R1 modules imported successfully")
```

---

## 常见问题解决

### 问题 1: Flash Attention 安装失败

**错误信息**: `error: invalid command 'bdist_wheel'`

**解决方案**:
```bash
pip install wheel setuptools
pip3 install flash-attn --no-build-isolation
```

### 问题 2: CUDA 版本不兼容

**错误信息**: `CUDA driver version is insufficient`

**解决方案**:
1. 检查 CUDA 驱动版本: `nvidia-smi`
2. 安装对应版本的 PyTorch:
```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# 或
pip3 install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

### 问题 3: Git submodule 初始化失败

**错误信息**: `fatal: not a git repository`

**解决方案**:
```bash
# 确保在仓库根目录
cd Agent-R1

# 重新初始化
git submodule deinit -f --all
git submodule update --init --recursive
```

### 问题 4: 内存不足 (OOM)

**错误信息**: `CUDA out of memory`

**解决方案**:
1. 减小 batch size
2. 启用梯度检查点
3. 使用更小的模型
4. 配置 `gpu_memory_utilization` 参数（在训练脚本中）

### 问题 5: Ray 初始化失败

**错误信息**: `Ray failed to connect to GCS`

**解决方案**:
```bash
# 停止现有 Ray 进程
ray stop

# 重新启动
ray start --head
```

---

## 环境变量配置

根据需要设置以下环境变量：

```bash
# HuggingFace 缓存目录（可选）
export HF_HOME=/path/to/huggingface/cache

# 模型下载镜像（中国用户推荐）
export HF_ENDPOINT=https://hf-mirror.com

# CUDA 设备可见性
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定可用 GPU

# Ray 临时目录
export RAY_TMPDIR=/tmp/ray

# 日志级别
export VERL_LOG_LEVEL=INFO
```

可以将这些添加到 `~/.bashrc` 或 `~/.zshrc` 中：

```bash
echo 'export HF_HOME=/path/to/huggingface/cache' >> ~/.bashrc
source ~/.bashrc
```

---

## 多 GPU 配置

### 单节点多卡

```bash
# 设置可见 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 在训练脚本中配置
# --nnodes 1 --nproc_per_node 4
```

### 多节点配置

```bash
# 主节点
ray start --head --port=6379

# 工作节点
ray start --address='<head-node-ip>:6379'
```

---

## 下一步

环境配置完成后，请继续阅读：
- [02_core_concepts.md](./02_core_concepts.md) - 了解核心概念
- [08_data_preparation.md](./08_data_preparation.md) - 准备训练数据
- [09_running_experiments.md](./09_running_experiments.md) - 运行实验
