# Vast.ai Docker 模板创建指南

本文档将指导你在 Vast.ai 上创建一个适合运行 Agent-R1 强化学习训练的 Docker 模板。

---

## 什么是 Vast.ai 模板？

Vast.ai 模板是一个预配置的 Docker 环境定义，包含：
- **基础镜像**：操作系统 + CUDA + PyTorch 等
- **启动模式**：SSH / Jupyter 访问方式
- **环境变量**：自动配置的变量
- **端口映射**：外部访问端口

创建模板后，你可以快速启动多个相同配置的实例，无需每次手动配置。

---

## 推荐的基础镜像策略

### 方案 A：使用 Vast.ai 官方 PyTorch 镜像（推荐）

```
vastai/pytorch:2.4.0-cuda12.4.0-cudnn8-devel
```

**优点**：
- PyTorch + CUDA + cuDNN 已预装
- 镜像经过优化，启动快
- 与 Vast.ai 平台兼容性好

**为什么选择这个版本**：
- CUDA 12.4 兼容最新的 vLLM 和 flash-attn
- PyTorch 2.4 支持最新的优化特性
- devel 版本包含编译工具，方便安装 flash-attn

### 方案 B：使用 NVIDIA 官方镜像（备选）

```
nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
```

**适用场景**：需要完全自定义环境时使用

---

## 创建模板步骤

### 步骤 1：访问模板管理页面

1. 登录 [cloud.vast.ai](https://cloud.vast.ai)
2. 点击左侧菜单的 **Templates**
3. 点击 **Create Template** 或编辑现有模板

### 步骤 2：填写基本信息

| 字段 | 推荐值 | 说明 |
|------|--------|------|
| **Template Name** | `agent-r1-training` | 便于识别的名称 |
| **Template Description** | `Agent-R1 RL Training with verl` | 可选描述 |

### 步骤 3：配置 Docker 镜像

**Image Path:Tag** 字段填写：

```
vastai/pytorch:2.4.0-cuda12.4.0-cudnn8-devel
```

> **注意**：如果该镜像不可用，可尝试：
> - `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`
> - `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`

### 步骤 4：选择启动模式

选择 **Jupyter + SSH**（推荐）

| 模式 | 适用场景 |
|------|----------|
| Jupyter + SSH | 开发调试、交互式训练（推荐新手） |
| SSH only | 纯命令行操作 |
| Docker ENTRYPOINT | 自动化流水线 |

### 步骤 5：配置端口

添加以下端口映射（如需要）：

| 端口 | 用途 |
|------|------|
| 8888 | Jupyter Notebook |
| 22 | SSH 连接 |
| 6006 | TensorBoard（可选） |

### 步骤 6：配置环境变量

点击 **Add Environment Variable** 添加以下变量：

```bash
# 自动执行的初始化脚本 URL（见下一节文档）
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_GIST/main/provision_agent_r1.sh

# HuggingFace 缓存目录
HF_HOME=/workspace/cache/huggingface

# Torch Hub 缓存目录
TORCH_HOME=/workspace/cache/torch

# WandB API Key（可选，也可在运行时设置）
# WANDB_API_KEY=your_wandb_api_key

# 中国用户可设置 HuggingFace 镜像
# HF_ENDPOINT=https://hf-mirror.com
```

> **安全提示**：不要在公开模板中填写 API Key，应在运行时通过命令行设置。

### 步骤 7：配置磁盘空间

**Disk Space** 推荐设置：

| 用途 | 最小空间 | 推荐空间 |
|------|----------|----------|
| 测试小模型 (1.5B) | 100 GB | 200 GB |
| 训练中等模型 (7B) | 200 GB | 500 GB |
| 多模型/多数据集 | 500 GB | 1 TB |

空间用于：
- 模型权重下载（1.5B ~3GB，7B ~14GB）
- 数据集存储（HotpotQA ~2GB）
- 检查点保存（每个 ~模型大小）
- HuggingFace/Torch 缓存

### 步骤 8：保存模板

- **Create**：仅保存模板，稍后使用
- **Create & Use**：保存并立即租用实例

---

## 代码部署选项

### 选项 A：通过 Provisioning Script 自动克隆（推荐）

在 provisioning script 中加入 git clone 命令，实例启动时自动拉取代码。

**优点**：
- 每次获取最新代码
- 无需本地准备

**缺点**：
- 启动时间略长
- 依赖网络

### 选项 B：手动克隆（适合迭代开发）

启动实例后手动执行：

```bash
cd /workspace
git clone https://github.com/0russwest0/Agent-R1.git
cd Agent-R1
git submodule update --init --recursive
```

**优点**：
- 可以切换分支/版本
- 便于修改代码

---

## 模板配置检查清单

在创建模板前，确认以下项目：

- [ ] 镜像路径正确（包含 CUDA 12.x + PyTorch 2.x）
- [ ] 启动模式选择 Jupyter + SSH
- [ ] 磁盘空间 >= 200GB
- [ ] 已设置 HF_HOME 环境变量
- [ ] 如有 provisioning script，URL 可访问

---

## 常见问题

### Q1: 镜像拉取失败

**可能原因**：镜像名称拼写错误或不存在

**解决方法**：
1. 在 Docker Hub 搜索确认镜像存在
2. 尝试使用备选镜像

### Q2: 找不到 CUDA

**可能原因**：选择了 CPU-only 镜像

**解决方法**：确保镜像名称包含 `cuda` 关键字

### Q3: 磁盘空间不足

**现象**：下载模型时报错 "No space left on device"

**解决方法**：
1. 租用时选择更大的磁盘
2. 清理不需要的缓存：`rm -rf /root/.cache/pip`

---

## 下一步

模板创建完成后，请阅读下一篇文档：

→ [02_provisioning_script.md](./02_provisioning_script.md) - 编写自动化初始化脚本
