# Vast.ai 云 GPU 训练完整指南

本目录包含在 Vast.ai 云平台上运行 Agent-R1 强化学习训练的完整教程。

---

## 适用读者

- LLM 与云平台新手
- 希望在 Vast.ai 上运行 RL 训练的用户
- 需要了解 GPU 配置优化的研究人员

---

## 文档结构

| 文档 | 内容 | 阅读时间 |
|------|------|----------|
| [01_vast_template_setup.md](./01_vast_template_setup.md) | Vast.ai Docker 模板创建 | 10 分钟 |
| [02_provisioning_script.md](./02_provisioning_script.md) | 自动化环境配置脚本 | 15 分钟 |
| [03_run_training_workflow.md](./03_run_training_workflow.md) | 从租机到训练的完整流程 | 20 分钟 |
| [04_gpu_selection_guide.md](./04_gpu_selection_guide.md) | GPU 选择与问题排查 | 15 分钟 |
| [05_config_system.md](./05_config_system.md) | 配置系统详解（Hydra/OmegaConf） | 10 分钟 |
| [06_parameter_reference.md](./06_parameter_reference.md) | 参数详解手册 | 20 分钟 |
| [07_gpu_config_templates.md](./07_gpu_config_templates.md) | GPU 配置模板（完整脚本） | 15 分钟 |

---

## 快速开始路径

### 路径 A：首次使用（推荐）

按顺序阅读所有文档：

```
01_vast_template_setup.md
        ↓
02_provisioning_script.md
        ↓
03_run_training_workflow.md
        ↓
04_gpu_selection_guide.md（遇到问题时参考）
```

### 路径 B：急于上手

直接阅读核心流程：

```
03_run_training_workflow.md（主要步骤）
        ↓
04_gpu_selection_guide.md（遇到问题时参考）
```

---

## 前置要求

- Vast.ai 账户（需充值）
- WandB 账户（免费）
- 基本的 Linux 命令行知识
- 了解 Python 和 Git 基础

---

## 推荐配置（入门）

| 项目 | 推荐值 |
|------|--------|
| 模型 | Qwen2.5-1.5B-Instruct |
| 算法 | GRPO |
| GPU | 2-4x RTX 4090 |
| 磁盘 | 200GB |
| 预计费用 | ~$1-2/小时 |

---

## 相关项目文档

- [docs/getting_started/installation.md](../getting_started/installation.md) - 环境安装
- [docs/getting_started/quickstart.md](../getting_started/quickstart.md) - 快速开始
- [docs/readme/09_running_experiments.md](../readme/09_running_experiments.md) - 实验运行详解
- [docs/readme/00_overview.md](../readme/00_overview.md) - 项目总览

---

## 联系与反馈

如有问题，请参考：
- GitHub Issues: https://github.com/0russwest0/Agent-R1/issues
- 项目文档: https://github.com/0russwest0/Agent-R1
