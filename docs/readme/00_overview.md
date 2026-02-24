# Agent-R1 代码库总览

## 项目简介

**Agent-R1** 是一个开源框架，旨在加速**强化学习（RL）**与**智能代理（Agent）**交叉领域的研究与开发。该框架采用**端到端强化学习**在特定环境中训练代理。开发者只需定义特定领域的工具和奖励函数，即可将 Agent-R1 扩展到自己的应用场景，无需复杂的工作流工程。

> **论文**: [Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning](https://arxiv.org/abs/2511.14460)

### 核心特性

- **多轮工具调用**: 在完整交互轨迹上进行端到端强化学习，使代理能从动作序列中学习
- **多工具协调**: 训练代理有效协调和使用多个工具来解决复杂任务
- **过程奖励**: 根据每个工具调用的有效性分配奖励，通过归一化与结果奖励平衡
- **自定义工具与环境**: 兼容主流 LLM 工具调用格式，易于扩展自定义工具和场景
- **多种 RL 算法**: 支持 `PPO`、`GRPO`、`REINFORCE++`、`RLOO` 等多种强化学习算法
- **多模态支持**: 兼容视觉-语言模型（VLMs）和多模态强化学习

---

## 项目目录结构

```
Agent-R1/
├── agent_r1/                    # 核心 Python 包
│   ├── llm_agent/              # LLM 代理模块
│   │   ├── generation.py       # 多轮生成管理器 (ToolGenerationManager)
│   │   └── tensor_helper.py    # 张量操作辅助函数
│   │
│   ├── src/                    # 核心训练与 RL 逻辑
│   │   ├── main_agent.py       # 主入口点 (run_agent)
│   │   ├── agent_ray_trainer.py # Ray 分布式训练器 (RayAgentTrainer)
│   │   ├── agent_dp_actor.py   # 数据并行 Actor
│   │   ├── agent_dp_critic.py  # 数据并行 Critic
│   │   ├── agent_reward_manager.py # 奖励管理器
│   │   ├── agent_rl_dataset.py # RL 数据集处理
│   │   ├── core_algos.py       # 核心 RL 算法实现
│   │   ├── fsdp_workers.py     # FSDP 分布式工作器
│   │   ├── reward.py           # 奖励函数加载
│   │   ├── sglang_rollout.py   # SGLang rollout 实现
│   │   ├── vllm_rollout_spmd.py # vLLM SPMD rollout 实现
│   │   ├── metric_utils.py     # 指标工具
│   │   ├── config/             # 配置文件
│   │   │   └── agent_trainer.yaml # 训练配置模板
│   │   └── reward_score/       # 奖励计算函数
│   │       ├── gsm8k.py        # GSM8K 数学任务奖励
│   │       ├── math.py         # 通用数学任务奖励
│   │       ├── qa_em_and_format.py # QA 精确匹配与格式奖励
│   │       └── retool.py       # ReTool 任务奖励
│   │
│   ├── tool/                   # 工具框架
│   │   ├── base.py             # BaseTool 和 BaseToolEnv 基类
│   │   ├── utils.py            # 工具辅助函数
│   │   ├── tools/              # 工具实现
│   │   │   ├── python_tool.py  # Python 代码执行工具
│   │   │   ├── search_tool.py  # 通用搜索工具
│   │   │   └── wiki_search_tool.py # Wikipedia 搜索工具
│   │   └── envs/               # 工具环境实现
│   │       ├── nous.py         # Nous 环境 (多跳 QA)
│   │       ├── mathtir.py      # 数学任务环境
│   │       └── retool.py       # ReTool 环境 (代码执行)
│   │
│   └── vllm_infer/             # 推理工具
│       ├── run.py              # 推理运行脚本
│       ├── config.py           # 推理配置
│       └── chat.py             # 交互式对话接口
│
├── docs/                       # 文档
│   ├── getting_started/        # 入门指南
│   ├── algorithm/              # 算法说明
│   ├── tutorial/               # 教程
│   ├── inference/              # 推理指南
│   └── readme/                 # 详细代码文档 (本系列)
│
├── examples/                   # 示例
│   ├── trainer/                # 训练脚本
│   │   ├── run_ppo_hotpotqa.sh
│   │   ├── run_ppo_multihopqa.sh
│   │   ├── run_ppo_retool.sh
│   │   ├── run_grpo_hotpotqa.sh
│   │   ├── run_grpo_multihopqa.sh
│   │   ├── run_grpo_retool.sh
│   │   └── run_rpp_hotpotqa.sh
│   └── data_preprocess/        # 数据预处理
│       ├── hotpotqa.py
│       ├── 2wikimultihopqa.py
│       ├── musique.py
│       ├── gsm8k.py
│       └── retool.py
│
├── scripts/                    # 实用脚本
│   ├── model_merge.sh          # 模型合并
│   ├── vllm_serve.sh           # vLLM 服务
│   ├── hotpotqa_search/        # HotpotQA 搜索索引
│   ├── wiki_search_server/     # Wikipedia 搜索服务
│   └── kilt_search_server/     # KILT 搜索服务
│
├── verl/                       # verl 框架 (git submodule)
├── image/                      # 项目图片
└── README.md                   # 项目说明
```

---

## 核心概念

### 1. MDP 框架扩展

Agent-R1 将传统的 MDP（马尔可夫决策过程）框架扩展到 LLM Agent 场景：

| 组件 | 静态 LLM | LLM Agent |
|------|----------|-----------|
| **状态空间 (S)** | 当前文本序列 | 多轮交互历史 + 环境反馈 |
| **动作空间 (A)** | 生成下一个 token | 生成 token + 工具调用命令 |
| **状态转移 (P)** | 确定性：追加 token | 随机性：工具执行结果不确定 |
| **奖励函数 (R)** | 稀疏：仅最终奖励 | 密集：过程奖励 + 结果奖励 |

### 2. Tool 与 ToolEnv

- **Tool**: 工具执行器，封装具体功能（如 API 调用、代码执行）
- **ToolEnv**: 环境编排器，管理状态转换和奖励计算

### 3. Action Mask

区分 LLM 生成的 token 与环境反馈的 token，确保只对代理的动作计算梯度。

### 4. 支持的 RL 算法

- **PPO** (Proximal Policy Optimization)
- **GRPO** (Group Relative Policy Optimization)
- **REINFORCE++**
- **REINFORCE++ Baseline**
- **RLOO** (Leave-One-Out)
- **ReMax**

---

## 推荐阅读顺序

为了系统地理解 Agent-R1 代码库，建议按以下顺序阅读文档：

```
┌─────────────────────────────────────────────────────────────┐
│  第一阶段：环境准备                                           │
│  ├── 01_environment_setup.md  - 环境配置                     │
│  └── 08_data_preparation.md   - 数据准备                     │
├─────────────────────────────────────────────────────────────┤
│  第二阶段：理论基础                                           │
│  └── 02_core_concepts.md      - 核心概念与 MDP 理论          │
├─────────────────────────────────────────────────────────────┤
│  第三阶段：代码架构                                           │
│  ├── 03_project_structure.md  - 代码结构详解                 │
│  ├── 04_tool_system.md        - Tool 系统                    │
│  └── 05_environment_system.md - ToolEnv 系统                 │
├─────────────────────────────────────────────────────────────┤
│  第四阶段：训练与奖励                                         │
│  ├── 06_training_pipeline.md  - 训练流程                     │
│  └── 07_reward_system.md      - 奖励系统                     │
├─────────────────────────────────────────────────────────────┤
│  第五阶段：实践应用                                           │
│  ├── 09_running_experiments.md - 实验运行                    │
│  └── 10_inference.md           - 推理部署                    │
└─────────────────────────────────────────────────────────────┘
```

### 快速上手路径

如果你想快速运行实验，可以按以下顺序：

1. **01_environment_setup.md** → 安装环境
2. **08_data_preparation.md** → 准备数据
3. **09_running_experiments.md** → 运行训练

### 深入理解路径

如果你想深入理解框架设计：

1. **02_core_concepts.md** → 理解理论基础
2. **03_project_structure.md** → 了解代码架构
3. **04_tool_system.md** + **05_environment_system.md** → 掌握工具系统
4. **06_training_pipeline.md** → 理解训练流程

---

## 文档导航索引

| 文档 | 内容概述 | 适合读者 |
|------|----------|----------|
| [00_overview.md](./00_overview.md) | 代码库总览（本文档） | 所有读者 |
| [01_environment_setup.md](./01_environment_setup.md) | 环境安装、依赖配置 | 新用户 |
| [02_core_concepts.md](./02_core_concepts.md) | MDP 理论、核心概念 | 研究人员 |
| [03_project_structure.md](./03_project_structure.md) | 代码结构、模块职责 | 开发者 |
| [04_tool_system.md](./04_tool_system.md) | BaseTool 设计与扩展 | 工具开发者 |
| [05_environment_system.md](./05_environment_system.md) | BaseToolEnv 设计与扩展 | 环境开发者 |
| [06_training_pipeline.md](./06_training_pipeline.md) | 训练流程、RL 算法 | 训练工程师 |
| [07_reward_system.md](./07_reward_system.md) | 奖励设计、自定义奖励 | 奖励设计者 |
| [08_data_preparation.md](./08_data_preparation.md) | 数据格式、预处理 | 数据工程师 |
| [09_running_experiments.md](./09_running_experiments.md) | 实验运行、参数配置 | 实验人员 |
| [10_inference.md](./10_inference.md) | 推理部署、模型服务 | 部署工程师 |

---

## 快速链接

- **GitHub**: https://github.com/0russwest0/Agent-R1
- **论文**: https://arxiv.org/abs/2511.14460
- **verl 框架**: https://github.com/volcengine/verl
- **Awesome-Agent-RL**: https://github.com/0russwest0/Awesome-Agent-RL

---

## 下一步

开始阅读 [01_environment_setup.md](./01_environment_setup.md) 配置你的开发环境。
