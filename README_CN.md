<h1 align="center"> Agent-R1：通过端到端强化学习训练强大的 LLM 智能体 </h1>

<p align="center">
  <a href="https://arxiv.org/abs/2511.14460">
  <img src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?logo=arxiv&logoColor=white" alt="Paper Arxiv">
</a>
  <a href="https://deepwiki.com/0russwest0/Agent-R1"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" height="20"/></a>
  <a href="https://github.com/0russwest0/Agent-R1/stargazers"><img src="https://img.shields.io/github/stars/0russwest0/Agent-R1" alt="GitHub Repo stars"></a>
  <a href="https://github.com/0russwest0/Agent-R1/network/members"><img src="https://img.shields.io/github/forks/0russwest0/Agent-R1" alt="GitHub forks"></a>
  <a href="https://raw.githubusercontent.com/0russwest0/Agent-R1-Community/refs/heads/main/Wechat.jpg"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
  <a href="https://discord.gg/kW3UZU2e"><img src="https://img.shields.io/badge/Discord-blue?logo=discord&amp"></a>
</p>

<p align="center"><img src="./image/agent.png" width="800px" alt="Agent vs Workflow" /></p>

## 新闻动态

<details open>
<summary><b>最近更新</b></summary>

- [2025.11.18] **技术报告**：我们已在 arXiv 上发布技术报告。点击[此处](https://arxiv.org/abs/2511.14460)阅读论文。
  
- [2025.05.06] **工具环境重新设计**：全面重新设计和抽象工具环境，以支持更灵活多样的智能体-工具交互模式。

- [2025.05.06] **关键 Bug 修复**：修复了导致训练期间出现 NaN 值的 GRPO 和 Reinforce++ 训练崩溃问题。详见 [issue #30](https://github.com/0russwest0/Agent-R1/issues/30)。

- [2025.05.06] **新教程**：添加了创建自定义工具和工具环境的综合教程，包括 ReTool 的首个开源可运行实现。

</details>

<details>
<summary><b>早期更新</b></summary>

- [2025.04.01] 添加了基础**推理脚本**和简单的交互式聊天界面。您现在可以轻松部署和与训练好的模型进行交互。详见[推理指南](docs/inference/inference.md)。

- [2025.03.18] 添加了全面的**多模态支持**！Agent-R1 现在可以无缝集成视觉语言模型（VLM），使智能体能够在丰富的多模态环境中处理和推理文本与视觉输入。

- [2025.03.18] 重构了代码库以提高可维护性！我们将 verl 从静态文件夹转换为 **git 子模块**，并分离了自定义代码扩展。这使得更新 `verl` 和理解项目结构更加容易。
  > **重要提示：** 拉取此更新后，您需要重新初始化环境。运行 `git submodule update --init --recursive` 并从此目录本地重新安装 verl。

- [2025.03.16] 添加了对**过程奖励**的支持！您现在可以根据每个工具调用的有效性为其分配奖励。为了平衡过程奖励和结果奖励，我们实现了受 [PRIME](https://github.com/PRIME-RL/PRIME) 启发的奖励归一化。

</details>

## 概述

**Agent-R1** 是一个开源框架，旨在加速**强化学习**与**智能体**关键交叉领域的研究和开发。我们的框架采用**端到端**强化学习在特定环境中训练智能体。开发者只需定义特定领域的工具和奖励函数即可将 Agent-R1 扩展到其独特的用例，无需复杂的工作流工程。我们希望我们的微薄贡献能够惠及开源社区，使研究人员和开发者更容易在自己的领域创建和探索智能体，共同推进自主智能体的发展。有关算法的更多详情，请参阅[算法文档](https://github.com/0russwest0/Agent-R1/blob/main/docs/algorithm/algorithm.md)。

> **另请查看 [Awesome-Agent-RL](https://github.com/0russwest0/Awesome-Agent-RL)**：我们精心整理的通过强化学习释放智能体潜力的论文和资源合集。

<p align="center"><img src="./image/framework.png" width="800px" alt="RICO Framework" /></p>

## 核心特性

- **多轮工具调用**：在完整交互轨迹上进行端到端强化学习，使智能体能够从动作序列中学习
- **多工具协调**：训练智能体有效协调和使用多个工具来解决复杂任务
- **过程奖励**：根据每个工具调用的有效性分配奖励，通过归一化与结果奖励进行平衡
- **自定义工具和环境**：兼容主流 LLM 工具调用格式，便于使用您自己的工具和场景进行扩展
- **多种强化学习算法**：支持多种强化学习方法，包括 `PPO`、`GRPO` 和 `REINFORCE++`
- **多模态支持**：兼容视觉语言模型（VLM）和多模态强化学习

## 即将推出的功能

- **扩展模型支持**：集成更多基础模型，不仅限于目前支持的 Qwen
- **更多用例**：在更多场景和领域提供更多示例实现

## 快速开始
- [环境配置](https://github.com/0russwest0/Agent-R1/blob/main/docs/getting_started/installation.md)
- [快速入门：在 HotpotQA 上尝试默认搜索工具](https://github.com/0russwest0/Agent-R1/blob/main/docs/getting_started/quickstart.md)（另见：[HotpotQA 上的结果](https://github.com/0russwest0/Agent-R1/blob/main/docs/getting_started/quickstart.md#5-results-on-hotpotqa)）

## 使用自定义工具和环境扩展 Agent-R1

Agent-R1 提供了灵活的架构来创建自定义工具和工具环境，以适应各种智能体应用。我们的框架建立在两个关键抽象之上：

1. **BaseTool**：智能体可用于与外部系统交互的单个工具
2. **BaseToolEnv**：定义智能体-工具交互状态转换函数的工具环境

有关扩展 Agent-R1 的详细指南，请参阅我们的教程：

- [为多跳问答定制工具](https://github.com/0russwest0/Agent-R1/blob/main/docs/tutorial/multihopqa.md)：学习如何创建和定制用于跨多个知识源检索信息的工具
- [为 ReTool 定制工具环境](https://github.com/0russwest0/Agent-R1/blob/main/docs/tutorial/retool.md)：了解如何实现将代码执行与 LLM 推理集成的工具环境

代码库中还有其他资源可用：
- 示例工具：`agent_r1/tool/tools/`
- 示例环境：`agent_r1/tool/envs/`
- 数据预处理：`examples/data_preprocess/`
- 奖励函数：`verl/utils/reward_score/`

## 反馈

我们欢迎各种形式的反馈！请针对 bug、问题或建议提出 issue。这有助于我们的团队高效解决常见问题，并建立更有成效的社区。

**加入我们的社区**：在我们的[微信群](https://raw.githubusercontent.com/0russwest0/Agent-R1-Community/refs/heads/main/Wechat.jpg)或 [Discord 服务器](https://discord.gg/kW3UZU2e)与其他用户和我们的开发团队交流。

## 贡献者

**学生贡献者**：[**欧阳杰**\*](https://github.com/0russwest0)、[**闫瑞然**\*](https://github.com/RuiranYan)、[**罗宇聪**\*](https://github.com/GodFire66666)、刘子睿、余硕、王道雨

**指导老师**：[**刘淇**](http://staff.ustc.edu.cn/~qiliuql/)、[**程明月**](https://mingyue-cheng.github.io/)

**所属单位**：**中国科学技术大学 认知智能全国重点实验室**

## 致谢
我们衷心感谢 [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) 提供 DeepSeek-R1 模型和启发性的想法。我们也感谢 [veRL](https://github.com/volcengine/verl) 团队提供的强大基础设施支持。此外，我们感谢 [RAGEN](https://github.com/ZihanWang314/ragen) 团队的开创性发现，这对我们的早期探索产生了重大影响。最后，我们深深感谢欧阳杰、闫瑞然、罗宇聪、刘子睿、余硕和王道雨富有洞察力的讨论和贡献。

## 引用
**Agent-R1**
```md
@misc{cheng2025agentr1trainingpowerfulllm,
      title={Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning}, 
      author={Mingyue Cheng and Jie Ouyang and Shuo Yu and Ruiran Yan and Yucong Luo and Zirui Liu and Daoyu Wang and Qi Liu and Enhong Chen},
      year={2025},
      eprint={2511.14460},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.14460}, 
}
```
**TableMind**（WSDM 2026）

*TableMind 基于 Agent-R1 框架构建，利用其端到端强化学习流程训练专门用于结构化表格推理的工具增强智能体。*
```md
@article{jiang2025tablemind,
  title={TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning},
  author={Jiang, Chuang and Cheng, Mingyue and Tao, Xiaoyu and Mao, Qingyang and Ouyang, Jie and Liu, Qi},
  journal={arXiv preprint arXiv:2509.06278},
  year={2025}
}
```
## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=0russwest0/Agent-R1&type=Date)](https://www.star-history.com/#0russwest0/Agent-R1&Date)

