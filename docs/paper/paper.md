# Agent-R1：使用端到端强化学习训练强大的 LLM 智能体

**Mingyue Cheng, Jie Ouyang, Shuo Yu, Ruiran Yan, Yucong Luo, Zirui Liu, Daoyu Wang, Qi Liu, Enhong Chen**  
认知智能国家重点实验室，中国科学技术大学，合肥，中国

---

## 摘要

大型语言模型（LLM）正被越来越多地用于构建能够主动与环境交互（例如通过工具调用）的智能体（Agent），以解决复杂问题。强化学习（RL）被认为是训练此类智能体的一项关键技术，具有巨大潜力；然而，将 RL 有效应用于 LLM 智能体仍处于起步阶段，并面临诸多挑战。目前，该新兴方向缺乏针对 LLM 智能体场景量身定制的 RL 方法学的深入探索，同时也缺少灵活且易扩展的训练框架来支撑相关研究与落地。

为推进这一领域，本文首先回顾并澄清面向 LLM 智能体的强化学习方法，通过系统性扩展马尔可夫决策过程（MDP）框架，全面定义 LLM 智能体的关键组成要素。其次，我们提出 **Agent-R1**：一个模块化、灵活且易用的基于 RL 的 LLM 智能体训练框架，可便捷适配多种任务场景与交互式环境。我们在多跳问答（Multihop QA）基准任务上进行了实验，为所提出的方法与框架的有效性提供了初步验证。

项目地址：§ https://github.com/0russwest0/Agent-R1

> 注：本文为 Agent-R1 项目的技术报告。  
> 预印本（Preprint），正在审稿。  
> arXiv:2511.14460v1 [cs.CL] 18 Nov 2025

---

## 1 引言

近年来，大型语言模型（LLM）在自然语言理解与生成方面展现出卓越能力 [2, 4]，并被越来越多地应用于更复杂的智能任务 [24]。当 LLM 被赋予“智能体（agent）”角色时，它不仅需要完成推理与决策等认知任务，还应能够自主行动、持续学习，并适应交互环境中的变化 [35, 25, 16]。不同于传统的静态推理任务 [30]，作为智能体的 LLM 必须在多轮对话中保持记忆 [37]、具备序列决策能力，并能对环境反馈做出有效响应——这使其更接近真实世界的自主智能系统 [27, 13]。这一方向为构建具备自我演化与问题求解能力的通用人工智能打开了新的可能性 [31, 9]。

尽管强化学习（RL）[5, 26, 6] 已在数学解题与代码生成等相对定义清晰的任务上显著提升了 LLM 能力 [15, 19, 1]，但将 RL 用于把 LLM 发展为自主、可交互的智能体仍较为初级。智能体设置天然要求模型进行序列决策、跨轮次维护记忆，并适应随机的环境反馈 [35, 27]，这些挑战与更静态的任务 [16, 30] 明显不同。因此，在多轮交互场景中应用 RL 会遭遇一些特定困难：例如训练不稳定、奖励信号设计复杂、以及泛化能力受限 [3, 10, 20]。这也意味着我们仍需要更细致地探索如何系统性地将 RL 方法应用并适配到 LLM 智能体，同时还需要灵活可扩展的训练框架 [14, 18]。

为系统性解决上述问题，本文从概念与实践两个层面做出贡献。在概念层面，我们聚焦于澄清 RL 在 LLM 智能体上的应用：通过扩展标准 MDP 框架 [22]，详细说明其核心组件——状态空间、动作空间、状态转移概率与奖励函数——如何被改造以全面刻画 LLM 智能体的多轮交互特性 [7, 36]。在此基础上，我们进一步阐述如何从多轮轨迹中优化智能体策略 [17]，强调需要区分智能体生成的动作与环境反馈，并引入中间（过程）奖励 [11, 27] 来更有效地引导学习。

在实践层面，为便于上述理念落地，我们构建了 **Agent-R1**：一个灵活、易用的基于 RL 的 LLM 智能体训练平台。得益于其模块化架构，Agent-R1 支持快速集成多种环境接口与任务场景，并能根据算力资源需求动态适配，从而易于扩展到更复杂与多样的应用 [32, 12]。

我们在具有挑战性的多跳问答（Multi-hop QA）任务上进行了系统实验，以验证方法与框架的有效性 [16, 33]。该任务需要跨文档的逻辑链式推理与信息检索，对智能体的多步决策能力、对环境反馈的适应性，以及知识构建过程提出了较高要求。实验结果表明，我们的方法与框架能够在这种动态交互环境中提升模型表现 [29]。

**图 1：工作流、代理式工作流与自主智能体的对比。**  
- 工作流（Workflow）：依赖人类设计的路由或规划  
- 代理式工作流（Agentic Workflow，例如 ReAct）：引入迭代的“推理—行动”循环  
- 自主智能体（Agent）：去除预定义工作流，通过端到端“行动—反馈”闭环主动与环境交互  
（图中要点：工作流依赖人类设计与提示工程；自主智能体不依赖预设工作流，不依赖提示工程，主动与环境交互。）

---

## 2 从大型语言模型到智能体：MDP 视角

LLM 的序列决策过程，无论是简单文本生成还是复杂的智能体交互，都可以在马尔可夫决策过程（MDP）框架下建模。然而，将 MDP 从“静态、单轮文本生成任务”（如数学或代码生成）扩展到“动态、多轮、强交互的 LLM 智能体对话”，需要对 MDP 做出实质性的扩展。本节通过对比“静态 LLM”与“LLM 智能体”的 MDP 组件，说明这些关键差异。

### 表 1：静态 LLM 与 LLM 智能体的 MDP 组件对比（为多轮交互场景所需的扩展）

| MDP 组件 | 静态 LLM | LLM 智能体 |
|---|---|---|
| 状态空间（S） | 状态 $s_t$ 主要包含当前文本上下文：初始提示 $w_p$ 与已生成 token 序列。重点是预测下一个连贯 token。<br> $s_t=(w_p, w_1,\ldots,w_t)$ | 状态 $s_t$ 更全面：保留多轮交互历史与环境反馈，使决策能基于完整对话历史。每个 $T_i$ 表示一整轮“智能体动作 + 环境反馈”。<br> $s_t=(w_p, T_1,\ldots,T_k, T^{\\text{partial}}_{k+1})$ |
| 动作空间（A） | 动作 $a_t$ 即从词表 $V$ 中选择下一个 token $w_{t+1}$ | 基本动作仍为从 $V$ 生成 token；但某些 token 序列可被解释为调用外部工具的命令，从而主动干预环境 |
| 状态转移（P） | 状态转移是确定性的：把选中的 token 追加到序列即可得到下一状态 | 转移机制引入与环境的交互，可能是随机的；区分确定性的生成转移 $P_G$ 与由工具使用触发、可能随机的环境转移 $P_E$ |
| 奖励函数（R） | 奖励通常稀疏，在完整生成结束时给出，评估最终输出整体质量 | 奖励更丰富更稠密：除最终结果奖励 $r_f$ 外，还可给出中间过程奖励 $r_p$（如成功调用工具），提供更频繁反馈 |

### 表 2：静态 LLM 与 LLM 智能体的 MDP 公式化差异（核心总结）

| 组件 | 静态 LLM | LLM 智能体 |
|---|---|---|
| 状态（S） | 仅捕获当前文本序列 | 捕获完整多轮交互历史与环境反馈 |
| 动作（A） | 生成下一个 token | 生成 token，同时可能形成调用外部工具的命令 |
| 状态转移（P） | 确定性：追加 token 决定下一状态 | 随机性：下一状态依赖环境的非确定反馈 |
| 奖励（R） | 仅在结束时获得一次稀疏奖励 | 过程奖励更稠密，除最终奖励外包含中间步骤奖励 |

---

### 2.1 状态空间（S）

**静态 LLM：** 在单轮文本生成中，状态 $s_t$ 主要表示当前文本上下文，包括初始提示 $w_p$ 与已生成 token 序列 $w_1,w_2,\ldots,w_t$：

$$
s_t = (w_p, w_1, w_2, \ldots, w_t).
\tag{1}
$$

该状态空间的重点是捕捉预测下一个 token 所需的信息，使生成序列保持连贯。

**LLM 智能体：** 对于进行多轮交互的智能体，状态 $s_t$ 必须更全面，既要包含文本上下文，也要保留交互历史与环境反馈，因此扩展为：

$$
s_t = (w_p, T_1, T_2, \ldots, T_k, T^{\text{partial}}_{k+1}).
\tag{2}
$$

其中，每个 $T_i$ 表示完整的一轮交互，由智能体生成 token 序列 $(w_{i1},\ldots,w_{iT_i})$ 与随后的环境反馈 $w^e_i$ 组成，即  
$T_i=(w_{i1},\ldots,w_{iT_i}, w^e_i)$。而 $T^{\text{partial}}_{k+1}$ 表示当前正在进行的这一轮中“部分生成”的序列。该增强状态表示使智能体能基于完整对话历史与环境结果（例如工具调用返回）来决策。

---

### 2.2 动作空间（A）

**静态 LLM：** 动作 $a_t$ 对应从词表 $V$ 中选择下一个 token $w_{t+1}$。因此动作空间 $A(s_t)$ 通常就是 $V$。

**LLM 智能体：** 智能体的动作同样是从 $V$ 中选择下一个 token。但动作序列的含义更广：智能体生成的特定 token 序列可能被解释为调用外部工具或 API 的指令。因此，尽管基础动作仍是 token 生成，其功能结果可能超越纯文本输出，扩展为主动干预环境。

---

### 2.3 状态转移概率（P）

**静态 LLM：** 文本生成的状态转移是确定性的。给定当前状态 $s_t$ 与动作 $a_t$（选择 token $w_{t+1}$），下一状态 $s_{t+1}$ 唯一确定为“把 $w_{t+1}$ 追加到当前序列”：

$$
P(s_{t+1}\mid s_t, a_t)=
\begin{cases}
1, & \text{若 } s_{t+1}=s_t \oplus a_t,\\
0, & \text{否则},
\end{cases}
\tag{3}
$$

其中 $\oplus$ 表示序列拼接。

**LLM 智能体：** 智能体的状态转移需要纳入与环境交互（可能随机）。转移可按“动作是否触发工具/环境交互”分类：

$$
P(s_{t+1}\mid s_t, a_t)=
\begin{cases}
P_E(s_{t+1}\mid s_t, a_t), & \text{若 } a_t \text{ 触发工具/环境交互},\\
P_G(s_{t+1}\mid s_t, a_t), & \text{否则（常规 token 生成）}.
\end{cases}
\tag{4}
$$

其中，$P_G$（生成转移）与静态 LLM 相同：当 $s_{t+1}=s_t\oplus a_t$ 时概率为 1，否则为 0；而 $P_E$（环境转移）反映工具执行与环境响应的不确定性。此时 $s_{t+1}$ 不仅取决于智能体动作，还取决于外部环境返回（如 API 响应、计算结果），这些会形成环境反馈 $w^e_i$ 的一部分。

---

### 2.4 奖励函数（R）

**静态 LLM：** 奖励通常稀疏，仅在到达终止状态 $s_T$（生成结束）时给出，常见为结果型奖励 $R(s_T)$，用于评估最终文本整体质量（例如连贯性、相关性）。

**LLM 智能体：** 智能体的奖励结构通常更丰富更稠密，以适配多轮任务。可定义：

$$
R(s_t, a_t, s_{t+1})=
\begin{cases}
r_f(s_{t+1}), & \text{若 } s_{t+1} \text{ 为终止状态},\\
r_p(s_t, a_t, s_{t+1}), & \text{若 } a_t \text{ 触发重要的中间事件},\\
0, & \text{否则（例如常规 token 生成）}.
\end{cases}
\tag{5}
$$

其中 $r_f(s_{t+1})$ 是任务完成时的最终结果奖励；关键在于，智能体还可因成功完成中间步骤（如有效工具调用、朝目标取得实质进展）获得过程奖励 $r_p(s_t,a_t,s_{t+1})$。这些中间信号提供更频繁的反馈，从而更有效地引导学习。

**小结：** 将 MDP 从静态 LLM 扩展到 LLM 智能体，需要在：状态（加入交互历史与环境反馈）、动作（可触发外部效应）、转移（纳入环境随机性）、奖励（引入过程奖励）等方面做出关键增强，以便 RL 能训练出能在动态环境中进行复杂多步推理与交互的智能体。

---

## 3 Agent-R1 框架

为更好满足 LLM 智能体的强化学习训练需求，我们提出 **Agent-R1**：一个灵活且高度可扩展的智能体强化学习训练框架（见图 2）。借鉴现有高效 RL 基础设施，我们将传统“单轮 RL 训练框架”扩展为可充分适配智能体“多轮交互”特性的训练框架，使其能无缝对接多样化任务环境，并支持随着智能体设置复杂度提升而扩展训练。

图 3 与图 4 分别对比了传统单轮 RL 与 Agent-R1 多轮 RL 在 **生成阶段** 与 **学习阶段** 的流程差异。单轮与多轮 RL 的最大不同在 rollout（采样/展开）阶段：单轮 rollout 仅需 Actor 模型生成一次响应；多轮 rollout 则包含多次复杂交互。为实现灵活、易扩展的多轮 rollout，我们设计了两个核心模块：**Tool** 与 **ToolEnv**。

---

### 图 2：Agent-R1 训练轨迹示意

智能体在 rollout 中进行多轮推理与基于工具的行动，接收环境反馈，并将工具响应追加到状态中形成下一状态。包含“思考步骤、动作、反馈”的轨迹将作为 Agent-R1 中 RL 更新的基础。

示意中包含（概念）：  
- 用户指令 $q$  
- 多轮：$t_1,a_1,r_1,\ldots,t_k,a_k,r_k$  
- 可能输出 `<think>...` 思考过程、`<tool_call>...` 工具调用、`<tool_response>...` 工具返回、`<answer>...` 最终答案  
- 环境在多轮过程中返回反馈；智能体决定 stop/continue

---

### 3.1 Tool 与 ToolEnv：交互式 Rollout 的核心模块

交互式 rollout 是训练 LLM 智能体的核心过程，强依赖两大组件：**Tool** 与 **ToolEnv**。二者职责划分清晰，是 Agent-R1 的重要设计哲学。

- **Tool**：被视为执行特定、原子动作的执行器。它封装某一项明确能力，例如调用外部 API、执行代码、访问数据库等。Tool 被调用后执行动作，并返回该动作的直接、原始结果，本质上是客观汇报“发生了什么”。

- **ToolEnv**：作为 RL 环境中的编排者与解释器。它接收 Tool 的原始输出，并决定该输出如何影响智能体感知的状态与任务进度。ToolEnv 负责：管理 RL 循环中的状态转移、基于转移与工具结果计算奖励信号、并将新的状态信息打包返回给智能体。本质上它决定“这个结果对智能体与任务意味着什么”。

---

#### 3.1.1 Tool 设计

Tools 是连接智能体与外部环境/功能的关键接口。在 Agent-R1 中，我们用 Tools 作为“智能体—环境交互”的统一接口：所有外部功能都被封装为标准化、可被智能体直接调用的“工具”。

受 OpenAI Function Calling 范式启发，Agent-R1 通过抽象基类 **BaseTool** 为 Tools 提供高层抽象与标准化，设计聚焦两部分：

1) **核心执行逻辑（Core Execution Logic）**  
BaseTool 中最关键的抽象方法是 `execute`。所有具体工具子类都必须实现该执行方法：它定义如何处理输入参数、执行具体操作（如调用外部 API、执行代码、访问数据库等），并返回结构化结果。

2) **工具元数据规范（Tool Metadata Specification）**  
为保证工具调用的标准化与可解析性，定义如下元数据属性：  
- **标识与描述（Identification and Description）**：`name`（唯一字符串标识）与 `description`（详细说明功能、使用场景与预期效果）。智能体通过它们在当前上下文中识别并选择合适工具。  
- **参数结构定义（Parameter Structure Definition）**：`parameters` 使用 JSON Schema 规范定义工具调用所需的输入结构，包括参数名、数据类型、详细描述、是否必填等。参数标准化确保智能体生成的调用参数符合预期格式。

该设计通过 `execute` 实现动作执行、通过清晰元数据让智能体理解工具，从而使 LLM 智能体能以结构化接口与外部环境交互。工具执行的结果随后由 ToolEnv 处理，负责管理对应的环境状态转移。二者协同，是智能体在多轮交互中解决复杂问题的基础，也与 ToolEnv 的状态管理设计形成闭环。

---

#### 3.1.2 ToolEnv 设计

ToolEnv 模块在 Agent-R1 强化学习框架中充当动态环境，负责管理智能体与世界的交互，尤其是涉及工具调用时。它实现 RL 环境的两项核心功能：**状态转移** 与 **奖励计算**，并特别处理多轮交互与工具使用带来的非确定性结果。该设计通过抽象基类 **BaseToolEnv** 形式化。

1) **核心状态转移与奖励逻辑**  
最关键的抽象方法是 `step`。它是环境交互的主引擎：接收智能体的原始输出（例如可能包含工具调用的生成文本），解析并与 Tool 模块协同编排工具调用。随后，基于智能体动作与工具执行反馈，更新环境内部状态，计算反映动作结果与新状态的奖励信号，并将新状态、奖励与其他信息（如成功标志、活动标志等）返回给智能体。该方法同时覆盖常规生成式状态转移与由工具交互引入的更复杂、可能随机的转移。

2) **交互管理的辅助机制**  
为支撑 `step` 的综合职责并处理工具交互细节，BaseToolEnv 定义了若干关键辅助方法：  
- `process_responses_ids`：提供可定制逻辑，用于在 LLM 生成的 token id 序列中识别工具调用触发点并确定精确调用位置。  
- `extract_tool_calls`：解析原始 LLM 响应，抽取并结构化工具调用请求（工具名与参数）。  
- `format_tool_response`：将 Tool.execute 的原始结果转换为适合拼接回新状态、呈现给 LLM 的字符串格式。  
- `stop`：实现轨迹终止条件逻辑，根据 LLM 输出、任务完成情况、错误状态或预设限制判断是否结束当前交互。

该设计以 `step` 驱动环境动态，并配合清晰的工具调用与轨迹生命周期管理机制，使 Agent-R1 能有效模拟复杂交互场景，明确区分“确定性的文本生成”与“工具使用引入的非确定性、会改变环境的状态变化”，从而为智能体学习提供关键支撑。

---

### 3.2 从多轮轨迹优化智能体策略

rollout 阶段结束后，我们得到完整的多轮交互轨迹。每条轨迹包含：状态序列、智能体动作（生成文本片段）以及奖励信号。如前所述，环境会为每轮交互提供奖励信号，称为过程奖励（$r_p$），同时还可能在终止时提供最终结果奖励（$r_f$）。

为在轨迹中清晰地区分“LLM 智能体生成的 token（即其动作）”与“环境反馈或初始提示”等非可学习部分，我们引入 **Action Mask（动作掩码）**，用于精确标记序列中哪些部分对应智能体可学习的动作 token。

强化学习通过优化策略模型的动作以最大化期望累计回报。Agent-R1 在学习阶段（见图 4）利用多轮轨迹中的细粒度信息（包括动作掩码与过程奖励）来完成优化，关键点包括：

1) **更精炼且对齐的优势函数（Advantage）计算**  
如生成阶段流程（图 3）所示，“优势（Advantages）”不再仅由最终结果奖励与 Critic 模型的价值估计决定。ToolEnv 在 rollout 收集到的“过程奖励（Process Rewards）”会被显式纳入。也就是说，在轨迹中每个相关时间步 $t$ 的优势估计 $\hat{A}_t$，不仅反映未来折扣回报（来自最终奖励与价值函数估计），还反映中间步骤的即时成功（例如有效工具调用带来的过程奖励）。

更重要的是，优势计算（例如用广义优势估计 GAE）会与动作掩码对齐：尽管奖励基于状态转移累积、价值函数评估状态好坏，但用于更新策略的最终优势必须对应智能体实际生成动作的时间步。这样，信用分配（正/负优势）会归因到智能体真正做出的决策，而非它无法控制的部分（如提示 token 或固定的环境响应）。这些“动作对齐”的优势随后被传入学习阶段（图 4）。

2) **掩码化策略优化（Actor Loss）**  
在学习阶段（图 4），Actor（策略）模型被更新以提高高优势动作的概率。轨迹数据输入 Actor 得到新的动作 logits。动作掩码在此至关重要：计算 Actor Loss（如 PPO 的 clipped surrogate objective）时，仅在智能体生成的 token 上计算损失。新旧策略动作概率比值（由新 logits 与生成阶段保存的旧 logits 得到）会与对齐后的优势结合，并在掩码约束下完成梯度更新。

3) **价值函数更新（Critic Loss）**  
Critic 模型学习更准确地估计不同状态下的期望累计回报（价值）。基于轨迹数据它生成新的 value 预测。Critic Loss 通常是新 value 与观测回报（包含过程奖励与结果奖励）之间的均方误差，或与基于 TD 学习构造的目标值之间的误差。这使 Critic 能为后续迭代提供更好的 baseline，从而改进优势计算。

通过确保优势与智能体真实动作严格对齐，并在策略优化中配合动作掩码使用，Agent-R1 提供了更精确、更有效的学习信号，使 Actor 与 Critic 能更高效地从长对话与工具使用等复杂场景中学习，推动智能体掌握更高难度任务。

---

## 4 实证研究

我们在一个具有挑战性的多跳问答（multi-hop QA）场景中对 Agent-R1 的有效性与设计贡献进行实证评估。在该场景中，LLM 需要使用外部搜索工具。本研究首先验证框架能否使用多种 RL 算法训练适用于多轮交互任务的 LLM 智能体；其次通过消融实验分析关键策略优化细节的影响：用于损失计算的动作掩码（“loss mask”）与用于优势对齐的动作掩码（“advantage mask”）。总体目标是评估 LLM 在工具调用与信息检索方面的学习能力，从而体现 Agent-R1 的实用性。

---

### 4.1 实验设置

**任务与数据集**  
我们使用多跳问答（MultihopQA）数据集。训练集包含 51,200 个样本，从 HotpotQA [34] 与 2WikiMultihopQA [8] 的训练集划分中等量随机抽取。评估使用 HotpotQA 与 2WikiMultihopQA 的完整验证集（域内 in-domain），以及 Musique [28]（域外 out-of-domain），这些数据集都需要多步检索与推理。

**模型与工具**  
实验使用 Qwen2.5-3B-Instruct [23]，在 NousToolEnv 中使用其原生 function calling。智能体使用单一的 `wikisearch` 工具查询 KILT Wikipedia 语料库（约 3600 万段落 [21]，向量嵌入为 bge-large-en-v1.5），返回 top 5 文档。

**RL 算法与基线**  
我们评估 PPO、GRPO、REINFORCE++、REINFORCE++Baseline 与 RLOO，以检验 Agent-R1 的适配性。对比两个基线：  
- **Naive RAG**（单次检索）  
- **Base Tool Call**（使用 `wikisearch` 工具的原生 function calling）

**奖励设计**  
使用稀疏的最终结果奖励 $r_f$，定义为：

$$
r_f =
\begin{cases}
r_{\text{answer}}, & \text{若 } r_{\text{format}}=1,\\
r_{\text{format}} - 1, & \text{若 } r_{\text{format}} < 1.
\end{cases}
\tag{6}
$$

其中，$r_{\text{answer}}=\text{EM}(a_{\text{pred}}, a_{\text{gold}})$ 为 Exact Match 分数。格式得分 $r_{\text{format}}=(r_{\text{format}a}+r_{\text{format}t})/2$，为两个二值指标的平均：最终答案展示是否正确（$r_{\text{format}a}$）与工具调用语法是否有效（$r_{\text{format}t}$）。该奖励严格奖励“格式完全正确且答案正确”的输出，并惩罚任何格式错误。

---

### 4.2 主要结果

框架验证的主要结果见表 3：比较 Agent-R1 支持的多种 RL 算法与基线方法在三个多跳问答数据集上的表现。HotpotQA 与 2Wiki 为域内数据集，Musique 为域外数据集。报告指标为 Exact Match（EM）。

**表 3：在 MultihopQA 数据集上，各 RL 算法与基线的 EM 对比**  
† 表示域内数据集；* 表示域外数据集。对 RL 算法而言：每列最好为 **加粗**，第二好用 <u>下划线</u> 表示。

| 方法 | HotpotQA† | 2Wiki† | Musique* | 平均 |
|---|---:|---:|---:|---:|
| Base Tool Call | 0.1372 | 0.0891 | 0.0277 | 0.0847 |
| Naive RAG | 0.1916 | 0.1792 | 0.0277 | 0.1328 |
| PPO | <u>0.4136</u> | 0.5468 | **0.1552** | <u>0.3719</u> |
| GRPO | **0.4405** | **0.5741** | <u>0.1485</u> | **0.3877** |
| REINFORCE++ | 0.3768 | 0.4796 | 0.1336 | 0.3300 |
| REINFORCE++Baseline | 0.3966 | 0.5406 | <u>0.1485</u> | 0.3619 |
| RLOO | 0.4089 | <u>0.5641</u> | 0.1419 | 0.3716 |

表 3 清晰表明：所有 RL 训练得到的智能体都显著优于两个基线（Base Tool Call 的 0.0847 与 Naive RAG 的 0.1328）。例如，表现最弱的 RL 智能体（REINFORCE++，平均 EM 0.3300）仍比 RAG 高约 2.5 倍。这一显著差距凸显：RL 对训练具备复杂多轮决策与有效工具使用能力的 LLM 智能体至关重要，能够超越更简单的启发式或单次方法。

在 RL 方法中，GRPO（平均 EM 0.3877）整体最佳，其次是 PPO（0.3719）与 RLOO（0.3716）。PPO 在挑战性的域外 Musique 数据集上表现尤其突出。REINFORCE++（0.3300）最弱，但加入 baseline 的 REINFORCE++Baseline（0.3619）带来明显提升，不过仍不及顶尖算法。总体上，这些结果有力验证了 Agent-R1 能通过端到端 RL 训练出强大的 LLM 智能体，并在多数据集、多算法上稳定显著超越基线。

---

### 4.3 策略优化细节的消融实验

为研究 Agent-R1 中若干策略优化细节的重要性——即用于损失计算的动作掩码（“loss mask”）与用于优势对齐的动作掩码（“advantage mask”）——我们在 PPO 与 GRPO 上做了消融实验。结果（EM）见表 4。每个消融步骤均相对同一算法组上一行配置，通过禁用某组件得到。

**表 4：策略优化组件的消融实验（EM）**  
符号说明：✓ 表示启用；“→” 表示从上一行配置禁用某组件得到。该设置下，GRPO 不单独消融 advantage mask。

| 配置 | HotpotQA | 2Wiki | Musique | 平均 |
|---|---:|---:|---:|---:|
| PPO（loss mask ✓，adv. mask ✓） | 0.4136 | 0.5468 | 0.1552 | 0.3719 |
| → 禁用 advantage mask | 0.3630 | 0.4641 | 0.1138 | 0.3136 |
| → 再禁用 loss mask | 0.3429 | 0.4631 | 0.1005 | 0.3022 |
| GRPO（loss mask ✓） | 0.4405 | 0.5741 | 0.1485 | 0.3877 |
| → 禁用 loss mask | 0.4260 | 0.5485 | 0.1422 | 0.3722 |

消融结果（表 4）表明 loss mask 与 advantage mask 都非常关键。禁用 loss mask 会在 PPO 与 GRPO 上稳定降低表现：例如在 PPO 中（advantage mask 已禁用的条件下），再移除 loss mask 会使平均 EM 从 0.3136 降到 0.3022；在 GRPO 中平均 EM 从 0.3877 降到 0.3722。这说明 loss mask 对于把梯度聚焦在智能体生成 token 上是必要的。同样地，对于 PPO，在启用 loss mask 的前提下禁用 advantage mask，会使平均 EM 从 0.3719 大幅降到 0.3136，验证了准确信用分配的重要性。这些发现表明：这些掩码策略是 Agent-R1 在交互式 LLM 智能体中实现有效策略优化的关键设计点。

---

## 5 结论

本文通过扩展经典 MDP 框架，澄清了如何将强化学习有效应用于 LLM 智能体：捕获多轮交互、环境反馈与过程奖励。在此基础上，我们提出 Agent-R1——一个模块化、可扩展的框架，支持多轮 rollout、精确信用分配，以及工具与环境的灵活集成。在多跳问答任务上的实验表明，Agent-R1 能让 LLM 智能体相较基线方法获得显著提升；消融实验进一步确认了关键策略优化组件的重要性。我们希望 Agent-R1 能为未来在更可扩展、更统一的智能体化 LLM 强化学习训练方向上提供基础。

---

## 参考文献

[1] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.  
[2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.  
[3] Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei Cao, Jie Ma, et al. A survey on knowledge-oriented retrieval-augmented generation. arXiv preprint arXiv:2503.10677, 2025.  
[4] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.  
[5] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.  
[6] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.  
[7] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680, 2024.  
[8] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060, 2020.  
[9] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, 2023.  
[10] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022.  
[11] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.  
[12] Chuang Jiang, Mingyue Cheng, Xiaoyu Tao, Qingyang Mao, Jie Ouyang, and Qi Liu. Tablemind: An autonomous programmatic agent for tool-augmented table reasoning. arXiv preprint arXiv:2509.06278, 2025.  
[13] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.  
[14] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. CAMEL: Communicative agents for "mind" exploration of large language model society. Advances in Neural Information Processing Systems, 36:51991–52008, 2023.  
[15] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.  
[16] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. AgentBench: Evaluating llms as agents. arXiv preprint arXiv:2308.03688, 2023.  
[17] Yucong Luo, Yitong Zhou, Mingyue Cheng, Jiahao Wang, Daoyu Wang, Tingyue Pan, and Jintao Zhang. Time series forecasting as reasoning: A slow-thinking approach with reinforced llms. arXiv preprint arXiv:2506.10630, 2025.  
[18] Jie Ouyang, Tingyue Pan, Mingyue Cheng, Ruiran Yan, Yucong Luo, Jiaying Lin, and Qi Liu. HOH: A dynamic benchmark for evaluating the impact of outdated information on retrieval-augmented generation. arXiv preprint arXiv:2503.04800, 2025.  
[19] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.  
[20] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.  
[21] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, et al. KILT: a benchmark for knowledge intensive language tasks. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2523–2544, 2021.  
[22] Jiahao Qiu, Xuan Qi, Tongcheng Zhang, Xinzhe Juan, Jiacheng Guo, Yifu Lu, Yimin Wang, Zixin Yao, Qihan Ren, Xun Jiang, et al. Alita: Generalist agent enabling scalable agentic reasoning with minimal predefinition and maximal self-evolution. arXiv preprint arXiv:2505.20286, 2025.  
[23] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, et al. Qwen2.5 technical report, 2025.  
[24] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.  
[25] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551, 2023.  
[26] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.  
[27] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.  
[28] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.  
[29] Daoyu Wang, Mingyue Cheng, Qi Liu, Shuo Yu, Zirui Liu, and Ze Guo. Paperarena: An evaluation benchmark for tool-augmented agentic reasoning on scientific literature. arXiv preprint arXiv:2510.10909, 2025.  
[30] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.  
[31] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.  
[32] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. AutoGen: Enabling next-gen llm applications via multi-agent conversations. In First Conference on Language Modeling, 2024.  
[33] Zhiheng Xi, Yiwen Ding, Wenxiang Chen, Boyang Hong, Honglin Guo, Junzhe Wang, Xin Guo, Dingwen Yang, Chenyang Liao, Wei He, et al. AgentGym: Evaluating and training large language model-based agents across diverse environments. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27914–27961, 2025.  
[34] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.  
[35] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.  
[36] Shuo Yu, Mingyue Cheng, Qi Liu, Daoyu Wang, Jiqian Yang, Jie Ouyang, Yucong Luo, Chenyi Lei, and Enhong Chen. Multi-source knowledge pruning for retrieval-augmented generation: A benchmark and empirical study. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, pages 3931–3941, 2025.  
[37] Shuo Yu, Mingyue Cheng, Daoyu Wang, Qi Liu, Zirui Liu, Ze Guo, and Xiaoyu Tao. Memweaver: A hierarchical memory from textual interactive behaviors for personalized generation. arXiv preprint arXiv:2510.07713, 2025.
