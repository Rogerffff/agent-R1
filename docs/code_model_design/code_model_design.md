下面是一套**可以直接照做**、并且**最后能写进简历**的 RLVR Coding Agent / Coding Model 实验设计（对齐你现有背景：**verl + GRPO + Python 沙盒 + 单测可验证奖励**，并吸收你 Deep Research 里提到的 VeRPO / RLTF / μCode / SWE-agent / DAPO 等关键思想）。

我按“从 0 到 1 全流程”写：**数据准备 → 环境/工具 → SFT →（可选）DPO → GRPO（单轮）→（可选）多轮修复 → 评测回归 → 最小实验矩阵 → 最终简历可写点**。

---

# 0) 项目定位与最终产出（你要在简历上证明什么）

**项目名建议**：RLVR Coding model（Unit-test Reward, GRPO on verl）

**你要产出的硬结果（可复现）**：

1. 在 **HumanEval / MBPP** 上的 **pass@1 / pass@k** 提升曲线（SFT vs DPO vs GRPO）
2. 稳定性：至少 **2 个 seed**（均值±方差）
3. 成本指标：平均每题 **token**、平均测试执行耗时、tokens/s 或 cost-per-solved
4. 消融：稠密奖励 vs 稀疏奖励（最值钱）、（可选）长度惩罚/超时惩罚

---

# 1) 数据准备（用哪些数据集、每一步怎么用）

## 1.1 数据集选择（高性价比组合）

**训练/调试（便宜、快）**

* **MBPP**：小、题短、测试简单 → 最适合把 pipeline 跑通、做 SFT 起步、做小规模 RL

**主训练（更像业务、可持续）**

* **APPS（建议先用 easy/intro 子集）** 或 **CodeContests**：题更难、测试更丰富 → 更适合 RLVR（执行反馈更有信息量）

**最终展示（强背书、简历识别度最高）**

* **HumanEval**：最终报告 pass@1/pass@k（最好把它只用于评测，不混进训练）

> 推荐顺序：**MBPP（跑通）→ APPS easy（RL）→ HumanEval（最终评测）**

---

## 1.2 统一样本格式（你要做的“数据工程”）

把不同数据集统一成同一个内部格式，后面所有训练/评测/rollout 都吃这一个结构：

* `task_id`
* `prompt`：题面 + 明确要求（函数签名、输入输出、限制）
* `entry_point`：函数名（HumanEval/MBPP 常有）
* `tests`：单测代码（如果有）
* `timeout_s`：默认 2s（可按数据集调整）
* `difficulty`：可空（APPS/CodeContests 常有）

**prompt 模板建议固定**（重要：减少分布漂移）：

* 强制要求输出“**只包含代码**”
* 明确函数签名
* 明确不要读写文件、不要网络、不要随机数（或固定 seed）

---

## 1.3 数据清洗/过滤（省钱、提高信号）

* 过滤明显不可运行/依赖复杂的题（缺库、需要 IO 文件、测试超慢）
* 把测试执行超时的题打标，训练时降低采样概率（课程学习）
* 去重：同一道题可能有多个变体，保留一个

---

# 2) 环境与工具设计（ACI-lite，决定你训练能不能稳）

## 2.1 沙盒执行（必须）

建议用 Docker 或同等隔离：

* 禁网
* 只读挂载 tests
* 内存限制（如 512MB/1GB）
* 超时 kill（2s/5s）
* 禁止访问宿主文件系统

## 2.2 工具接口（两种模式：先做 A，再做 B）

### 模式 A（推荐先做）：环境每轮自动跑测试（更稳、更省 token）

* 模型 action：输出 `code`（完整代码或函数实现）
* 环境 step：自动 `run_tests(code)`
* 返回 observation（结构化、短）：

  * `passed`, `total`, `pass_ratio`
  * `error_type`: `SYNTAX/RUNTIME/TIMEOUT/ASSERT`
  * `failed_tests`：最多 K 条名字
  * `error_summary`：截断到最后 N 行

### 模式 B（可选加分）：模型显式决定何时 run_tests（更 agentic）

* 模型输出可以包含：`Action: RUN_TESTS` 或 `Action: SUBMIT_CODE`
* 每次 RUN_TESTS 扣一点成本（reward penalty），让模型学会“必要时才测”

> 你先把 **模式 A 跑通**，出结果后再加模式 B 做亮点（成本可控）。

---

# 3) 训练全流程（SFT → DPO → GRPO），每一步怎么做

## Phase 0：Baseline 评测（必须有）

对你选的 base model（建议 coder 模型，如 1.5B/3B 级别先跑通）：

* 在 MBPP / HumanEval 跑 pass@1、exec success rate、平均 token、平均耗时
* 形成 baseline 表格（简历写“从基线到提升”）

---

## Phase 1：SFT（先让模型“能写对格式、能跑起来”）

**数据**：MBPP +（可选）APPS easy 的参考解
**目标**：显著降低 SyntaxError/RuntimeError，提升 exec success rate

训练要点：

* 强约束输出只含代码（SFT 阶段非常关键）
* max_new_tokens 控制在合理范围（比如 256~512，视题目）
* 训练完成后立刻跑回归评测（见第 6 节）

你应该看到：

* exec success rate ↑
* pass@1 小幅↑或持平（主要先把“能跑”解决）

---

## Phase 2（可选但强烈推荐）：DPO 初始化（把“会通过测试的解”变成偏好）

目的：**用较低成本**提高 GRPO 起点，让在线 RL 更稳、更快收敛。

### 2.1 自动构造偏好对（关键工程点）

对每道训练题：

1. 用 SFT 模型采样 N 个候选代码（N=2~4 足够）
2. 对每个候选跑测试得到 `pass_ratio`
3. 选：

* `chosen`：pass_ratio 最高（若并列选更短的）
* `rejected`：pass_ratio 最低（或超时/语法错优先作为 rejected）

得到 DPO 数据：`(prompt, chosen, rejected)`
**数据集**：先用 MBPP 或 APPS easy 子集构造（省钱）

### 2.2 DPO 训练

* 训练轮数不需要多（1~2 epoch 往往就有收益）
* DPO 后再评测一次（应该比纯 SFT 更稳一点）

---

## Phase 3：GRPO（单轮 RLVR，主菜）

这是你简历里最重要的一步：**可验证奖励 + 在线 RL**。

### 3.1 rollout 采样设置（GRPO 的“组内对比”）

* 每个 prompt 采样 `G` 条候选（建议从 **4** 开始，资源够再到 8）
* 温度 `T` 稍高一点鼓励多样性（训练用），评测用低温/贪心
* 每条候选执行 `run_tests` 得 reward

### 3.2 Reward 设计（直接可用）

核心（稠密、稳定）：

* `r_main = pass_ratio = passed/total`（0~1）

加一点“全通过奖励”拉开差距：

* `r = r_main + 0.2 * I[p_passed_all]`

惩罚项（控制成本与投机，别加太多）：

* 超时：`r -= 0.5`（或直接置 0）
* 语法错误：`r -= 0.2`
* 长度惩罚：`r -= α * (output_tokens/1000)`（α 很小，如 0.02 量级）

> 这套就是你 Deep Research 里 VeRPO/RLTF 的核心思想：**稠密可验证奖励 + 错误类型惩罚 + 成本约束**。

### 3.3 稳定训练细节（照 DAPO/GRPO 思路做工程化）

你在 verl 里建议“默认就开”的东西：

* KL 约束（参考模型用 SFT 或 DPO checkpoint）
* 优势标准化 / clip
* 梯度裁剪（防发散）
* 过滤超长输出（超过 max tokens 的样本直接丢弃或强惩罚）
* 课程学习：先在 MBPP / APPS easy 训练，成功率上来后再放难题

---

## Phase 4（可选加分）：多轮修复（2~3 轮，按 μCode 思路省钱做）

目的：让你的项目“更 agentic”，并且很贴工业 coding agent。

**关键是别把它做得很重**：建议只做 **最多 2~3 轮**，并且 observation **只保留上一轮反馈摘要**（不要堆全历史）。

流程（模式 A 自动测试）：

* Round 1：生成 code → 测试反馈
* Round 2：带上“失败摘要”再生成修复版 code → 测试
* Round 3：可选

奖励分配（简单稳）：

* 每轮 reward = 当前轮 pass_ratio（即时奖励）
* 加一个“轮数成本”：每多一轮扣一点（鼓励更快解决）

你最终可以报：

* 多轮后 pass@k 或 success rate ↑
* 平均轮数 / 平均 token 成本（trade-off）

---

# 4) 评测与回归（你要把它做成“工程化系统”）

## 4.1 固定回归集（训练过程必须反复跑）

* 从 MBPP/HumanEval 各抽一小撮（例如 50~100 题）
* 每隔固定 step 评测一次，画曲线：

  * pass@1
  * exec success rate
  * avg tokens
  * avg runtime
  * reward 平均值、KL

## 4.2 最终评测（用于简历）

* HumanEval：pass@1、pass@5 / pass@10（选一个 k）
* MBPP：同样指标作为补充
* 报 2 个 seed（均值±std）

---

# 5) 最小可行实验矩阵（控制 GPU 成本，但能写简历）

建议你至少完成下面 4 个（3~6 次跑就够）：

1. **Baseline**：未训练 / base model → HumanEval pass@1
2. **SFT**：SFT 后 → pass@1 + exec success rate（通常很明显）
3. **SFT + GRPO（稠密奖励）**：核心结果
4. **消融：稀疏 vs 稠密**

   * 稀疏：全通过=1 否则 0
   * 稠密：passed/total
     这条消融最能证明你“懂 RLVR 关键点”

可选加分（看预算）：
5) **加 DPO 初始化**：SFT→DPO→GRPO（通常收敛更快更稳）
6) **单轮 vs 多轮**：2 轮修复是否值得（提升 vs token 成本）

---

# 6) 你最终在简历里能写的“项目亮点”（对应上面每个模块）

你做完上述流程后，简历 bullet 很好写（示例方向）：

* 搭建 RLVR Coding 训练基座：在隔离 Python 沙盒中执行单测，以通过率构造稠密可验证奖励，支持 SFT→DPO→GRPO 全链路训练与回归评测。
* 设计奖励与稳定化训练：采用 pass_ratio 稠密奖励 + 超时/错误/长度惩罚，并通过 KL 约束、优势归一化、课程学习提升 GRPO 稳定性。
* 在 HumanEval 上将 pass@1 从 X% 提升到 Y%（平均±std，2 seeds），同时将平均 token/题降低 Z% 或控制在可接受范围。
* （可选）引入多轮执行反馈修复（2~3 轮），提升难题成功率并量化 token 成本与轮数 trade-off。

---

# 7) 你下一步怎么开工（不需要额外澄清也能直接动手）

按这个顺序做最稳：

1. 先把 **统一数据格式 + run_tests 沙盒 + 评测脚本**做出来（这是基座）
2. 跑 baseline（HumanEval/MBPP）
3. SFT（先让能跑起来）
4. GRPO（稠密奖励）
5. 做“稀疏 vs 稠密”消融（最值）
6. 再考虑 DPO / 多轮修复做加分


