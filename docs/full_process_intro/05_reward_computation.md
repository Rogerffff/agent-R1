# 奖励计算系统

本文档详细介绍 Agent-R1 的奖励计算系统，包括 `AgentRewardManager` 的实现、ReTool 任务的奖励函数设计、奖励张量的形状和存储方式，以及具体的奖励计算示例。

---

## 目录

1. [奖励系统概览](#1-奖励系统概览)
2. [AgentRewardManager 类](#2-agentrewardmanager-类)
3. [ReTool 奖励函数](#3-retool-奖励函数)
4. [奖励路由机制](#4-奖励路由机制)
5. [奖励张量存储](#5-奖励张量存储)
6. [完整奖励计算示例](#6-完整奖励计算示例)
7. [奖励函数扩展](#7-奖励函数扩展)
8. [常见问题](#8-常见问题)

---

## 1. 奖励系统概览

Agent-R1 的奖励系统采用**规则奖励**（Rule-based Reward）机制，根据模型输出的格式和答案正确性计算奖励：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           奖励计算系统架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     AgentRewardManager                              │   │
│  │                                                                     │   │
│  │  输入: DataProto (包含 prompt, response, ground_truth)              │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  for i in range(batch_size):                                │   │   │
│  │  │      │                                                       │   │   │
│  │  │      ├─▶ 1. 解码 prompt + response → sequences_str          │   │   │
│  │  │      │                                                       │   │   │
│  │  │      ├─▶ 2. 获取 ground_truth                               │   │   │
│  │  │      │                                                       │   │   │
│  │  │      ├─▶ 3. 根据 data_source 路由到对应的奖励函数            │   │   │
│  │  │      │       ↓                                               │   │   │
│  │  │      │   ┌──────────────────────────────────────────────┐   │   │   │
│  │  │      │   │  _default_compute_score()                     │   │   │   │
│  │  │      │   │  • compute_score_format()    → 格式奖励       │   │   │   │
│  │  │      │   │  • compute_score_answer()    → 答案奖励       │   │   │   │
│  │  │      │   │  • compute_score_format_answer() → 组合奖励   │   │   │   │
│  │  │      │   └──────────────────────────────────────────────┘   │   │   │
│  │  │      │                                                       │   │   │
│  │  │      └─▶ 4. 将奖励放置在 response 最后一个有效 token 位置    │   │   │
│  │  │                                                              │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  输出: reward_tensor [batch_size, response_length]                 │   │
│  │        reward_extra_info (acc, format 等额外信息)                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `agent_r1/src/agent_reward_manager.py` | AgentRewardManager 实现 |
| `agent_r1/src/reward_score/__init__.py` | 奖励路由函数 |
| `agent_r1/src/reward_score/retool.py` | ReTool 任务奖励函数 |
| `agent_r1/src/reward.py` | 奖励加载和异步计算工具 |

### 奖励类型

| 奖励类型 | 说明 | 值范围 |
|----------|------|--------|
| **格式奖励** (`format`) | 检查输出是否符合格式要求 | 0 或 1 |
| **答案奖励** (`acc`) | 检查答案是否正确 | 0 或 1 |
| **组合奖励** (`score`) | 格式正确时返回答案奖励，否则返回 -1 | -1, 0 或 1 |

---

## 2. AgentRewardManager 类

### 2.1 类定义

**文件**: `agent_r1/src/agent_reward_manager.py:23-30`

```python
class AgentRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,           # HuggingFace tokenizer
        num_examine,         # 打印到控制台的样本数
        compute_score=None,  # 自定义计算函数（可选）
        reward_fn_key="data_source"  # 用于路由的键名
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
```

### 2.2 奖励计算流程

**文件**: `agent_r1/src/agent_reward_manager.py:32-112`

```python
def __call__(self, data: DataProto, return_dict=False):
    """计算奖励"""

    # 如果已有 rm_scores，直接返回
    if "rm_scores" in data.batch.keys():
        if return_dict:
            return {"reward_tensor": data.batch["rm_scores"]}
        else:
            return data.batch["rm_scores"]

    # 初始化奖励张量
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    reward_extra_info = defaultdict(list)

    already_print_data_sources = {}

    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        # 1. 提取 prompt 和 response
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # 2. 解码为字符串
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)

        # 清理 pad token
        pad_token_id = self.tokenizer.pad_token_id
        sequences_str = sequences_str.split(self.tokenizer.decode([pad_token_id]))[0]
        if not sequences_str.endswith(self.tokenizer.eos_token):
            sequences_str += self.tokenizer.eos_token

        # 3. 获取 ground_truth 和 data_source
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", None)

        # 4. 计算奖励
        score = self.compute_score(
            data_source=data_source,
            solution_str=sequences_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        # 5. 处理返回值
        if isinstance(score, dict):
            reward = score["score"]
            for key, value in score.items():
                reward_extra_info[key].append(value)
        else:
            reward = score

        # 6. 放置奖励到最后一个有效 token 位置
        reward_tensor[i, valid_response_length - 1] = reward

        # 7. 打印调试信息（前 num_examine 个样本）
        if data_source not in already_print_data_sources:
            already_print_data_sources[data_source] = 0

        if already_print_data_sources[data_source] < self.num_examine:
            already_print_data_sources[data_source] += 1
            print("[prompt+response]", sequences_str)
            print("[ground_truth]", ground_truth)
            if isinstance(score, dict):
                for key, value in score.items():
                    print(f"[{key}]", value)
            else:
                print("[score]", score)

    # 返回结果
    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }
    else:
        return reward_tensor
```

### 2.3 奖励放置位置

奖励被放置在 response 的**最后一个有效 token 位置**：

```python
reward_tensor[i, valid_response_length - 1] = reward
```

**示例**：
```
response_ids = [tok1, tok2, tok3, tok4, tok5, PAD, PAD, PAD, PAD, PAD]
                ^     ^     ^     ^     ^
                0     1     2     3     4     (valid_response_length = 5)

reward_tensor = [0,    0,    0,    0,    1.0,  0,    0,    0,    0,    0]
                                         ^
                                   位置 4 = valid_response_length - 1
```

这样设计的原因：
1. **稀疏奖励**：只在序列末尾给予奖励，符合 RL 的稀疏奖励设定
2. **梯度传播**：奖励位置与 EOS token 对应，确保梯度能传播到整个序列
3. **与 Action Mask 配合**：只有模型生成的 token 才会收到奖励信号

---

## 3. ReTool 奖励函数

### 3.1 格式奖励 (compute_score_format)

**文件**: `agent_r1/src/reward_score/retool.py:13-47`

```python
def compute_score_format(solution_str):
    """
    格式奖励函数

    检查输出是否符合以下格式：
    1. 包含 <|im_start|>assistant 和 <|im_end|> 标签
    2. 最后一个 assistant 块包含 <answer>\boxed{...}</answer>

    Returns:
        1.0: 格式正确
        0.0: 格式不正确
    """
    if solution_str is None:
        return 0.0

    try:
        # 提取所有 assistant 块
        assistant_blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
            solution_str,
            re.DOTALL
        )

        format_reward = 0.0

        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0

        # 检查最后一个 assistant 块是否包含正确格式的答案
        last_assistant_block = assistant_blocks[-1]
        think_answer_match = re.search(
            r'<answer>.*\\boxed\{.*\}.*</answer>$',
            last_assistant_block,
            re.DOTALL
        )
        if think_answer_match:
            format_reward = 1.0

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0

    return format_reward
```

**格式要求**：
```
<|im_start|>assistant
<think>思考过程...</think>
<answer>
\boxed{最终答案}
</answer><|im_end|>
```

### 3.2 答案奖励 (compute_score_answer)

**文件**: `agent_r1/src/reward_score/retool.py:50-60`

```python
def compute_score_answer(solution_str: str, ground_truth: str) -> float:
    """
    答案奖励函数

    检查模型的答案是否与 ground_truth 匹配

    Returns:
        1.0: 答案正确
        0.0: 答案错误或无法提取答案
    """
    # 提取所有 assistant 块
    assistant_blocks = re.findall(
        r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
        solution_str,
        re.DOTALL
    )
    if len(assistant_blocks) == 0:
        return 0.0

    last_assistant_block = assistant_blocks[-1]

    # 提取 <answer>...</answer> 中的内容
    answer = extract_solution(last_assistant_block)
    if answer is None:
        return 0.0

    # 提取 \boxed{...} 中的内容
    answer = extract_boxed_content(answer)

    # 使用 mathruler 的 grade_answer 函数比较答案
    return 1.0 if grade_answer(answer, ground_truth) else 0.0
```

**答案提取流程**：
```
输入: "<think>...</think><answer>\boxed{-7/25}</answer>"

1. extract_solution() → "\boxed{-7/25}"
2. extract_boxed_content() → "-7/25"
3. grade_answer("-7/25", "-7/25") → True
4. 返回 1.0
```

### 3.3 辅助函数

**提取答案**：
```python
def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)

    if match:
        return match.group(1).strip()
    return None
```

**提取 boxed 内容**（来自 mathruler 库）：
```python
from mathruler.grader import extract_boxed_content, grade_answer
# extract_boxed_content("\boxed{-7/25}") → "-7/25"
```

### 3.4 组合奖励 (compute_score_format_answer)

**文件**: `agent_r1/src/reward_score/retool.py:63-83`

```python
def compute_score_format_answer(solution_str, ground_truth):
    """
    组合奖励函数

    奖励策略：
    - 格式正确 + 答案正确 → 1.0
    - 格式正确 + 答案错误 → 0.0
    - 格式不正确 → -1.0 (惩罚)

    Returns:
        1.0: 格式正确且答案正确
        0.0: 格式正确但答案错误
        -1.0: 格式不正确
    """
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        if format_reward == 1.0:
            return answer_reward  # 0.0 或 1.0
        else:
            return -1.0  # 格式惩罚
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return -1.0
```

**奖励矩阵**：

| 格式 | 答案 | 组合奖励 |
|------|------|----------|
| 正确 (1.0) | 正确 (1.0) | 1.0 |
| 正确 (1.0) | 错误 (0.0) | 0.0 |
| 错误 (0.0) | - | -1.0 |

---

## 4. 奖励路由机制

### 4.1 _default_compute_score 函数

**文件**: `agent_r1/src/reward_score/__init__.py:55-60`

```python
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    默认的奖励计算函数

    返回包含三个指标的字典：
    - score: 组合奖励（用于 RL 训练）
    - acc: 答案准确率（用于评估）
    - format: 格式正确率（用于评估）
    """
    return {
        "score": _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info),
        "acc": _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info),
        "format": _default_compute_score_format(data_source, solution_str, extra_info),
    }
```

### 4.2 根据 data_source 路由

**文件**: `agent_r1/src/reward_score/__init__.py:37-53`

```python
def _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info=None):
    """根据 data_source 路由到对应的奖励函数"""

    if data_source == 'hotpotqa/hotpot_qa' or \
       data_source == 'bdsaglam/musique' or \
       data_source == 'xanhho/2WikiMultihopQA':
        # 多跳 QA 任务
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_format_answer(solution_str, ground_truth)

    elif data_source == 'openai/gsm8k':
        # GSM8K 数学任务
        from . import gsm8k
        res = gsm8k.compute_score_format_answer(solution_str, ground_truth)

    elif data_source == 'BytedTsinghua-SIA/DAPO-Math-17k':
        # ReTool 数学任务 (DAPO-Math)
        from . import retool
        res = retool.compute_score_format_answer(solution_str, ground_truth)

    else:
        raise NotImplementedError

    # 确保返回 float
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
```

### 4.3 支持的数据源

| data_source | 奖励模块 | 任务类型 |
|-------------|----------|----------|
| `hotpotqa/hotpot_qa` | `qa_em_and_format` | 多跳问答 |
| `bdsaglam/musique` | `qa_em_and_format` | 多跳问答 |
| `xanhho/2WikiMultihopQA` | `qa_em_and_format` | 多跳问答 |
| `openai/gsm8k` | `gsm8k` | 数学推理 |
| `BytedTsinghua-SIA/DAPO-Math-17k` | `retool` | 数学+代码 |

---

## 5. 奖励张量存储

### 5.1 张量形状

```python
# 输入
data.batch["responses"]  # shape: [batch_size, response_length]
# 例如: [128, 16384]

# 输出
reward_tensor  # shape: [batch_size, response_length]
# 例如: [128, 16384]
```

### 5.2 奖励放置示意图

```
样本 0 (response_length=16384, valid_response_length=500):

response_ids:
[tok1, tok2, ..., tok500, PAD, PAD, ..., PAD]
|<----- 500 个有效 ----->||<--- 15884 个 PAD --->|

reward_tensor:
[0, 0, ..., 0, reward, 0, 0, ..., 0]
             ^
          位置 499 (= 500 - 1)

attention_mask (response 部分):
[1, 1, ..., 1, 0, 0, ..., 0]
|<--- 500 --->||<--- 15884 --->|

action_mask:
[1, 1, ..., 1, 0, 0, ..., 0, 1, 1, ..., 1, 0, ..., 0]
|<- model ->||<- tool ->||<- model ->||<- pad ->|
```

### 5.3 reward_extra_info 结构

```python
reward_extra_info = {
    "score": [1.0, -1.0, 0.0, 1.0, ...],   # 组合奖励列表
    "acc": [1.0, 0.0, 0.0, 1.0, ...],      # 答案准确率列表
    "format": [1.0, 0.0, 1.0, 1.0, ...],   # 格式正确率列表
}
```

这些额外信息用于：
- 记录训练指标
- WandB 日志
- 验证集评估

---

## 6. 完整奖励计算示例

### 6.1 输入数据

```python
# 假设 batch_size=2, response_length=100

data = DataProto(
    batch = {
        "prompts": tensor([2, 1024]),
        "responses": tensor([2, 100]),
        "attention_mask": tensor([2, 1124]),  # prompt + response
    },
    non_tensor_batch = {
        "reward_model": array([
            {"style": "rule", "ground_truth": "-7/25"},
            {"style": "rule", "ground_truth": "42"},
        ], dtype=object),
        "data_source": array([
            "BytedTsinghua-SIA/DAPO-Math-17k",
            "BytedTsinghua-SIA/DAPO-Math-17k",
        ], dtype=object),
    }
)
```

### 6.2 样本 0 处理

**解码后的字符串**：
```
<|im_start|>user
Solve the following problem step by step...
*user question:*
In triangle ABC, sin∠A = 4/5 and ∠A < 90°. Find the value of cos2A.
<|im_end|>
<|im_start|>assistant
<think>
我需要计算 cos2A。使用公式 cos2A = 1 - 2sin²A
</think>
<code>
```python
sin_A = 4/5
cos_2A = 1 - 2 * sin_A**2
print(cos_2A)
```
</code><|im_end|>
<|im_start|>user
<interpreter>
-0.28
</interpreter><|im_end|>
<|im_start|>assistant
<think>
结果是 -0.28，转换为分数是 -7/25
</think>
<answer>
\boxed{-7/25}
</answer><|im_end|>
```

**奖励计算**：
```python
# 1. compute_score_format()
#    检查最后一个 assistant 块: "<think>...</think><answer>\boxed{-7/25}</answer>"
#    包含 <answer>\boxed{...}</answer> → format_reward = 1.0

# 2. compute_score_answer()
#    extract_solution() → "\boxed{-7/25}"
#    extract_boxed_content() → "-7/25"
#    grade_answer("-7/25", "-7/25") → True
#    → answer_reward = 1.0

# 3. compute_score_format_answer()
#    format_reward == 1.0 → 返回 answer_reward = 1.0

# 最终
score = {"score": 1.0, "acc": 1.0, "format": 1.0}
```

**奖励放置**：
```python
valid_response_length = 55  # 假设有效响应长度
reward_tensor[0, 54] = 1.0  # 位置 54 = 55 - 1
```

### 6.3 样本 1 处理（格式错误示例）

**解码后的字符串**：
```
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
The answer is 42.<|im_end|>
```

**奖励计算**：
```python
# 1. compute_score_format()
#    检查最后一个 assistant 块: "The answer is 42."
#    不包含 <answer>\boxed{...}</answer> → format_reward = 0.0

# 2. compute_score_answer()
#    没有 <answer> 标签 → answer_reward = 0.0

# 3. compute_score_format_answer()
#    format_reward == 0.0 → 返回 -1.0 (惩罚)

# 最终
score = {"score": -1.0, "acc": 0.0, "format": 0.0}
```

**奖励放置**：
```python
valid_response_length = 20  # 假设有效响应长度
reward_tensor[1, 19] = -1.0  # 位置 19 = 20 - 1
```

### 6.4 最终奖励张量

```python
reward_tensor = tensor([
    # 样本 0: 位置 54 有奖励 1.0
    [0, 0, ..., 0, 1.0, 0, ..., 0],

    # 样本 1: 位置 19 有奖励 -1.0
    [0, 0, ..., 0, -1.0, 0, ..., 0],
])  # shape: [2, 100]

reward_extra_info = {
    "score": [1.0, -1.0],
    "acc": [1.0, 0.0],
    "format": [1.0, 0.0],
}
```

---

## 7. 奖励函数扩展

### 7.1 添加新的数据源支持

1. **创建奖励函数文件**：

```python
# agent_r1/src/reward_score/my_task.py

def compute_score_format(solution_str):
    """格式奖励"""
    # 实现你的格式检查逻辑
    pass

def compute_score_answer(solution_str, ground_truth):
    """答案奖励"""
    # 实现你的答案检查逻辑
    pass

def compute_score_format_answer(solution_str, ground_truth):
    """组合奖励"""
    format_reward = compute_score_format(solution_str)
    answer_reward = compute_score_answer(solution_str, ground_truth)

    if format_reward == 1.0:
        return answer_reward
    else:
        return -1.0
```

2. **注册路由**：

```python
# agent_r1/src/reward_score/__init__.py

def _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info=None):
    # ... 现有代码 ...

    elif data_source == 'my_org/my_dataset':
        from . import my_task
        res = my_task.compute_score_format_answer(solution_str, ground_truth)

    # ...
```

### 7.2 使用自定义奖励函数

通过配置文件指定自定义奖励函数：

```yaml
# agent_trainer.yaml
custom_reward_function:
  path: "/path/to/my_reward.py"
  name: "my_compute_score"
  reward_kwargs:
    param1: value1
```

**加载逻辑**（`agent_r1/src/reward.py:22-54`）：

```python
def get_custom_reward_fn(config):
    """加载自定义奖励函数"""
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    # 动态加载模块
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取函数
    function_name = reward_fn_config.get("name")
    raw_fn = getattr(module, function_name)

    # 包装额外参数
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn
```

---

## 8. 常见问题

### 8.1 奖励为什么放在最后一个有效 token？

**原因**：
1. **稀疏奖励设定**：符合 RL 中的延迟奖励（只在 episode 结束时给予奖励）
2. **梯度传播**：通过反向传播，奖励信号会影响整个生成序列
3. **与 PPO/GRPO 算法配合**：这些算法期望奖励在序列末尾

### 8.2 为什么格式错误给 -1.0 惩罚？

**原因**：
1. **强制格式遵循**：鼓励模型学习正确的输出格式
2. **区分错误类型**：
   - 格式正确但答案错误 → 0.0（中性）
   - 格式错误 → -1.0（惩罚）
3. **加速学习**：负奖励让模型更快学会避免格式错误

### 8.3 如何处理多轮工具调用的奖励？

当前实现中，奖励只在**最终响应末尾**给予，不对中间的工具调用单独给奖励：

```
Turn 1: [LLM 生成] [工具响应]
Turn 2: [LLM 生成] [工具响应]
Turn 3: [LLM 生成 + 答案] ← 奖励放在这里

action_mask: [1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1]
reward:      [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,R]
                                              ^
                                        最后一个有效位置
```

### 8.4 如何添加过程奖励（Process Reward）？

如果需要对中间步骤给予奖励，可以修改 `AgentRewardManager`：

```python
# 在 __call__ 方法中
for turn_idx, turn_end_pos in enumerate(turn_positions):
    # 对每轮工具调用给予中间奖励
    process_reward = compute_process_reward(...)
    reward_tensor[i, turn_end_pos] = process_reward

# 最终答案奖励
reward_tensor[i, valid_response_length - 1] = final_reward
```

---

## 总结

本文档详细介绍了 Agent-R1 的奖励计算系统：

1. **AgentRewardManager** 管理整个奖励计算流程
2. **ReTool 奖励函数** 包含格式奖励、答案奖励和组合奖励
3. **奖励路由机制** 根据 `data_source` 选择对应的奖励函数
4. **奖励张量** 将奖励放置在 response 的最后一个有效 token 位置

关键要点：
- 组合奖励：格式正确时返回答案奖励，否则返回 -1.0 惩罚
- 奖励位置：`reward_tensor[i, valid_response_length - 1]`
- 额外信息：`score`、`acc`、`format` 用于监控和评估
- 可扩展：支持自定义奖励函数和新数据源

下一篇文档将介绍训练循环详解 (`06_training_loop.md`)，详细讲解 `RayAgentTrainer.fit()` 方法的完整训练流程。
