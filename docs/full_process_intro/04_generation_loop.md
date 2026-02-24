# 多轮生成循环

本文档详细介绍 Agent-R1 的多轮生成循环机制，包括 `ToolGenerationManager` 的核心实现、`run_llm_loop()` 主循环的每个步骤、Action Mask 的创建机制，以及完整的多轮工具调用示例。

---

## 目录

1. [多轮生成概览](#1-多轮生成概览)
2. [ToolGenerationManager 类](#2-toolgenerationmanager-类)
3. [run_llm_loop() 主循环详解](#3-run_llm_loop-主循环详解)
4. [Active Mask 机制](#4-active-mask-机制)
5. [滚动状态更新](#5-滚动状态更新)
6. [Action Mask 创建](#6-action-mask-创建)
7. [完整两轮调用示例](#7-完整两轮调用示例)
8. [张量辅助类](#8-张量辅助类)
9. [数据形状变化](#9-数据形状变化)

---

## 1. 多轮生成概览

多轮生成是 Agent-R1 的核心机制，它实现了 LLM 与工具环境之间的交互循环：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           多轮生成循环概览                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        ToolGenerationManager                        │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                     run_llm_loop()                          │   │   │
│  │  │                                                             │   │   │
│  │  │    for turn in range(max_turns):                           │   │   │
│  │  │        │                                                    │   │   │
│  │  │        ├─▶ 1. 检查序列长度                                  │   │   │
│  │  │        │                                                    │   │   │
│  │  │        ├─▶ 2. 筛选活跃样本                                  │   │   │
│  │  │        │       ↓                                            │   │   │
│  │  │        │   ┌────────────────────┐                          │   │   │
│  │  │        │   │  generate_sequences │ ← Actor Worker           │   │   │
│  │  │        │   └────────────────────┘                          │   │   │
│  │  │        │       ↓                                            │   │   │
│  │  │        ├─▶ 3. 分布式生成响应                                │   │   │
│  │  │        │       ↓                                            │   │   │
│  │  │        │   ┌────────────────────┐                          │   │   │
│  │  │        │   │    env.step()      │ ← Tool Environment        │   │   │
│  │  │        │   └────────────────────┘                          │   │   │
│  │  │        │       ↓                                            │   │   │
│  │  │        ├─▶ 4. 执行工具调用                                  │   │   │
│  │  │        │                                                    │   │   │
│  │  │        ├─▶ 5. 更新 active_mask                             │   │   │
│  │  │        │                                                    │   │   │
│  │  │        └─▶ 6. 更新滚动状态                                  │   │   │
│  │  │                                                             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  输出:                                                              │   │
│  │  • prompts: 原始 prompt                                            │   │
│  │  • responses: 完整响应（含工具输出）                                 │   │
│  │  • action_mask: 区分模型生成 vs 环境反馈                            │   │
│  │  • turns: 每个样本的轮数                                            │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `agent_r1/llm_agent/generation.py` | ToolGenerationManager 和 run_llm_loop() 实现 |
| `agent_r1/llm_agent/tensor_helper.py` | TensorHelper 张量操作辅助类 |

### 核心概念

| 概念 | 说明 |
|------|------|
| **Turn** | 一轮生成 = LLM 生成 + 工具执行 |
| **Active Mask** | 标记哪些样本仍需要继续生成 |
| **Action Mask** | 标记哪些 token 是模型生成的（1）还是环境反馈的（0） |
| **Rolling State** | 滚动状态，包含当前序列和元数据 |

---

## 2. ToolGenerationManager 类

### 2.1 类定义

**文件**: `agent_r1/llm_agent/generation.py:118-164`

```python
class ToolGenerationManager:
    """
    工具生成管理器

    负责管理 LLM 与工具环境之间的多轮交互
    """

    def __init__(
        self,
        tokenizer,           # HuggingFace tokenizer
        processor,           # 多模态处理器（可选）
        actor_rollout_wg,    # Actor/Rollout Worker 组
        config: ToolGenerationConfig,  # 生成配置
        is_validation: bool = False,   # 是否为验证模式
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        # 初始化张量操作辅助类
        self.tensor_fn = TensorHelper(TensorConfig(pad_token_id=tokenizer.pad_token_id))
```

### 2.2 配置类

**文件**: `agent_r1/llm_agent/generation.py:84-111`

```python
@dataclass
class ToolGenerationConfig:
    """
    工具生成配置

    属性：
        max_turns: 最大交互轮数（默认 5）
        max_prompt_length: 最大 prompt 长度（默认 16384）
        max_response_length: 最大响应总长度（默认 16384）
        max_response_length_single_turn: 单轮最大响应长度（默认 8192）
        use_batch_tool_calls: 是否批量执行工具调用（默认 False）
    """
    max_turns: int                      # 如：5
    max_prompt_length: int              # 如：16384
    max_response_length: int            # 如：16384
    max_response_length_single_turn: int  # 如：8192
    use_batch_tool_calls: bool = False
```

来自 `run_grpo_retool.sh` 的配置：
```bash
data.max_prompt_length=16384 \
data.max_response_length=16384 \
data.max_response_length_single_turn=8192 \
tool.max_turns=5 \
```

---

## 3. run_llm_loop() 主循环详解

### 3.1 函数签名

**文件**: `agent_r1/llm_agent/generation.py:513-849`

```python
def run_llm_loop(
    self, gen_batch: DataProto, env: Union[BaseToolEnv, BaseImageToolEnv]
) -> DataProto:
    """
    执行 LLM 多轮生成主循环

    参数：
        gen_batch: 输入批次（包含 prompt 的 DataProto）
        env: 工具环境

    返回：
        final_output: 完整生成结果（DataProto）
    """
```

### 3.2 输入 DataProto 结构

```python
gen_batch = DataProto(
    batch = {
        "input_ids": tensor([batch_size, max_prompt_length]),
        "attention_mask": tensor([batch_size, max_prompt_length]),
        "position_ids": tensor([batch_size, max_prompt_length]),
    },
    non_tensor_batch = {
        "raw_prompt_ids": array([list, list, ...]),  # 变长 token ID 列表
        "reward_model": array([dict, dict, ...]),     # 奖励配置
        "data_source": array([str, str, ...]),
        ...
    },
    meta_info = {
        "n": 1,                    # 每个 prompt 生成数
        "temperature": 0.7,        # 采样温度
        "do_sample": True,         # 是否采样
        ...
    }
)
```

### 3.3 初始化阶段

**文件**: `agent_r1/llm_agent/generation.py:613-636`

```python
# 获取批次大小
batch_size = gen_batch.batch["input_ids"].shape[0]

# 初始化 active_mask：标记哪些样本仍需要继续生成
# True = 继续生成, False = 已完成
active_mask = torch.ones(batch_size, dtype=torch.bool)

# turns: 记录每个样本的交互轮数
turns = torch.zeros(batch_size, dtype=torch.int32)

# 记录每轮活跃样本数（用于调试）
active_num_list = [active_mask.sum().item()]  # 初始：[128]

# rollings: 滚动状态，每轮更新
rollings = gen_batch

# 保存原始 prompt
prompts = gen_batch.batch["input_ids"][:, -self.config.max_prompt_length:].clone()
```

### 3.4 主循环结构

```python
for turn in range(self.config.max_turns):  # 最多 max_turns 轮
    # 步骤 1: 检查序列长度
    # 步骤 2: 筛选活跃样本
    # 步骤 3: 分布式生成响应
    # 步骤 4: 执行工具调用
    # 步骤 5: 更新 active_mask
    # 步骤 6: 更新滚动状态
```

### 3.5 步骤 1：检查序列长度

**文件**: `agent_r1/llm_agent/generation.py:645-672`

```python
# 检查 attention_mask 指示的有效长度
effective_len = rollings.batch["attention_mask"].sum(dim=1)
length_exceeded = effective_len > self.config.max_prompt_length

if length_exceeded.sum() > 0:
    print("[WARNING] SEQUENCE LENGTH EXCEEDED MAX PROMPT LENGTH")
    active_mask[length_exceeded] = 0  # 超长的样本停止生成

# 检查 raw_prompt_ids 长度
raw_prompt_ids = rollings.non_tensor_batch["raw_prompt_ids"]
length_exceeded = [
    len(prompt_id) > self.config.max_prompt_length
    for prompt_id in raw_prompt_ids
]
if any(length_exceeded):
    active_mask[length_exceeded] = 0

# 如果没有活跃样本，提前退出
if not active_mask.sum():
    print("[WARNING] NO ACTIVE SEQUENCES")
    break
```

### 3.6 步骤 2：筛选活跃样本

**文件**: `agent_r1/llm_agent/generation.py:674-691`

```python
# 只对 active_mask=True 的样本继续生成，节省计算资源
if hasattr(rollings, "non_tensor_batch") and rollings.non_tensor_batch:
    rollings_active = DataProto.from_dict(
        tensors={k: v[active_mask] for k, v in rollings.batch.items()},
        non_tensors={
            k: v[active_mask.numpy()]
            for k, v in rollings.non_tensor_batch.items()
        },
        meta_info=meta_info,
    )
else:
    rollings_active = DataProto.from_dict(
        tensors={k: v[active_mask] for k, v in rollings.batch.items()},
        meta_info=meta_info,
    )
```

**筛选示意图**：

```
原始批次 (batch_size=128):
active_mask = [T, T, F, T, F, F, T, ...]  # 80 个 True, 48 个 False

筛选后 (active_batch_size=80):
rollings_active 只包含 active_mask=True 的 80 个样本
```

### 3.7 步骤 3：分布式生成响应

**文件**: `agent_r1/llm_agent/generation.py:693-705`

```python
# 填充到 world_size 的整数倍（分布式数据并行需要）
rollings_active, pad_size = pad_dataproto_to_divisor(
    rollings_active, self.actor_rollout_wg.world_size
)

# 调用 Actor Worker 生成序列
gen_output = self.actor_rollout_wg.generate_sequences(rollings_active)

# 去除填充
gen_output = unpad_dataproto(gen_output, pad_size=pad_size)
```

**分布式生成流程**：

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        分布式生成流程                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  rollings_active (80 个样本)                                               │
│           │                                                                │
│           ▼ pad_dataproto_to_divisor (world_size=8)                       │
│                                                                            │
│  rollings_padded (80 → 80 个样本，已可被 8 整除)                            │
│           │                                                                │
│           ▼ 分发到 8 个 GPU                                                │
│                                                                            │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│  │GPU 0 │ │GPU 1 │ │GPU 2 │ │GPU 3 │ │GPU 4 │ │GPU 5 │ │GPU 6 │ │GPU 7 │ │
│  │10个  │ │10个  │ │10个  │ │10个  │ │10个  │ │10个  │ │10个  │ │10个  │ │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │
│           │                                                                │
│           ▼ 并行生成 (vLLM/SGLang)                                         │
│           │                                                                │
│           ▼ 收集结果                                                       │
│                                                                            │
│  gen_output (80 个样本的生成结果)                                           │
│           │                                                                │
│           ▼ unpad_dataproto                                                │
│                                                                            │
│  gen_output (80 个样本)                                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.8 步骤 4：执行工具调用

**文件**: `agent_r1/llm_agent/generation.py:707-750`

```python
# 获取生成的响应
raw_responses_ids = gen_output.batch["responses"]
responses_ids = env.process_responses_ids(self.tokenizer, raw_responses_ids)
raw_responses = self.tokenizer.batch_decode(
    responses_ids, skip_special_tokens=True
)

# 执行工具调用
if isinstance(env, BaseToolEnv):
    if self.config.use_batch_tool_calls:
        # 批量执行（更高效）
        tool_responses, _, new_active_masks = env.batch_step(raw_responses)
    else:
        # 逐个执行
        tool_responses = []
        new_active_masks = []
        for raw_response in raw_responses:
            tool_response, _, active = env.step(raw_response)
            tool_responses.append(tool_response)
            new_active_masks.append(active)
    tool_images = [[]] * len(raw_responses)  # 文本环境无图像
```

**env.step() / env.batch_step() 返回值**：
- `tool_response`: 工具响应文本（如 `"<interpreter>\n4.0\n</interpreter>\n"`）
- `success_list`: 工具执行是否成功
- `active`: 是否继续生成（`True` = 检测到工具调用，继续；`False` = 未检测到，停止）

### 3.9 步骤 5：更新 active_mask

**文件**: `agent_r1/llm_agent/generation.py:751-772`

```python
# 将活跃样本的数据填充回完整批次
responses_ids = self._example_level_pad(responses_ids, active_mask)
tool_responses = self._example_level_pad(
    tool_responses, active_mask, pad_value=""
)
tool_images = self._example_level_pad(
    tool_images, active_mask, pad_value=[]
)

# 更新 active_mask
# new_active_masks: True=继续生成, False=停止
active_mask[active_mask.clone()] = torch.tensor(
    new_active_masks, dtype=torch.bool
)

# 增加仍活跃样本的轮数计数
turns[active_mask] += 1

active_num_list.append(active_mask.sum().item())
```

**更新逻辑示意**：

```
Turn 1 前:
active_mask = [T, T, T, T]  # 4 个全活跃
active_num_list = [4]

Turn 1 后:
new_active_masks = [T, F, T, T]  # 样本 1 检测到答案，停止
active_mask = [T, F, T, T]
turns = [1, 1, 1, 1]  # 所有样本 +1
active_num_list = [4, 3]

Turn 2 后:
new_active_masks = [F, T, F]  # 对应 active_mask=T 的位置
active_mask = [F, F, T, F]
turns = [1, 1, 2, 1]  # 只有活跃的样本 +1
active_num_list = [4, 3, 1]
```

### 3.10 步骤 6：更新滚动状态

**文件**: `agent_r1/llm_agent/generation.py:774-780`

```python
# 将新生成的响应和工具响应追加到序列中
rollings = self._update_rolling_state(
    rollings, responses_ids, tool_responses, tool_images
)
```

详见 [第 5 节：滚动状态更新](#5-滚动状态更新)。

### 3.11 组装最终输出

**文件**: `agent_r1/llm_agent/generation.py:786-849`

```python
# 组装最终输出
final_output = {}

# 基本输出
final_output["turns"] = turns                         # 每个样本的轮数
final_output["prompts"] = prompts                     # 原始 prompt
final_output["responses"] = rollings.batch["responses"].long()  # 完整响应

# 组合完整序列
final_output["input_ids"] = torch.cat(
    [prompts, rollings.batch["responses"].long()], dim=1
)

# 创建注意力掩码和位置编码
final_output["attention_mask"] = self.tensor_fn.create_attention_mask(
    final_output["input_ids"]
)
final_output["position_ids"] = self.tensor_fn.create_position_ids(
    final_output["attention_mask"]
)

# 创建最终的 Action Mask
response_length = final_output["responses"].shape[-1]
response_mask = final_output["attention_mask"][:, -response_length:]
final_output["action_mask"] = response_mask.clone()

# 应用实际的 action_mask
for i, action_mask in enumerate(rollings.non_tensor_batch["action_mask"]):
    mask_len = min(len(action_mask), response_mask.shape[1])
    final_output["action_mask"][i, :mask_len] = (
        torch.tensor(action_mask[:mask_len]) * response_mask[i, :mask_len]
    )

# 封装为 DataProto 并返回
final_output = DataProto.from_dict(final_output)
final_output.non_tensor_batch = rollings.non_tensor_batch

return final_output
```

---

## 4. Active Mask 机制

### 4.1 Active Mask 的作用

Active Mask 是一个布尔张量，标记哪些样本仍需要继续生成：

```python
active_mask = torch.ones(batch_size, dtype=torch.bool)  # 初始：全 True
```

| 值 | 含义 |
|----|------|
| `True` | 样本仍活跃，需要继续生成 |
| `False` | 样本已完成（检测到答案、超长、错误等） |

### 4.2 Active Mask 的更新时机

1. **序列长度超限**（步骤 1）：
   ```python
   length_exceeded = effective_len > self.config.max_prompt_length
   active_mask[length_exceeded] = 0
   ```

2. **工具调用结果**（步骤 5）：
   ```python
   active_mask[active_mask.clone()] = torch.tensor(new_active_masks)
   ```
   - `new_active_masks[i] = True`：检测到工具调用，继续生成
   - `new_active_masks[i] = False`：未检测到工具调用（如检测到答案），停止生成

### 4.3 样本级别填充 (_example_level_pad)

**文件**: `agent_r1/llm_agent/generation.py:182-250`

由于只对活跃样本进行生成，需要将结果填充回完整批次：

```python
def _example_level_pad(
    self,
    data: Union[List[Any], np.ndarray, torch.Tensor],
    active_mask: torch.Tensor,
    pad_value: Any = None,
) -> Union[List[Any], np.ndarray, torch.Tensor]:
    """
    根据 active_mask 进行样本级别的填充

    示例：
        active_mask = [True, False, True, False]  # 4 个样本，2 个活跃
        data = ["response1", "response3"]          # 只有 2 个活跃样本的数据

        结果 = ["response1", "", "response3", ""]  # 填充后的 4 个样本
    """
    batch_size = active_mask.shape[0]

    # 自动推断填充值
    if pad_value is None:
        first_elem = data[0]
        if isinstance(first_elem, str):
            pad_value = ""
        elif isinstance(first_elem, list):
            pad_value = []
        elif isinstance(first_elem, torch.Tensor):
            pad_value = torch.full_like(first_elem, fill_value=self.tokenizer.pad_token_id)

    # 创建填充后的输出
    padded_data = [pad_value] * batch_size

    # 将活跃位置的数据填入
    s = 0
    for i, is_active in enumerate(active_mask):
        if is_active:
            padded_data[i] = data[s]
            s += 1

    return padded_data
```

**示意图**：

```
输入:
  active_mask = [T,  F,  T,  F,  T,  F]
  data = ["resp0", "resp2", "resp4"]  # 3 个活跃样本的数据

处理过程:
  padded_data = ["", "", "", "", "", ""]  # 初始化为 pad_value
  i=0, active=T: padded_data[0] = "resp0"
  i=1, active=F: 跳过
  i=2, active=T: padded_data[2] = "resp2"
  i=3, active=F: 跳过
  i=4, active=T: padded_data[4] = "resp4"
  i=5, active=F: 跳过

输出:
  padded_data = ["resp0", "", "resp2", "", "resp4", ""]
```

---

## 5. 滚动状态更新

### 5.1 _update_rolling_state 方法

**文件**: `agent_r1/llm_agent/generation.py:286-511`

```python
def _update_rolling_state(
    self,
    rollings,                  # 当前滚动状态
    responses_ids: torch.Tensor,      # 模型新生成的响应 token ID
    tool_responses: List[str],        # 工具响应文本列表
    tool_responses_images: List[List[Image.Image]],  # 工具响应图像（多模态）
):
    """
    更新滚动状态

    状态更新示意：
    Turn N 前:
        input_ids = [Prompt + 之前的响应]
        responses = [之前的响应]
        action_mask = [之前的 mask]

    Turn N 后:
        input_ids = [Prompt + 之前的响应 + 新响应 + 工具响应]
        responses = [之前的响应 + 新响应 + 工具响应]
        action_mask = [之前的 mask + 新响应 mask(1) + 工具响应 mask(0)]
    """
```

### 5.2 更新流程

**1. 分词工具响应**：
```python
tool_responses_ids = self._batch_tokenize(formatted_tool_responses)
```

**2. 更新 responses**：
```python
if "responses" not in rollings.batch.keys():
    # 第一轮：创建 responses
    rollings.batch["responses"] = self.tensor_fn.concatenate_with_padding(
        [responses_ids, tool_responses_ids], pad_to_left=False
    )
else:
    # 后续轮：追加到 responses
    rollings.batch["responses"] = self.tensor_fn.concatenate_with_padding(
        [rollings.batch["responses"], responses_ids, tool_responses_ids],
        pad_to_left=False,
    )

# 截断到最大响应长度
rollings.batch["responses"] = rollings.batch["responses"][
    :, : self.config.max_response_length
]
```

**3. 创建并更新 action_mask**：
```python
action_masks = self._create_response_action_mask(
    responses_ids_list, tool_responses_ids_list
)

if "action_mask" not in rollings.non_tensor_batch.keys():
    # 第一轮
    rollings.non_tensor_batch["action_mask"] = np.array(action_masks, dtype=object)
else:
    # 后续轮：追加
    new_action_masks = []
    for i, action_mask in enumerate(rollings.non_tensor_batch["action_mask"]):
        new_action_masks.append(action_mask + action_masks[i])
    rollings.non_tensor_batch["action_mask"] = np.array(new_action_masks, dtype=object)
```

**4. 更新 input_ids**：
```python
new_input_ids = self.tensor_fn.concatenate_with_padding(
    [rollings.batch["input_ids"], responses_ids, tool_responses_ids]
)
```

**5. 更新 attention_mask 和 position_ids**：
```python
new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

rollings.batch["input_ids"] = new_input_ids
rollings.batch["position_ids"] = new_position_ids
rollings.batch["attention_mask"] = new_attention_mask
```

**6. 更新 raw_prompt_ids**：
```python
new_raw_prompt_ids = []
for raw_prompt_id, responses_ids_, raw_tool_response in zip(
    raw_prompt_ids, responses_ids_list, raw_tool_responses
):
    tool_response_ids = self.tokenizer.encode(
        raw_tool_response, add_special_tokens=False
    )
    new_raw_prompt_ids.append(
        raw_prompt_id + responses_ids_ + tool_response_ids
    )

rollings.non_tensor_batch["raw_prompt_ids"] = np.array(new_raw_prompt_ids, dtype=object)
```

---

## 6. Action Mask 创建

### 6.1 _create_response_action_mask 方法

**文件**: `agent_r1/llm_agent/generation.py:252-284`

```python
def _create_response_action_mask(
    self,
    responses_ids_list: List[List[int]],      # 模型生成的 token ID
    tool_responses_ids_list: List[List[int]], # 工具响应的 token ID
) -> List[List[int]]:
    """
    创建响应的 Action Mask

    规则：
    - 模型生成的 token → 1（需要计算梯度）
    - 工具响应的 token → 0（不计算梯度）
    """
    action_masks = []

    for model_ids, tool_ids in zip(responses_ids_list, tool_responses_ids_list):
        # 模型生成 → 1，工具响应 → 0
        action_mask = [1] * len(model_ids) + [0] * len(tool_ids)
        action_masks.append(action_mask)

    return action_masks
```

### 6.2 Action Mask 示例

```
样本响应:
  "<think>让我计算...</think><code>print(1+1)</code>"  (模型生成, 20 token)
  "<interpreter>2</interpreter>"                      (工具响应, 10 token)
  "<think>结果是2</think><answer>\\boxed{2}</answer>" (模型生成, 25 token)

Action Mask:
  Turn 1: [1,1,1,...,1,1,1,1,  0,0,0,0,0,0,0,0,0,0]
          |<-- 模型 20 -->|    |<-- 工具 10 -->|

  Turn 2: [1,1,1,...,1,1,1,1,  0,0,0,0,0,0,0,0,0,0,  1,1,1,...,1,1,1,1,1]
          |<-- 模型 20 -->|    |<-- 工具 10 -->|      |<-- 模型 25 -->|

最终 Action Mask (长度 55):
  [1,1,1,...,1, 0,0,0,...,0, 1,1,1,...,1]
   |<- 20 ->|   |<- 10 ->|   |<- 25 ->|
```

### 6.3 最终 Action Mask 处理

**文件**: `agent_r1/llm_agent/generation.py:828-843`

```python
# 获取响应部分的 mask
response_length = final_output["responses"].shape[-1]
response_mask = final_output["attention_mask"][:, -response_length:]

# 初始化为 attention_mask（处理 padding）
final_output["action_mask"] = response_mask.clone()

# 应用实际的 action_mask
for i, action_mask in enumerate(rollings.non_tensor_batch["action_mask"]):
    mask_len = min(len(action_mask), response_mask.shape[1])
    # 同时考虑 action_mask 和 response_mask（padding 位置）
    final_output["action_mask"][i, :mask_len] = (
        torch.tensor(action_mask[:mask_len]) * response_mask[i, :mask_len]
    )
```

**处理逻辑**：

```
假设 max_response_length = 100

样本 0 (实际响应 55 token):
  action_mask (逻辑) = [1,...,1, 0,...,0, 1,...,1]  # 长度 55
  response_mask      = [0,...,0, 1,1,1,...,1,1,1]   # 长度 100, 前 45 个是 padding

  最终 action_mask   = [0,...,0, 1,...,1, 0,...,0, 1,...,1]  # 长度 100
                       |pad 45| |<-- 实际 55 -->|

样本 1 (实际响应 30 token):
  action_mask (逻辑) = [1,...,1, 0,...,0]           # 长度 30
  response_mask      = [0,...,0, 1,1,1,...,1]       # 长度 100, 前 70 个是 padding

  最终 action_mask   = [0,...,0, 1,...,1, 0,...,0]  # 长度 100
                       |pad 70| |<-- 实际 30 -->|
```

---

## 7. 完整两轮调用示例

### 7.1 输入数据

```python
# 输入 DataProto
gen_batch = DataProto(
    batch = {
        "input_ids": tensor([2, 1024]),       # 2 个样本, max_prompt_length=1024
        "attention_mask": tensor([2, 1024]),
        "position_ids": tensor([2, 1024]),
    },
    non_tensor_batch = {
        "raw_prompt_ids": array([
            [151644, 872, 198, ...],  # 样本 0: "Solve the following..."
            [151644, 872, 198, ...],  # 样本 1: "Solve the following..."
        ], dtype=object),
        "reward_model": array([
            {"style": "rule", "ground_truth": "-7/25"},
            {"style": "rule", "ground_truth": "42"},
        ], dtype=object),
    },
    meta_info = {"temperature": 0.7, "do_sample": True}
)
```

### 7.2 Turn 1

**初始化**:
```python
batch_size = 2
active_mask = [True, True]
turns = [0, 0]
active_num_list = [2]
```

**生成响应**:
```
样本 0: "<think>我需要计算...</think><code>
```python
sin_A = 4/5
cos_2A = 1 - 2 * sin_A**2
print(cos_2A)
```
</code>"

样本 1: "<think>这是简单问题...</think><answer>\boxed{42}</answer>"
```

**执行工具**:
```python
# 样本 0: 检测到 <code>...</code>
tool_responses[0] = "<interpreter>\n-0.28\n</interpreter>\n"
new_active_masks[0] = True  # 继续生成

# 样本 1: 未检测到代码，检测到答案
tool_responses[1] = ""
new_active_masks[1] = False  # 停止生成
```

**更新状态**:
```python
active_mask = [True, False]
turns = [1, 1]
active_num_list = [2, 1]

# 样本 0 的 action_mask (Turn 1)
action_mask[0] = [1,1,1,...,1,  0,0,0,...,0]
#                |<- model ->|  |<- tool ->|
```

### 7.3 Turn 2

**筛选活跃样本**:
```python
# 只有样本 0 活跃
rollings_active = DataProto(
    batch = {
        "input_ids": tensor([1, seq_len]),  # 只有样本 0
        ...
    }
)
```

**生成响应**:
```
样本 0: "<think>结果是 -0.28 = -7/25</think><answer>\boxed{-7/25}</answer>"
```

**执行工具**:
```python
# 样本 0: 未检测到代码，检测到答案
tool_responses[0] = ""
new_active_masks[0] = False  # 停止生成
```

**填充并更新状态**:
```python
# 填充回完整批次
responses_ids = _example_level_pad([tensor_0], [True, False])
# = [tensor_0, pad_tensor]

active_mask = [False, False]  # 所有样本都完成
turns = [2, 1]
active_num_list = [2, 1, 0]

# 样本 0 的 action_mask (最终)
action_mask[0] = [1,1,...,1,  0,0,...,0,  1,1,...,1]
#                |<- T1 ->|  |<- tool ->| |<- T2 ->|
```

### 7.4 最终输出

```python
final_output = DataProto(
    batch = {
        "prompts": tensor([2, 1024]),      # 原始 prompt
        "responses": tensor([2, 512]),     # 完整响应
        "input_ids": tensor([2, 1536]),    # prompt + response
        "attention_mask": tensor([2, 1536]),
        "position_ids": tensor([2, 1536]),
        "action_mask": tensor([2, 512]),   # 1=模型, 0=工具/padding
        "turns": tensor([2, 1]),           # [2轮, 1轮]
    },
    non_tensor_batch = {
        "raw_prompt_ids": array([...]),
        "reward_model": array([...]),
        "action_mask": array([             # 逻辑 mask (变长)
            [1,...,1, 0,...,0, 1,...,1],   # 样本 0: 55 个
            [1,...,1],                      # 样本 1: 30 个
        ], dtype=object),
    }
)
```

---

## 8. 张量辅助类

### 8.1 TensorHelper 类

**文件**: `agent_r1/llm_agent/tensor_helper.py:14-73`

```python
class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config  # 包含 pad_token_id

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """从 input_ids 创建 attention_mask"""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """从 attention_mask 创建 position_ids"""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(
        self, tensors: List[torch.Tensor], pad_to_left: bool = True
    ) -> torch.Tensor:
        """拼接张量并处理 padding"""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def convert_pad_structure(
        self, tensor: torch.Tensor, pad_to_left: bool = True
    ) -> torch.Tensor:
        """转换 padding 结构（左/右填充）"""
        # 创建 mask: content=1, padding=0 (pad_to_left) 或相反
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id

        # 排序使内容移动到期望的一侧
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        sorted_tensor = tensor.gather(1, sorted_indices)

        # 计算有效长度
        effective_len = (tensor != self.config.pad_token_id).sum(dim=1).max().item()

        # 保留内容侧
        if pad_to_left:
            return sorted_tensor[:, -effective_len:]
        else:
            return sorted_tensor[:, :effective_len]
```

### 8.2 concatenate_with_padding 示例

```
输入:
  tensors = [
      tensor([[PAD, PAD, 1, 2, 3]]),           # 左填充的 prompt
      tensor([[4, 5, 6, PAD, PAD]]),           # 右填充的 response
      tensor([[7, 8, PAD, PAD, PAD]])          # 右填充的 tool_response
  ]

拼接:
  concatenated = tensor([[PAD, PAD, 1, 2, 3, 4, 5, 6, PAD, PAD, 7, 8, PAD, PAD, PAD]])

转换为左填充 (pad_to_left=True):
  result = tensor([[PAD, PAD, PAD, PAD, PAD, PAD, PAD, 1, 2, 3, 4, 5, 6, 7, 8]])
                   |<-------- padding -------->|    |<-- content -->|
```

---

## 9. 数据形状变化

### 9.1 各阶段数据形状

假设 `batch_size=128`, `max_prompt_length=16384`, `max_response_length=16384`：

| 阶段 | 字段 | 形状 | 说明 |
|------|------|------|------|
| **输入** | `input_ids` | `[128, 16384]` | 左填充的 prompt |
| | `attention_mask` | `[128, 16384]` | 0/1 掩码 |
| | `position_ids` | `[128, 16384]` | 位置编码 |
| **Turn 1 生成后** | `responses` | `[128, ~500]` | 单轮生成 |
| | `action_mask` (逻辑) | `[128, 变长]` | 列表数组 |
| **Turn 1 工具后** | `responses` | `[128, ~600]` | +工具响应 |
| | `action_mask` (逻辑) | `[128, 变长]` | +工具 mask |
| **最终输出** | `prompts` | `[128, 16384]` | 原始 prompt |
| | `responses` | `[128, ≤16384]` | 完整响应 |
| | `input_ids` | `[128, ≤32768]` | prompt + response |
| | `attention_mask` | `[128, ≤32768]` | 完整掩码 |
| | `position_ids` | `[128, ≤32768]` | 完整位置 |
| | `action_mask` | `[128, ≤16384]` | 1=模型, 0=工具 |
| | `turns` | `[128]` | 轮数 |

### 9.2 Action Mask 形状变化

```
Turn 1:
  responses_ids_list[i] = [t1, t2, ..., t20]    # 模型生成 20 token
  tool_responses_ids_list[i] = [t1, ..., t10]  # 工具响应 10 token

  action_mask[i] = [1,1,...,1, 0,0,...,0]       # 长度 30
                   |<- 20 ->| |<- 10 ->|

Turn 2:
  responses_ids_list[i] = [t1, t2, ..., t25]    # 模型生成 25 token
  tool_responses_ids_list[i] = []               # 无工具调用

  action_mask[i] = [...之前 30..., 1,1,...,1]   # 长度 55
                                    |<- 25 ->|

最终:
  action_mask (逻辑)[i] = [1×20, 0×10, 1×25]    # 长度 55, 存储在 non_tensor_batch
  action_mask (张量)[i] = [0×45, 1×20, 0×10, 1×25]  # 长度 100, padding 后
```

---

## 总结

本文档详细介绍了 Agent-R1 的多轮生成循环机制：

1. **ToolGenerationManager** 管理 LLM 与工具环境的交互
2. **run_llm_loop()** 主循环包含 6 个步骤：
   - 检查序列长度
   - 筛选活跃样本
   - 分布式生成响应
   - 执行工具调用
   - 更新 active_mask
   - 更新滚动状态
3. **Active Mask** 标记哪些样本需要继续生成
4. **Action Mask** 区分模型生成（1）和工具响应（0）
5. **滚动状态更新** 将每轮的响应追加到序列

关键要点：
- 每轮只对活跃样本进行生成，节省计算资源
- Action Mask 确保只对模型生成的 token 计算梯度
- 滚动状态维护完整的序列和元数据
- 支持多模态（图像）工具响应

下一篇文档将介绍奖励计算系统 (`05_reward_computation.md`)，详细讲解 `AgentRewardManager` 和 ReTool 奖励函数的实现。
