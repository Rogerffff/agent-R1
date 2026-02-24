# 数据准备与加载

本文档详细介绍 Agent-R1 训练流程中的数据准备阶段，包括原始数据集预处理、数据集类的实现、Chat Template 应用、分词填充、以及最终转换为 DataProto 的完整流程。

---

## 目录

1. [数据准备概览](#1-数据准备概览)
2. [原始数据集预处理](#2-原始数据集预处理)
3. [数据集类详解](#3-数据集类详解)
4. [Chat Template 与工具描述注入](#4-chat-template-与工具描述注入)
5. [分词与填充处理](#5-分词与填充处理)
6. [批处理与 collate_fn](#6-批处理与-collate_fn)
7. [DataLoader 创建](#7-dataloader-创建)
8. [从 DataLoader 到 DataProto](#8-从-dataloader-到-dataproto)
9. [完整数据流示例](#9-完整数据流示例)
10. [数据字段参考](#10-数据字段参考)

---

## 1. 数据准备概览

数据准备是训练流程的第一步，其目标是将原始数据集转换为模型可以使用的格式。整个流程如下：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据准备完整流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │   原始数据集     │    │   预处理脚本     │    │    Parquet 文件         │ │
│  │ DAPO-Math-17K   │───▶│   retool.py     │───▶│ train/test.parquet     │ │
│  │ (HuggingFace)   │    │                 │    │                         │ │
│  └─────────────────┘    └─────────────────┘    └───────────┬─────────────┘ │
│                                                             │               │
│                                                             ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ToolRLDataset                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ 读取 Parquet │  │ 应用 Chat    │  │ 分词与填充   │               │   │
│  │  │    文件      │─▶│ Template     │─▶│ (Left Pad)   │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └──────────────────────────────────────┬──────────────────────────────┘   │
│                                          │                                  │
│                                          ▼                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     DataLoader + collate_fn                           │ │
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │ │
│  │  │ 采样多个样本  │───▶│ Stack 张量    │───▶│ 返回 batch dict       │ │ │
│  │  │ (__getitem__) │    │ 收集非张量    │    │ {tensors, non_tensors}│ │ │
│  │  └───────────────┘    └───────────────┘    └───────────────────────┘ │ │
│  └───────────────────────────────────────┬───────────────────────────────┘ │
│                                           │                                 │
│                                           ▼                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                          DataProto                                    │ │
│  │  batch = {input_ids, attention_mask, position_ids}                    │ │
│  │  non_tensor_batch = {raw_prompt_ids, reward_model, data_source, ...}  │ │
│  │  meta_info = {}                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `examples/data_preprocess/retool.py` | 数据预处理脚本 |
| `agent_r1/src/agent_rl_dataset.py` | ToolRLDataset 数据集类 |
| `verl/verl/utils/dataset/rl_dataset.py` | 父类 RLHFDataset |
| `verl/verl/utils/torch_functional.py` | 填充和截断工具函数 |
| `agent_r1/src/main_agent.py:523` | create_rl_dataset 函数 |
| `agent_r1/src/agent_ray_trainer.py:998` | _create_dataloader 方法 |

---

## 2. 原始数据集预处理

### 2.1 原始数据集结构

ReTool 使用 DAPO-Math-17K 数据集，原始结构如下：

```python
# 原始数据集字段
{
    "prompt": "In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$...",  # 数学问题
    "target": "34"  # 正确答案
}
```

原始数据集来自 HuggingFace: `haizhongzheng/DAPO-Math-17K-cleaned`

### 2.2 预处理脚本详解

**文件**: `examples/data_preprocess/retool.py`

```python
# 第 33-36 行：加载原始数据集
dapo_dataset = datasets.load_dataset(
    "haizhongzheng/DAPO-Math-17K-cleaned"
)

# 第 39 行：获取训练集
dataset = dapo_dataset["train"]  # 共 17,917 条

# 第 41-43 行：划分训练集和测试集
test_dataset = dataset.shuffle(seed=42).select(range(100))        # 100 条测试
train_dataset = dataset.shuffle(seed=42).select(range(100, len(dataset)))  # 17,817 条训练
```

### 2.3 Instruction 模板

预处理脚本定义了一个详细的 instruction 模板（第 45 行）：

```python
instruction = """Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.
Each code snippet is wrapped with `<code>
```python
code snippet
```
</code>`.
The last part of your response should be in the following format:
<answer>
\\boxed{{'The final answer goes here.'}}
</answer>

*user question:*
{question}

Remember to place the final answer in the last part using the format:
<answer>
\\boxed{{'The final answer goes here.'}}
</answer>"""
```

**Instruction 关键要素**：
1. **代码格式**: 使用 `<code>...</code>` 标签包裹 Python 代码
2. **输出格式**: 代码执行结果通过 `<interpreter>...</interpreter>` 返回
3. **答案格式**: 最终答案使用 `<answer>\boxed{...}</answer>` 格式
4. **占位符**: `{question}` 将被替换为实际的数学问题

### 2.4 数据转换函数

```python
# 第 48-78 行：处理函数
def process_dapo(example, idx):
    # 从原始 prompt 中获取问题
    prompt = example.get("prompt", "")
    question = instruction.format(question=prompt)  # 将问题嵌入 instruction

    # 获取正确答案
    ground_truth = example.get("target", "")

    # 构建标准化的数据结构
    data = {
        "data_source": "BytedTsinghua-SIA/DAPO-Math-17k",
        "prompt": [
            {
                "role": "user",
                "content": question,  # 完整的 instruction + 问题
            }
        ],
        "ability": "",
        "reward_model": {
            "style": "rule",           # 使用规则奖励
            "ground_truth": ground_truth,  # 正确答案
        },
        "extra_info": {
            "split": "train",
            "index": idx,  # 样本索引
        },
    }
    return data
```

### 2.5 保存为 Parquet 文件

```python
# 第 80-92 行：处理并保存
processed_train = train_dataset.map(function=process_dapo, with_indices=True)
processed_test = test_dataset.map(function=process_dapo, with_indices=True)

# 保存到本地
processed_train.to_parquet(os.path.join(local_dir, "train.parquet"))
processed_test.to_parquet(os.path.join(local_dir, "test.parquet"))
```

### 2.6 预处理后的数据示例

```json
{
  "data_source": "BytedTsinghua-SIA/DAPO-Math-17k",
  "prompt": [
    {
      "role": "user",
      "content": "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$...\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
    }
  ],
  "ability": "",
  "reward_model": {
    "style": "rule",
    "ground_truth": "34"
  },
  "extra_info": {
    "split": "train",
    "index": 0
  }
}
```

---

## 3. 数据集类详解

### 3.1 类继承关系

```
┌───────────────────────────────────────────────────────┐
│                    RLHFDataset                        │
│           verl/verl/utils/dataset/rl_dataset.py       │
│  ┌─────────────────────────────────────────────────┐ │
│  │ - dataframe: datasets.Dataset                   │ │
│  │ - tokenizer: PreTrainedTokenizer                │ │
│  │ - max_prompt_length: int                        │ │
│  │ - truncation: str                               │ │
│  │                                                  │ │
│  │ + __init__(data_files, tokenizer, config)       │ │
│  │ + __getitem__(item) -> dict                     │ │
│  │ + __len__() -> int                              │ │
│  │ + _build_messages(example) -> list              │ │
│  │ + _read_files_and_tokenize()                    │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────┬────────────────────────────┘
                           │ 继承
                           ▼
┌───────────────────────────────────────────────────────┐
│                   ToolRLDataset                       │
│          agent_r1/src/agent_rl_dataset.py             │
│  ┌─────────────────────────────────────────────────┐ │
│  │ - env: BaseToolEnv                              │ │
│  │ - tools: List[dict]  (工具描述)                  │ │
│  │ - use_default_tool_template: bool               │ │
│  │ - use_custom_system_prompt: bool                │ │
│  │                                                  │ │
│  │ + __init__(..., env)                            │ │
│  │ + __getitem__(item) -> dict (重写)              │ │
│  │ + _build_messages(example) -> list (重写)       │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

### 3.2 ToolRLDataset 初始化

**文件**: `agent_r1/src/agent_rl_dataset.py:31-44`

```python
class ToolRLDataset(RLHFDataset):
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        env: Optional[BaseToolEnv] = None,  # 新增：工具环境
    ):
        self.env = env
        self.use_default_tool_template = config.get("use_default_tool_template", True)

        # 如果使用默认工具模板，提取工具描述
        if self.use_default_tool_template and self.env is not None:
            self.tools = [tool.tool_description for tool in self.env.tools]

        self.use_custom_system_prompt = config.get("use_custom_system_prompt", False)

        # 调用父类初始化
        super().__init__(data_files, tokenizer, config, processor)
```

**关键配置项**：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `use_default_tool_template` | `True` | 是否使用 tokenizer 的工具模板 |
| `use_custom_system_prompt` | `False` | 是否使用自定义系统提示 |

### 3.3 父类 RLHFDataset 初始化

**文件**: `verl/verl/utils/dataset/rl_dataset.py:63-96`

```python
class RLHFDataset(Dataset):
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        # 配置参数
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        # 下载并读取数据文件
        self._download()
        self._read_files_and_tokenize()
```

### 3.4 数据读取流程

```python
# 第 104-124 行：读取 Parquet 文件
def _read_files_and_tokenize(self):
    dataframes = []
    for parquet_file in self.data_files:
        # 使用 HuggingFace datasets 库读取 parquet
        dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
        dataframes.append(dataframe)

    # 合并多个数据集
    self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

    print(f"dataset len: {len(self.dataframe)}")  # 输出：dataset len: 17817

    # 过滤超长 prompt（可选）
    if self.filter_overlong_prompts:
        self.dataframe = self.dataframe.filter(
            lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
                        <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )
```

---

## 4. Chat Template 与工具描述注入

### 4.1 消息构建 (_build_messages)

**文件**: `agent_r1/src/agent_rl_dataset.py:46-73`

```python
def _build_messages(self, example: dict):
    # 从数据中提取 prompt（消息列表）
    messages = example.pop(self.prompt_key)

    # 如果使用自定义系统提示，注入工具环境的 system_prompt
    if self.use_custom_system_prompt and self.env is not None:
        if isinstance(messages, list):
            if messages[0]["role"] == "system":
                # 如果已有系统消息，追加内容
                messages[0]["content"] = messages[0]["content"] + self.env.system_prompt
            else:
                # 否则，在开头插入系统消息
                system_msg = [{"role": "system", "content": self.env.system_prompt}]
                messages = system_msg + messages

    return messages
```

### 4.2 Chat Template 应用

**文件**: `agent_r1/src/agent_rl_dataset.py:83-87`

```python
# 根据配置选择是否使用工具模板
if self.use_default_tool_template and hasattr(self, 'tools'):
    # 使用工具模板：将工具描述注入到 prompt 中
    raw_prompt = self.tokenizer.apply_chat_template(
        messages,
        tools=self.tools,  # 工具描述列表
        add_generation_prompt=True,
        tokenize=False
    )
else:
    # 普通模板
    raw_prompt = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
```

### 4.3 工具描述格式

ReTool 使用 `PythonTool`，其工具描述定义在 `agent_r1/tool/tools/python_tool.py`：

```python
tool_description = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code in a sandboxed environment. Use this for calculations, data processing, or any computation that would benefit from code execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
}
```

### 4.4 应用 Chat Template 后的示例

**输入消息**：
```python
messages = [
    {
        "role": "user",
        "content": "Solve the following problem step by step..."
    }
]
```

**应用 Chat Template 后的 raw_prompt（以 Qwen 为例）**：
```
<|im_start|>system
You are Qwen, a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "python", "description": "Execute Python code in a sandboxed environment...", "parameters": {...}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Solve the following problem step by step. You now have the ability to selectively write executable Python code...

*user question:*
In triangle $ABC$, $\sin \angle A = \frac{4}{5}$...

Remember to place the final answer in the last part using the format:
<answer>
\boxed{{'The final answer goes here.'}}
</answer><|im_end|>
<|im_start|>assistant
```

**注意**：ReTool 实际上设置 `use_default_tool_template=False`（见训练脚本第 13 行），因为 instruction 中已经包含了工具使用说明，不需要额外的工具模板。

---

## 5. 分词与填充处理

### 5.1 分词过程

**文件**: `agent_r1/src/agent_rl_dataset.py:119-122`

```python
# 使用 tokenizer 进行分词
model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
input_ids = model_inputs.pop("input_ids")       # shape: [1, seq_len]
attention_mask = model_inputs.pop("attention_mask")  # shape: [1, seq_len]
```

### 5.2 填充与截断

**文件**: `agent_r1/src/agent_rl_dataset.py:124-131`

```python
# 使用 verl 的 postprocess_data 函数进行填充/截断
input_ids, attention_mask = verl_F.postprocess_data(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=self.max_prompt_length,      # 从配置获取，如 16384
    pad_token_id=self.tokenizer.pad_token_id,
    left_pad=True,                          # 左填充
    truncation=self.truncation,             # 截断策略
)
```

### 5.3 postprocess_data 函数详解

**文件**: `verl/verl/utils/torch_functional.py:258-288`

```python
def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad=True,
    truncation="error",
):
    """
    对 tokenizer 输出进行后处理：填充或截断到指定长度
    """
    assert truncation in ["left", "right", "error"]
    assert input_ids.ndim == 2  # [batch_size, seq_len]

    sequence_length = input_ids.shape[-1]

    if sequence_length < max_length:
        # 填充到 max_length
        input_ids = pad_sequence_to_length(
            input_ids,
            max_seq_len=max_length,
            pad_token_id=pad_token_id,
            left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask,
            max_seq_len=max_length,
            pad_token_id=0,  # attention_mask 填充 0
            left_pad=left_pad
        )
    elif sequence_length > max_length:
        # 截断处理
        if truncation == "left":
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == "error":
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")

    return input_ids, attention_mask
```

### 5.4 左填充 (Left Padding) 示意图

```
原始序列 (长度 6):
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ tok1│ tok2│ tok3│ tok4│ tok5│ tok6│
└─────┴─────┴─────┴─────┴─────┴─────┘

左填充到 max_length=10:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ PAD │ PAD │ PAD │ PAD │ tok1│ tok2│ tok3│ tok4│ tok5│ tok6│
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
position:  0     1     2     3     4     5     6     7     8     9

attention_mask:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │  0  │  0  │  0  │  1  │  1  │  1  │  1  │  1  │  1  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

**为什么使用左填充？**
- 生成时从序列末尾开始，左填充保证真实 token 在末尾
- 方便拼接生成的 response（直接在末尾添加）
- attention_mask 为 0 的位置不参与计算

### 5.5 Position IDs 计算

**文件**: `agent_r1/src/agent_rl_dataset.py:147-152`

```python
# 根据 attention_mask 计算 position_ids
position_ids = compute_position_id_with_mask(attention_mask)
```

`compute_position_id_with_mask` 函数会根据 attention_mask 生成正确的位置编码：

```python
# 示例
attention_mask = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
position_ids   = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
#                 ^padding 区域     ^真实 token 区域
```

---

## 6. 批处理与 collate_fn

### 6.1 __getitem__ 返回值

**文件**: `agent_r1/src/agent_rl_dataset.py:75-172`

```python
def __getitem__(self, item):
    # 从 dataframe 获取一行数据
    row_dict = self.dataframe[item]

    # 构建消息并应用 chat template
    messages = self._build_messages(row_dict)

    # 分词和填充...

    # 设置返回字段
    row_dict["input_ids"] = input_ids[0]        # [max_prompt_length]
    row_dict["attention_mask"] = attention_mask[0]  # [max_prompt_length]
    row_dict["position_ids"] = position_ids[0]      # [max_prompt_length]

    # 原始（未填充）的 token IDs，用于后续处理
    raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
    row_dict["raw_prompt_ids"] = raw_prompt_ids  # List[int]，变长

    # 从 extra_info 提取索引
    index = row_dict.get("extra_info", {}).get("index", 0)
    row_dict["index"] = index

    return row_dict
```

### 6.2 单样本返回示例

```python
{
    # 张量字段 (将被 stack)
    "input_ids": tensor([151643, 151643, ..., 2610, 382, ...]),  # shape: [16384]
    "attention_mask": tensor([0, 0, ..., 1, 1, 1, ...]),          # shape: [16384]
    "position_ids": tensor([0, 0, ..., 0, 1, 2, ...]),            # shape: [16384]

    # 非张量字段 (将被收集为数组)
    "raw_prompt_ids": [2610, 382, 1495, ...],                      # List[int], 变长
    "data_source": "BytedTsinghua-SIA/DAPO-Math-17k",
    "ability": "",
    "reward_model": {"style": "rule", "ground_truth": "34"},
    "extra_info": {"split": "train", "index": 0},
    "index": 0,
}
```

### 6.3 collate_fn 详解

**文件**: `verl/verl/utils/dataset/rl_dataset.py:37-55`

```python
def collate_fn(data_list: list[dict]) -> dict:
    """
    将多个样本的 dict 合并成一个 batch dict
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    # 分离张量和非张量
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    # 张量字段：stack 成 batch
    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)  # [batch_size, seq_len]

    # 非张量字段：转为 numpy 数组（dtype=object 保持原始类型）
    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}
```

### 6.4 collate_fn 处理示意图

```
输入：data_list = [sample_0, sample_1, sample_2, ...]

sample_0 = {
    "input_ids": tensor([...]),        # [16384]
    "attention_mask": tensor([...]),   # [16384]
    "raw_prompt_ids": [tok1, tok2, ...],
    "reward_model": {"style": "rule", "ground_truth": "34"},
}

sample_1 = {
    "input_ids": tensor([...]),        # [16384]
    "attention_mask": tensor([...]),   # [16384]
    "raw_prompt_ids": [tok1, tok2, tok3, ...],  # 长度可能不同
    "reward_model": {"style": "rule", "ground_truth": "56"},
}

                    │
                    ▼ collate_fn

输出：batch = {
    # 张量字段 - stack 后
    "input_ids": tensor([...]),        # [batch_size, 16384]
    "attention_mask": tensor([...]),   # [batch_size, 16384]

    # 非张量字段 - numpy array (dtype=object)
    "raw_prompt_ids": array([[tok1, tok2, ...], [tok1, tok2, tok3, ...]], dtype=object),
    "reward_model": array([{"style": "rule", "ground_truth": "34"}, {...}], dtype=object),
}
```

---

## 7. DataLoader 创建

### 7.1 创建数据集函数

**文件**: `agent_r1/src/main_agent.py:523-543`

```python
def create_rl_dataset(data_paths, data_config, tokenizer, processor, env=None):
    """
    创建强化学习数据集。

    参数：
        data_paths: 数据文件路径（支持 parquet、jsonl 等格式）
        data_config: 数据配置
        tokenizer: HuggingFace tokenizer
        processor: 多模态处理器（可选）
        env: 工具环境，用于在数据处理时应用聊天模板（包含工具描述）

    返回：
        dataset: ToolRLDataset 实例
    """
    from agent_r1.src.agent_rl_dataset import ToolRLDataset

    dataset = ToolRLDataset(
        data_files=data_paths,
        tokenizer=tokenizer,
        config=data_config,
        processor=processor,
        env=env,
    )
    return dataset
```

### 7.2 创建 DataLoader

**文件**: `agent_r1/src/agent_ray_trainer.py:998-1048`

```python
def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
    """
    创建训练和验证 DataLoader
    """
    from .main_agent import create_rl_dataset, create_rl_sampler

    # 创建训练数据集
    if train_dataset is None:
        train_dataset = create_rl_dataset(
            self.config.data.train_files,     # ['data/retool/train.parquet']
            self.config.data,
            self.tokenizer,
            self.processor,
            self.env,                          # ReToolEnv 实例
        )

    # 创建验证数据集
    if val_dataset is None:
        val_dataset = create_rl_dataset(
            self.config.data.val_files,        # ['data/retool/test.parquet']
            self.config.data,
            self.tokenizer,
            self.processor,
            self.val_env,
        )

    self.train_dataset, self.val_dataset = train_dataset, val_dataset

    # 创建采样器
    if train_sampler is None:
        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)

    # 使用默认 collate_fn
    if collate_fn is None:
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
        collate_fn = default_collate_fn

    # 创建训练 DataLoader
    self.train_dataloader = StatefulDataLoader(
        dataset=self.train_dataset,
        batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
        num_workers=self.config.data.get("dataloader_num_workers", 8),
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )

    # 创建验证 DataLoader
    val_batch_size = self.config.data.val_batch_size or len(self.val_dataset)
    self.val_dataloader = StatefulDataLoader(
        dataset=self.val_dataset,
        batch_size=val_batch_size,
        num_workers=self.config.data.get("dataloader_num_workers", 8),
        drop_last=False,
        collate_fn=collate_fn,
    )
```

### 7.3 关键配置参数（来自 run_grpo_retool.sh）

| 参数 | 值 | 说明 |
|------|-----|------|
| `data.train_files` | `['data/retool/train.parquet']` | 训练数据路径 |
| `data.val_files` | `['data/retool/test.parquet']` | 验证数据路径 |
| `data.train_batch_size` | 128 | 训练批次大小 |
| `data.max_prompt_length` | 16384 | 最大 prompt 长度 |
| `data.max_response_length` | 16384 | 最大响应长度 |
| `data.use_default_tool_template` | False | 不使用默认工具模板 |

---

## 8. 从 DataLoader 到 DataProto

### 8.1 训练循环中的数据获取

**文件**: `agent_r1/src/agent_ray_trainer.py:1933-1960`

```python
# fit() 方法中的数据循环
for batch_dict in self.train_dataloader:
    # batch_dict 是 collate_fn 返回的字典
    # 包含张量和非张量数据

    # 转换为 DataProto
    batch = DataProto.from_dict(batch_dict)

    # ... 后续处理
```

### 8.2 DataProto 创建

**文件**: `verl/verl/protocol.py` (from_dict 方法)

```python
@classmethod
def from_dict(cls, data: dict) -> "DataProto":
    """
    从字典创建 DataProto
    自动分离张量和非张量数据
    """
    batch = {}
    non_tensor_batch = {}

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val
        elif isinstance(val, np.ndarray):
            # 检查是否为张量数组
            if val.dtype == object:
                non_tensor_batch[key] = val
            else:
                batch[key] = torch.from_numpy(val)
        else:
            non_tensor_batch[key] = val

    return cls(batch=batch, non_tensor_batch=non_tensor_batch)
```

### 8.3 DataProto 结构示例

```python
# 假设 batch_size=128, max_prompt_length=16384

DataProto(
    batch = {
        "input_ids": tensor([128, 16384]),         # 填充后的 token IDs
        "attention_mask": tensor([128, 16384]),    # 注意力掩码
        "position_ids": tensor([128, 16384]),      # 位置编码
    },
    non_tensor_batch = {
        "raw_prompt_ids": array([list, list, ...], dtype=object),  # 128 个变长列表
        "data_source": array(["BytedTsinghua-SIA/DAPO-Math-17k", ...], dtype=object),
        "ability": array(["", "", ...], dtype=object),
        "reward_model": array([
            {"style": "rule", "ground_truth": "34"},
            {"style": "rule", "ground_truth": "56"},
            ...
        ], dtype=object),
        "extra_info": array([
            {"split": "train", "index": 0},
            {"split": "train", "index": 1},
            ...
        ], dtype=object),
        "index": array([0, 1, 2, ...], dtype=object),
    },
    meta_info = {}
)
```

---

## 9. 完整数据流示例

以一条具体数据为例，展示从原始数据到 DataProto 的完整变换：

### 9.1 原始 HuggingFace 数据

```json
{
  "prompt": "In triangle ABC, sin∠A = 4/5 and ∠A < 90°. Find the value of cos2A.",
  "target": "-7/25"
}
```

### 9.2 预处理后 (train.parquet)

```json
{
  "data_source": "BytedTsinghua-SIA/DAPO-Math-17k",
  "prompt": [
    {
      "role": "user",
      "content": "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\nIn triangle ABC, sin∠A = 4/5 and ∠A < 90°. Find the value of cos2A.\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
    }
  ],
  "ability": "",
  "reward_model": {
    "style": "rule",
    "ground_truth": "-7/25"
  },
  "extra_info": {
    "split": "train",
    "index": 42
  }
}
```

### 9.3 应用 Chat Template 后

由于 `use_default_tool_template=False`，直接应用标准 chat template：

```
<|im_start|>user
Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.
Each code snippet is wrapped with `<code>
```python
code snippet
```
</code>`.
The last part of your response should be in the following format:
<answer>
\boxed{{'The final answer goes here.'}}
</answer>

*user question:*
In triangle ABC, sin∠A = 4/5 and ∠A < 90°. Find the value of cos2A.

Remember to place the final answer in the last part using the format:
<answer>
\boxed{{'The final answer goes here.'}}
</answer><|im_end|>
<|im_start|>assistant
```

### 9.4 分词后

```python
# raw_prompt_ids (变长，实际长度约 300-500 tokens)
raw_prompt_ids = [151644, 872, 198, 50115, 279, 2768, ...]  # List[int]

# input_ids (左填充到 max_prompt_length=16384)
input_ids = tensor([
    151643, 151643, 151643, ...,  # 前面是 PAD tokens (约 15900 个)
    151644, 872, 198, 50115, ...   # 真实 tokens
])  # shape: [16384]

# attention_mask
attention_mask = tensor([
    0, 0, 0, ...,  # PAD 位置为 0
    1, 1, 1, ...   # 真实 token 位置为 1
])  # shape: [16384]

# position_ids
position_ids = tensor([
    0, 0, 0, ...,  # PAD 位置为 0
    0, 1, 2, ...   # 真实 token 从 0 开始递增
])  # shape: [16384]
```

### 9.5 __getitem__ 返回值

```python
{
    # 张量（形状固定）
    "input_ids": tensor([16384]),         # 左填充
    "attention_mask": tensor([16384]),
    "position_ids": tensor([16384]),

    # 非张量（保持原始格式）
    "raw_prompt_ids": [151644, 872, 198, ...],  # 变长列表
    "data_source": "BytedTsinghua-SIA/DAPO-Math-17k",
    "ability": "",
    "reward_model": {"style": "rule", "ground_truth": "-7/25"},
    "extra_info": {"split": "train", "index": 42},
    "index": 42,
}
```

### 9.6 collate_fn 后的 batch

```python
# batch_size = 128
batch_dict = {
    # 张量（stacked）
    "input_ids": tensor([128, 16384]),
    "attention_mask": tensor([128, 16384]),
    "position_ids": tensor([128, 16384]),

    # 非张量（numpy array with dtype=object）
    "raw_prompt_ids": array([
        [151644, 872, 198, ...],     # 样本 0
        [151644, 872, 198, ...],     # 样本 1
        ...                           # 共 128 个
    ], dtype=object),
    "data_source": array(["BytedTsinghua-SIA/DAPO-Math-17k", ...], dtype=object),
    "ability": array(["", "", ...], dtype=object),
    "reward_model": array([
        {"style": "rule", "ground_truth": "-7/25"},
        {"style": "rule", "ground_truth": "34"},
        ...
    ], dtype=object),
    "extra_info": array([...], dtype=object),
    "index": array([42, 0, 15, ...], dtype=object),
}
```

### 9.7 DataProto.from_dict() 后

```python
DataProto(
    batch = {
        "input_ids": tensor([128, 16384]),
        "attention_mask": tensor([128, 16384]),
        "position_ids": tensor([128, 16384]),
    },
    non_tensor_batch = {
        "raw_prompt_ids": array([list, list, ...], dtype=object),
        "data_source": array([str, str, ...], dtype=object),
        "ability": array([str, str, ...], dtype=object),
        "reward_model": array([dict, dict, ...], dtype=object),
        "extra_info": array([dict, dict, ...], dtype=object),
        "index": array([int, int, ...], dtype=object),
    },
    meta_info = {}
)
```

---

## 10. 数据字段参考

### 10.1 原始数据字段（Parquet 文件）

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `data_source` | str | 数据来源标识 | `"BytedTsinghua-SIA/DAPO-Math-17k"` |
| `prompt` | List[dict] | 消息列表 | `[{"role": "user", "content": "..."}]` |
| `ability` | str | 能力分类（可为空） | `""` |
| `reward_model` | dict | 奖励配置 | `{"style": "rule", "ground_truth": "34"}` |
| `extra_info` | dict | 额外信息 | `{"split": "train", "index": 0}` |

### 10.2 Dataset __getitem__ 返回字段

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `input_ids` | Tensor | `[max_prompt_length]` | 左填充的 token IDs |
| `attention_mask` | Tensor | `[max_prompt_length]` | 注意力掩码 (0/1) |
| `position_ids` | Tensor | `[max_prompt_length]` | 位置编码 |
| `raw_prompt_ids` | List[int] | 变长 | 未填充的原始 token IDs |
| `data_source` | str | - | 数据来源 |
| `ability` | str | - | 能力分类 |
| `reward_model` | dict | - | 奖励配置 |
| `extra_info` | dict | - | 额外信息 |
| `index` | int | - | 样本索引 |

### 10.3 DataProto 字段

**batch (张量)**：
| 字段 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `[batch_size, max_prompt_length]` | 批量的 token IDs |
| `attention_mask` | `[batch_size, max_prompt_length]` | 批量的注意力掩码 |
| `position_ids` | `[batch_size, max_prompt_length]` | 批量的位置编码 |

**non_tensor_batch (非张量)**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `raw_prompt_ids` | `np.ndarray[object]` | 变长 token ID 列表数组 |
| `data_source` | `np.ndarray[object]` | 数据来源字符串数组 |
| `ability` | `np.ndarray[object]` | 能力分类数组 |
| `reward_model` | `np.ndarray[object]` | 奖励配置字典数组 |
| `extra_info` | `np.ndarray[object]` | 额外信息字典数组 |
| `index` | `np.ndarray[object]` | 样本索引数组 |

---

## 总结

本文档详细介绍了 Agent-R1 ReTool 训练的数据准备流程：

1. **预处理脚本** (`retool.py`) 将 DAPO-Math-17K 数据集转换为标准格式，添加 instruction 模板和奖励配置
2. **ToolRLDataset** 类负责加载数据、应用 Chat Template、分词和填充
3. **collate_fn** 将多个样本合并为 batch，分离张量和非张量数据
4. **DataProto** 提供统一的数据协议，便于在 Driver 和 Worker 之间传递数据

关键要点：
- 使用**左填充**以便生成时在末尾追加 token
- `raw_prompt_ids` 保留未填充的原始 token，用于后续灵活处理
- `reward_model` 字段包含奖励计算所需的 ground truth
- ReTool 的 instruction 模板已包含工具使用说明，因此 `use_default_tool_template=False`

下一篇文档将介绍工具环境系统 (`03_tool_environment.md`)，详细讲解 `PythonTool` 和 `ReToolEnv` 的实现。
