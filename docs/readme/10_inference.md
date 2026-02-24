# 推理部署指南

本文档详细介绍如何将训练好的 Agent-R1 模型部署为推理服务，包括检查点转换、vLLM 服务部署、交互式推理和 API 调用。

---

## 推理流程概览

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   训练检查点     │ ──> │   HF 格式模型    │ ──> │   vLLM 服务     │
│   (FSDP shards) │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                              ┌─────────────────────────┼─────────────────────────┐
                              │                         │                         │
                              ▼                         ▼                         ▼
                    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
                    │   交互式聊天     │     │   单次查询       │     │   自定义客户端   │
                    │   (chat.py)     │     │   (run.py)      │     │   (OpenAI API)  │
                    └─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 步骤 1: 检查点转换

### 训练检查点结构

训练完成后，检查点保存在 `checkpoints/<project>/<experiment>/global_step_<N>/` 目录：

```
checkpoints/
└── hotpotqa/
    └── ppo-qwen2.5-1.5b/
        └── global_step_100/
            ├── actor/
            │   ├── model/          # FSDP 分片模型
            │   ├── optimizer/      # 优化器状态
            │   └── extra/          # 额外元数据
            └── critic/             # Critic 模型（如果有）
```

### 确保检查点已保存

在训练时，需要设置 `trainer.save_freq > 0` 才会保存检查点：

```bash
# 训练时每 100 步保存一次
trainer.save_freq=100
```

### 使用 model_merge.sh 转换

**脚本位置**: `scripts/model_merge.sh`

```bash
#!/bin/bash
export CHECKPOINT_DIR=<your_checkpoint_dir>
export HF_MODEL_PATH=<your_hf_model_path>
export TARGET_DIR=<your_target_dir>

python3 verl/scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path $HF_MODEL_PATH \
    --local_dir $CHECKPOINT_DIR \
    --target_dir $TARGET_DIR
```

### 转换示例

```bash
# 设置路径
export CHECKPOINT_DIR=./checkpoints/hotpotqa/ppo-qwen2.5-1.5b/global_step_100/actor
export HF_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
export TARGET_DIR=./models/hotpotqa-ppo-1.5b

# 复制并运行脚本
cp scripts/model_merge.sh ./
bash model_merge.sh
```

### 转换参数说明

| 参数 | 说明 |
|------|------|
| `--backend` | 分布式后端，使用 `fsdp` |
| `--hf_model_path` | 原始 HuggingFace 模型路径 |
| `--local_dir` | 训练检查点目录（actor 文件夹） |
| `--target_dir` | 输出的 HF 格式模型目录 |

### 验证转换结果

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载转换后的模型
model = AutoModelForCausalLM.from_pretrained("./models/hotpotqa-ppo-1.5b")
tokenizer = AutoTokenizer.from_pretrained("./models/hotpotqa-ppo-1.5b")

print(f"Model loaded: {model.config.model_type}")
print(f"Vocab size: {model.config.vocab_size}")
```

---

## 步骤 2: 部署 vLLM 服务

### vLLM 服务脚本

**脚本位置**: `scripts/vllm_serve.sh`

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME=<your_model_name>

vllm serve $MODEL_NAME \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name agent \
    --port 8000
```

### 启动服务

```bash
# 设置模型路径
export MODEL_NAME=./models/hotpotqa-ppo-1.5b

# 复制并修改脚本
cp scripts/vllm_serve.sh ./

# 启动服务
bash vllm_serve.sh
```

### vLLM 服务参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable-auto-tool-choice` | 启用自动工具选择 | - |
| `--tool-call-parser` | 工具调用解析器 | `hermes` |
| `--served-model-name` | API 中的模型名称 | `agent` |
| `--port` | 服务端口 | `8000` |

### 多 GPU 部署

```bash
# 使用多个 GPU 进行张量并行
export CUDA_VISIBLE_DEVICES=0,1

vllm serve $MODEL_NAME \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name agent \
    --port 8000 \
    --tensor-parallel-size 2
```

### 更多 vLLM 服务参数

```bash
vllm serve $MODEL_NAME \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name agent \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --enforce-eager  # 禁用 CUDA graph，用于调试
```

---

## 步骤 3: 推理接口

### 交互式聊天模式

**脚本位置**: `agent_r1/vllm_infer/chat.py`

启动交互式聊天：

```bash
python3 -m agent_r1.vllm_infer.chat
```

**使用示例**：

```
Starting interactive chat with model: agent
Type 'exit', 'quit', or 'q' to end the conversation
==================================================

 User  Who directed the movie "The Departed"?

 Assistant  <think>Let me search for information about the movie "The Departed"...</think>

<tool_call>
{"name": "search", "arguments": {"query": "The Departed director"}}
</tool_call>

 Tool Call  Function: search
Arguments:
{
  "query": "The Departed director"
}

 Tool  Martin Scorsese directed "The Departed" (2006), a crime thriller film...

 Assistant  <think>Based on the search results, I found the answer.</think>

<answer>Martin Scorsese</answer>

 User  exit
Ending conversation. Goodbye!
```

### 单次查询模式

**脚本位置**: `agent_r1/vllm_infer/run.py`

运行单次查询：

```bash
python3 -m agent_r1.vllm_infer.run --question "Who is older, James Harden or Stephen Curry?"
```

**命令行参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--question` | 要问的问题 | 示例问题 |
| `--tools` | 使用的工具 | `search` |
| `--model` | 模型名称 | `agent` |
| `--api-base` | API 地址 | `http://localhost:8000/v1` |
| `--temperature` | 采样温度 | 0.7 |
| `--top-p` | Top-P 采样 | 0.8 |
| `--max-tokens` | 最大生成长度 | 512 |
| `--no-color` | 禁用彩色输出 | False |
| `--config` | 自定义配置文件 | None |

---

## 配置说明

### 默认配置

**配置文件**: `agent_r1/vllm_infer/config.py`

```python
# 环境和 API 设置
TOOLS = ["search"]
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "agent"

# 模型推理参数
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_TOKENS = 512
REPETITION_PENALTY = 1.05

# 指令跟随提示
INSTRUCTION_FOLLOWING = (
    r'You FIRST think about the reasoning process as an internal monologue '
    r'and then provide the final answer. '
    r'The reasoning process MUST BE enclosed within <think> </think> tags. '
    r'The final answer MUST BE put in <answer> </answer> tags.'
)
```

### 自定义配置

创建自定义配置文件：

```python
# my_config.py
TOOLS = ["wiki_search"]           # 使用 Wikipedia 搜索
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "my_agent"           # 自定义模型名称

TEMPERATURE = 0.5                 # 降低随机性
TOP_P = 0.9
MAX_TOKENS = 1024                 # 增加生成长度
REPETITION_PENALTY = 1.1

INSTRUCTION_FOLLOWING = (
    r'Please think step by step and provide a detailed answer. '
    r'Enclose your reasoning in <think> </think> tags. '
    r'Put your final answer in <answer> </answer> tags.'
)
```

使用自定义配置：

```bash
# 交互式聊天
python3 -m agent_r1.vllm_infer.chat --config my_config.py

# 单次查询
python3 -m agent_r1.vllm_infer.run --config my_config.py --question "What is quantum computing?"
```

---

## OpenAI 兼容 API

vLLM 服务提供 OpenAI 兼容的 API 接口，可以使用标准 OpenAI 客户端调用。

### Python 客户端示例

```python
from openai import OpenAI

# 连接到 vLLM 服务
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# 定义工具
tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}]

# 发送请求
messages = [
    {"role": "user", "content": "Who directed The Departed?"}
]

response = client.chat.completions.create(
    model="agent",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0.7,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

### 处理工具调用

```python
import json
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

def execute_tool(tool_name, arguments):
    """执行工具调用"""
    # 这里实现实际的工具逻辑
    if tool_name == "search":
        return f"Search results for: {arguments['query']}"
    return "Unknown tool"

def chat_with_tools(question, max_turns=5):
    """带工具调用的对话"""
    messages = [{"role": "user", "content": question}]

    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="agent",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
        )

        assistant_message = response.choices[0].message
        messages.append({
            "role": "assistant",
            "content": assistant_message.content
        })

        # 检查是否有工具调用
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                # 执行工具
                result = execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )

                # 添加工具结果
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.id
                })
        else:
            # 没有工具调用，返回最终答案
            return assistant_message.content

    return messages[-1]["content"]

# 使用
answer = chat_with_tools("Who is the CEO of Apple?")
print(answer)
```

### cURL 调用示例

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "agent",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }'
```

---

## 高级配置

### 使用 Wiki 搜索服务

对于需要实时 Wikipedia 搜索的场景：

1. 启动 Wikipedia 搜索服务：

```bash
# 启动 KILT 搜索服务
cd scripts/kilt_search_server
python server.py --port 5000
```

2. 配置推理使用 wiki_search：

```python
# config.py
TOOLS = ["wiki_search"]
```

### 使用 Python 执行工具

对于 ReTool 任务：

```python
# config.py
TOOLS = ["python"]
```

### 批量推理

```python
import json
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

questions = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
]

results = []
for question in questions:
    response = client.chat.completions.create(
        model="agent",
        messages=[{"role": "user", "content": question}],
        temperature=0.7,
    )
    results.append({
        "question": question,
        "answer": response.choices[0].message.content
    })

# 保存结果
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 推理性能优化

### 显存优化

```bash
# 降低 GPU 显存使用
vllm serve $MODEL_NAME \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096  # 限制最大序列长度
```

### 吞吐量优化

```bash
# 启用 chunked prefill
vllm serve $MODEL_NAME \
    --enable-chunked-prefill \
    --max-num-batched-tokens 65536
```

### 延迟优化

```bash
# 禁用 eager 模式以使用 CUDA graph
vllm serve $MODEL_NAME \
    --enforce-eager=False
```

---

## 常见问题

### Q1: 服务启动失败

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 检查端口是否被占用
lsof -i :8000

# 尝试使用不同端口
vllm serve $MODEL_NAME --port 8001
```

### Q2: 工具调用不生效

确保 vLLM 服务启动时包含以下参数：
```bash
--enable-auto-tool-choice \
--tool-call-parser hermes
```

### Q3: 响应质量差

- 检查模型是否正确加载
- 调整 temperature 和 top_p 参数
- 确保使用正确的 INSTRUCTION_FOLLOWING 提示

### Q4: 显存不足

```bash
# 使用更多 GPU
export CUDA_VISIBLE_DEVICES=0,1
vllm serve $MODEL_NAME --tensor-parallel-size 2

# 或降低显存使用
vllm serve $MODEL_NAME --gpu-memory-utilization 0.7
```

### Q5: 连接超时

```python
# 增加超时时间
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=120.0  # 增加到 120 秒
)
```

---

## 部署架构示例

### 单机部署

```
┌─────────────────────────────────────────────┐
│                 服务器                       │
│  ┌─────────────┐     ┌─────────────────┐   │
│  │  vLLM 服务   │ <── │  推理客户端      │   │
│  │  (GPU 0-3)  │     │  (chat.py)      │   │
│  └─────────────┘     └─────────────────┘   │
└─────────────────────────────────────────────┘
```

### 分布式部署

```
┌─────────────────┐     ┌─────────────────┐
│   GPU 服务器 1   │     │   GPU 服务器 2   │
│   vLLM 实例 1   │     │   vLLM 实例 2   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────┴──────┐
              │   负载均衡    │
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │   API 网关   │
              └─────────────┘
```

---

## 快速参考

### 完整部署流程

```bash
# 1. 转换检查点
export CHECKPOINT_DIR=./checkpoints/hotpotqa/ppo-qwen2.5-1.5b/global_step_100/actor
export HF_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
export TARGET_DIR=./models/hotpotqa-ppo-1.5b
python3 verl/scripts/model_merger.py --backend fsdp --hf_model_path $HF_MODEL_PATH --local_dir $CHECKPOINT_DIR --target_dir $TARGET_DIR

# 2. 启动 vLLM 服务
export MODEL_NAME=./models/hotpotqa-ppo-1.5b
vllm serve $MODEL_NAME --enable-auto-tool-choice --tool-call-parser hermes --served-model-name agent --port 8000

# 3. 运行推理（新终端）
python3 -m agent_r1.vllm_infer.chat
```

### 常用命令速查

| 操作 | 命令 |
|------|------|
| 转换检查点 | `python3 verl/scripts/model_merger.py --backend fsdp ...` |
| 启动服务 | `vllm serve $MODEL_NAME --enable-auto-tool-choice ...` |
| 交互式聊天 | `python3 -m agent_r1.vllm_infer.chat` |
| 单次查询 | `python3 -m agent_r1.vllm_infer.run --question "..."` |
| 使用自定义配置 | `--config my_config.py` |

---

## 相关文档

- [09_running_experiments.md](./09_running_experiments.md) - 训练实验指南
- [01_environment_setup.md](./01_environment_setup.md) - 环境配置
- [04_tool_system.md](./04_tool_system.md) - 工具系统详解
