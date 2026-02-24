# 模型导出和部署

本文档介绍如何将训练完成的 Agent-R1 模型导出、转换和部署用于推理。

## 目录

- [从 Checkpoint 合并模型](#从-checkpoint-合并模型)
- [模型格式转换](#模型格式转换)
- [推理部署](#推理部署)
- [模型评估](#模型评估)
- [常见问题](#常见问题)

---

## 从 Checkpoint 合并模型

### FSDP 检查点说明

Agent-R1 使用 FSDP（Fully Sharded Data Parallel）进行分布式训练。训练完成后，模型权重以分片形式存储：

```
checkpoints/my_project/exp01/global_step_400/actor/
├── model/
│   ├── __0_0.distcp    # GPU 0 的分片
│   ├── __1_0.distcp    # GPU 1 的分片
│   ├── .metadata
│   └── ...
└── ...
```

要进行推理，需要先将这些分片合并为完整模型。

### 方法 1：使用模型合并脚本

Agent-R1 提供了模型合并脚本：

```bash
# 基本用法
bash scripts/model_merge.sh \
    --checkpoint_path checkpoints/my_project/exp01/global_step_400/actor/model \
    --output_path ./merged_model \
    --model_name Qwen/Qwen2.5-1.5B-Instruct

# 参数说明
# --checkpoint_path: FSDP 检查点路径
# --output_path: 输出的 HuggingFace 格式模型路径
# --model_name: 原始模型名称（用于加载配置）
```

### 方法 2：Python 脚本合并

```python
import torch
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_fsdp_checkpoint(checkpoint_path, model_name, output_path):
    """
    合并 FSDP 检查点为 HuggingFace 格式
    """
    # 加载原始模型结构
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 加载 FSDP 检查点
    from torch.distributed.checkpoint import load
    state_dict = {}
    load(
        state_dict=state_dict,
        storage_reader=FileSystemReader(checkpoint_path)
    )

    # 加载到模型
    model.load_state_dict(state_dict)

    # 保存为 HuggingFace 格式
    model.save_pretrained(output_path)

    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"模型已保存到: {output_path}")

# 使用示例
merge_fsdp_checkpoint(
    checkpoint_path="checkpoints/my_project/exp01/global_step_400/actor/model",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    output_path="./merged_model"
)
```

### 方法 3：使用 verl 工具

```python
from verl.utils.checkpoint import merge_fsdp_checkpoint

merge_fsdp_checkpoint(
    fsdp_checkpoint_path="checkpoints/.../actor/model",
    hf_model_path="Qwen/Qwen2.5-1.5B-Instruct",
    output_path="./merged_model"
)
```

### 验证合并结果

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载合并后的模型
model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# 测试推理
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

---

## 模型格式转换

### 转换为 vLLM 格式

vLLM 可以直接加载 HuggingFace 格式的模型：

```python
from vllm import LLM

# 直接加载
llm = LLM(
    model="./merged_model",
    dtype="bfloat16",
    tensor_parallel_size=1
)
```

### 模型量化

#### AWQ 量化

```bash
# 安装 autoawq
pip install autoawq

# 量化
python -m awq.entry \
    --model_path ./merged_model \
    --w_bit 4 \
    --q_group_size 128 \
    --output_path ./merged_model_awq
```

```python
from awq import AutoAWQForCausalLM

# 量化模型
model = AutoAWQForCausalLM.from_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# 配置量化参数
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4
}

# 执行量化
model.quantize(tokenizer, quant_config=quant_config)

# 保存
model.save_quantized("./merged_model_awq")
tokenizer.save_pretrained("./merged_model_awq")
```

#### GPTQ 量化

```bash
# 安装 auto-gptq
pip install auto-gptq

# 量化
python -m auto_gptq.quantize \
    --model_path ./merged_model \
    --output_path ./merged_model_gptq \
    --bits 4 \
    --group_size 128
```

---

## 推理部署

### 使用 Agent-R1 推理模块

Agent-R1 提供了专门的推理模块：

#### 单次推理

```python
from agent_r1.vllm_infer.run import run_inference

results = run_inference(
    model_path="./merged_model",
    prompts=["What is the capital of France?"],
    temperature=0.7,
    max_tokens=512
)
print(results)
```

#### 交互式对话

```bash
python -m agent_r1.vllm_infer.chat \
    --model_path ./merged_model \
    --tools search \
    --max_turns 5
```

### 使用 vLLM 服务

#### 启动服务

```bash
# 基本启动
python -m vllm.entrypoints.openai.api_server \
    --model ./merged_model \
    --dtype bfloat16 \
    --port 8000

# 带张量并行
python -m vllm.entrypoints.openai.api_server \
    --model ./merged_model \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --port 8000
```

或者使用提供的脚本：

```bash
bash scripts/vllm_serve.sh \
    --model_path ./merged_model \
    --tensor_parallel_size 2 \
    --port 8000
```

#### 调用 API

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "./merged_model",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
)
print(response.json())
```

### 带工具调用的推理

```python
from agent_r1.vllm_infer.run import AgentInference
from agent_r1.tool.tools import _default_tool
from agent_r1.tool.envs import _default_env

# 初始化工具
tools = [_default_tool("search")]
env = _default_env("nous")(tools=tools, max_tool_response_length=1024)

# 初始化推理
agent = AgentInference(
    model_path="./merged_model",
    env=env,
    max_turns=5
)

# 执行推理
result = agent.chat("Who won the 2024 Nobel Prize in Physics?")
print(result)
```

---

## 模型评估

### 使用 Agent-R1 验证

```bash
python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=./merged_model \
    data.val_files="['data/hotpotqa/test.parquet']" \
    trainer.total_training_steps=0 \
    trainer.val_before_train=True \
    trainer.test_freq=1 \
    tool.tools="['search']" \
    ...
```

### 批量评估脚本

```python
import json
from agent_r1.vllm_infer.run import AgentInference
from agent_r1.tool.tools import _default_tool
from agent_r1.tool.envs import _default_env

def evaluate_model(model_path, test_data_path, output_path):
    # 初始化
    tools = [_default_tool("search")]
    env = _default_env("nous")(tools=tools, max_tool_response_length=1024)
    agent = AgentInference(model_path=model_path, env=env, max_turns=5)

    # 加载测试数据
    with open(test_data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]

    results = []
    correct = 0
    total = 0

    for item in test_data:
        question = item['question']
        ground_truth = item['answer']

        # 推理
        response = agent.chat(question)

        # 评估（简单精确匹配）
        is_correct = ground_truth.lower() in response.lower()

        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'response': response,
            'correct': is_correct
        })

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    # 保存结果
    with open(output_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'results': results
        }, f, indent=2)

    return accuracy

# 使用
evaluate_model(
    model_path="./merged_model",
    test_data_path="data/hotpotqa/test.jsonl",
    output_path="evaluation_results.json"
)
```

### 评估指标

| 指标 | 说明 | 计算方式 |
|-----|------|---------|
| Accuracy | 答案准确率 | 正确回答数 / 总问题数 |
| F1 Score | 答案重叠度 | 2 × Precision × Recall / (P + R) |
| EM (Exact Match) | 精确匹配 | 完全匹配的比例 |
| Format Rate | 格式正确率 | 格式正确数 / 总数 |
| Avg Turns | 平均轮数 | 工具调用的平均轮数 |

---

## 常见问题

### Q1: 合并后的模型推理结果不对？

**可能原因**：
1. 检查点不完整
2. 合并过程出错
3. tokenizer 配置不匹配

**解决方案**：
```python
# 验证模型加载
model = AutoModelForCausalLM.from_pretrained("./merged_model")
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 对比原始模型
original_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
print(f"原始模型参数数量: {sum(p.numel() for p in original_model.parameters())}")

# 应该相等
```

### Q2: 量化后精度下降严重？

**解决方案**：
- 使用 4-bit 而不是 3-bit 量化
- 使用较大的 group_size（128 而不是 64）
- 对关键层跳过量化

### Q3: vLLM 服务启动失败？

**检查项**：
```bash
# 检查 GPU 可用性
nvidia-smi

# 检查模型文件完整性
ls -la ./merged_model/

# 检查 config.json
cat ./merged_model/config.json
```

### Q4: 推理速度慢？

**优化建议**：
```python
# 使用更大的 batch size
llm = LLM(model_path, max_num_seqs=256)

# 启用张量并行
llm = LLM(model_path, tensor_parallel_size=2)

# 增加 GPU 内存使用
llm = LLM(model_path, gpu_memory_utilization=0.9)
```

### Q5: 如何部署到生产环境？

**建议**：
1. 使用量化模型减少资源需求
2. 使用负载均衡部署多个实例
3. 添加请求队列处理突发流量
4. 监控延迟和吞吐量指标
5. 设置合理的超时和重试策略

---

## 部署检查清单

- [ ] 模型已正确合并
- [ ] 推理结果与训练时一致
- [ ] 量化后精度可接受
- [ ] 服务可以正常启动
- [ ] API 接口可以正常调用
- [ ] 工具调用功能正常
- [ ] 监控和日志已配置
- [ ] 资源使用在预算内

---

## 下一步

- [09_troubleshooting.md](./09_troubleshooting.md) - 问题排查指南
- [05_training_configuration.md](./05_training_configuration.md) - 回顾训练配置
