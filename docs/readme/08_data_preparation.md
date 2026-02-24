# 数据准备指南

本文档详细介绍如何为 Agent-R1 准备训练数据，包括数据格式规范、预处理脚本使用和支持的数据集。

---

## 数据格式规范

Agent-R1 使用 Parquet 格式存储训练数据。每条数据记录包含以下字段：

### 必需字段

```python
{
    "data_source": str,          # 数据源标识符
    "prompt": List[Dict],        # 对话历史（符合聊天模板格式）
    "ability": str,              # 任务类型/能力标签
    "reward_model": {
        "style": str,            # 奖励类型："rule" 或 "model"
        "ground_truth": str      # 正确答案
    }
}
```

### 可选字段

```python
{
    "extra_info": Dict,          # 额外元数据
    "images": List[str],         # 图片路径（多模态任务）
    "videos": List[str]          # 视频路径（多模态任务）
}
```

### 数据示例

```json
{
    "data_source": "hotpotqa/hotpot_qa",
    "prompt": [
        {
            "role": "user",
            "content": "Question: Who was the director of the movie starring the actor born in 1962 who played a role in \"The Departed\"?\n\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags."
        }
    ],
    "ability": "multihop_qa",
    "reward_model": {
        "style": "rule",
        "ground_truth": "Martin Scorsese"
    }
}
```

---

## Prompt 格式

### 聊天模板格式

`prompt` 字段是一个消息列表，遵循 OpenAI 聊天格式：

```python
prompt = [
    {"role": "system", "content": "You are a helpful assistant."},  # 可选
    {"role": "user", "content": "用户问题..."},
]
```

### 指令格式建议

为了获得最佳训练效果，建议在用户问题中包含格式指令：

```python
instruction = (
    'You FIRST think about the reasoning process as an internal monologue '
    'and then provide the final answer. '
    'The reasoning process MUST BE enclosed within <think> </think> tags. '
    'The final answer MUST BE put in <answer> </answer> tags.'
)

prompt = [{
    "role": "user",
    "content": f"Question: {question}\n\n{instruction}"
}]
```

---

## 数据源标识符

数据源标识符用于选择正确的奖励函数：

| 数据源 | 标识符 |
|--------|--------|
| HotpotQA | `hotpotqa/hotpot_qa` |
| MuSiQue | `bdsaglam/musique` |
| 2WikiMultihopQA | `xanhho/2WikiMultihopQA` |
| GSM8K | `openai/gsm8k` |
| ReTool/DAPO | `BytedTsinghua-SIA/DAPO-Math-17k` |

---

## 预处理脚本

### HotpotQA 预处理

**脚本路径**: `examples/data_preprocess/hotpotqa.py`

```bash
# 下载并预处理 HotpotQA 数据
python examples/data_preprocess/hotpotqa.py \
    --local_dir ./data/hotpotqa \
    --download_method direct \
    --train_size 25600 \
    --val_size 128
```

**参数说明**:
- `--local_dir`: 输出目录
- `--download_method`: 下载方式（`direct` 或 `huggingface`）
- `--train_size`: 训练集大小
- `--val_size`: 验证集大小

### 2WikiMultihopQA 预处理

**脚本路径**: `examples/data_preprocess/2wikimultihopqa.py`

```bash
python examples/data_preprocess/2wikimultihopqa.py \
    --local_dir ./data/2wikimultihopqa \
    --train_size 25600 \
    --val_size 128
```

### MuSiQue 预处理

**脚本路径**: `examples/data_preprocess/musique.py`

```bash
python examples/data_preprocess/musique.py \
    --local_dir ./data/musique \
    --train_size 25600 \
    --val_size 128
```

### GSM8K 预处理

**脚本路径**: `examples/data_preprocess/gsm8k.py`

```bash
python examples/data_preprocess/gsm8k.py \
    --local_dir ./data/gsm8k
```

### ReTool 预处理

**脚本路径**: `examples/data_preprocess/retool.py`

```bash
python examples/data_preprocess/retool.py \
    --local_dir ./data/retool
```

---

## 预处理脚本详解

### 核心处理函数

```python
def make_map_fn(split):
    def process_fn(example, idx):
        question_raw = example.pop('question')
        answer_raw = example.pop('answer')

        # 构建指令
        instruction = (
            r'You FIRST think about the reasoning process... '
            r'The final answer MUST BE put in <answer> </answer> tags.'
        )
        question = f"Question: {question_raw}\n{instruction}"

        # 构建数据记录
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "multihop_qa",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer_raw
            }
        }
        return data

    return process_fn

# 应用处理函数
train_dataset = train_dataset.map(
    function=make_map_fn('train'),
    with_indices=True
)

# 保存为 Parquet
train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
```

---

## 创建自定义数据集

### 步骤 1: 准备原始数据

```python
import json

# 加载原始数据
with open('my_data.json', 'r') as f:
    raw_data = json.load(f)
```

### 步骤 2: 定义处理函数

```python
def process_item(item):
    """处理单条数据"""
    return {
        "data_source": "my_namespace/my_dataset",
        "prompt": [{
            "role": "user",
            "content": f"Question: {item['question']}\n\n{INSTRUCTION}"
        }],
        "ability": "my_task_type",
        "reward_model": {
            "style": "rule",
            "ground_truth": item['answer']
        },
        "extra_info": {
            "original_id": item.get('id'),
            "category": item.get('category')
        }
    }
```

### 步骤 3: 批量处理并保存

```python
import datasets

# 处理所有数据
processed_data = [process_item(item) for item in raw_data]

# 转换为 Dataset
dataset = datasets.Dataset.from_list(processed_data)

# 划分训练集和验证集
split_dataset = dataset.train_test_split(test_size=0.05, seed=42)

# 保存为 Parquet
split_dataset['train'].to_parquet('data/my_dataset/train.parquet')
split_dataset['test'].to_parquet('data/my_dataset/validation.parquet')
```

### 步骤 4: 添加奖励函数支持

在 `agent_r1/src/reward_score/__init__.py` 中添加：

```python
def _default_compute_score_format(data_source, solution_str, extra_info=None):
    # ... 现有代码 ...
    elif data_source == 'my_namespace/my_dataset':
        from . import my_reward
        res = my_reward.compute_score_format(solution_str)
    # ...
```

---

## 多模态数据

### 图片数据格式

```python
{
    "data_source": "my_vqa_dataset",
    "prompt": [{
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "描述这张图片中发生了什么？"}
        ]
    }],
    "ability": "vqa",
    "reward_model": {
        "style": "rule",
        "ground_truth": "答案描述"
    }
}
```

### 视频数据格式

```python
{
    "data_source": "my_video_dataset",
    "prompt": [{
        "role": "user",
        "content": [
            {"type": "video", "video": "path/to/video.mp4"},
            {"type": "text", "text": "总结这段视频的主要内容。"}
        ]
    }],
    "ability": "video_qa",
    "reward_model": {
        "style": "rule",
        "ground_truth": "视频内容描述"
    }
}
```

---

## 数据验证

### 验证数据格式

```python
import pyarrow.parquet as pq

def validate_dataset(file_path):
    """验证数据集格式"""
    table = pq.read_table(file_path)
    df = table.to_pandas()

    required_fields = ['data_source', 'prompt', 'ability', 'reward_model']

    for field in required_fields:
        if field not in df.columns:
            raise ValueError(f"Missing required field: {field}")

    # 验证 prompt 格式
    for i, row in df.iterrows():
        prompt = row['prompt']
        if not isinstance(prompt, list):
            raise ValueError(f"Row {i}: prompt must be a list")
        for msg in prompt:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Row {i}: invalid message format")

    print(f"Dataset validation passed: {len(df)} samples")
    return True

# 使用
validate_dataset('data/my_dataset/train.parquet')
```

### 检查数据分布

```python
import pandas as pd

def analyze_dataset(file_path):
    """分析数据集"""
    df = pd.read_parquet(file_path)

    print(f"Total samples: {len(df)}")
    print(f"\nData source distribution:")
    print(df['data_source'].value_counts())
    print(f"\nAbility distribution:")
    print(df['ability'].value_counts())

    # 计算 prompt 长度分布
    df['prompt_length'] = df['prompt'].apply(
        lambda x: sum(len(m['content']) for m in x)
    )
    print(f"\nPrompt length statistics:")
    print(df['prompt_length'].describe())

analyze_dataset('data/my_dataset/train.parquet')
```

---

## 数据配置

在训练配置中指定数据路径：

```yaml
data:
  train_files:
    - ./data/hotpotqa/train.parquet
  val_files:
    - ./data/hotpotqa/validation.parquet
  train_batch_size: 128
  max_prompt_length: 8192
  max_response_length: 8192
  shuffle: true
  seed: 42
```

### 多数据集训练

```yaml
data:
  train_files:
    - ./data/hotpotqa/train.parquet
    - ./data/2wikimultihopqa/train.parquet
    - ./data/musique/train.parquet
  val_files:
    - ./data/hotpotqa/validation.parquet
```

---

## 数据集大小建议

| 任务类型 | 推荐训练集大小 | 推荐验证集大小 |
|----------|----------------|----------------|
| 多跳 QA | 25,000 - 50,000 | 100 - 500 |
| 数学推理 | 10,000 - 30,000 | 100 - 500 |
| 代码生成 | 20,000 - 50,000 | 100 - 500 |

---

## 常见问题

### Q1: Parquet 文件读取错误

```python
# 确保安装了必要的依赖
pip install pyarrow pandas
```

### Q2: 内存不足

```python
# 使用分块处理大文件
import pyarrow.parquet as pq

parquet_file = pq.ParquetFile('large_file.parquet')
for batch in parquet_file.iter_batches(batch_size=1000):
    process_batch(batch.to_pandas())
```

### Q3: 数据格式不兼容

确保 `prompt` 字段是列表格式，而不是字符串：

```python
# 错误
prompt = "user question"

# 正确
prompt = [{"role": "user", "content": "user question"}]
```

---

## 下一步

- [09_running_experiments.md](./09_running_experiments.md) - 使用准备好的数据运行实验
- [07_reward_system.md](./07_reward_system.md) - 为自定义数据集添加奖励函数
