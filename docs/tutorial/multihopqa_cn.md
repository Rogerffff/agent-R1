### 教程：为多跳问答任务定制工具

本教程演示如何为基于 Agent 的多跳问答创建和定制工具。多跳问答是一项具有挑战性的任务，需要跨多个文档进行推理以得出答案，这使其成为展示工具定制的绝佳用例。

本教程涵盖四个主要组成部分：
1. 设置知识访问的检索服务
2. 创建用于 Agent 交互的自定义工具
3. 准备数据集（HotpotQA、2Wiki 和 MusiQue）
4. 使用 GRPO 训练模型

#### 1. 设置检索服务

##### 下载索引和语料库
从 Hugging Face 下载所需数据：
- 语料库：`corag/kilt-corpus`
- 索引：`russwest404/kilt_index`

```bash
# 创建数据目录
mkdir -p data/corpus/kilt

# 下载语料库和索引（使用 Hugging Face CLI 或网页下载）
huggingface-cli download corag/kilt-corpus
huggingface-cli download russwest404/kilt_index --local-dir data/corpus/kilt
```

##### 配置并启动检索服务
修改 `scripts/kilt_search_server/run_search_api.sh` 中的 `INDEX_PATH`，使其指向正确的索引文件位置：

```bash
# 编辑脚本以修改 INDEX_PATH
export INDEX_PATH="../../data/corpus/kilt/kilt_index_IVF16384_PQ64.bin"
```

运行脚本以启动检索服务：

```bash
cd scripts/kilt_search_server
bash run_search_api.sh
```

##### 可选：构建自己的索引
默认索引使用倒排索引和 PQ 量化，这可能会降低检索性能。如果您的硬件资源允许，可以考虑构建自己的索引。

我们在 `scripts/kilt_search_server/process_kilt.py` 中提供了示例代码，您可以为不同的索引类型进行配置：

```bash
python process_kilt.py --index_type "HNSW64"
```

支持的索引类型包括：
- `Flat`：精确但占用大量内存
- `IVF4096,Flat`：性能和准确性之间的平衡
- `IVF4096,PQ96`：压缩索引，牺牲一些准确性
- `HNSW32`/`HNSW64`：高效的近似最近邻搜索

#### 2. 创建用于 Agent 交互的自定义工具

自定义工具允许您的 Agent 与外部服务交互并执行特定操作。对于多跳问答任务，我们将创建一个可以跨多个知识源搜索信息的工具。

##### 工具结构

每个工具都基于 `BaseTool` 类（位于 `agent_r1/tool/base.py`），该类定义了一个通用接口：

```python
class BaseTool(ABC):
    name: str = ''              # 用于函数调用的工具名称
    description: str = ''       # 解释工具功能的描述
    parameters: dict = {}       # 定义预期参数的 JSON schema
    
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        # 工具逻辑的实现
        pass
```

##### 创建维基百科搜索工具

让我们来看看如何实现一个用于搜索维基百科的自定义工具。这个实现可以在 `agent_r1/tool/tools/wiki_search_tool.py` 中找到。Agent 将使用此工具检索回答多跳问题所需的信息。

1. 使用元数据定义工具类：

```python
class WikiSearchTool(BaseTool):
    name = "search"
    description = "使用维基百科作为知识源在互联网上搜索信息。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索查询"}
        },
        "required": ["query"]
    }
```

2. 实现 `execute` 方法来处理搜索请求：

```python
def execute(self, args: Dict) -> Dict[str, Any]:
    """
    执行搜索查询
    
    Args:
        args: 工具参数，包含：
            - "query": 搜索查询字符串
        
    Returns:
        格式为 {"content": result_content, "success": bool} 的字典
    """
    query = args.get("query", "").strip()
    
    # 调用搜索 API
    response = requests.get(
        f"{self.api_url}/search",
        params={"query": query, "top_k": limit}
    )
    
    # 处理和格式化结果
    if response.status_code == 200:
        result = response.json()
        formatted_result = self._format_results(result)
        return {"content": formatted_result, "success": True}
    else:
        return {"content": "搜索失败", "success": False}
```

##### 集成自定义工具

要使您的工具对 Agent 可用，您需要在工具注册文件 `agent_r1/tool/tools/__init__.py` 中注册它们：

```python
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool

def _default_tool(name):
    # ...
    elif name == "wiki_search":
        return WikiSearchTool()
    # ...
```

这种工厂函数方法允许按名称按需创建工具。在设置环境时，您可以通过配置指定要启用哪些工具，环境将使用此注册表来创建适当的工具实例。

##### 创建自己的工具

要创建您自己的自定义工具：

1. 继承 `BaseTool` 并定义必需的属性：
   - `name`：工具的唯一标识符
   - `description`：关于工具功能和使用时机的清晰说明
   - `parameters`：定义预期输入参数的 JSON schema

2. 实现 `execute` 方法来处理工具的逻辑，并以一致的格式返回结果。

3. 可选：如果您的工具可能一次处理多个类似请求，可以实现 `batch_execute` 以进行优化的批处理。

对于更复杂的工具，请考虑如何正确处理错误、为 Agent 格式化响应，以及在需要时维护与外部服务的有状态连接。

#### 3. 准备数据

##### HotpotQA

```bash
# 创建数据目录
mkdir -p data/hotpotqa

# 运行预处理脚本
python examples/data_preprocess/hotpotqa.py --local_dir ./data/hotpotqa
```

预处理脚本将自动下载 HotpotQA 数据集并将其转换为训练所需的格式，保存为 `train.parquet` 和 `validation.parquet`。

##### 2Wiki

```bash
# 创建数据目录
mkdir -p data/2wiki

# 运行预处理脚本
python examples/data_preprocess/2wikimultihopqa.py --local_dir ./data/2wiki
```

预处理脚本将自动下载 2WikiMultihopQA 数据集并将其转换为所需格式，保存为 `train_processed.parquet` 和 `validation_processed.parquet`。

##### MusiQue

```bash
# 创建数据目录
mkdir -p data/musique

# 运行预处理脚本，默认使用 answerable 配置
python examples/data_preprocess/musique.py --local_dir ./data/musique --config answerable
```

预处理将生成 `train_answerable_processed.parquet` 和 `validation_answerable_processed.parquet` 文件。

#### 4. 训练模型

##### 准备训练脚本
将训练脚本复制到主目录：

```bash
cp examples/trainer/run_grpo_multihopqa.sh ./
```

##### 配置训练参数
编辑 `run_grpo_multihopqa.sh` 脚本，根据您的需要调整参数。

##### 运行训练
执行训练脚本：

```bash
bash run_grpo_multihopqa.sh
```

训练进度和日志将保存到 Wandb（如果已配置）和控制台。

