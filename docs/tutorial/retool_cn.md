### 教程：为 Qwen3-4B 上的 ReTool 定制工具环境

本教程演示如何为实现 [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) 创建和定制工具环境 - 这是一种通过工具集成学习增强长形式推理的方法。ReTool 在自然语言推理过程中动态地交错实时代码执行，使其成为定制工具环境的绝佳展示。

本教程涵盖四个主要组成部分：
1. 设置用于代码执行的沙箱环境
2. 为 ReTool 创建自定义工具环境
3. 准备数据集
4. 使用强化学习训练模型

#### 1. 设置沙箱环境

##### 安装 SandboxFusion
我们将使用 SandboxFusion 为训练期间的代码执行提供安全环境：

```bash
# 克隆 SandboxFusion 仓库
git clone https://github.com/bytedance/SandboxFusion.git
cd SandboxFusion

# 创建名为 "sandbox-runtime" 的 conda 环境
conda create -n sandbox-runtime python==3.11
conda activate sandbox-runtime

# 安装依赖
pip install -r runtime/python/requirement.txt
pip install poetry
poetry install

# 准备并运行沙箱
mkdir -p docs/build
make run-online
```

沙箱服务现在应该正在运行并准备处理代码执行请求。

#### 2. 为 ReTool 创建自定义工具环境

ReTool 实现了一个独特的状态转换函数，将代码执行与语言模型推理集成：在推理过程中，语言模型可以通过 XML 标签 `<code></code>` 请求代码执行。这些标签之间的代码被提取并在沙箱环境中执行，执行结果通过 `<interpreter></interpreter>` 返回。在收到结果后，语言模型通过整合结果继续推理，决定是执行更多代码还是提供最终答案。

##### 工具环境结构

基础工具环境在 `agent_r1/tool/base.py` 中定义，并为所有自定义工具环境提供接口：

```python
class BaseToolEnv(ABC):
    @abstractmethod
    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        """
        环境的状态转换函数

        Args:
            raw_response: 来自 LLM 的原始响应
            
        Returns:
            tool_response: 来自环境的工具响应
            success: 工具调用是否成功
            active: 轨迹是否处于活跃状态
        """
        pass
    
    # 其他重要方法包括 batch_step、stop、extract_tool_calls 和 format_tool_response
```

##### 将工具环境理解为状态转换函数

要完全理解如何创建自定义工具环境，关键是要认识到它们在为基于 Agent 的 LLM 实现马尔可夫决策过程（MDP）框架中的状态转换函数所起的作用。

在传统的 LLM 中，状态转换是确定性的且简单明了：
1. 模型生成一个 token
2. 该 token 被添加到现有序列中
3. 这个增强的序列成为新状态

然而，对于可以与外部工具交互的基于 Agent 的 LLM，这个过程变得更加复杂和随机：
1. 模型生成可能包含工具触发模式的 token（如 ReTool 中的 `<code>...</code>`）
2. 当检测到此类模式时，环境提取工具调用
3. 工具在外部环境中执行请求（引入非确定性）
4. 工具的响应被格式化并返回给模型
5. 模型根据这些新信息继续生成

工具环境正是这个非确定性状态转换过程的抽象。它定义了：
- 如何识别何时应该调用工具（`process_responses_ids`）- 或等效地，何时终止单轮生成以执行工具
- 如何从模型的响应中提取用于工具执行的参数（`extract_tool_calls`）
- 如何格式化供模型使用的结果（`format_tool_response`）
- 何时停止生成过程（`stop`）

然后，这些抽象方法在核心 `step` 方法中组合在一起，该方法实现了完整的状态转换函数。`step` 方法：
1. 将原始 LLM 响应作为输入
2. 使用 `extract_tool_calls` 提取工具参数
3. 将这些参数传递给适当的工具进行执行
4. 使用 `format_tool_response` 格式化执行结果
5. 返回格式化的响应以及成功和活动状态标志

这种统一的方法允许清晰地分离关注点，同时保持可以为不同工具交互模式定制的内聚状态转换过程。

因此，当创建自定义工具环境时，您本质上是在实现一种特定类型的状态转换函数，该函数规定了您的 Agent 如何与外部工具交互并将它们的响应整合到其推理过程中。

##### 实现 ReTool 环境

现在我们理解了作为状态转换函数的工具环境的理论基础，让我们看看 ReTool 如何具体实现这些概念。ReTool 环境是将 MDP 框架转换为代码的实际示例。

```python
class ReToolEnv(BaseToolEnv):
    def __init__(self, tools: List[BaseTool], max_tool_response_length: int):
        self.tool = tools[0]
        assert self.tool.name == "python"
        self.max_tool_response_length = max_tool_response_length
```

在初始化中，ReTool 接受工具（特别是 Python 执行器）并设置响应长度限制，建立环境的参数。

让我们看看 ReTool 如何实现非确定性状态转换过程的每个组件：

> **关于 `process_responses_ids` 的特别说明**：在大多数情况下，工具调用在模型生成特定 token 或特殊字符串时触发。对于这些常见场景，我们可以简单地配置生成参数，如 `stop_token_ids` 或 `stop`（例如，在 vLLM 中）在适当的点终止生成，而无需实现自定义的 `process_responses_ids` 方法。在 ReTool 中，我们可以简单地设置 `stop=["</code>"]` 在代码块结束时终止生成。`process_responses_ids` 方法仅在需要复杂的模式识别来识别何时触发工具执行的更复杂情况下才需要。

1. **提取工具执行参数**（`extract_tool_calls`）：

```python
def extract_tool_calls(self, raw_response: str) -> List[Any]:
    """
    提取 "<code>" 之后和 "</code>" 之前的代码
    """
    code = ''
    start = False
    for line in raw_response.split('\n'):
        if line.startswith('<code>'):
            code += '\n# ========\n'
            start = True
        elif line.startswith('</code>'):
            start = False
        elif start:
            if line.startswith('```'):
                continue
            code += line + '\n'
    if start or len(code) == 0:
        # 代码不完整
        return []
    return [code]
```

此方法识别并提取 LLM 响应中由 `<code>...</code>` 标签包围的代码。它不执行代码本身，而是准备参数（要执行的代码）以供稍后实际工具执行使用。

2. **为模型格式化结果**（`format_tool_response`）：

```python
def format_tool_response(self, tool_responses: List[str]) -> str:
    if len(tool_responses) == 0:
        return ""
    if len(tool_responses[0]) > self.max_tool_response_length:
        tool_responses[0] = tool_responses[0][:self.max_tool_response_length] + "..."
    return "\n<interpreter>\n" + tool_responses[0] + "\n</interpreter>\n"
```

此方法将执行结果格式化为 LLM 可以识别并整合到其推理中的结构，将输出包装在 `<interpreter></interpreter>` 标签中。

3. **确定何时停止生成**（`stop`）：

```python
def stop(self, raw_response: str) -> bool:
    tool_calls = self.extract_tool_calls(raw_response)
    if len(tool_calls) == 0:
        return True
    else:
        return False
```

`stop` 方法决定何时结束生成 - 在这种情况下，当没有更多代码块要执行时，表明模型已完成推理或提供了最终答案。

4. **编排完整的状态转换**（`step` 方法）：

```python
def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
    code = self.extract_tool_calls(raw_response)
    if len(code) == 0:
        return "", [], False
    code = code[0]
    tool_response, tool_success = self.tool.execute({"code": code})
    tool_response = self.format_tool_response([tool_response])
    return tool_response, [tool_success], True
```

`step` 方法协调整个状态转换过程：
- 首先，它使用 `extract_tool_calls` 提取代码参数
- 如果找到有效代码，它将这些参数传递给实际工具进行执行
- 执行发生在工具本身（`self.tool.execute`），而不是在环境中
- 然后格式化结果并返回带有状态信息的结果

ReTool 环境还实现了 `batch_step` 方法，用于训练期间的高效批处理。通过这些组件，ReTool 创建了代码执行 Agent 所需的非确定性状态转换函数的完整实现。

这种实现启用了 ReTool 核心的"思考-执行-思考"循环：LLM 可以推理问题，编写代码来解决其中的部分问题，观察执行结果，并基于这些新信息继续推理。

##### 注册工具环境

工具环境在 `agent_r1/tool/envs/__init__.py` 中注册，以使其可供使用：

```python
from agent_r1.tool.envs.retool import ReToolEnv

def _default_env(name):
    # ...
    elif name == "retool":
        return ReToolEnv
    # ...
```

这种工厂模式允许您的代码根据配置创建适当的工具环境。

#### 3. 准备数据

```bash
# 创建数据目录
mkdir -p data/retool

# 运行预处理脚本
python examples/data_preprocess/retool.py --local_dir ./data/retool
```

预处理脚本将自动下载并准备 ReTool 数据集以进行训练。

#### 4. 训练模型

ReTool 使用特殊格式来触发代码执行工具，如原始论文中所述。我们将首先使用少量数据进行冷启动。微调模型可在 `russwest404/Qwen3-4B-ReTool-SFT` 下载。

##### 准备训练脚本
将训练脚本复制到主目录：

```bash
cp examples/trainer/run_ppo_retool.sh ./
```

##### 重要配置更新
在运行训练脚本之前，您需要修改配置文件以正确处理 `</code>` 停止字符。在 bash 脚本中，此字符可能被错误解析。

```bash
# 编辑 agent trainer 配置文件
nano agent_r1/src/config/agent_trainer.yaml
```

在配置文件中，找到 `actor_rollout_ref.rollout` 部分并更新 `stop` 参数。您应该将 `"</code>"` 添加到停止列表：

```yaml
  rollout:
    # ... 现有配置 ...
    stop: ["</code>"]
```

进行此更改后保存文件。

> **注意**：更优雅的解决方案是直接在训练脚本本身中配置此参数，避免手动配置的需要。如果您发现了直接在脚本中设置 `actor_rollout_ref.rollout.stop` 的方法（例如，通过使用环境变量或命令行参数），我们欢迎您提交 pull request 来改进此工作流程。

##### 配置训练参数
编辑 `run_ppo_retool.sh` 脚本，根据您的需要调整参数，确保工具环境设置为 "retool"。

##### 运行训练
执行训练脚本：

```bash
bash run_ppo_retool.sh
```

训练进度和日志将保存到 Wandb（如果已配置）和控制台。

