# 工具环境系统

本文档详细介绍 Agent-R1 的工具环境系统，包括基类设计、PythonTool 实现、ReToolEnv 环境的状态转换机制，以及完整的工具调用流程示例。

---

## 目录

1. [工具系统概览](#1-工具系统概览)
2. [基类设计](#2-基类设计)
3. [PythonTool 实现](#3-pythontool-实现)
4. [ReToolEnv 环境实现](#4-retoolenv-环境实现)
5. [工具注册机制](#5-工具注册机制)
6. [完整工具调用流程](#6-完整工具调用流程)
7. [与 NousToolEnv 的对比](#7-与-noustoolenv-的对比)
8. [批量执行优化](#8-批量执行优化)
9. [扩展指南](#9-扩展指南)

---

## 1. 工具系统概览

Agent-R1 的工具系统采用 **Tool + ToolEnv** 的双层架构：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           工具系统架构                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       ToolEnv (环境编排器)                           │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ ReToolEnv / NousToolEnv / ...                               │   │   │
│  │  │                                                             │   │   │
│  │  │ 职责:                                                        │   │   │
│  │  │ • 从 LLM 输出中提取工具调用 (extract_tool_calls)              │   │   │
│  │  │ • 判断是否需要工具调用 (stop)                                 │   │   │
│  │  │ • 编排工具执行流程 (step/batch_step)                         │   │   │
│  │  │ • 格式化工具响应 (format_tool_response)                      │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    │ 调用                                   │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Tool (工具执行器)                            │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ PythonTool / SearchTool / WikiSearchTool / ...              │   │   │
│  │  │                                                             │   │   │
│  │  │ 职责:                                                        │   │   │
│  │  │ • 执行具体的工具功能 (execute/batch_execute)                  │   │   │
│  │  │ • 与外部服务交互 (API、沙箱等)                                │   │   │
│  │  │ • 返回执行结果和状态                                         │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `agent_r1/tool/base.py` | BaseTool 和 BaseToolEnv 基类定义 |
| `agent_r1/tool/tools/python_tool.py` | PythonTool 实现 |
| `agent_r1/tool/tools/search_tool.py` | SearchTool 实现 |
| `agent_r1/tool/envs/retool.py` | ReToolEnv 环境实现 |
| `agent_r1/tool/envs/nous.py` | NousToolEnv 环境实现 |
| `agent_r1/tool/tools/__init__.py` | 工具注册工厂 |
| `agent_r1/tool/envs/__init__.py` | 环境注册工厂 |

### MDP 框架中的角色

工具系统在 MDP（马尔可夫决策过程）中扮演 **状态转换函数 P(s'|s,a)** 的角色：

| 组件 | MDP 角色 | 说明 |
|------|----------|------|
| LLM 输出 | 动作 a | 模型生成的文本，可能包含工具调用 |
| ToolEnv.step() | 转换函数 P | 执行工具调用，返回新状态 |
| Tool 响应 | 新状态 s' | 工具执行结果追加到序列 |

---

## 2. 基类设计

### 2.1 BaseTool 类

**文件**: `agent_r1/tool/base.py:7-50`

```python
class BaseTool(ABC):
    # 类属性：工具元数据
    name: str = ''           # 工具名称
    description: str = ''    # 工具描述
    parameters: dict = {}    # OpenAI 兼容的 JSON Schema

    def __init__(self):
        # 验证工具名称
        if not self.name:
            raise ValueError('Tool name must be provided')
        # 验证参数 schema 格式
        if not is_tool_schema({
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }):
            raise ValueError(
                'The parameters must confirm to a valid openai-compatible JSON schema.')

    @abstractmethod
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        """
        执行工具（单个调用）

        Args:
            args: 工具参数字典

        Returns:
            {"content": str, "success": bool}
        """
        pass

    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        """
        批量执行工具（默认实现：循环调用 execute）

        子类可重写以实现更高效的批量执行
        """
        return [self.execute(args, **kwargs) for args in args_list]

    @property
    def tool_info(self) -> Dict:
        """返回工具信息（简化格式）"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }

    @property
    def tool_description(self) -> Dict:
        """返回 OpenAI 函数调用格式的工具描述"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def validate_args(self, args: Dict) -> bool:
        """验证工具参数是否符合 schema"""
        try:
            validate(instance=args, schema=self.parameters)
            return True
        except ValidationError:
            return False
```

### 2.2 BaseToolEnv 类

**文件**: `agent_r1/tool/base.py:52-90`

```python
class BaseToolEnv(ABC):
    @abstractmethod
    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        """
        环境的状态转换函数

        Args:
            raw_response: LLM 的原始输出文本

        Returns:
            tool_response: 工具响应文本（将追加到序列）
            success: 每个工具调用是否成功的列表
            active: 是否触发了工具调用（True=触发，False=未触发/结束）
        """
        pass

    def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[bool]], List[bool]]:
        """
        批量状态转换（默认实现：循环调用 step）
        """
        results = [self.step(raw_response) for raw_response in raw_responses]
        return (
            [result[0] for result in results],   # tool_responses
            [result[1] for result in results],   # successes
            [result[2] for result in results]    # actives
        )

    def process_responses_ids(self, tokenizer, raw_responses_ids: torch.Tensor) -> torch.Tensor:
        """
        后处理响应 token IDs（可选，默认不处理）
        """
        return raw_responses_ids

    @abstractmethod
    def stop(self, raw_response: str) -> bool:
        """
        判断是否应该停止生成

        Returns:
            True: 没有工具调用，停止生成
            False: 有工具调用，继续生成
        """
        pass

    @abstractmethod
    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        """从 LLM 输出中提取工具调用参数"""
        pass

    @abstractmethod
    def format_tool_response(self, tool_response: str) -> str:
        """将工具执行结果格式化为追加到序列的文本"""
        pass

    @property
    def system_prompt(self) -> str:
        """系统提示（可选）"""
        return ""
```

### 2.3 step() 返回值详解

```python
tool_response, success_list, active = env.step(raw_response)
```

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `tool_response` | `str` | 格式化后的工具响应文本，将追加到当前序列 |
| `success_list` | `List[bool]` | 每个工具调用的执行状态（成功/失败） |
| `active` | `bool` | 本次是否触发了工具调用。`True` 表示需要继续生成，`False` 表示应该停止 |

**active 的判断逻辑**：
```python
active = True   # 检测到工具调用 → 执行后继续生成
active = False  # 未检测到工具调用 → 停止当前样本的生成
```

---

## 3. PythonTool 实现

### 3.1 类定义

**文件**: `agent_r1/tool/tools/python_tool.py:12-55`

```python
class PythonTool(BaseTool):
    name = "python"
    description = "Python code sandbox, which can be used to execute Python code."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. The Python code should be "
                               "complete scripts, including necessary imports. "
                               "IMPORTANT: Use print() statements to output any results "
                               "you want to see, otherwise they won't be visible.",
            }
        },
        "required": ["code"],
    }

    def __init__(self):
        super().__init__()
        # 设置 SandboxFusion 沙箱服务端点
        set_sandbox_endpoint('http://localhost:8080')
        self.run_timeout = 10      # 执行超时（秒）
        self.concurrency = 32      # 并发执行数
        self.max_attempts = 5      # 重试次数
```

### 3.2 批量执行实现

```python
def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
    """
    批量执行 Python 代码

    使用 SandboxFusion 的 run_concurrent API 实现高效并发执行
    """
    # 提取所有代码
    batch_code = [args.get("code", "") for args in args_list]
    batch_results = []

    # 并发执行所有代码
    results = run_concurrent(
        run_code,
        kwargs=[{
            "request": RunCodeRequest(
                run_timeout=self.run_timeout,
                code=c,
                language='python'
            ),
            'max_attempts': self.max_attempts
        } for c in batch_code],
        concurrency=self.concurrency
    )

    # 处理执行结果
    for result in results:
        if result.status == RunStatus.Success:
            # 执行成功
            if result.run_result and result.run_result.stdout and len(result.run_result.stdout) > 0:
                batch_results.append({
                    "content": result.run_result.stdout,
                    "success": True
                })
            else:
                batch_results.append({
                    "content": "Execution successful but no output",
                    "success": True
                })
        else:
            # 执行失败
            error_message = result.message or "Unknown error"
            if result.run_result and result.run_result.stderr:
                error_message = result.run_result.stderr
            elif result.compile_result and result.compile_result.stderr:
                error_message = result.compile_result.stderr
            batch_results.append({
                "content": error_message,
                "success": False
            })

    # 清理输出中的空白字符
    for result in batch_results:
        result['content'] = result['content'].strip()

    return batch_results

def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
    """单个执行（委托给批量执行）"""
    return self.batch_execute([args])[0]
```

### 3.3 SandboxFusion 沙箱

PythonTool 使用 [SandboxFusion](https://github.com/bytedance/SandboxFusion) 作为代码执行沙箱：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SandboxFusion 架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     HTTP POST      ┌─────────────────────┐                │
│  │             │  ─────────────────▶│                     │                │
│  │  PythonTool │     /run_code      │   SandboxFusion     │                │
│  │             │◀───────────────────│   (localhost:8080)  │                │
│  └─────────────┘     JSON 响应      └─────────────────────┘                │
│                                              │                              │
│                                              ▼                              │
│                                     ┌─────────────────────┐                │
│                                     │   隔离的 Python     │                │
│                                     │   执行环境          │                │
│                                     │   • 超时控制        │                │
│                                     │   • 资源限制        │                │
│                                     │   • 安全隔离        │                │
│                                     └─────────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**请求格式**：
```python
RunCodeRequest(
    run_timeout=10,        # 执行超时（秒）
    code="print(1+1)",     # 要执行的代码
    language='python'      # 语言类型
)
```

**响应格式**：
```python
RunCodeResponse(
    status=RunStatus.Success,  # 或 RunStatus.Failed
    run_result=RunResult(
        stdout="2\n",          # 标准输出
        stderr="",             # 标准错误
    ),
    message=None               # 错误信息
)
```

---

## 4. ReToolEnv 环境实现

### 4.1 初始化

**文件**: `agent_r1/tool/envs/retool.py:4-8`

```python
class ReToolEnv(BaseToolEnv):
    def __init__(self, tools: List[BaseTool], max_tool_response_length: int):
        self.tool = tools[0]  # ReTool 只使用第一个工具（PythonTool）
        assert self.tool.name == "python"
        self.max_tool_response_length = max_tool_response_length  # 响应最大长度（默认 512）
```

### 4.2 代码提取 (extract_tool_calls)

**文件**: `agent_r1/tool/envs/retool.py:49-68`

```python
def extract_tool_calls(self, raw_response: str) -> List[Any]:
    """
    从 LLM 输出中提取 <code>...</code> 标签内的 Python 代码

    解析逻辑：
    1. 逐行扫描
    2. 遇到 <code> 开始收集
    3. 遇到 </code> 停止收集
    4. 跳过 ``` 标记行（Markdown 代码块标记）
    """
    code = ''
    start = False

    for line in raw_response.split('\n'):
        if line.startswith('<code>'):
            code += '\n# ========\n'  # 添加分隔标记
            start = True
        elif line.startswith('</code>'):
            start = False
        elif start:
            if line.startswith('```'):
                continue  # 跳过 Markdown 代码块标记
            code += line + '\n'

    # 检查代码完整性
    if start or len(code) == 0:
        # 代码不完整（没有 </code>）或没有代码
        return []

    return [code]
```

**解析示例**：

输入：
```
<think>让我计算一下</think>
<code>
```python
import math
result = math.sqrt(16)
print(result)
```
</code>
```

提取结果：
```python
["\n# ========\nimport math\nresult = math.sqrt(16)\nprint(result)\n"]
```

### 4.3 停止判断 (stop)

**文件**: `agent_r1/tool/envs/retool.py:42-47`

```python
def stop(self, raw_response: str) -> bool:
    """
    判断是否应该停止生成

    Returns:
        True: 没有检测到工具调用（或代码不完整），停止生成
        False: 检测到完整的工具调用，继续生成
    """
    tool_calls = self.extract_tool_calls(raw_response)
    if len(tool_calls) == 0:
        return True   # 停止
    else:
        return False  # 继续
```

### 4.4 单步执行 (step)

**文件**: `agent_r1/tool/envs/retool.py:10-17`

```python
def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
    """
    单样本状态转换

    流程：
    1. 提取代码
    2. 执行代码
    3. 格式化响应

    Returns:
        (tool_response, [success], active)
    """
    # 1. 提取代码
    code = self.extract_tool_calls(raw_response)
    if len(code) == 0:
        # 没有检测到代码 → 返回空响应，标记为非活跃
        return "", [], False

    code = code[0]  # 取第一个代码块

    # 2. 执行代码
    tool_response, tool_success = self.tool.execute({"code": code})

    # 3. 格式化响应
    tool_response = self.format_tool_response([tool_response])

    return tool_response, [tool_success], True
```

### 4.5 批量执行 (batch_step)

**文件**: `agent_r1/tool/envs/retool.py:19-40`

```python
def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[bool]], List[bool]]:
    """
    批量状态转换（优化版本）

    优化策略：
    1. 先解析所有样本，找出有工具调用的样本
    2. 只对有工具调用的样本进行批量执行
    3. 合并结果

    这样可以利用 PythonTool.batch_execute 的并发能力
    """
    # 初始化结果列表
    batch_tool_response = [""] * len(raw_responses)
    batch_tool_successes = [[]] * len(raw_responses)
    batch_active = [True] * len(raw_responses)

    # 收集需要执行的代码
    codes = []
    for i, raw_response in enumerate(raw_responses):
        code = self.extract_tool_calls(raw_response)
        if len(code) == 0:
            # 没有检测到代码
            batch_tool_response[i] = ""
            batch_tool_successes[i] = []
            batch_active[i] = False
            continue
        codes.append({"code": code[0]})

    # 批量执行（利用 SandboxFusion 的并发能力）
    results = self.tool.batch_execute(codes)

    # 分配结果
    i = 0
    for j in range(len(raw_responses)):
        if batch_active[j]:
            result = results[i]
            batch_tool_response[j] = self.format_tool_response([result["content"]])
            batch_tool_successes[j] = [result["success"]]
            i += 1

    return batch_tool_response, batch_tool_successes, batch_active
```

### 4.6 响应格式化 (format_tool_response)

**文件**: `agent_r1/tool/envs/retool.py:70-75`

```python
def format_tool_response(self, tool_responses: List[str]) -> str:
    """
    将工具执行结果格式化为追加到序列的文本

    格式：<interpreter>\n{output}\n</interpreter>\n
    """
    if len(tool_responses) == 0:
        return ""

    # 截断过长的响应
    if len(tool_responses[0]) > self.max_tool_response_length:
        tool_responses[0] = tool_responses[0][:self.max_tool_response_length] + "..."

    return "\n<interpreter>\n" + tool_responses[0] + "\n</interpreter>\n"
```

**格式化示例**：

执行结果：`"4.0"`

格式化后：
```
<interpreter>
4.0
</interpreter>
```

---

## 5. 工具注册机制

### 5.1 工具注册工厂

**文件**: `agent_r1/tool/tools/__init__.py`

```python
def _default_tool(name):
    """
    工具注册工厂

    根据名称返回对应的工具实例
    """
    if name == "search":
        from agent_r1.tool.tools.search_tool import SearchTool
        return SearchTool()
    elif name == "wiki_search":
        from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
        return WikiSearchTool()
    elif name == "python":
        from agent_r1.tool.tools.python_tool import PythonTool
        return PythonTool()
    else:
        raise NotImplementedError(f"Tool {name} not implemented")
```

### 5.2 环境注册工厂

**文件**: `agent_r1/tool/envs/__init__.py`

```python
def _default_env(name):
    """
    环境注册工厂

    根据名称返回对应的环境类（注意：返回的是类，不是实例）
    """
    if name == "nous":
        from agent_r1.tool.envs.nous import NousToolEnv
        return NousToolEnv
    elif name == "retool":
        from agent_r1.tool.envs.retool import ReToolEnv
        return ReToolEnv
    else:
        raise NotImplementedError(f"Tool environment {name} is not implemented")
```

### 5.3 使用示例

```python
from agent_r1.tool.tools import _default_tool
from agent_r1.tool.envs import _default_env

# 创建工具
tools = [_default_tool("python")]  # [PythonTool()]

# 创建环境
env_class = _default_env("retool")  # ReToolEnv 类
env = env_class(
    tools=tools,
    max_tool_response_length=512
)  # ReToolEnv 实例
```

---

## 6. 完整工具调用流程

### 6.1 单轮工具调用示例

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         完整工具调用流程                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入 prompt:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Solve the following problem step by step...                         │   │
│  │ *user question:*                                                    │   │
│  │ In triangle ABC, sin∠A = 4/5 and ∠A < 90°. Find the value of cos2A.│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  LLM 第一轮生成:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ <think>                                                             │   │
│  │ 我需要计算 cos2A。已知 sin∠A = 4/5，可以用公式 cos2A = 1 - 2sin²A   │   │
│  │ </think>                                                            │   │
│  │ <code>                                                              │   │
│  │ ```python                                                           │   │
│  │ import math                                                         │   │
│  │ sin_A = 4/5                                                         │   │
│  │ cos_2A = 1 - 2 * sin_A**2                                          │   │
│  │ print(f"cos2A = {cos_2A}")                                         │   │
│  │ ```                                                                 │   │
│  │ </code>                                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  │ extract_tool_calls()                     │
│                                  ▼                                          │
│  提取的代码:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ # ========                                                          │   │
│  │ import math                                                         │   │
│  │ sin_A = 4/5                                                         │   │
│  │ cos_2A = 1 - 2 * sin_A**2                                          │   │
│  │ print(f"cos2A = {cos_2A}")                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  │ PythonTool.execute()                     │
│                                  ▼                                          │
│  SandboxFusion 执行:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ stdout: "cos2A = -0.28"                                            │   │
│  │ status: Success                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  │ format_tool_response()                   │
│                                  ▼                                          │
│  格式化响应:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ <interpreter>                                                       │   │
│  │ cos2A = -0.28                                                       │   │
│  │ </interpreter>                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  │ 追加到序列                               │
│                                  ▼                                          │
│  更新后的序列:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ [prompt] + [LLM 第一轮] + [interpreter 响应]                        │   │
│  │                                                                     │   │
│  │ action_mask:                                                        │   │
│  │ [0,0,...,0] + [1,1,...,1] + [0,0,...,0]                            │   │
│  │  ^prompt     ^LLM生成       ^工具响应                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  │ 继续生成                                 │
│                                  ▼                                          │
│  LLM 第二轮生成:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ <think>                                                             │   │
│  │ 计算结果是 -0.28，转换为分数是 -7/25                                │   │
│  │ </think>                                                            │   │
│  │ <answer>                                                            │   │
│  │ \boxed{-7/25}                                                       │   │
│  │ </answer>                                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  │ extract_tool_calls() → []               │
│                                  │ stop() → True                            │
│                                  ▼                                          │
│  最终序列:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ [prompt] + [Turn1] + [interpreter] + [Turn2]                        │   │
│  │                                                                     │   │
│  │ action_mask:                                                        │   │
│  │ [0,...,0] + [1,...,1] + [0,...,0] + [1,...,1]                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 step() 返回值示例

```python
# 第一轮调用
tool_response, success_list, active = env.step(llm_output_turn1)

# 返回值
tool_response = "\n<interpreter>\ncos2A = -0.28\n</interpreter>\n"
success_list = [True]
active = True  # 继续生成

# 第二轮调用
tool_response, success_list, active = env.step(llm_output_turn2)

# 返回值
tool_response = ""
success_list = []
active = False  # 停止生成
```

---

## 7. 与 NousToolEnv 的对比

### 7.1 设计差异

| 特性 | ReToolEnv | NousToolEnv |
|------|-----------|-------------|
| **工具数量** | 单工具（PythonTool） | 多工具支持 |
| **调用格式** | `<code>...</code>` | `<tool_call>{"name": ..., "arguments": ...}</tool_call>` |
| **响应格式** | `<interpreter>...</interpreter>` | `<tool_response>...</tool_response>` |
| **并行调用** | 不支持 | 可配置 |
| **参数验证** | 无需（只有 code 参数） | JSON Schema 验证 |

### 7.2 NousToolEnv 代码解析

**文件**: `agent_r1/tool/envs/nous.py:128-138`

```python
def extract_tool_calls(self, raw_response: str) -> List[Any]:
    """
    从 LLM 输出中提取 <tool_call>...</tool_call> 标签内的 JSON
    """
    tool_calls = []
    pattern = re.compile(
        f"{re.escape(self.tool_call_start)}(.*?){re.escape(self.tool_call_end)}",
        re.DOTALL
    )
    for tool_call in re.findall(pattern, raw_response):
        try:
            tool_call = json.loads(tool_call)
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            tool_calls.append(None)  # 解析失败

    return tool_calls
```

### 7.3 NousToolEnv 调用示例

```
LLM 输出:
<tool_call>
{"name": "search", "arguments": {"query": "Python list comprehension"}}
</tool_call>

解析结果:
[{"name": "search", "arguments": {"query": "Python list comprehension"}}]

工具响应:
<|im_end|>
<|im_start|>user
<tool_response>
Python list comprehension is a concise way to create lists...
</tool_response><|im_end|>
<|im_start|>assistant
<think>
```

---

## 8. 批量执行优化

### 8.1 批量执行流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        批量执行优化示意图                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: batch_size = 128 个 LLM 响应                                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 第 1 步: 解析所有响应                                                │   │
│  │                                                                     │   │
│  │ responses[0] → extract_tool_calls() → code[0]  (有代码)            │   │
│  │ responses[1] → extract_tool_calls() → []       (无代码)            │   │
│  │ responses[2] → extract_tool_calls() → code[2]  (有代码)            │   │
│  │ ...                                                                 │   │
│  │ responses[127] → extract_tool_calls() → code[127] (有代码)         │   │
│  │                                                                     │   │
│  │ 结果: 100 个有代码, 28 个无代码                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 第 2 步: 批量执行有代码的样本                                        │   │
│  │                                                                     │   │
│  │ PythonTool.batch_execute([code[0], code[2], ..., code[127]])       │   │
│  │                                                                     │   │
│  │            ┌─────────────────────────────────────────┐             │   │
│  │            │       SandboxFusion                     │             │   │
│  │            │  ┌───┐ ┌───┐ ┌───┐ ... ┌───┐          │             │   │
│  │            │  │ 1 │ │ 2 │ │ 3 │     │32 │          │             │   │
│  │            │  └───┘ └───┘ └───┘     └───┘          │             │   │
│  │            │     并发执行 (concurrency=32)          │             │   │
│  │            └─────────────────────────────────────────┘             │   │
│  │                                                                     │   │
│  │ 返回 100 个执行结果                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 第 3 步: 分配结果                                                    │   │
│  │                                                                     │   │
│  │ batch_tool_response[0] = format(results[0])                        │   │
│  │ batch_tool_response[1] = ""  (无代码)                              │   │
│  │ batch_tool_response[2] = format(results[1])                        │   │
│  │ ...                                                                 │   │
│  │                                                                     │   │
│  │ batch_active[0] = True   (继续生成)                                │   │
│  │ batch_active[1] = False  (停止生成)                                │   │
│  │ batch_active[2] = True   (继续生成)                                │   │
│  │ ...                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 性能对比

| 方式 | 描述 | 时间复杂度 |
|------|------|------------|
| 串行执行 | 每个样本独立调用 execute() | O(n × t)，n=样本数，t=单次执行时间 |
| 批量执行 | 使用 batch_execute() 并发 | O(n/c × t)，c=并发数 |

假设：
- batch_size = 128
- 单次执行时间 t = 1 秒
- 并发数 c = 32

则：
- 串行执行：128 × 1 = 128 秒
- 批量执行：128 / 32 × 1 = 4 秒

---

## 9. 扩展指南

### 9.1 添加新工具

1. **创建工具类**：

```python
# agent_r1/tool/tools/my_tool.py
from agent_r1.tool.base import BaseTool
from typing import Dict, Any

class MyTool(BaseTool):
    name = "my_tool"
    description = "描述你的工具功能"
    parameters = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "参数1的描述"
            },
            "param2": {
                "type": "integer",
                "description": "参数2的描述"
            }
        },
        "required": ["param1"]
    }

    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        param1 = args.get("param1", "")
        param2 = args.get("param2", 0)

        # 执行工具逻辑
        result = f"处理 {param1} 和 {param2}"

        return {"content": result, "success": True}

    def batch_execute(self, args_list, **kwargs):
        # 可选：实现高效的批量执行
        return [self.execute(args) for args in args_list]
```

2. **注册工具**：

```python
# agent_r1/tool/tools/__init__.py
def _default_tool(name):
    # ... 现有代码 ...
    elif name == "my_tool":
        from agent_r1.tool.tools.my_tool import MyTool
        return MyTool()
    # ...
```

### 9.2 添加新环境

1. **创建环境类**：

```python
# agent_r1/tool/envs/my_env.py
from agent_r1.tool.base import BaseToolEnv, BaseTool
from typing import List, Tuple, Any

class MyToolEnv(BaseToolEnv):
    def __init__(self, tools: List[BaseTool], max_tool_response_length: int):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.max_tool_response_length = max_tool_response_length

    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        tool_calls = self.extract_tool_calls(raw_response)
        if len(tool_calls) == 0:
            return "", [], False

        # 执行工具调用
        # ...

        return tool_response, success_list, True

    def stop(self, raw_response: str) -> bool:
        return len(self.extract_tool_calls(raw_response)) == 0

    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        # 实现你的解析逻辑
        pass

    def format_tool_response(self, tool_responses: List[str]) -> str:
        # 实现你的格式化逻辑
        pass
```

2. **注册环境**：

```python
# agent_r1/tool/envs/__init__.py
def _default_env(name):
    # ... 现有代码 ...
    elif name == "my_env":
        from agent_r1.tool.envs.my_env import MyToolEnv
        return MyToolEnv
    # ...
```

3. **使用新环境**：

```bash
python -m agent_r1.src.main_agent \
    tool.env=my_env \
    tool.tools=['my_tool'] \
    # ... 其他参数
```

---

## 总结

本文档详细介绍了 Agent-R1 的工具环境系统：

1. **双层架构**：Tool（工具执行器）+ ToolEnv（环境编排器）
2. **BaseTool**：定义工具的元数据、执行接口和参数验证
3. **BaseToolEnv**：定义状态转换、代码提取、响应格式化等接口
4. **PythonTool**：通过 SandboxFusion 沙箱执行 Python 代码
5. **ReToolEnv**：使用 `<code>...</code>` 和 `<interpreter>...</interpreter>` 标签
6. **批量执行优化**：利用并发能力显著提升性能
7. **扩展机制**：通过工厂模式轻松添加新工具和环境

关键要点：
- `step()` 返回 `(tool_response, success_list, active)`，其中 `active=False` 表示停止生成
- `extract_tool_calls()` 负责从 LLM 输出中解析工具调用参数
- `format_tool_response()` 负责格式化工具响应以追加到序列
- 批量执行可以利用 SandboxFusion 的并发能力提升性能

下一篇文档将介绍多轮生成循环 (`04_generation_loop.md`)，详细讲解 `ToolGenerationManager.run_llm_loop()` 的实现。
