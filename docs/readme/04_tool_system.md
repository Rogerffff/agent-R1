# Tool 系统详解

本文档详细介绍 Agent-R1 的 Tool 系统设计，包括 `BaseTool` 抽象类、现有工具实现以及如何创建自定义工具。

---

## 概述

在 Agent-R1 框架中，**Tool** 是连接 Agent 与外部环境的关键接口。Tool 作为原子操作的执行器，封装了特定的功能能力，如：
- 调用外部 API
- 执行代码
- 访问数据库
- 搜索信息

当被调用时，Tool 执行其操作并返回直接的原始结果。

---

## BaseTool 抽象类

**路径**: `agent_r1/tool/base.py`

### 类定义

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from jsonschema import validate, ValidationError

class BaseTool(ABC):
    # 工具元数据属性
    name: str = ''           # 工具唯一标识符
    description: str = ''    # 工具功能描述
    parameters: dict = {}    # JSON Schema 参数定义

    def __init__(self):
        if not self.name:
            raise ValueError('Tool name must be provided')
        # 验证参数符合 OpenAI 兼容的 JSON Schema
        if not is_tool_schema({...}):
            raise ValueError('Invalid JSON schema')

    @abstractmethod
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        """执行工具的核心方法"""
        pass

    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        """批量执行（默认实现：循环调用 execute）"""
        return [self.execute(args, **kwargs) for args in args_list]

    @property
    def tool_info(self) -> Dict:
        """返回工具信息"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }

    @property
    def tool_description(self) -> Dict:
        """返回 OpenAI 函数调用格式的描述"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def validate_args(self, args: Dict) -> bool:
        """验证参数是否符合 JSON Schema"""
        try:
            validate(instance=args, schema=self.parameters)
            return True
        except ValidationError:
            return False
```

### 核心组件说明

#### 1. 工具元数据

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 工具的唯一标识符，Agent 用此名称调用工具 |
| `description` | `str` | 详细描述工具功能、使用场景和预期效果 |
| `parameters` | `dict` | 遵循 JSON Schema 规范的参数结构定义 |

#### 2. 核心方法

| 方法 | 说明 |
|------|------|
| `execute()` | **必须实现** - 工具的核心执行逻辑 |
| `batch_execute()` | 批量执行，可重写以优化批处理性能 |
| `validate_args()` | 参数验证 |

#### 3. 属性方法

| 属性 | 说明 |
|------|------|
| `tool_info` | 返回工具基本信息字典 |
| `tool_description` | 返回 OpenAI Function Calling 格式的描述 |

---

## 工具参数规范

工具参数必须遵循 JSON Schema 规范，以确保 Agent 能够正确生成参数：

```python
parameters = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "搜索查询字符串"
        },
        "limit": {
            "type": "integer",
            "description": "返回结果数量",
            "default": 5
        }
    },
    "required": ["query"]
}
```

### 参数类型支持

| JSON Schema 类型 | Python 类型 | 示例 |
|------------------|-------------|------|
| `string` | `str` | `"hello world"` |
| `integer` | `int` | `42` |
| `number` | `float` | `3.14` |
| `boolean` | `bool` | `true` |
| `array` | `list` | `[1, 2, 3]` |
| `object` | `dict` | `{"key": "value"}` |

---

## 现有工具实现

### 1. WikiSearchTool - Wikipedia 搜索工具

**路径**: `agent_r1/tool/tools/wiki_search_tool.py`

**功能**: 通过 Wikipedia 知识库搜索信息

```python
class WikiSearchTool(BaseTool):
    name = "search"
    description = "Search for information on the internet using Wikipedia as a knowledge source."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Number of results to return", "default": 5}
        },
        "required": ["query"]
    }

    def __init__(self):
        super().__init__()
        # 从环境变量获取 API URL
        self.api_url = os.environ.get("WIKI_SEARCH_API_URL", "http://localhost:8000")

    def execute(self, args: Dict) -> Dict[str, Any]:
        query = args.get("query", "").strip()
        limit = args.get("limit", 5)

        response = requests.get(
            f"{self.api_url}/search",
            params={"query": query, "top_k": limit}
        )

        if response.status_code == 200:
            result = response.json()
            formatted_result = self._format_results(result)
            return {"content": formatted_result, "success": True}
        else:
            return {"content": error_msg, "success": False}

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """优化的批量搜索实现"""
        queries = [args.get("query", "").strip() for args in args_list]

        response = requests.post(
            f"{self.api_url}/search",
            json={"queries": queries, "top_k": max_limit}
        )
        # ...
```

**使用示例**:

```json
{
    "name": "search",
    "arguments": {
        "query": "Who was the first president of the United States?"
    }
}
```

### 2. SearchTool - 通用搜索工具

**路径**: `agent_r1/tool/tools/search_tool.py`

**功能**: 基于本地 FAISS 索引的搜索工具

```python
class SearchTool(BaseTool):
    name = "search"
    description = "Search for relevant information from the knowledge base."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
```

### 3. PythonTool - Python 代码执行工具

**路径**: `agent_r1/tool/tools/python_tool.py`

**功能**: 在沙箱环境中执行 Python 代码

```python
class PythonTool(BaseTool):
    name = "python"
    description = "Execute Python code and return the result."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
        },
        "required": ["code"]
    }

    def execute(self, args: Dict) -> Dict[str, Any]:
        code = args.get("code", "")
        # 在沙箱中执行代码
        result = self._execute_in_sandbox(code)
        return {"content": result, "success": True}
```

---

## 工具注册

工具通过工厂函数注册和获取：

**路径**: `agent_r1/tool/tools/__init__.py`

```python
def _default_tool(name):
    """工具工厂函数"""
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

在配置中使用工具名称：

```yaml
tool:
  tools:
    - search        # 使用 SearchTool
    - wiki_search   # 使用 WikiSearchTool
    - python        # 使用 PythonTool
```

---

## 创建自定义工具

### 步骤 1: 创建工具类

在 `agent_r1/tool/tools/` 目录下创建新文件，例如 `calculator_tool.py`：

```python
from typing import Dict, Any
from agent_r1.tool.base import BaseTool

class CalculatorTool(BaseTool):
    # 定义工具元数据
    name = "calculator"
    description = "Perform mathematical calculations. Supports basic arithmetic operations."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate, e.g., '2 + 3 * 4'"
            }
        },
        "required": ["expression"]
    }

    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        """
        执行数学计算

        Args:
            args: 包含 "expression" 键的字典

        Returns:
            {"content": 计算结果, "success": bool}
        """
        expression = args.get("expression", "")

        try:
            # 安全地评估数学表达式
            # 注意：实际应用中应使用更安全的解析器
            result = eval(expression, {"__builtins__": {}}, {})
            return {
                "content": f"Result: {result}",
                "success": True
            }
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "success": False
            }
```

### 步骤 2: 注册工具

修改 `agent_r1/tool/tools/__init__.py`：

```python
def _default_tool(name):
    if name == "search":
        from agent_r1.tool.tools.search_tool import SearchTool
        return SearchTool()
    elif name == "wiki_search":
        from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
        return WikiSearchTool()
    elif name == "python":
        from agent_r1.tool.tools.python_tool import PythonTool
        return PythonTool()
    # 添加新工具
    elif name == "calculator":
        from agent_r1.tool.tools.calculator_tool import CalculatorTool
        return CalculatorTool()
    else:
        raise NotImplementedError(f"Tool {name} not implemented")
```

### 步骤 3: 在配置中使用

```yaml
tool:
  tools:
    - search
    - calculator  # 新添加的工具
```

---

## 工具调用格式

Agent-R1 支持 OpenAI 兼容的函数调用格式：

### 工具描述格式

```json
{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information...",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
}
```

### 工具调用格式

```xml
<tool_call>
{"name": "search", "arguments": {"query": "example query"}}
</tool_call>
```

### 工具响应格式

```xml
<tool_response>
{"results": [{"title": "...", "content": "..."}]}
</tool_response>
```

---

## 批量执行优化

对于需要处理大量请求的工具，可以重写 `batch_execute` 方法以优化性能：

```python
class OptimizedSearchTool(BaseTool):
    # ...

    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        """
        优化的批量执行：一次 API 调用处理所有查询
        """
        queries = [args.get("query", "") for args in args_list]

        # 单次批量 API 调用
        response = requests.post(
            f"{self.api_url}/batch_search",
            json={"queries": queries}
        )

        if response.status_code == 200:
            results = response.json()["results"]
            return [
                {"content": r, "success": True}
                for r in results
            ]
        else:
            return [
                {"content": "Error", "success": False}
                for _ in queries
            ]
```

---

## 工具执行结果格式

所有工具的 `execute` 方法应返回统一格式的字典：

```python
{
    "content": str,    # 工具执行结果的文本内容
    "success": bool    # 执行是否成功
}
```

### 成功示例

```python
{
    "content": '{"results": [{"title": "George Washington", "content": "..."}]}',
    "success": True
}
```

### 失败示例

```python
{
    "content": "Error: Connection timeout",
    "success": False
}
```

---

## 工具设计最佳实践

### 1. 明确的功能边界

每个工具应该有单一、明确的职责：

```python
# 好的设计：单一职责
class WikiSearchTool(BaseTool):
    name = "wiki_search"
    description = "Search Wikipedia for factual information."

# 避免：职责过于宽泛
class SuperTool(BaseTool):
    name = "super_tool"
    description = "Search, calculate, and execute code."
```

### 2. 详细的参数描述

参数描述应该足够详细，帮助 Agent 理解如何使用：

```python
parameters = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query. Be specific and include relevant keywords. "
                          "Example: 'capital city of France' rather than 'France'."
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return. Default is 5, max is 20.",
            "default": 5,
            "minimum": 1,
            "maximum": 20
        }
    },
    "required": ["query"]
}
```

### 3. 健壮的错误处理

```python
def execute(self, args: Dict) -> Dict[str, Any]:
    try:
        # 验证参数
        if not self.validate_args(args):
            return {"content": "Invalid parameters", "success": False}

        # 执行操作
        result = self._do_operation(args)
        return {"content": result, "success": True}

    except ConnectionError as e:
        return {"content": f"Connection error: {e}", "success": False}
    except TimeoutError as e:
        return {"content": f"Timeout: {e}", "success": False}
    except Exception as e:
        return {"content": f"Unexpected error: {e}", "success": False}
```

### 4. 有意义的返回内容

返回内容应该结构化且易于 Agent 解析：

```python
def _format_results(self, raw_results):
    """格式化搜索结果，使其易于 Agent 理解"""
    formatted = []
    for r in raw_results:
        formatted.append({
            "title": r.get("title", ""),
            "content": r.get("content", "")[:500],  # 限制长度
            "url": r.get("url", "")
        })
    return json.dumps({"results": formatted}, ensure_ascii=False)
```

---

## 下一步

- [05_environment_system.md](./05_environment_system.md) - 了解 ToolEnv 如何编排工具调用
- [07_reward_system.md](./07_reward_system.md) - 了解如何为工具调用设计奖励
