# verl 框架详解

本文档详细介绍 Agent-R1 使用的 verl 分布式训练框架的核心组件，包括 DataProto 数据协议、Ray 分布式架构和 Worker 系统。

## 目录

1. [verl 框架概述](#1-verl-框架概述)
2. [DataProto 数据协议](#2-dataproto-数据协议)
3. [DataProto 核心方法详解](#3-dataproto-核心方法详解)
4. [Ray 分布式架构](#4-ray-分布式架构)
5. [Worker 类型详解](#5-worker-类型详解)
6. [数据分发与收集](#6-数据分发与收集)
7. [实战示例](#7-实战示例)

---

## 1. verl 框架概述

### 1.1 什么是 verl？

**verl** (Volcano Engine Reinforcement Learning) 是字节跳动开发的分布式强化学习训练框架，为 Agent-R1 提供了以下核心功能：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           verl 框架核心功能                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐ │
│  │   DataProto         │    │   Ray 分布式        │    │   Worker 系统   │ │
│  │   数据协议          │    │   资源管理           │    │   分布式计算    │ │
│  │                     │    │                     │    │                 │ │
│  │ • 统一数据格式      │    │ • GPU 资源池        │    │ • ActorRollout │ │
│  │ • 张量/非张量分离   │    │ • 远程调用          │    │ • Critic       │ │
│  │ • 批处理操作        │    │ • 共置 Worker       │    │ • RefPolicy    │ │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 在 Agent-R1 中的位置

```
Agent-R1
├── agent_r1/                    # Agent-R1 核心代码
│   ├── src/
│   │   ├── agent_ray_trainer.py # 使用 verl 的训练器
│   │   └── fsdp_workers.py      # 继承 verl Worker
│   └── llm_agent/
│       └── generation.py        # 使用 DataProto
│
└── verl/                        # verl 框架 (git submodule)
    └── verl/
        ├── protocol.py          # DataProto 定义
        ├── single_controller/   # Ray 分布式控制
        │   └── ray/
        │       ├── base.py      # RayWorkerGroup
        │       └── decorator.py # 远程调用装饰器
        └── workers/             # Worker 实现
            └── fsdp_workers.py  # FSDP Worker 基类
```

---

## 2. DataProto 数据协议

### 2.1 DataProto 结构

**文件**: `verl/verl/protocol.py`，第 199-210 行

```python
@dataclass
class DataProto:
    """
    verl 框架的核心数据传输协议。

    用于在 Driver 和 Worker 之间传递数据，支持张量和非张量数据的统一管理。
    """
    batch: TensorDict = None                              # 张量数据字典
    non_tensor_batch: Dict = field(default_factory=dict)  # 非张量数据字典
    meta_info: Dict = field(default_factory=dict)         # 元数据字典
```

### 2.2 三个核心组件

| 组件 | 类型 | 用途 | 示例 |
|------|------|------|------|
| `batch` | `TensorDict` | 存储 PyTorch 张量，有统一的 batch 维度 | `{"input_ids": tensor[B, L], "attention_mask": tensor[B, L]}` |
| `non_tensor_batch` | `Dict[str, np.ndarray]` | 存储非张量数据（Python 对象） | `{"reward_model": array[{"ground_truth": "42"}, ...]}` |
| `meta_info` | `Dict` | 存储整个批次共享的元数据 | `{"eos_token_id": 2, "pad_token_id": 0}` |

### 2.3 具体数据示例

**训练过程中的 DataProto 内容：**

```python
# 初始状态（从 DataLoader 加载后）
data_proto = DataProto(
    batch=TensorDict({
        "input_ids": tensor([
            [0, 0, 0, 101, 2345, 6789, 102],  # 样本 1，左填充
            [0, 101, 2345, 6789, 1234, 5678, 102],  # 样本 2
        ]),  # shape: (2, 7)

        "attention_mask": tensor([
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
        ]),  # shape: (2, 7)

        "position_ids": tensor([
            [0, 0, 0, 0, 1, 2, 3],
            [0, 0, 1, 2, 3, 4, 5],
        ]),  # shape: (2, 7)
    }, batch_size=(2,)),

    non_tensor_batch={
        "raw_prompt_ids": np.array([
            [101, 2345, 6789, 102],      # 无填充
            [101, 2345, 6789, 1234, 5678, 102],
        ], dtype=object),

        "reward_model": np.array([
            {"ground_truth": "42", "style": "rule"},
            {"ground_truth": "Paris", "style": "rule"},
        ], dtype=object),

        "data_source": np.array([
            "haizhongzheng/DAPO-Math-17K-cleaned",
            "haizhongzheng/DAPO-Math-17K-cleaned",
        ], dtype=object),
    },

    meta_info={
        "eos_token_id": 102,
        "pad_token_id": 0,
    }
)

# 生成后新增字段
data_proto.batch["responses"] = tensor([
    [201, 202, 203, 204, 205, 0, 0],
    [301, 302, 303, 0, 0, 0, 0],
])  # shape: (2, 7)

data_proto.batch["action_mask"] = tensor([
    [1, 1, 1, 0, 0, 0, 0],  # 工具响应位置为 0
    [1, 1, 1, 0, 0, 0, 0],
])  # shape: (2, 7)
```

### 2.4 DataProtoItem 结构

当通过整数索引访问 DataProto 时，返回 DataProtoItem：

**文件**: `verl/verl/protocol.py`，第 191-196 行

```python
@dataclass
class DataProtoItem:
    """单个样本的数据容器"""
    batch: TensorDict = None          # 单样本的张量（无 batch 维度）
    non_tensor_batch: Dict = field(default_factory=dict)  # 单样本的非张量数据
    meta_info: Dict = field(default_factory=dict)         # 共享的元数据
```

```python
# 使用示例
item = data_proto[0]  # 返回 DataProtoItem

print(item.batch["input_ids"].shape)  # torch.Size([7]) - 无 batch 维度
print(item.non_tensor_batch["reward_model"])  # {"ground_truth": "42", "style": "rule"}
```

---

## 3. DataProto 核心方法详解

### 3.1 创建方法

#### `from_dict()` - 从字典创建

**文件**: `verl/verl/protocol.py`，第 343-378 行

```python
@classmethod
def from_dict(cls,
              tensors: Dict[str, torch.Tensor],
              non_tensors=None,
              meta_info=None,
              num_batch_dims=1):
    """
    从张量字典创建 DataProto。

    参数:
        tensors: 张量字典，所有张量必须有相同的 batch size
        non_tensors: 非张量字典（可选）
        meta_info: 元数据字典（可选）
        num_batch_dims: batch 维度数（默认 1）
    """
```

**使用示例**:

```python
# 从张量字典创建
obs = torch.randn(100, 10)
act = torch.randn(100, 3)
data = DataProto.from_dict(tensors={"obs": obs, "act": act})

# 包含非张量数据
labels = np.array(["a", "b", "c"] * 33 + ["d"], dtype=object)
data = DataProto.from_dict(
    tensors={"obs": obs},
    non_tensors={"labels": labels}
)
```

#### `from_single_dict()` - 从混合字典创建

**文件**: `verl/verl/protocol.py`，第 327-341 行

```python
@classmethod
def from_single_dict(cls,
                    data: Dict[str, Union[torch.Tensor, np.ndarray]],
                    meta_info=None):
    """
    从混合字典创建 DataProto。

    自动将 torch.Tensor 归类到 batch，其他归类到 non_tensor_batch。
    """
```

**Agent-R1 中的使用**（`agent_ray_trainer.py` 第 1959 行）:

```python
# 从 DataLoader 获取的 batch_dict 创建 DataProto
batch: DataProto = DataProto.from_single_dict(batch_dict)
```

### 3.2 索引方法

#### `__getitem__()` - 灵活索引

**文件**: `verl/verl/protocol.py`，第 225-257 行

支持多种索引方式：

| 索引类型 | 返回类型 | 示例 |
|---------|---------|------|
| `int` | `DataProtoItem` | `data[0]` → 单个样本 |
| `slice` | `DataProto` | `data[10:20]` → 切片 |
| `list[int]` | `DataProto` | `data[[1, 3, 5]]` → 选择索引 |
| `np.ndarray` | `DataProto` | `data[np.array([1, 3, 5])]` |
| `torch.Tensor` | `DataProto` | `data[torch.tensor([1, 3, 5])]` |
| `bool array` | `DataProto` | `data[mask]` → 布尔索引 |

```python
# 示例
data = DataProto.from_dict({"obs": torch.randn(100, 10)})

# 切片
subset = data[10:20]  # 返回 DataProto，batch_size=10

# 列表索引
selected = data[[0, 5, 10, 15]]  # 返回 DataProto，batch_size=4

# 单元素索引
item = data[0]  # 返回 DataProtoItem

# 布尔索引
mask = torch.tensor([True, False] * 50)
filtered = data[mask]  # 返回 DataProto，batch_size=50
```

#### `__len__()` - 获取批次大小

**文件**: `verl/verl/protocol.py`，第 216-223 行

```python
def __len__(self):
    if self.batch is not None:
        return self.batch.batch_size[0]  # 优先从 batch 获取
    elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
        random_key = list(self.non_tensor_batch.keys())[0]
        return self.non_tensor_batch[random_key].shape[0]  # 回退到 non_tensor_batch
    else:
        return 0
```

### 3.3 操作方法

#### `select()` - 选择特定字段

**文件**: `verl/verl/protocol.py`，第 394-427 行

```python
def select(self,
          batch_keys=None,
          non_tensor_batch_keys=None,
          meta_info_keys=None) -> "DataProto":
    """
    选择特定的字段，返回新的 DataProto。
    不修改原始数据。
    """
```

**Agent-R1 中的使用**（`agent_dp_actor.py`）:

```python
# 只选择 Actor 更新需要的字段
select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "loss_mask"]
batch = data.select(batch_keys=select_keys).batch
```

#### `pop()` - 提取并移除字段

**文件**: `verl/verl/protocol.py`，第 511-541 行

```python
def pop(self,
       batch_keys=None,
       non_tensor_batch_keys=None,
       meta_info_keys=None) -> "DataProto":
    """
    从原始 DataProto 中提取并移除指定字段。

    ⚠️ 重要：此方法会修改原始 DataProto！
    """
```

**Agent-R1 中的使用**（`agent_ray_trainer.py` 第 1984-1987 行）:

```python
# 分离生成所需的字段
batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

gen_batch = batch.pop(
    batch_keys=batch_keys_to_pop,
    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
)
# 现在 batch 中已经没有这些字段了
# gen_batch 包含这些字段
```

#### `union()` - 合并两个 DataProto

**文件**: `verl/verl/protocol.py`，第 568-585 行

```python
def union(self, other: "DataProto") -> "DataProto":
    """
    合并另一个 DataProto 的数据到当前对象。

    ⚠️ 重要：此方法会修改原始 DataProto！

    规则：
    - batch_size 必须相同
    - 相同 key 的值必须完全相等，否则抛出 AssertionError
    """
```

**Agent-R1 中的使用**（`agent_ray_trainer.py` 第 2029 行）:

```python
# 合并生成输出到批次
batch = batch.union(gen_batch_output)
# 现在 batch 包含了 gen_batch_output 的所有字段
```

#### `repeat()` - 重复数据

**文件**: `verl/verl/protocol.py`，第 713-750 行

```python
def repeat(self, repeat_times=2, interleave=True) -> "DataProto":
    """
    重复数据。

    参数:
        repeat_times: 重复次数
        interleave:
            True  → [a, b, c] → [a, a, b, b, c, c]（交错）
            False → [a, b, c] → [a, b, c, a, b, c]（连续）
    """
```

**Agent-R1 中的使用**（`agent_ray_trainer.py` 第 1968-1971 行）:

```python
# 每个 prompt 重复生成 n_repeat 个响应
batch = batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n_repeat,  # 例如 5
    interleave=True,  # 交错排列
)
# 原来 batch_size=128，现在变成 128*5=640
```

**数据变化示例**:

```
原始数据（batch_size=3）:
  input_ids: [[prompt_1], [prompt_2], [prompt_3]]
  uid: ["uid_1", "uid_2", "uid_3"]

repeat(repeat_times=2, interleave=True) 后（batch_size=6）:
  input_ids: [[prompt_1], [prompt_1], [prompt_2], [prompt_2], [prompt_3], [prompt_3]]
  uid: ["uid_1", "uid_1", "uid_2", "uid_2", "uid_3", "uid_3"]

用途：GRPO 算法需要同一 prompt 的多个响应来计算组内相对优势
```

#### `reorder()` - 重排序数据

**文件**: `verl/verl/protocol.py`，第 705-711 行

```python
def reorder(self, indices):
    """
    按给定索引重排序数据。

    ⚠️ 重要：此方法会原地修改数据！
    """
    indices_np = indices.detach().numpy()
    self.batch = self.batch[indices]
    self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}
```

**Agent-R1 中的使用**（`agent_ray_trainer.py` 第 1709 行）:

```python
# 序列长度平衡：重排序使各 GPU 获得相似的总 token 数
global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
batch.reorder(global_idx)
```

### 3.4 批处理方法

#### `chunk()` - 分割为多个块

**文件**: `verl/verl/protocol.py`，第 646-681 行

```python
def chunk(self, chunks: int) -> List["DataProto"]:
    """
    将 DataProto 沿 batch 维度分割为多个块。

    用于分布式训练中的数据分发。
    """
```

**使用示例**:

```python
data = DataProto.from_dict({"obs": torch.randn(100, 10)})

# 分成 4 块，每块 25 个样本
chunks = data.chunk(4)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: batch_size = {len(chunk)}")  # 25
```

#### `concat()` - 合并多个 DataProto

**文件**: `verl/verl/protocol.py`，第 683-703 行

```python
@staticmethod
def concat(data: List["DataProto"]) -> "DataProto":
    """
    将多个 DataProto 沿 batch 维度合并。

    用于收集分布式计算结果。
    """
```

**使用示例**:

```python
# chunk 的逆操作
chunks = data.chunk(4)
merged = DataProto.concat(chunks)
assert len(merged) == len(data)  # 100
```

---

## 4. Ray 分布式架构

### 4.1 核心组件

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Ray 分布式架构组件                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ResourcePoolManager                               │   │
│  │                    (资源池管理器)                                      │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                        │   │
│  │  │  RayResourcePool │    │  RayResourcePool │                        │   │
│  │  │  "global_pool"   │    │  "critic_pool"   │                        │   │
│  │  │  [8, 8] GPUs     │    │  [4, 4] GPUs     │                        │   │
│  │  └─────────────────┘    └─────────────────┘                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              │ create_colocated_worker_cls()               │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     RayWorkerGroup                                   │   │
│  │                     (Worker 组管理)                                   │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                        │   │
│  │  │  Worker[0]       │    │  Worker[1]       │    ...                 │   │
│  │  │  GPU 0           │    │  GPU 1           │                        │   │
│  │  │                  │    │                  │                        │   │
│  │  │ ActorRollout    │    │ ActorRollout    │                        │   │
│  │  │ Critic          │    │ Critic          │                        │   │
│  │  │ RefPolicy       │    │ RefPolicy       │                        │   │
│  │  └─────────────────┘    └─────────────────┘                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 ResourcePoolManager

**文件**: `agent_r1/src/agent_ray_trainer.py`，第 313-426 行

```python
@dataclass
class ResourcePoolManager:
    """
    GPU 资源池管理器。

    属性:
        resource_pool_spec: 资源池规格
            格式: {pool_name: [gpus_per_node] * num_nodes}
            示例: {"global_pool": [8, 8]} → 2 节点，每节点 8 GPU

        mapping: 角色到资源池的映射
            格式: {Role: pool_name}
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
```

**Agent-R1 中的配置**:

```python
# agent_ray_trainer.py 第 366-375 行
global_pool_id = "global_pool"
resource_pool_spec = {
    global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    # 例如: {"global_pool": [8, 8]} 表示 2 个节点，每节点 8 个 GPU
}

mapping = {
    Role.ActorRollout: global_pool_id,
    Role.Critic: global_pool_id,  # 所有角色共享同一资源池
}
```

### 4.3 RayClassWithInitArgs

**文件**: `verl/verl/single_controller/ray/base.py`

```python
class RayClassWithInitArgs:
    """
    包装 Ray 远程类及其初始化参数。

    用于延迟创建 Worker 实例，直到分配了 GPU 资源。
    """
    def __init__(self, cls, config, role=None, **kwargs):
        self.cls = cls
        self.config = config
        self.role = role
        self.kwargs = kwargs
```

**Agent-R1 中的使用**（`agent_ray_trainer.py` 第 1406-1414 行）:

```python
# 创建 Actor + Rollout Worker 类
actor_rollout_cls = RayClassWithInitArgs(
    cls=self.role_worker_mapping[Role.ActorRollout],  # ActorRolloutRefWorker
    config=self.config.actor_rollout_ref,
    role="actor_rollout",
)

self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
```

### 4.4 create_colocated_worker_cls

**文件**: `verl/verl/single_controller/ray/base.py`

```python
def create_colocated_worker_cls(class_dict: Dict[str, RayClassWithInitArgs]):
    """
    创建共置 Worker 类。

    将多个 Worker 组合到同一个 GPU 上运行，共享内存和计算资源。

    参数:
        class_dict: {"actor_rollout": cls1, "critic": cls2, ...}

    返回:
        组合后的 Worker 类
    """
```

**为什么需要共置 Worker？**

1. **内存效率**: 多个 Worker 共享 GPU 内存
2. **通信效率**: 同 GPU 的 Worker 间无需数据传输
3. **简化管理**: 一个进程管理多个角色

### 4.5 RayWorkerGroup

**文件**: `verl/verl/single_controller/ray/__init__.py`

```python
class RayWorkerGroup:
    """
    Ray Worker 组管理器。

    管理一组分布在多个 GPU 上的 Worker 实例。
    """

    def __init__(self, resource_pool, ray_cls_with_init):
        self.resource_pool = resource_pool
        self.ray_cls_with_init = ray_cls_with_init
        self.workers = []

    @property
    def world_size(self):
        """返回 Worker 数量（等于 GPU 数量）"""
        return len(self.workers)

    def spawn(self, prefix_set):
        """
        创建 Worker 实例。

        返回: {prefix: WorkerGroup} 字典
        """
        pass
```

---

## 5. Worker 类型详解

### 5.1 Worker 角色枚举

**文件**: `agent_r1/src/agent_ray_trainer.py`，第 228-258 行

```python
class Role(Enum):
    """
    训练角色枚举。

    每个角色对应不同的模型或功能。
    """
    Actor = 0           # 策略模型（训练用）
    Rollout = 1         # 生成器（推理用）
    ActorRollout = 2    # Actor + Rollout 混合（混合引擎模式）
    Critic = 3          # 值函数模型（仅 GAE 需要）
    RefPolicy = 4       # 参考策略（计算 KL 散度）
    RewardModel = 5     # 奖励模型
    ActorRolloutRef = 6 # Actor + Rollout + RefPolicy
```

### 5.2 ActorRolloutRefWorker

**文件**: `agent_r1/src/fsdp_workers.py`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ActorRolloutRefWorker                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           核心组件                                    │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │   Actor Model   │  │  Rollout Engine │  │   RefPolicy Model   │  │   │
│  │  │   (FSDP 分片)   │  │    (vLLM)       │  │   (FSDP 分片)       │  │   │
│  │  │                 │  │                 │  │                     │  │   │
│  │  │ • forward()     │  │ • generate()    │  │ • compute_log_prob()│  │   │
│  │  │ • backward()    │  │ • sample()      │  │                     │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           核心方法                                    │   │
│  │                                                                      │   │
│  │  • init_model()           初始化模型权重                              │   │
│  │  • generate_sequences()   生成序列（调用 vLLM）                       │   │
│  │  • compute_log_prob()     计算当前策略对数概率                         │   │
│  │  • compute_ref_log_prob() 计算参考策略对数概率                         │   │
│  │  • update_actor()         更新策略网络                                │   │
│  │  • save_checkpoint()      保存检查点                                  │   │
│  │  • load_checkpoint()      加载检查点                                  │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**核心方法签名**:

```python
class ActorRolloutRefWorker:
    def init_model(self):
        """初始化模型，加载预训练权重"""
        pass

    def generate_sequences(self, data: DataProto) -> DataProto:
        """
        使用 vLLM 生成序列。

        输入:
            data.batch: {"input_ids", "attention_mask", "position_ids"}
            data.non_tensor_batch: {"raw_prompt_ids"}
            data.meta_info: {"eos_token_id", "pad_token_id", "do_sample", ...}

        输出:
            DataProto with {"responses", "log_probs", ...}
        """
        pass

    def compute_log_prob(self, data: DataProto) -> DataProto:
        """
        计算当前策略下生成序列的对数概率。

        输入:
            data.batch: {"input_ids", "responses", "attention_mask", "action_mask"}

        输出:
            DataProto with {"old_log_probs", "entropys"}
        """
        pass

    def update_actor(self, data: DataProto) -> DataProto:
        """
        使用 PPO/GRPO 损失更新策略网络。

        输入:
            data.batch: {"input_ids", "responses", "old_log_probs", "advantages", ...}

        输出:
            DataProto with meta_info["metrics"] containing loss values
        """
        pass
```

### 5.3 CriticWorker

**文件**: `verl/verl/workers/fsdp_workers.py`

```python
class CriticWorker:
    """
    值函数模型 Worker。

    仅在使用 GAE 优势估计时需要。
    GRPO、REINFORCE++ 等算法不需要 Critic。
    """

    def init_model(self):
        """初始化 Critic 模型"""
        pass

    def compute_values(self, data: DataProto) -> DataProto:
        """
        计算状态值函数 V(s)。

        输入:
            data.batch: {"input_ids", "attention_mask", "responses"}

        输出:
            DataProto with {"values": tensor[B, R]}
        """
        pass

    def update_critic(self, data: DataProto) -> DataProto:
        """
        使用 MSE 损失更新 Critic。

        损失: L = E[(V(s) - returns)^2]

        输入:
            data.batch: {"values", "returns"}

        输出:
            DataProto with meta_info["metrics"]
        """
        pass
```

### 5.4 Worker 初始化顺序

**文件**: `agent_r1/src/agent_ray_trainer.py`，第 1480-1498 行

```python
# 按顺序初始化模型：Critic -> RefPolicy -> ActorRollout
# 原因：ActorRollout 包含 vLLM，最后初始化让它能正确估计可用 GPU 内存

if self.use_critic:
    self.critic_wg = all_wg["critic"]
    self.critic_wg.init_model()  # ① 先初始化 Critic

if self.use_reference_policy:
    self.ref_policy_wg = all_wg["ref"]
    self.ref_policy_wg.init_model()  # ② 再初始化 RefPolicy

# ActorRollout 最后初始化
self.actor_rollout_wg = all_wg["actor_rollout"]
self.actor_rollout_wg.init_model()  # ③ 最后初始化 ActorRollout
```

---

## 6. 数据分发与收集

### 6.1 数据填充

**文件**: `verl/verl/protocol.py`，第 69-94 行

```python
def pad_dataproto_to_divisor(data: DataProto, size_divisor: int):
    """
    将 DataProto 填充到能被 size_divisor 整除的大小。

    用于分布式训练中确保数据能均匀分配到各 GPU。

    参数:
        data: 原始 DataProto
        size_divisor: 除数（通常是 world_size）

    返回:
        data_padded: 填充后的 DataProto
        pad_size: 填充的样本数

    填充策略: 循环使用现有数据
    """
```

**使用示例**:

```python
# 原始数据: batch_size = 100
# world_size = 8
# 100 % 8 = 4，需要填充 4 个样本

data_padded, pad_size = pad_dataproto_to_divisor(data, 8)
# data_padded.batch_size = 104
# pad_size = 4
```

### 6.2 数据去填充

**文件**: `verl/verl/protocol.py`，第 97-100 行

```python
def unpad_dataproto(data: DataProto, pad_size):
    """
    移除填充的数据。

    是 pad_dataproto_to_divisor 的逆操作。
    """
    if pad_size != 0:
        data = data[:-pad_size]  # 切掉最后 pad_size 个样本
    return data
```

### 6.3 序列长度平衡

**文件**: `verl/verl/utils/seqlen_balancing.py`

```python
def get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size=True):
    """
    平衡各分区的序列总长度。

    目标: 使每个 GPU 处理的总 token 数尽量接近。

    参数:
        seqlen_list: 每个样本的序列长度列表
        k_partitions: 分区数（通常是 GPU 数）
        equal_size: 是否要求各分区样本数相等

    返回:
        partitions: 每个分区的样本索引列表
    """
```

**Agent-R1 中的使用**（`agent_ray_trainer.py` 第 1694-1715 行）:

```python
def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
    """重排序数据，使各 GPU 获得相似的总 token 数"""
    attention_mask = batch.batch["attention_mask"]
    batch_size = attention_mask.shape[0]

    # 计算每个样本的有效 token 数
    global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()

    world_size = self.actor_rollout_wg.world_size

    # 获取平衡后的分区
    global_partition_lst = get_seqlen_balanced_partitions(
        global_seqlen_lst, k_partitions=world_size, equal_size=True
    )

    # 按新顺序重排
    global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
    batch.reorder(global_idx)
```

---

## 7. 实战示例

### 7.1 完整的数据流示例

```python
# 步骤 1: 从 DataLoader 获取数据
batch_dict = next(iter(train_dataloader))
# batch_dict = {
#     "input_ids": tensor[128, 1024],
#     "attention_mask": tensor[128, 1024],
#     "position_ids": tensor[128, 1024],
#     "raw_prompt_ids": list[128],
#     "reward_model": list[128],
#     "data_source": list[128],
# }

# 步骤 2: 转换为 DataProto
batch = DataProto.from_single_dict(batch_dict)
# batch.batch = TensorDict with tensors
# batch.non_tensor_batch = dict with numpy arrays

# 步骤 3: 分配唯一 ID（用于 GRPO 分组）
batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(batch))], dtype=object
)

# 步骤 4: 重复数据（每个 prompt 生成 5 个响应）
batch = batch.repeat(repeat_times=5, interleave=True)
# batch_size: 128 -> 640

# 步骤 5: 分离生成所需字段
gen_batch = batch.pop(
    batch_keys=["input_ids", "attention_mask", "position_ids"],
    non_tensor_batch_keys=["raw_prompt_ids"],
)
# gen_batch 包含生成所需字段
# batch 现在只包含其他字段（reward_model, data_source, uid）

# 步骤 6: 填充到 GPU 数量的倍数
gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, world_size=8)
# batch_size: 640 -> 640 (已经是 8 的倍数)

# 步骤 7: 分布式生成
gen_output = actor_rollout_wg.generate_sequences(gen_batch_padded)
# gen_output.batch = {"responses", "log_probs", ...}

# 步骤 8: 去填充
gen_output = unpad_dataproto(gen_output, pad_size)

# 步骤 9: 合并回主批次
batch = batch.union(gen_output)
# batch 现在包含所有字段

# 步骤 10: 计算奖励
reward_tensor, _ = compute_reward(batch, reward_fn)
batch.batch["token_level_scores"] = reward_tensor

# 步骤 11: 计算对数概率
old_log_prob = actor_rollout_wg.compute_log_prob(batch)
batch = batch.union(old_log_prob)

# 步骤 12: 计算优势
batch = compute_advantage(batch, adv_estimator="grpo")
# batch.batch 新增 {"advantages", "returns"}

# 步骤 13: 更新模型
actor_output = actor_rollout_wg.update_actor(batch)
```

### 7.2 DataProto 字段追踪表

| 阶段 | batch 字段 | non_tensor_batch 字段 |
|------|-----------|----------------------|
| **DataLoader 输出** | `input_ids`, `attention_mask`, `position_ids` | `raw_prompt_ids`, `reward_model`, `data_source` |
| **添加 uid** | - | + `uid` |
| **repeat** | 形状: B → B*5 | 形状: B → B*5 |
| **pop 生成字段** | - `input_ids`, `attention_mask`, `position_ids` | - `raw_prompt_ids` |
| **生成后** | + `responses`, `prompts`, `action_mask`, `turns` | + `action_mask` (详细版) |
| **union 合并** | 所有字段合并 | 所有字段合并 |
| **奖励计算** | + `token_level_scores` | - |
| **log_prob** | + `old_log_probs`, + `entropys` | - |
| **ref_log_prob** | + `ref_log_prob` | - |
| **values (GAE)** | + `values` | - |
| **advantage** | + `advantages`, + `returns`, + `token_level_rewards` | - |

---

## 下一步

继续阅读 [02_data_preparation.md](./02_data_preparation.md) 了解数据准备与加载的详细流程。
