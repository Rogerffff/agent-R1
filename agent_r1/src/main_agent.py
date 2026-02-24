# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
==============================================================================
Agent-R1 主入口模块 (Main Entry Point)
==============================================================================

这是 Agent-R1 框架的主入口文件，负责整个训练流程的初始化和启动。

主要功能：
1. 初始化 Ray 分布式计算集群
2. 加载模型、tokenizer 和处理器
3. 配置工具(Tools)和环境(Environment)
4. 设置奖励函数(Reward Function)
5. 创建数据集和数据采样器
6. 实例化并启动训练器(RayAgentTrainer)

文件结构：
- get_custom_reward_fn(): 加载用户自定义的奖励函数
- main(): Hydra 配置入口点
- run_agent(): 初始化 Ray 并启动任务
- TaskRunner: Ray 远程 Actor，执行主要的训练设置逻辑
- create_rl_dataset(): 创建强化学习数据集
- create_rl_sampler(): 创建数据采样器

注意：我们没有将 main 与 ray_trainer 合并，因为 ray_trainer 会被其他 main 使用。
"""

# ============================================================================
# 导入部分
# ============================================================================

# 从本地模块导入核心训练器
from .agent_ray_trainer import RayAgentTrainer

# 导入工具系统：_default_env 用于获取环境类，_default_tool 用于获取工具实例
from agent_r1.tool.envs import _default_env
from agent_r1.tool.tools import _default_tool

import os

# Hydra: Facebook 开发的配置管理框架，支持命令行参数覆盖和配置组合
import hydra

# Ray: 分布式计算框架，用于并行化训练过程
import ray

# 从本地模块导入奖励管理器加载函数
from .reward import load_reward_manager


# ============================================================================
# 自定义奖励函数加载器
# ============================================================================


def get_custom_reward_fn(config):
    """
    动态加载用户自定义的奖励函数。

    这个函数允许用户在不修改框架代码的情况下，通过配置文件指定自己的奖励函数。
    奖励函数是强化学习中的核心组件，决定了代理的学习目标。

    参数：
        config: Hydra 配置对象，包含以下字段：
            - custom_reward_function.path: 奖励函数所在的 Python 文件路径
            - custom_reward_function.name: 奖励函数的名称
            - custom_reward_function.reward_kwargs: 传递给奖励函数的额外参数

    返回：
        wrapped_fn: 包装后的奖励函数，如果没有配置则返回 None

    工作流程：
        1. 从配置中提取奖励函数文件路径
        2. 使用 importlib 动态加载指定的 Python 模块
        3. 从模块中获取指定名称的函数
        4. 使用 reward_kwargs 包装函数，实现参数预绑定

    使用示例（配置文件）：
        custom_reward_function:
            path: "/path/to/my_rewards.py"
            name: "my_custom_reward"
            reward_kwargs:
                weight: 0.5
    """
    import importlib.util
    import sys

    # 从配置中获取自定义奖励函数的配置，如果没有则返回空字典
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")

    # 如果没有配置路径，直接返回 None，使用默认奖励函数
    if not file_path:
        return None

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    # 使用 importlib 动态加载 Python 模块
    # spec_from_file_location: 从文件路径创建模块规格
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)

    try:
        # 将模块添加到 sys.modules 以便后续导入
        sys.modules["custom_module"] = module
        # 执行模块代码，加载其中的函数和类
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    # 获取函数名称并检查是否存在
    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(
            f"Reward function '{function_name}' not found in '{file_path}'."
        )

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    # 获取额外的奖励函数参数
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    # 创建包装函数，将 reward_kwargs 预绑定到奖励函数
    # 这样在调用时不需要每次都传递这些参数
    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


# ============================================================================
# Hydra 入口点
# ============================================================================


@hydra.main(config_path="config", config_name="agent_trainer", version_base=None)
def main(config):
    """
    Hydra 装饰的主入口函数。

    Hydra 会自动：
    1. 加载 config/agent_trainer.yaml 作为基础配置
    2. 解析命令行参数并覆盖配置
    3. 将合并后的配置传递给此函数

    参数：
        config: OmegaConf 配置对象，包含所有训练参数

    使用示例：
        # 使用默认配置运行
        python main_agent.py

        # 覆盖特定参数
        python main_agent.py trainer.n_gpus_per_node=4 algorithm.adv_estimator=grpo

        # 使用不同的配置组
        python main_agent.py +experiment=hotpotqa
    """
    run_agent(config)


# ============================================================================
# Ray 集群初始化
# ============================================================================


def run_agent(config) -> None:
    """
    初始化 Ray 分布式集群并启动训练任务。

    这个函数是训练流程的第一步，它：
    1. 检查 Ray 是否已初始化（支持外部 Ray 集群）
    2. 如果未初始化，创建本地 Ray 集群
    3. 启动 TaskRunner 远程 Actor 执行主要训练逻辑

    参数：
        config: Hydra 配置对象

    Ray 初始化参数说明：
        - TOKENIZERS_PARALLELISM: 启用 tokenizer 并行化
        - NCCL_DEBUG: NCCL（GPU 通信库）调试级别设为 WARN
        - VLLM_LOGGING_LEVEL: vLLM 日志级别设为 WARN
        - num_cpus: 可用 CPU 核心数

    设计考虑：
        使用 TaskRunner 远程 Actor 而不是直接运行训练代码，是为了确保
        主要训练任务不会被调度到 Ray head 节点上，从而避免资源竞争。
    """
    if not ray.is_initialized():
        # 本地 Ray 集群初始化
        # runtime_env 设置环境变量，控制日志级别和并行行为
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",  # 启用 HuggingFace tokenizer 并行
                    "NCCL_DEBUG": "WARN",  # 减少 NCCL 调试输出
                    "VLLM_LOGGING_LEVEL": "WARN",  # 减少 vLLM 日志输出
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    # 创建 TaskRunner 远程 Actor 实例
    # .remote() 表示这是一个远程调用，会在 Ray 集群中分配资源执行
    runner = TaskRunner.remote()

    # 启动训练任务并等待完成
    # ray.get() 会阻塞直到远程任务完成
    ray.get(runner.run.remote(config))


# ============================================================================
# 核心训练任务执行器
# ============================================================================


@ray.remote(num_cpus=1)  # 确保 TaskRunner 不会被调度到 head 节点
class TaskRunner:
    """
    Ray 远程 Actor，负责执行主要的训练设置和启动逻辑。

    为什么使用远程 Actor？
    1. 避免在 Ray head 节点上运行繁重任务
    2. 实现资源隔离，确保训练任务有独立的 CPU 资源
    3. 支持分布式环境中的任务调度

    这个类是整个训练流程的核心编排者，它协调：
    - 模型加载（Actor、Critic、Reference Policy）
    - 工具和环境配置
    - 奖励函数设置
    - 数据集准备
    - 训练器实例化
    """

    def run(self, config):
        """
        执行完整的训练初始化和启动流程。

        参数：
            config: Hydra 配置对象，包含所有训练超参数

        执行步骤：
            1. 打印配置信息
            2. 下载/加载模型检查点
            3. 初始化 tokenizer 和 processor
            4. 根据分布式策略选择 Worker 类
            5. 配置资源池和角色映射
            6. 设置奖励模型和参考策略
            7. 加载工具和环境
            8. 创建数据集和采样器
            9. 实例化训练器并开始训练
        """
        # ====================================================================
        # 步骤 1: 打印配置信息
        # ====================================================================
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        # 将 OmegaConf 配置转换为 Python 字典并打印
        # resolve=True 会解析所有符号引用（如 ${other_config.value}）
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # ====================================================================
        # 步骤 2: 下载/加载模型检查点
        # ====================================================================
        # copy_to_local 支持从 HDFS 或其他远程存储下载模型到本地
        # 如果路径已经是本地路径，则直接返回
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # ====================================================================
        # 步骤 3: 初始化 tokenizer 和 processor
        # ====================================================================
        from verl.utils import hf_processor, hf_tokenizer

        # trust_remote_code: 是否信任模型仓库中的自定义代码
        # 某些模型（如 Qwen）需要执行自定义的 Python 代码
        trust_remote_code = config.data.get("trust_remote_code", False)

        # tokenizer: 将文本转换为 token ID
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # processor: 用于多模态 LLM（如视觉-语言模型），处理图像等输入
        # 如果模型不支持多模态，processor 可能为 None
        processor = hf_processor(local_path, use_fast=True)

        # ====================================================================
        # 步骤 4: 根据分布式策略选择 Worker 类
        # ====================================================================
        # Agent-R1 支持两种分布式训练策略：
        # - FSDP (Fully Sharded Data Parallel): PyTorch 原生分布式策略
        # - Megatron: NVIDIA 的大模型并行训练框架

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            # FSDP 策略：使用 PyTorch FSDP 进行模型分片
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy

            # 导入 FSDP 工作器
            # ActorRolloutRefWorker: 处理 Actor（策略模型）、Rollout（生成）和 RefPolicy（参考策略）
            # CriticWorker: 处理 Critic（值函数模型）
            from .fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            # 根据 rollout 模式选择同步或异步 Actor
            # async 模式：生成和训练可以并行进行，提高 GPU 利用率
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # Megatron 策略：使用 NVIDIA Megatron-LM 进行模型并行
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy

            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker,
                CriticWorker,
            )

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        # ====================================================================
        # 步骤 5: 配置资源池和角色映射
        # ====================================================================
        from .agent_ray_trainer import ResourcePoolManager, Role

        # 角色到 Worker 类的映射
        # Role 枚举定义了不同的训练角色：Actor、Critic、RefPolicy 等
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(
                actor_rollout_cls
            ),  # Actor + Rollout 共用一个 Worker
            Role.Critic: ray.remote(CriticWorker),  # Critic 值函数 Worker
        }

        # 资源池配置
        # global_pool: 全局 GPU 资源池
        # resource_pool_spec: 指定每个节点的 GPU 数量
        # 例如：{"global_pool": [8, 8]} 表示 2 个节点，每个节点 8 个 GPU
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # 角色到资源池的映射
        # 所有角色共享同一个全局资源池
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # ====================================================================
        # 步骤 6: 配置奖励模型（可选）
        # ====================================================================
        # Agent-R1 支持多种奖励来源：
        # - 基于规则的奖励：直接计算奖励分数（如精确匹配、格式检查）
        # - 基于模型的奖励：使用神经网络奖励模型
        # - 代码执行奖励：在沙箱中执行代码并评估结果
        # 最终奖励是多个来源的组合，具体取决于数据的标签

        if config.reward_model.enable:
            # 如果启用了模型奖励，加载对应的 RewardModelWorker
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # ====================================================================
        # 步骤 7: 配置参考策略（用于 KL 散度）
        # ====================================================================
        # 参考策略用于计算 KL 散度，防止训练后的策略偏离初始策略太远
        # 这是 PPO 等算法中常用的正则化技术

        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # ====================================================================
        # 步骤 8: 加载奖励函数
        # ====================================================================
        # reward_fn: 训练时使用的奖励函数
        # val_reward_fn: 验证时使用的奖励函数（可能有不同的评估标准）
        # num_examine: 打印多少个样本用于检查奖励计算是否正确
        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)

        # 创建资源池管理器
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # ====================================================================
        # 步骤 9: 加载工具和环境
        # ====================================================================
        # 工具(Tools): 代理可以调用的外部功能，如搜索、代码执行等
        # 环境(Environment): 管理工具调用、状态转换和奖励计算的编排器

        # 根据配置加载指定的工具列表
        # _default_tool 是工厂函数，根据工具名称返回对应的工具实例
        tools = [_default_tool(name) for name in config.tool.tools]

        # 创建训练环境
        # _default_env 返回环境类，然后实例化并传入工具列表
        env = _default_env(config.tool.env)(
            tools=tools, max_tool_response_length=config.tool.max_tool_response_length
        )

        # 检查验证环境是否与训练环境不同
        # 有时验证时需要使用不同的工具或环境配置
        if (
            config.tool.val_kwargs.tools != config.tool.tools
            or config.tool.val_kwargs.env != config.tool.env
            or config.tool.val_kwargs.max_tool_response_length
            != config.tool.max_tool_response_length
        ):
            val_env = _default_env(config.tool.val_kwargs.env)(
                tools=config.tool.val_kwargs.tools,
                max_tool_response_length=config.tool.val_kwargs.max_tool_response_length,
            )
        else:
            val_env = None  # 使用与训练相同的环境

        # ====================================================================
        # 步骤 10: 创建数据集
        # ====================================================================
        from verl.utils.dataset.rl_dataset import collate_fn

        # 创建训练数据集
        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor, env=env
        )

        # 创建验证数据集
        if val_env is not None:
            val_dataset = create_rl_dataset(
                config.data.val_files, config.data, tokenizer, processor, env=val_env
            )
        else:
            val_dataset = create_rl_dataset(
                config.data.val_files, config.data, tokenizer, processor, env=env
            )

        # 创建训练数据采样器
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # ====================================================================
        # 步骤 11: 创建并启动训练器
        # ====================================================================
        # RayAgentTrainer 是核心训练器，负责：
        # - 管理分布式工作组
        # - 执行生成-评估-优化循环
        # - 保存检查点
        # - 记录训练指标

        trainer = RayAgentTrainer(
            config=config,  # 完整配置
            tokenizer=tokenizer,  # 文本分词器
            processor=processor,  # 多模态处理器
            role_worker_mapping=role_worker_mapping,  # 角色到 Worker 的映射
            resource_pool_manager=resource_pool_manager,  # 资源池管理器
            ray_worker_group_cls=ray_worker_group_cls,  # Ray Worker 组类
            reward_fn=reward_fn,  # 训练奖励函数
            val_reward_fn=val_reward_fn,  # 验证奖励函数
            train_dataset=train_dataset,  # 训练数据集
            val_dataset=val_dataset,  # 验证数据集
            collate_fn=collate_fn,  # 数据批处理函数
            train_sampler=train_sampler,  # 训练数据采样器
            env=env,  # 训练环境
            val_env=val_env,  # 验证环境
        )

        # 初始化分布式工作组
        trainer.init_workers()

        # 开始训练循环
        # fit() 方法包含完整的 RL 训练循环：
        # 生成(Rollout) -> 奖励计算 -> 优势估计 -> 策略更新 -> 值函数更新
        trainer.fit()


# ============================================================================
# 数据集创建函数
# ============================================================================


def create_rl_dataset(data_paths, data_config, tokenizer, processor, env=None):
    """
    创建强化学习数据集。

    这个函数负责加载和预处理训练/验证数据，将原始数据转换为
    可供模型使用的格式。

    参数：
        data_paths: 数据文件路径（支持 parquet、jsonl 等格式）
        data_config: 数据配置，包含：
            - max_prompt_length: 最大 prompt 长度
            - max_response_length: 最大响应长度
            - truncation: 是否截断超长文本
        tokenizer: HuggingFace tokenizer，用于文本编码
        processor: 多模态处理器，用于处理图像等输入（可选）
        env: 工具环境，用于在数据处理时应用聊天模板（包含工具描述）

    返回：
        dataset: ToolRLDataset 实例，支持：
            - 多轮对话格式
            - 工具描述注入
            - 奖励信号提取

    数据格式说明：
        输入数据应为 parquet 或 jsonl 格式，包含以下字段：
        - prompt: 用户输入/问题
        - reward_model.ground_truth: 正确答案（用于计算奖励）
    """
    from torch.utils.data import Dataset
    from .agent_rl_dataset import ToolRLDataset

    # 注释掉的代码：支持自定义数据集类
    # 如果需要使用自定义数据集，可以在配置中指定 custom_cls.path 和 custom_cls.name
    # if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
    #     from verl.utils.import_utils import load_extern_type
    #
    #     dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    #     if not issubclass(dataset_cls, Dataset):
    #         raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    # else:

    # 使用默认的 ToolRLDataset 类
    dataset_cls = ToolRLDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # 实例化数据集
    dataset = dataset_cls(
        data_files=data_paths,  # 数据文件路径
        tokenizer=tokenizer,  # 分词器
        processor=processor,  # 多模态处理器
        config=data_config,  # 数据配置
        env=env,  # 工具环境（用于聊天模板）
    )

    return dataset


# ============================================================================
# 数据采样器创建函数
# ============================================================================


def create_rl_sampler(data_config, dataset):
    """
    创建数据采样器。

    采样器决定了数据在训练过程中的遍历顺序。支持两种模式：
    1. 随机采样（shuffle=True）：每个 epoch 打乱数据顺序
    2. 顺序采样（shuffle=False）：按原始顺序遍历数据

    参数：
        data_config: 数据配置，包含：
            - shuffle: 是否打乱数据
            - seed: 随机种子（用于可重复性）
        dataset: PyTorch Dataset 对象

    返回：
        sampler: RandomSampler 或 SequentialSampler 实例

    设计考虑：
        使用 sampler 而不是直接在 DataLoader 中设置 shuffle，
        是为了支持更好的检查点恢复。通过保存和恢复 sampler 的状态，
        可以在训练中断后从上次的位置继续。
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        # 创建随机数生成器并设置种子
        # 使用独立的 Generator 可以确保数据采样的可重复性
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))

        # RandomSampler: 每次迭代返回一个随机排列的索引序列
        sampler = RandomSampler(
            data_source=dataset, generator=train_dataloader_generator
        )
    else:
        # SequentialSampler: 按顺序返回索引 0, 1, 2, ...
        sampler = SequentialSampler(data_source=dataset)

    return sampler


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    # 当直接运行此脚本时，调用 main() 函数
    # Hydra 装饰器会自动处理配置加载和命令行参数解析
    main()
