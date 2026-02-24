# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Agent-R1 核心训练器模块 (Ray Agent Trainer)
==============================================================================

这是 Agent-R1 框架的核心训练模块，实现了基于 Ray 的分布式 PPO/GRPO 训练器。

==============================================================================
📖 阅读指南 (READING GUIDE)
==============================================================================

本文件较长（~1200行），建议按以下顺序阅读以快速理解核心逻辑：

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🚀 快速理解路径 (30分钟)                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 1️⃣  [第68-93行] Role 和 AdvantageEstimator 枚举                              │
│     理解训练中的角色定义和支持的优势估计算法                                      │
│                                                                             │
│ 2️⃣  [第265-336行] RayAgentTrainer.__init__()                                │
│     理解训练器如何初始化，有哪些核心组件                                         │
│                                                                             │
│ 3️⃣  [第946-1216行] RayAgentTrainer.fit() ⭐ 最重要                           │
│     这是主训练循环，理解整个训练流程的核心                                        │
│                                                                             │
│ 4️⃣  [第697-777行] RayAgentTrainer.init_workers()                            │
│     理解分布式 Worker 如何初始化和管理                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔬 深入理解路径 (2小时)                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 5️⃣  [第95-146行] ResourcePoolManager                                        │
│     理解 GPU 资源池管理和分配机制                                               │
│                                                                             │
│ 6️⃣  [第148-183行] apply_kl_penalty()                                        │
│     理解 KL 散度惩罚的实现（PPO 的核心技术）                                     │
│                                                                             │
│ 7️⃣  [第193-253行] compute_advantage()                                       │
│     理解各种优势估计算法（GAE、GRPO、REINFORCE++等）                             │
│                                                                             │
│ 8️⃣  [第564-695行] _validate()                                               │
│     理解验证流程和指标计算                                                     │
│                                                                             │
│ 9️⃣  [第778-860行] _save_checkpoint() 和 _load_checkpoint()                  │
│     理解检查点保存和恢复机制                                                   │
│                                                                             │
│ 🔟  [第1218-1245行] _create_action_mask()                                    │
│     理解 Action Mask 的创建（区分模型生成 vs 环境反馈）                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

==============================================================================
📊 训练循环流程图 (fit() 方法)
==============================================================================

    ┌──────────────────────────────────────────────────────────────────────┐
    │                        训练主循环 (fit)                               │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  1. 生成阶段 (Generation)                                             │
    │     - ToolGenerationManager.run_llm_loop()                           │
    │     - 多轮对话生成，包含工具调用                                        │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  2. 奖励计算 (Reward)                                                 │
    │     - reward_fn() 计算结果奖励                                        │
    │     - 可选：模型奖励 (rm_wg.compute_rm_score)                          │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  3. 对数概率计算 (Log Probabilities)                                   │
    │     - actor_rollout_wg.compute_log_prob() 当前策略                    │
    │     - ref_policy_wg.compute_ref_log_prob() 参考策略（可选）            │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  4. 值函数估计 (Values) - 仅 PPO/GAE                                   │
    │     - critic_wg.compute_values()                                     │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  5. 优势计算 (Advantage)                                              │
    │     - compute_advantage() 根据选择的算法计算优势                        │
    │     - 支持 GAE/GRPO/REINFORCE++/RLOO 等                               │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  6. 策略更新 (Policy Update)                                          │
    │     - critic_wg.update_critic() 更新值函数（仅 PPO）                   │
    │     - actor_rollout_wg.update_actor() 更新策略网络                     │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  7. 验证与保存                                                        │
    │     - _validate() 在验证集上评估                                      │
    │     - _save_checkpoint() 保存模型检查点                               │
    └──────────────────────────────────────────────────────────────────────┘

==============================================================================
🏗️ 核心类和函数概览
==============================================================================

枚举类：
  - Role: 定义训练角色（Actor、Critic、RefPolicy 等）
  - AdvantageEstimator: 定义优势估计算法（GAE、GRPO 等）

数据类：
  - ResourcePoolManager: GPU 资源池管理器

核心函数：
  - apply_kl_penalty(): 应用 KL 散度惩罚
  - compute_advantage(): 计算优势函数
  - compute_response_mask(): 计算响应掩码

核心类：
  - RayAgentTrainer: 主训练器类
    - __init__(): 初始化训练器
    - init_workers(): 初始化分布式工作组
    - fit(): 主训练循环
    - _validate(): 验证循环
    - _save_checkpoint(): 保存检查点
    - _load_checkpoint(): 加载检查点

==============================================================================
"""

# ============================================================================
# 导入部分
# ============================================================================

# 标准库导入
import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Tuple, Optional, Type

# 第三方库导入
import numpy as np
import ray  # 分布式计算框架
import torch  # 深度学习框架
from codetiming import Timer  # 代码计时工具
from omegaconf import OmegaConf, open_dict  # 配置管理
from torch.utils.data import Dataset, Sampler  # PyTorch 数据工具
from torchdata.stateful_dataloader import StatefulDataLoader  # 支持状态保存的数据加载器
from tqdm import tqdm  # 进度条

# verl 框架导入（核心分布式训练基础设施）
from verl import DataProto  # 数据协议类，用于在 Worker 间传递数据
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto  # 数据填充/去填充
from verl.single_controller.base import Worker  # Worker 基类
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)  # Ray 资源管理
from verl.single_controller.ray.base import (
    create_colocated_worker_cls,
)  # 创建共置 Worker
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path  # 检查点管理
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)  # 序列长度平衡
from verl.utils.torch_functional import masked_mean  # 带掩码的均值计算
from verl.utils.tracking import ValidationGenerationsLogger  # 验证日志记录
from verl.workers.rollout.async_server import AsyncLLMServerManager  # 异步推理服务管理

# Agent-R1 本地模块导入
from agent_r1.llm_agent.generation import (
    ToolGenerationManager,
    ToolGenerationConfig,
)  # 工具调用生成管理
from agent_r1.src import core_algos  # 核心 RL 算法
from agent_r1.src.core_algos import agg_loss  # 损失聚合函数
from agent_r1.src.metric_utils import (  # 指标计算工具
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from agent_r1.src.reward import compute_reward, compute_reward_async  # 奖励计算
from agent_r1.tool.base import BaseToolEnv  # 工具环境基类

# 类型别名
WorkerType = Type[Worker]


# ============================================================================
# 枚举定义
# ============================================================================


class Role(Enum):
    """
    训练角色枚举。

    定义了分布式训练中的不同角色，每个角色对应不同的模型或功能。
    这些角色会被映射到不同的 Worker 类和 GPU 资源池。

    角色说明：
        Actor: 策略模型，负责生成动作（token）
        Rollout: 生成器，负责采样生成完整序列
        ActorRollout: Actor 和 Rollout 的组合（混合引擎模式）
        Critic: 值函数模型，估计状态价值（仅 PPO/GAE 需要）
        RefPolicy: 参考策略，用于计算 KL 散度
        RewardModel: 奖励模型，用于模型打分
        ActorRolloutRef: Actor、Rollout 和 RefPolicy 的组合

    使用示例：
        role_worker_mapping = {
            Role.ActorRollout: ActorRolloutRefWorker,
            Role.Critic: CriticWorker,
        }
    """

    Actor = 0  # 策略模型（训练用）
    Rollout = 1  # 生成器（推理用）
    ActorRollout = 2  # Actor + Rollout 混合
    Critic = 3  # 值函数模型
    RefPolicy = 4  # 参考策略（冻结的初始策略）
    RewardModel = 5  # 奖励模型
    ActorRolloutRef = 6  # Actor + Rollout + RefPolicy


class AdvantageEstimator(str, Enum):
    """
    优势函数估计器枚举。

    优势函数 A(s,a) = Q(s,a) - V(s) 衡量某个动作相对于平均动作的好坏程度。
    不同的估计方法有不同的方差-偏差权衡。

    算法说明：
        GAE (Generalized Advantage Estimation):
            - 需要 Critic 模型
            - 使用 TD(λ) 方法，平衡偏差和方差
            - 适合长序列和复杂任务

        GRPO (Group Relative Policy Optimization):
            - 不需要 Critic 模型
            - 在同一 prompt 的多个采样中计算相对优势
            - 简单高效，适合大规模训练

        REINFORCE++:
            - 改进的 REINFORCE 算法
            - 使用基线减少方差

        REINFORCE++_BASELINE:
            - 带基线版本的 REINFORCE++
            - 使用同组样本的平均奖励作为基线

        RLOO (REINFORCE Leave-One-Out):
            - 使用留一法计算基线
            - 每个样本的基线是其他样本的平均奖励

        REMAX:
            - 使用 argmax 采样作为基线
            - 比较随机采样和贪婪采样的差异

    选择建议：
        - 有足够 GPU 内存 -> GAE (最稳定)
        - 追求效率 -> GRPO (无需 Critic)
        - 简单任务 -> REINFORCE++
    """

    GAE = "gae"  # 需要 Critic
    GRPO = "grpo"  # 不需要 Critic，组相对优势
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"  # 不需要 Critic
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"  # 不需要 Critic
    REMAX = "remax"  # 不需要 Critic
    RLOO = "rloo"  # 不需要 Critic，留一法


# ============================================================================
# 资源池管理器
# ============================================================================


@dataclass
class ResourcePoolManager:
    """
    GPU 资源池管理器。

    负责管理分布式训练中的 GPU 资源分配。支持：
    - 多节点多 GPU 配置
    - 不同角色共享或独占资源池
    - 资源可用性检查

    属性：
        resource_pool_spec: 资源池规格
            格式：{pool_name: [gpus_per_node] * num_nodes}
            示例：{"global_pool": [8, 8]} 表示 2 个节点，每节点 8 GPU

        mapping: 角色到资源池的映射
            格式：{Role: pool_name}
            示例：{Role.ActorRollout: "global_pool", Role.Critic: "global_pool"}

        resource_pool_dict: 实际创建的资源池实例

    使用示例：
        manager = ResourcePoolManager(
            resource_pool_spec={"global_pool": [8, 8]},
            mapping={Role.ActorRollout: "global_pool", Role.Critic: "global_pool"}
        )
        manager.create_resource_pool()  # 创建资源池
        pool = manager.get_resource_pool(Role.ActorRollout)  # 获取资源池
    """

    resource_pool_spec: dict[str, list[int]]  # 资源池规格
    mapping: dict[Role, str]  # 角色 -> 资源池名称
    resource_pool_dict: dict[str, RayResourcePool] = field(
        default_factory=dict
    )  # 资源池实例

    def create_resource_pool(self):
        """
        根据规格创建资源池。

        遍历 resource_pool_spec，为每个资源池创建 RayResourcePool 实例。
        然后检查集群中是否有足够的 GPU 资源。
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count: 每个资源池中可以共存的 WorkerGroup 数量
            # FSDP 后端推荐 max_colocate_count=1，将所有 WorkerGroup 合并
            # Megatron 后端推荐 max_colocate_count>1，不同模型使用不同 WorkerGroup
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        # 检查资源是否满足需求
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """获取指定角色对应的资源池。"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """获取集群中的总 GPU 数量。"""
        return sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )

    def _check_resource_available(self):
        """
        检查 Ray 集群中的 GPU 资源是否满足需求。

        如果资源不足，会抛出 ValueError 异常。
        """
        # 获取各节点的可用资源
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # 检查总 GPU 数量是否满足
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # 检查每个资源池是否能被满足（贪心分配算法）
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


# ============================================================================
# 核心辅助函数
# ============================================================================


def apply_kl_penalty(
    data: DataProto,
    kl_ctrl: core_algos.AdaptiveKLController,
    kl_penalty="kl",
    multi_turn=False,
):
    """
    应用 KL 散度惩罚到奖励中。

    KL 散度惩罚是 PPO 算法的核心技术，用于防止策略更新过大。
    通过在奖励中减去 KL 惩罚项，鼓励新策略与参考策略保持接近。

    公式：r_kl = r - β * KL(π || π_ref)

    参数：
        data: 包含以下字段的 DataProto 对象
            - old_log_probs: 当前策略的对数概率
            - ref_log_prob: 参考策略的对数概率
            - token_level_scores: 原始 token 级别奖励
        kl_ctrl: 自适应 KL 系数控制器
        kl_penalty: KL 惩罚类型（"kl", "abs", "mse" 等）
        multi_turn: 是否为多轮对话模式

    返回：
        data: 更新了 token_level_rewards 字段的数据
        metrics: 包含 KL 惩罚相关指标的字典
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # 根据是否多轮对话选择不同的掩码
    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # 使用 action_mask（如果存在），否则使用 response_mask
    if "action_mask" in data.batch:
        action_mask = data.batch["action_mask"]
    else:
        action_mask = response_mask

    # 计算当前策略和参考策略之间的 KL 散度
    # 当 apply_kl_penalty 时，use_kl_in_reward=True，参考模型已启用
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * action_mask  # 只计算有效 token 的 KL
    beta = kl_ctrl.value  # 当前 KL 惩罚系数

    # 计算带 KL 惩罚的奖励
    token_level_rewards = token_level_scores - beta * kld

    # 计算当前批次的平均 KL 散度
    current_kl = masked_mean(kld, mask=action_mask, axis=-1)  # 序列维度平均
    current_kl = torch.mean(current_kl, dim=0).item()

    # 自适应更新 KL 系数
    # 参考：https://github.com/huggingface/trl/blob/master/trl/trainer/ppo_trainer.py
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {
        "actor/reward_kl_penalty": current_kl,  # 当前 KL 散度
        "actor/reward_kl_penalty_coeff": beta,  # 当前 KL 系数
    }

    return data, metrics


def compute_response_mask(data: DataProto):
    """
    计算响应部分的掩码。

    从完整序列的 attention_mask 中提取响应部分的掩码。
    响应部分是模型生成的内容，prompt 部分是输入内容。

    参数：
        data: 包含 responses 和 attention_mask 的 DataProto

    返回：
        response_mask: 形状为 (batch_size, response_length) 的掩码张量
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
):
    """
    根据指定的算法计算优势函数。

    优势函数 A(s,a) 衡量某个动作相对于平均动作的好坏程度。
    不同的估计方法在方差和偏差之间有不同的权衡。

    参数：
        data: 包含奖励、值函数等信息的 DataProto
        adv_estimator: 优势估计算法（AdvantageEstimator 枚举值）
        gamma: 折扣因子，控制未来奖励的重要性（0-1）
        lam: GAE 的 λ 参数，控制偏差-方差权衡（0-1）
        num_repeat: 每个 prompt 的采样次数
        multi_turn: 是否为多轮对话模式
        norm_adv_by_std_in_grpo: GRPO 是否使用标准差归一化

    返回：
        data: 更新了 advantages 和 returns 字段的数据

    优势估计算法对比：
        ┌─────────────────┬─────────┬─────────┬──────────────────────────┐
        │ 算法             │ 偏差    │ 方差    │ 是否需要 Critic          │
        ├─────────────────┼─────────┼─────────┼──────────────────────────┤
        │ GAE             │ 低      │ 可调    │ ✓                        │
        │ GRPO            │ 低      │ 中      │ ✗                        │
        │ REINFORCE++     │ 高      │ 低      │ ✗                        │
        │ RLOO            │ 低      │ 中      │ ✗                        │
        │ REMAX           │ 中      │ 低      │ ✗                        │
        └─────────────────┴─────────┴─────────┴──────────────────────────┘
    """
    # 向后兼容：如果没有 response_mask，则计算
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)

    # 根据选择的算法计算优势
    if adv_estimator == AdvantageEstimator.GAE:
        # GAE (Generalized Advantage Estimation)
        # 使用 TD(λ) 方法，需要 Critic 提供值函数估计
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            action_mask=data.batch["action_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == AdvantageEstimator.GRPO:
        # GRPO (Group Relative Policy Optimization)
        # 在同一 prompt 的多个采样中计算相对优势
        # 不需要 Critic，使用组内相对比较
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            action_mask=data.batch["action_mask"],
            index=data.non_tensor_batch["uid"],  # 用于分组的 prompt ID
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        # REINFORCE++ with Baseline
        # 使用同组样本的平均奖励作为基线
        advantages, returns = (
            core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                action_mask=data.batch["action_mask"],
                index=data.non_tensor_batch["uid"],
            )
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        # REINFORCE++
        # 基本的策略梯度方法，使用折扣累积奖励
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            action_mask=data.batch["action_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == AdvantageEstimator.REMAX:
        # REMAX
        # 使用贪婪采样（argmax）作为基线
        # 比较随机采样和贪婪采样的差异
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],  # argmax 采样的奖励
            action_mask=data.batch["action_mask"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == AdvantageEstimator.RLOO:
        # RLOO (REINFORCE Leave-One-Out)
        # 使用留一法计算基线
        # 每个样本的基线是组内其他样本的平均奖励
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            action_mask=data.batch["action_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError(f"Unknown advantage estimator: {adv_estimator}")

    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """ 
    计时上下文管理器。

    用于测量代码块的执行时间，并将结果累加到 timing_raw 字典中。

    使用示例：
        timing_raw = {}
        with _timer("generation", timing_raw):
            # 生成代码
            ...
        print(f"生成耗时: {timing_raw['generation']}秒")
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


# ============================================================================
# 核心训练器类
# ============================================================================


class RayAgentTrainer(object):
    """
    Ray 分布式 Agent 训练器。

    这是 Agent-R1 的核心训练器，实现了完整的强化学习训练循环。
    训练器运行在 driver 进程（单个 CPU/GPU 节点），通过 RPC 调用
    分布式 Worker 执行计算密集型操作。

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        RayAgentTrainer 架构                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Driver Process (RayAgentTrainer)                                       │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  - 训练循环编排                                                  │    │
    │  │  - 优势计算（轻量级）                                            │    │
    │  │  - 指标记录和检查点                                              │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                              │                                          │
    │                              │ RPC 调用                                 │
    │                              ▼                                          │
    │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
    │  │ Actor Worker  │  │ Critic Worker │  │ RefPolicy     │               │
    │  │ (GPU 0-3)     │  │ (GPU 0-3)     │  │ Worker        │               │
    │  │               │  │               │  │ (GPU 0-3)     │               │
    │  │ - 生成序列    │  │ - 计算值函数  │  │ - 计算 ref    │               │
    │  │ - 计算 log_p  │  │ - 更新 Critic │  │   log_prob    │               │
    │  │ - 更新 Actor  │  │               │  │               │               │
    │  └───────────────┘  └───────────────┘  └───────────────┘               │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    主要职责：
        1. 初始化和管理分布式 Worker 组
        2. 执行训练主循环（生成 -> 奖励 -> 更新）
        3. 验证和评估模型性能
        4. 保存和加载检查点
        5. 记录训练指标

    核心方法：
        - __init__(): 初始化训练器
        - init_workers(): 创建分布式 Worker
        - fit(): 主训练循环
        - _validate(): 验证循环
        - _save_checkpoint(): 保存检查点
        - _load_checkpoint(): 加载检查点
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        env: Optional[BaseToolEnv] = None,
        val_env: Optional[BaseToolEnv] = None,
    ):
        """
        初始化训练器。

        参数：
            config: OmegaConf 配置对象，包含所有训练超参数
            tokenizer: HuggingFace tokenizer，用于文本编码/解码
            role_worker_mapping: 角色到 Worker 类的映射
                例如：{Role.ActorRollout: ActorRolloutRefWorker, Role.Critic: CriticWorker}
            resource_pool_manager: GPU 资源池管理器
            ray_worker_group_cls: Ray Worker 组类（用于创建 Worker 组）
            processor: 多模态处理器（可选，用于 VLM）
            reward_fn: 训练奖励函数
            val_reward_fn: 验证奖励函数
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            collate_fn: 数据批处理函数
            train_sampler: 训练数据采样器
            env: 训练工具环境
            val_env: 验证工具环境（可选，如果与训练环境不同）

        初始化步骤：
            1. 保存配置和组件引用
            2. 确定是否使用 Critic（取决于优势估计算法）
            3. 初始化 KL 控制器（如果启用）
            4. 验证配置合法性
            5. 创建数据加载器
        """
        # 保存基本组件
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.env = env
        self.val_env = val_env

        # 验证环境配置
        if val_env is not None:
            print(
                "[INFO] val env is different from train env, it means you are evaluating the model's generalization capabilities."
            )
        else:
            self.val_env = env

        # 检查混合引擎模式（Actor 和 Rollout 共用同一模型）
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"{role_worker_mapping.keys()=}"

        # 保存角色映射和资源管理器
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager

        # 确定可选组件
        self.use_reference_policy = (
            Role.RefPolicy in role_worker_mapping
        )  # 是否使用参考策略
        self.use_rm = Role.RewardModel in role_worker_mapping  # 是否使用奖励模型
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # 初始化 KL 惩罚控制器
        # 用于自适应调整 KL 散度惩罚系数
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(
                config.algorithm.kl_ctrl
            )

        # 根据优势估计算法确定是否需要 Critic
        # GAE 需要 Critic，其他算法不需要
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        # 验证配置并创建数据加载器
        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = (
            config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        )
        assert (
            real_train_batch_size % n_gpus == 0
        ), f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'"
                        + "is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size,
                config.critic.ppo_micro_batch_size_per_gpu,
                "critic",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size,
                config.reward_model.micro_batch_size_per_gpu,
                "reward_model",
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert (
                config.data.train_batch_size
                >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            )
            sp_size = config.actor_rollout_ref.actor.get(
                "ulysses_sequence_parallel_size", 1
            )
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert (
                    config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size
                    >= n_gpus
                )

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if (
            config.algorithm.use_kl_in_reward
            and config.actor_rollout_ref.actor.use_kl_loss
        ):
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert (
                    config.critic.ppo_mini_batch_size
                    % config.critic.ppo_micro_batch_size
                    == 0
                )
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert (
                config.actor_rollout_ref.model.use_remove_padding
            ), "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert (
                    config.critic.model.use_remove_padding
                ), "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert (
                config.actor_rollout_ref.rollout.temperature > 0
            ), "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
            ), "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO
            ], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from .main_agent import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                self.env,
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                self.val_env,
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get(
                "gen_batch_size", self.config.data.train_batch_size
            ),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}"
        )

        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                        total_training_steps
                    )
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(
                f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}"
            )

    def _dump_generations(
        self, inputs, outputs, scores, reward_extra_infos_dict, dump_path
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(
            self.config.trainer.logger, samples, self.global_steps
        )

    def _validate(self):
        """
        验证循环。

        在验证集上评估模型性能，包括：
        1. 使用当前策略生成响应
        2. 计算奖励/准确率
        3. 收集和聚合各种指标
        4. 可选：记录生成样本到日志/文件

        验证流程：

        ┌─────────────────────────────────────────────────────────────────────┐
        │                          验证流程                                    │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  for batch in val_dataloader:                                       │
        │      │                                                              │
        │      ├──► 1. 重复批次（每个 prompt 生成 n 个响应）                    │
        │      │                                                              │
        │      ├──► 2. 使用 ToolGenerationManager 生成响应                     │
        │      │       - 多轮工具调用                                         │
        │      │       - 可配置温度、采样策略                                  │
        │      │                                                              │
        │      ├──► 3. 使用 val_reward_fn 计算奖励                            │
        │      │       - 精确匹配、格式检查等                                  │
        │      │                                                              │
        │      └──► 4. 收集样本和指标                                         │
        │                                                                     │
        │  聚合指标：                                                          │
        │  - mean@N: N 个采样的平均分数                                        │
        │  - maj@N: 多数投票准确率                                            │
        │  - best@N: N 个采样中的最高分                                        │
        │                                                                     │
        └─────────────────────────────────────────────────────────────────────┘

        返回：
            metric_dict: 包含所有验证指标的字典，格式为：
                {
                    "val-core/{data_source}/{metric}": value,
                    "val-aux/{data_source}/{metric}": value,
                }
        """
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # 收集样本用于日志记录
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        # 配置验证生成参数
        gen_config = ToolGenerationConfig(
            max_turns=self.config.tool.val_kwargs.max_turns,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_response_length_single_turn=self.config.data.max_response_length_single_turn,
            use_batch_tool_calls=self.config.tool.val_kwargs.use_batch_tool_calls,
        )

        # 创建验证专用的生成管理器
        generation_manager = ToolGenerationManager(
            tokenizer=self.tokenizer,
            processor=self.processor,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=True,  # 标记为验证模式
        )

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_inputs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.extend(
                    ["multi_modal_data", "multi_modal_inputs"]
                )
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = generation_manager.run_llm_loop(
                test_gen_batch_padded,
                env=self.val_env,
            )

            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            # for key in test_output_gen_batch.batch.keys():
            #     test_output_gen_batch.batch[key] = test_output_gen_batch.batch[key].long()

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_infos_dict["turns"].extend(test_batch.batch["turns"].tolist())
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(
                sample_scores
            ), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(
            data_sources, sample_inputs, reward_extra_infos_dict
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(
                    [
                        int(name.split("@")[-1].split("/")[0])
                        for name in metric2val.keys()
                    ]
                )
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(
                            metric_name.startswith(pfx)
                            for pfx in ["mean", "maj", "best"]
                        )
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """
        初始化分布式 Worker 组。

        这是训练开始前的关键步骤，负责：
        1. 创建 GPU 资源池
        2. 实例化各角色的 Worker 类
        3. 将 Worker 部署到 GPU 上
        4. 初始化模型权重

        Worker 初始化流程：

        ┌─────────────────────────────────────────────────────────────────────┐
        │                      Worker 初始化流程                               │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  1. 创建资源池                                                       │
        │     ResourcePoolManager.create_resource_pool()                      │
        │                              │                                      │
        │                              ▼                                      │
        │  2. 注册 Worker 类                                                   │
        │     ┌──────────────┬──────────────┬──────────────┐                 │
        │     │ ActorRollout │   Critic     │  RefPolicy   │                 │
        │     │   (必需)      │ (GAE 需要)   │  (KL 需要)    │                 │
        │     └──────────────┴──────────────┴──────────────┘                 │
        │                              │                                      │
        │                              ▼                                      │
        │  3. 创建共置 Worker（同一 GPU 上运行多个 Worker）                     │
        │     create_colocated_worker_cls()                                   │
        │                              │                                      │
        │                              ▼                                      │
        │  4. 初始化模型                                                       │
        │     Critic -> RefPolicy -> ActorRollout (最后初始化以优化显存)        │
        │                                                                     │
        └─────────────────────────────────────────────────────────────────────┘

        注意：ActorRollout 最后初始化，以便 vLLM 可以更好地估计 KV cache 可用内存。
        """
        # 步骤 1：创建 GPU 资源池
        self.resource_pool_manager.create_resource_pool()

        # 初始化资源池到 Worker 类的映射
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        } 

        # 步骤 2：注册各角色的 Worker 类

        # 创建 Actor + Rollout Worker（混合引擎模式）
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            
            self.resource_pool_to_cls[resource_pool][
                "actor_rollout"
            ] = actor_rollout_cls
        else:
            raise NotImplementedError

        # 创建 Critic Worker（仅 GAE 算法需要）
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # 创建参考策略 Worker（KL 惩罚需要）
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # 创建奖励模型 Worker（如果启用）
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # 步骤 3：初始化 WorkerGroup
        # 注意：如果需要不同角色使用不同大小的并行度，不要使用 create_colocated_worker_cls
        # 而是直接为不同 worker group 传入不同的 resource pool
        # 参见：https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}

        # 设置 Worker 注册超时时间
        if (
            OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout")
            is not None
        ):
            wg_kwargs["ray_wait_register_center_timeout"] = (
                self.config.trainer.ray_wait_register_center_timeout
            )

        # 为每个资源池创建共置的 Worker 组
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # 创建共置 Worker 类（多个 Worker 共享同一 GPU）
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # 保持 WorkerDict 引用以支持 ray >= 2.31
            # 参见：https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        # 步骤 4：按顺序初始化模型
        # 顺序很重要：Critic -> RefPolicy -> ActorRollout
        # ActorRollout 最后初始化，让 vLLM 更好地估计 KV cache 可用内存

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # ActorRollout 最后初始化
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # 步骤 5：创建异步推理管理器（如果启用）
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        """
        保存训练检查点。

        保存内容包括：
        1. Actor 模型权重
        2. Critic 模型权重（如果使用）
        3. DataLoader 状态（用于恢复数据迭代位置）
        4. 最新检查点迭代号（用于自动恢复）

        目录结构：
            default_local_dir/
            ├── global_step_100/
            │   ├── actor/           # Actor 模型
            │   ├── critic/          # Critic 模型（可选）
            │   └── data.pt          # DataLoader 状态
            ├── global_step_200/
            │   └── ...
            └── latest_checkpointed_iteration.txt  # 最新检查点号
        """
        # 构建本地路径：基础目录 + global_step_{N}
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        # 构建远程路径（HDFS，可选）
        actor_remote_path = None
        if self.config.trainer.default_hdfs_dir is not None:
            actor_remote_path = os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )

        # 检查点保留策略
        remove_previous_ckpt_in_save = self.config.trainer.get(
            "remove_previous_ckpt_in_save", False
        )
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )

        # 保存 Actor 模型
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        # 保存 Critic 模型（如果使用）
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None
            if self.config.trainer.default_hdfs_dir is not None:
                critic_remote_path = os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        # 保存 DataLoader 状态（用于恢复数据迭代位置）
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # 记录最新检查点迭代号（用于原子性恢复）
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """
        加载训练检查点。

        支持三种恢复模式：
        1. disable: 不恢复，从头开始训练
        2. auto: 自动查找最新检查点，如果没有则从头开始
        3. resume_path: 从指定路径恢复

        加载内容：
        1. Actor 模型权重
        2. Critic 模型权重（如果使用）
        3. DataLoader 状态
        4. 设置 global_steps

        返回：
            加载的 global_steps，如果从头开始则返回 0
        """
        # 模式 1: 禁用恢复
        if self.config.trainer.resume_mode == "disable":
            return 0

        # 确定检查点文件夹
        if self.config.trainer.default_hdfs_dir is not None:
            # TODO: 从 HDFS 加载
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            # 确保使用绝对路径
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            # 查找最新检查点
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder
            )  # 没有则返回 None

        # 模式 2: 自动恢复
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        # 模式 3: 从指定路径恢复
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(
                    self.config.trainer.resume_from_path, str
                ), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_from_path
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)

        print(f"Load from checkpoint folder: {global_step_folder}")

        # 从文件夹名称解析 global_steps
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        # 加载 Actor 模型
        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # 加载 Critic 模型
        if self.use_critic:
            critic_path = os.path.join(global_step_folder, "critic")
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # 加载 DataLoader 状态
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(
                dataloader_local_path, weights_only=False
            )
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(
                f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch"
            )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        )  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def prime_norm(self, token_level_scores):
        """
        Apply normalization to token level scores.
        This helps stabilize training by considering cumulative effects and scaling rewards.

        Args:
            token_level_scores: Tensor of token level scores

        Returns:
            Normalized token level scores
        """
        # Compute reverse cumulative sum and normalize by its maximum absolute value
        reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(
            dims=[1]
        )
        token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
        return token_level_scores

    def _compute_process_rewards(self, batch, envs, reward_tensor) -> torch.Tensor:
        """
        Compute process rewards for tool calls.

        Args:
            batch: The batch data containing responses and attention masks
            envs: List of tool environments containing reward history
            reward_tensor: Tensor with same shape as responses for storing rewards

        Returns:
            process_rewards: Tensor containing rewards for each tool call
        """
        process_rewards = torch.zeros_like(reward_tensor)

        if not self.config.algorithm.get("use_process_rewards", False):
            return process_rewards

        responses = [
            self.tokenizer.decode(resp, skip_special_tokens=False)
            for resp in batch.batch["responses"]
        ]

        for i, (response, env) in enumerate(zip(responses, envs)):
            # Get valid response length
            prompt_ids = batch.batch["prompts"][i]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = (
                batch.batch["attention_mask"][i, prompt_length:].sum().item()
            )

            # Get rewards from env
            env_rewards = env.rewards

            # Find all tool call end positions
            tool_call_ends = []
            pos = 0
            while True:
                pos = response.find(self.config.tool.tool_call_end, pos)
                if pos == -1:
                    break
                tool_call_ends.append(pos)
                pos += 1

            # Convert character positions to token positions
            for tool_idx, end_pos in enumerate(tool_call_ends):
                if tool_idx >= len(env_rewards):  # Safety check
                    break

                # Get token position for the end of tool call
                prefix = response[: end_pos + len(self.config.tool.tool_call_end)]
                token_pos = (
                    len(self.tokenizer.encode(prefix, add_special_tokens=False)) - 1
                )

                # Only assign reward if token position is within valid response length
                if token_pos < valid_response_length:
                    process_rewards[i, token_pos] = env_rewards[tool_idx]

        # Apply normalization to process rewards
        process_rewards = self.prime_norm(process_rewards)

        return process_rewards

    def fit(self):
        """
        主训练循环。

        这是 Agent-R1 的核心训练方法，实现了完整的 PPO/GRPO 训练流程。
        Driver 进程通过 RPC 调用 Worker 组的计算函数来构建训练数据流，
        轻量级的优势计算在 Driver 进程上完成。

        ┌─────────────────────────────────────────────────────────────────────┐
        │                        训练循环详细流程                               │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  for epoch in total_epochs:                                         │
        │      for batch in dataloader:                                       │
        │                                                                     │
        │          ┌─────────────────────────────────────────────────────┐    │
        │          │ 1. 生成阶段 (gen)                                    │    │
        │          │    - ToolGenerationManager.run_llm_loop()           │    │
        │          │    - 多轮工具调用生成完整响应                         │    │
        │          │    - 构建 action_mask（区分模型生成 vs 环境反馈）     │    │
        │          └─────────────────────────────────────────────────────┘    │
        │                              │                                      │
        │                              ▼                                      │
        │          ┌─────────────────────────────────────────────────────┐    │
        │          │ 2. 奖励计算 (reward)                                 │    │
        │          │    - reward_fn() 计算结果奖励                        │    │
        │          │    - 可选：rm_wg.compute_rm_score() 模型奖励         │    │
        │          └─────────────────────────────────────────────────────┘    │
        │                              │                                      │
        │                              ▼                                      │
        │          ┌─────────────────────────────────────────────────────┐    │
        │          │ 3. 对数概率计算 (old_log_prob)                       │    │
        │          │    - actor_rollout_wg.compute_log_prob()            │    │
        │          │    - 计算当前策略下生成序列的对数概率                 │    │
        │          └─────────────────────────────────────────────────────┘    │
        │                              │                                      │
        │                              ▼                                      │
        │          ┌─────────────────────────────────────────────────────┐    │
        │          │ 4. 参考策略对数概率 (ref) - 可选                      │    │
        │          │    - ref_policy_wg.compute_ref_log_prob()           │    │
        │          │    - 用于计算 KL 散度惩罚                            │    │
        │          └─────────────────────────────────────────────────────┘    │
        │                              │                                      │
        │                              ▼                                      │
        │          ┌─────────────────────────────────────────────────────┐    │
        │          │ 5. 值函数估计 (values) - 仅 GAE                      │    │
        │          │    - critic_wg.compute_values()                     │    │
        │          └─────────────────────────────────────────────────────┘    │
        │                              │                                      │
        │                              ▼                                      │
        │          ┌─────────────────────────────────────────────────────┐    │
        │          │ 6. 优势计算 (adv) - 在 Driver 上执行                 │    │
        │          │    - apply_kl_penalty() 应用 KL 惩罚                │    │
        │          │    - compute_advantage() 计算优势函数               │    │
        │          └─────────────────────────────────────────────────────┘    │
        │                              │                                      │
        │                              ▼                                      │
        │          ┌─────────────────────────────────────────────────────┐    │
        │          │ 7. 模型更新                                          │    │
        │          │    - critic_wg.update_critic() 更新值函数（仅 GAE）  │    │
        │          │    - actor_rollout_wg.update_actor() 更新策略       │    │
        │          └─────────────────────────────────────────────────────┘    │
        │                              │                                      │
        │                              ▼                                      │
        │          ┌─────────────────────────────────────────────────────┐    │
        │          │ 8. 验证和保存                                        │    │
        │          │    - _validate() 按 test_freq 频率验证              │    │
        │          │    - _save_checkpoint() 按 save_freq 频率保存       │    │
        │          └─────────────────────────────────────────────────────┘    │
        │                                                                     │
        └─────────────────────────────────────────────────────────────────────┘

        计时统计：
            - gen: 生成阶段耗时
            - reward: 奖励计算耗时
            - old_log_prob: 对数概率计算耗时
            - ref: 参考策略计算耗时
            - values: 值函数计算耗时
            - adv: 优势计算耗时
            - update_critic: Critic 更新耗时
            - update_actor: Actor 更新耗时
            - testing: 验证耗时
            - save_checkpoint: 保存检查点耗时
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        # ====================================================================
        # 初始化日志记录器
        # ====================================================================
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,  # wandb, tensorboard 等
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # ====================================================================
        # 加载检查点（如果存在）
        # ====================================================================
        self._load_checkpoint()

        # ====================================================================
        # 训练前验证（可选）
        # ====================================================================
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

            # 如果只做验证，直接返回
            if self.config.trainer.get("val_only", False):
                return

        # 创建进度条
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # 从 step 1 开始
        self.global_steps += 1
        last_val_metrics = None

        # ====================================================================
        # 配置工具调用生成管理器
        # ====================================================================
        gen_config = ToolGenerationConfig(
            max_turns=self.config.tool.max_turns,  # 最大对话轮数
            max_prompt_length=self.config.data.max_prompt_length,  # 最大 prompt 长度
            max_response_length=self.config.data.max_response_length,  # 最大响应长度
            max_response_length_single_turn=self.config.data.max_response_length_single_turn,  # 单轮最大长度
            use_batch_tool_calls=self.config.tool.use_batch_tool_calls,  # 是否批量执行工具调用
        )

        generation_manager = ToolGenerationManager(
            tokenizer=self.tokenizer,
            processor=self.processor,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # ====================================================================
        # 主训练循环
        # ====================================================================
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # 初始化本轮的指标和计时字典
                metrics = {}
                timing_raw = {}

                # ============================================================
                # 数据准备
                # ============================================================
                # 将字典转换为 DataProto 对象（verl 的数据传输格式）
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 为每个样本分配唯一 ID（用于 GRPO/RLOO 的分组计算）
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # 重复批次（每个 prompt 生成多个响应）
                # interleave=True 表示交错排列：[p1, p1, p2, p2, ...]
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n_repeat,
                    interleave=True,
                )

                # 分离生成所需的字段
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(
                        ["multi_modal_data", "multi_modal_inputs"]
                    )
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # 检查是否是最后一步
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # ========================================================
                    # 步骤 1: 生成阶段
                    # ========================================================
                    # 使用工具调用生成管理器进行多轮对话生成
                    with _timer("gen", timing_raw):
                        gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            env=self.env,  # 工具环境
                        )

                    # ========================================================
                    # REMAX 基线生成（可选）
                    # ========================================================
                    # REMAX 算法需要额外的贪婪采样作为基线
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = (
                                False  # 贪婪采样
                            )
                            gen_baseline_output = (
                                self.actor_rollout_wg.generate_sequences(
                                    gen_baseline_batch
                                )
                            )

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # 合并生成输出到批次
                    batch = batch.union(gen_batch_output)

                    # 计算响应掩码
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # ========================================================
                    # 序列长度平衡（可选）
                    # ========================================================
                    # 平衡各 DP rank 上的有效 token 数量
                    # 注意：这会打乱批次内的数据顺序，影响 GRPO/RLOO 的分组计算
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # 计算全局有效 token 数量
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # ========================================================
                    # 创建 Action Mask
                    # ========================================================
                    # 区分模型生成的 token 和环境反馈的 token
                    batch, metrics = self._create_action_mask(batch, metrics)

                    # ========================================================
                    # 步骤 2: 奖励计算
                    # ========================================================
                    with _timer("reward", timing_raw):
                        # 使用奖励模型计算分数（如果启用）
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # 计算奖励（同步或异步）
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                batch, self.config, self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch, self.reward_fn
                            )

                    # ========================================================
                    # 步骤 3: 计算当前策略的对数概率
                    # ========================================================
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

                        # 计算熵损失（用于鼓励探索）
                        entropys = old_log_prob.batch["entropys"]
                        action_masks = batch.batch["action_mask"]
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
                        entropy_loss = agg_loss(
                            loss_mat=entropys,
                            loss_mask=action_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy_loss": entropy_loss.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)

                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # ========================================================
                    # 步骤 4: 计算参考策略的对数概率（用于 KL 惩罚）
                    # ========================================================
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    # ========================================================
                    # 步骤 5: 计算值函数（仅 GAE 算法）
                    # ========================================================
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # ========================================================
                    # 步骤 6: 优势计算
                    # ========================================================
                    with _timer("adv", timing_raw):
                        # 获取异步奖励计算结果
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(
                                future_reward
                            )

                        # 保存 token 级别的分数
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {
                                    k: np.array(v)
                                    for k, v in reward_extra_infos_dict.items()
                                }
                            )

                        # 应用 KL 惩罚（如果启用）
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]

                        # 计算优势函数（在 Driver 进程执行，轻量级）
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                        )

                    # ========================================================
                    # 步骤 7: 模型更新
                    # ========================================================

                    # 更新 Critic（仅 GAE 算法）
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # Critic 热身：在热身阶段只更新 Critic，不更新 Actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # 更新 Actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # ========================================================
                    # 步骤 8: 记录生成数据（可选）
                    # ========================================================
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True
                            )
                            outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )
                            scores = (
                                batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # ========================================================
                    # 步骤 9: 验证和保存检查点
                    # ========================================================

                    # 验证
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # 保存检查点
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # ============================================================
                # 收集和记录指标
                # ============================================================
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                # 数据指标（奖励统计、响应长度等）
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )

                # 计时指标
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )

                # 吞吐量指标
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                # 记录到日志系统
                logger.log(data=metrics, step=self.global_steps)

                # 检查是否完成训练
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

    def _create_action_mask(
        self, batch: DataProto, metrics: dict
    ) -> Tuple[DataProto, dict]:
        """
        创建动作掩码（Action Mask）用于优势计算。

        Action Mask 是 Agent RL 训练的关键技术，用于区分：
        - 模型生成的 token（action_mask=1）：需要计算梯度
        - 环境反馈的 token（action_mask=0）：不需要计算梯度

        ┌─────────────────────────────────────────────────────────────────────┐
        │                        Action Mask 示意图                            │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  完整序列:                                                           │
        │  ┌─────────────────────────────────────────────────────────────┐   │
        │  │ [Prompt] [Model Response] [Tool Output] [Model Response]    │   │
        │  └─────────────────────────────────────────────────────────────┘   │
        │                                                                     │
        │  Action Mask:                                                       │
        │  ┌─────────────────────────────────────────────────────────────┐   │
        │  │    0          1              0              1               │   │
        │  └─────────────────────────────────────────────────────────────┘   │
        │                                                                     │
        │  只对 action_mask=1 的 token 计算策略梯度                            │
        │                                                                     │
        └─────────────────────────────────────────────────────────────────────┘

        在多轮对话中，以下内容被视为外部交互（action_mask=0）：
        - 用户消息（包括 <|im_start|>user 等标签）
        - 工具调用的响应
        - 系统提示

        参数：
            batch: 包含响应和注意力掩码的 DataProto
            metrics: 用于记录指标的字典

        返回：
            batch: 更新了 action_mask 的数据
            metrics: 更新了动作比例指标的字典
        """
        response_length = batch.batch["responses"].shape[-1]
        response_mask = batch.batch["attention_mask"][:, -response_length:]

        # 检查是否已有 action_mask（由生成阶段创建）
        if "action_mask" not in batch.batch.keys():
            # 如果没有，使用全 1 掩码（所有 token 都视为动作）
            action_mask = torch.ones_like(response_mask)
            print("[WARNING] No action mask found in batch, using all ones")
        else:
            action_mask = batch.batch["action_mask"]

        # 记录动作相关指标
        # action_ratio: 动作 token 占有效 token 的比例
        action_ratio = action_mask.sum().item() / (response_mask.sum().item() + 1e-8)
        metrics["action/ratio"] = action_ratio
        metrics["action/length/max"] = action_mask.sum(dim=-1).max().item()
        metrics["action/length/min"] = action_mask.sum(dim=-1).min().item()
        metrics["action/length/mean"] = action_mask.sum(dim=-1).float().mean().item()

        return batch, metrics
