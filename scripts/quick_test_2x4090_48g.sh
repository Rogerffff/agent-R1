#!/bin/bash
# ============================================================
# 快速验证脚本 - 2x RTX 4090-48GB + 3B 模型
# 预计运行时间：3-10 分钟
# 用途：验证训练流程能否跑通
# ============================================================

set -e

export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-3b-quick-test'

echo "🚀 开始快速验证..."
echo "   模型: $BASE_MODEL"
echo "   GPU: 2x RTX 4090-48GB"
echo "   预计时间: 3-10 分钟"
echo ""

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="['data/hotpotqa_mini/train.parquet']" \
    data.val_files="['data/hotpotqa_mini/validation.parquet']" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.max_response_length_single_turn=256 \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n_repeat=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.stop_token_ids="[151658]" \
    actor_rollout_ref.rollout.stop="[]" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    trainer.logger="['console']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    \
    tool.max_turns=2 \
    tool.tools="['search']" \
    tool.max_tool_response_length=512 \
    \
    "$@"

echo ""
echo "✅ 快速验证完成！"
echo ""
echo "如果看到 'actor/loss' 等指标输出，说明训练流程正常。"
echo "现在可以使用完整配置进行正式训练了。"

