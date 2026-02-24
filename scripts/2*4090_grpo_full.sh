#!/bin/bash
# ============================================================
# 正式训练脚本 - 2x RTX 4090-24GB + Qwen2.5-1.5B-Instruct
# 数据集：hotpotqa (25600 训练 / 128 验证)
# 预计训练时间：根据 epoch 数而定
# ============================================================

set -e

export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
export PROJECT_NAME='hotpotqa'
export EXPERIMENT_NAME='grpo-1.5b-2x4090'

# WandB 配置 - 确保 Ray worker 可以访问认证信息
if [ -z "$WANDB_API_KEY" ]; then
    # 从 .netrc 文件读取 API key
    WANDB_API_KEY=$(grep -A2 "api.wandb.ai" ~/.netrc 2>/dev/null | grep password | awk '{print $2}' | head -c 86)
    if [ -n "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY
        echo "✓ 已从 ~/.netrc 加载 WandB API Key"
    else
        echo "⚠ 警告: 未找到 WandB API Key，将只使用 console logger"
    fi
fi

echo "=============================================="
echo "🚀 开始 GRPO 训练"
echo "   模型: $BASE_MODEL"
echo "   GPU: 2x RTX 4090-24GB"
echo "   数据集: hotpotqa (25600 train / 128 val)"
echo "=============================================="
echo ""

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="['data/hotpotqa/train.parquet']" \
    data.val_files="['data/hotpotqa/validation.parquet']" \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.max_response_length_single_turn=512 \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.stop_token_ids="[151658]" \
    actor_rollout_ref.rollout.stop="[]" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=400 \
    trainer.val_before_train=True \
    \
    tool.max_turns=5 \
    tool.tools="['search']" \
    tool.max_tool_response_length=1024 \
    \
    "$@"


echo ""
echo "=============================================="
echo "✅ 训练完成！"
echo "=============================================="