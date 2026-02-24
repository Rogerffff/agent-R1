#!/bin/bash
# ============================================================
# 生成快速验证用小数据集
# 只保留 32 条训练数据和 8 条验证数据
# ============================================================

set -e

echo "🔄 正在生成快速验证用小数据集..."

# 创建小数据集目录
mkdir -p data/hotpotqa_mini

# 运行数据预处理，限制样本数量
python examples/data_preprocess/hotpotqa.py \
    --local_dir data/hotpotqa_mini \
    --train_size 32 \
    --val_size 8

echo ""
echo "✅ 小数据集生成完成！"
echo "   - 训练集: data/hotpotqa_mini/train.parquet (32 条)"
echo "   - 验证集: data/hotpotqa_mini/validation.parquet (8 条)"
echo ""
echo "💡 现在可以运行快速验证脚本了："
echo "   bash scripts/quick_test_2x4090_48g.sh"

