"""
查看 DAPO-Math-17K-cleaned 原始数据集

该脚本展示如何加载和查看原始的 HuggingFace 数据集。

使用方法:
    source .venv-retool/bin/activate
    python examples/data_preprocess/learn_retool/view_raw_dataset.py
"""

import os
import sys
from datasets import load_dataset


def run_view_raw():
    """查看原始数据集的逻辑"""
    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset("haizhongzheng/DAPO-Math-17K-cleaned")

    # 显示数据集基本信息
    print("\n" + "=" * 60)
    print("数据集基本信息")
    print("=" * 60)
    print(f"分割: {list(dataset.keys())}")
    print(f"训练集大小: {len(dataset['train'])}")
    print(f"字段: {dataset['train'].column_names}")

    # 显示字段类型
    print("\n" + "=" * 60)
    print("字段类型")
    print("=" * 60)
    for feature_name, feature in dataset["train"].features.items():
        print(f"  {feature_name}: {feature}")

    # 显示前几条数据
    print("\n" + "=" * 60)
    print("前 3 条数据示例")
    print("=" * 60)

    for i in range(min(3, len(dataset["train"]))):
        example = dataset["train"][i]
        print(f"\n--- 第 {i + 1} 条数据 ---")

        # 显示 prompt（截断显示）
        prompt = example["prompt"]
        if len(prompt) > 500:
            print(f"prompt: {prompt[:500]}...")
        else:
            print(f"prompt: {prompt}")

        # 显示 target
        print(f"target: {example['target']}")

    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)

    # prompt 长度统计
    prompt_lengths = [len(ex["prompt"]) for ex in dataset["train"]]
    print(
        f"prompt 长度 - 最小: {min(prompt_lengths)}, 最大: {max(prompt_lengths)}, 平均: {sum(prompt_lengths) / len(prompt_lengths):.1f}"
    )

    # target 长度统计
    target_lengths = [len(ex["target"]) for ex in dataset["train"]]
    print(
        f"target 长度 - 最小: {min(target_lengths)}, 最大: {max(target_lengths)}, 平均: {sum(target_lengths) / len(target_lengths):.1f}"
    )


def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "raw_dataset_info.txt")

    print(f"正在将查询结果写入文件: {output_file}")

    # 重定向 stdout 到文件
    original_stdout = sys.stdout
    with open(output_file, "w", encoding="utf-8") as f:
        sys.stdout = f
        try:
            run_view_raw()
        finally:
            sys.stdout = original_stdout

    print("写入完成。")


if __name__ == "__main__":
    main()
