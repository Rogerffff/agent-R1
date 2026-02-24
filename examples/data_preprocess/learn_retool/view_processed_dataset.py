"""
查看处理后的 ReTool 数据集

该脚本展示如何加载和查看处理后的 Parquet 格式数据集。

使用方法:
    source .venv-retool/bin/activate
    python examples/data_preprocess/learn_retool/view_processed_dataset.py

前置条件:
    需要先运行数据处理脚本生成 parquet 文件:
    python examples/data_preprocess/retool.py --local_dir ./data/retool
"""

import os
import json
import sys
import pyarrow.parquet as pq


def view_parquet_file(file_path: str, name: str):
    """查看单个 parquet 文件的内容"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        print("请先运行数据处理脚本: python examples/data_preprocess/retool.py --local_dir ./data/retool")
        return None

    table = pq.read_table(file_path)

    print(f"\n{'=' * 60}")
    print(f"{name} - {file_path}")
    print("=" * 60)
    print(f"记录数: {table.num_rows}")
    print(f"字段: {table.column_names}")

    return table


def show_example(table, index: int):
    """显示指定索引的数据示例"""
    print(f"\n--- 第 {index + 1} 条数据 ---")

    for col_name in table.column_names:
        value = table[col_name][index].as_py()

        print(f"\n{col_name}:")
        if isinstance(value, (dict, list)):
            # 对于嵌套结构，格式化显示
            formatted = json.dumps(value, ensure_ascii=False, indent=2)
            # # 截断过长的内容
            # if len(formatted) > 1000:
            #     formatted = formatted[:1000] + "\n  ..."
            print(formatted)
        elif isinstance(value, str):
            # 对于字符串，截断显示
            if len(value) > 300:
                print(f"  {value[:300]}...")
            else:
                print(f"  {value}")
        else:
            print(f"  {value}")


def run_analysis():
    """执行数据查看和分析逻辑"""
    # 数据目录
    data_dir = "./data/retool"

    print("=" * 60)
    print("ReTool 处理后数据集查看工具")
    print("=" * 60)

    # 查看训练集
    train_path = os.path.join(data_dir, "train.parquet")
    train_table = view_parquet_file(train_path, "训练集")

    # 查看测试集
    test_path = os.path.join(data_dir, "test.parquet")
    test_table = view_parquet_file(test_path, "测试集")

    if train_table is None:
        return

    # 显示数据结构说明
    print("\n" + "=" * 60)
    print("数据结构说明")
    print("=" * 60)
    print("""
字段说明:
  - prompt: 消息列表，包含 role 和 content
    - role: "user"
    - content: 包含完整指令和问题的文本
  - target: 原始答案
  - data_source: 数据来源标识
  - ability: 能力分类（可为空）
  - reward_model: 奖励模型配置
    - style: "rule" (规则评分)
    - ground_truth: 正确答案
  - extra_info: 额外信息
    - split: 数据分割 (train/test)
    - index: 数据索引
""")

    # 显示训练集示例
    print("\n" + "=" * 60)
    print("训练集数据示例")
    print("=" * 60)
    show_example(train_table, 0)

    # 显示测试集示例
    if test_table:
        print("\n" + "=" * 60)
        print("测试集数据示例")
        print("=" * 60)
        show_example(test_table, 0)

    # 使用 pandas 进行更多分析
    print("\n" + "=" * 60)
    print("使用 Pandas 进行数据分析")
    print("=" * 60)

    df = train_table.to_pandas()

    print(f"\nDataFrame 形状: {df.shape}")
    print(f"\n列类型:\n{df.dtypes}")

    # 提取 reward_model 中的 ground_truth 进行分析
    df['ground_truth'] = df['reward_model'].apply(lambda x: x.get('ground_truth', ''))
    gt_lengths = df['ground_truth'].str.len()
    print(f"\nground_truth 长度统计:")
    print(f"  最小: {gt_lengths.min()}")
    print(f"  最大: {gt_lengths.max()}")
    print(f"  平均: {gt_lengths.mean():.1f}")


def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "processed_dataset_info.txt")
    
    print(f"正在将查询结果写入文件: {output_file}")
    
    # 重定向 stdout 到文件
    original_stdout = sys.stdout
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        try:
            run_analysis()
        finally:
            sys.stdout = original_stdout
            
    print("写入完成。")


if __name__ == "__main__":
    main()
