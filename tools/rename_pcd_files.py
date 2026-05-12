#!/usr/bin/env python3
"""
批量重命名 PCD 文件为 OpenPCDet 要求的格式

使用方法：
    python tools/rename_pcd_files.py <source_dir> <target_dir>

示例：
    python tools/rename_pcd_files.py /path/to/your/pcd/files data/custom/points
"""

import sys
import shutil
from pathlib import Path


def rename_pcd_files(source_dir, target_dir, start_index=0):
    """
    将 PCD 文件重命名为 6 位数字编号

    Args:
        source_dir: 原始 PCD 文件目录
        target_dir: 目标目录（data/custom/points）
        start_index: 起始索引（默认 0）
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        print(f"✗ 源目录不存在: {source_dir}")
        return False

    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 PCD 文件并排序
    pcd_files = sorted(source_dir.glob('*.pcd'))

    if not pcd_files:
        print(f"✗ 在 {source_dir} 中未找到 PCD 文件")
        return False

    print(f"找到 {len(pcd_files)} 个 PCD 文件")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"起始索引: {start_index}")
    print()

    # 复制并重命名文件
    for idx, pcd_file in enumerate(pcd_files):
        new_idx = start_index + idx
        target_file = target_dir / f"{new_idx:06d}.pcd"

        # 复制文件
        shutil.copy(pcd_file, target_file)

        # 显示进度
        print(f"[{idx + 1}/{len(pcd_files)}] {pcd_file.name} -> {target_file.name}")

    print()
    print(f"✓ 成功处理 {len(pcd_files)} 个文件")
    print(f"✓ 文件保存在: {target_dir}")

    return True


def create_sample_id_list(points_dir, output_dir):
    """
    根据 points 目录中的文件创建样本 ID 列表

    Args:
        points_dir: points 目录路径
        output_dir: ImageSets 目录路径
    """
    points_dir = Path(points_dir)
    output_dir = Path(output_dir)

    # 获取所有点云文件
    pcd_files = sorted(points_dir.glob('*.pcd'))
    bin_files = sorted(points_dir.glob('*.bin'))
    npy_files = sorted(points_dir.glob('*.npy'))

    # 合并所有文件并去重
    all_files = pcd_files + bin_files + npy_files
    sample_ids = sorted(set([f.stem for f in all_files]))

    if not sample_ids:
        print(f"✗ 在 {points_dir} 中未找到点云文件")
        return False

    # 创建 ImageSets 目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 划分训练集和验证集（80% 训练，20% 验证）
    import random
    random.seed(42)
    random.shuffle(sample_ids)

    split_idx = int(len(sample_ids) * 0.8)
    train_ids = sample_ids[:split_idx]
    val_ids = sample_ids[split_idx:]

    # 保存训练集列表
    train_file = output_dir / 'train.txt'
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_ids))
    print(f"✓ 训练集列表: {train_file} ({len(train_ids)} 个样本)")

    # 保存验证集列表
    val_file = output_dir / 'val.txt'
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_ids))
    print(f"✓ 验证集列表: {val_file} ({len(val_ids)} 个样本)")

    return True


def main():
    print("=" * 60)
    print("PCD 文件批量重命名工具")
    print("=" * 60)
    print()

    if len(sys.argv) < 3:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <source_dir> <target_dir> [start_index]")
        print()
        print("示例:")
        print(f"  python {sys.argv[0]} /path/to/your/pcd/files data/custom/points")
        print(f"  python {sys.argv[0]} /path/to/your/pcd/files data/custom/points 100")
        print()
        sys.exit(1)

    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    start_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # 重命名文件
    success = rename_pcd_files(source_dir, target_dir, start_index)

    if not success:
        sys.exit(1)

    # 询问是否创建样本 ID 列表
    print()
    response = input("是否自动创建 train.txt 和 val.txt？(y/n): ").strip().lower()

    if response == 'y':
        print()
        imagesets_dir = Path(target_dir).parent / 'ImageSets'
        create_sample_id_list(target_dir, imagesets_dir)

    print()
    print("=" * 60)
    print("完成！")
    print("=" * 60)
    print()
    print("下一步:")
    print("  1. 检查 data/custom/points/ 目录中的文件")
    print("  2. 准备对应的标注文件到 data/custom/labels/")
    print("  3. 继续按照 user_data_training.md 的步骤进行")


if __name__ == '__main__':
    main()
