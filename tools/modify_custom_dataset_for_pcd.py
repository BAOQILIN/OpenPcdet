#!/usr/bin/env python3
"""
修改 custom_dataset.py 以支持 PCD 格式点云文件

使用方法：
    python tools/modify_custom_dataset_for_pcd.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def create_backup(file_path):
    """创建备份文件"""
    backup_path = file_path.with_suffix('.py.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy(file_path, backup_path)
        print(f"✓ 已创建备份: {backup_path}")
    else:
        print(f"✓ 备份已存在: {backup_path}")


def modify_custom_dataset():
    """修改 custom_dataset.py 以支持 PCD 格式"""

    dataset_file = ROOT_DIR / 'pcdet' / 'datasets' / 'custom' / 'custom_dataset.py'

    if not dataset_file.exists():
        print(f"✗ 文件不存在: {dataset_file}")
        return False

    # 创建备份
    create_backup(dataset_file)

    # 读取原文件
    with open(dataset_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已经修改过
    if '_load_pcd' in content:
        print("✓ 文件已经支持 PCD 格式，无需重复修改")
        return True

    # 定义新的 get_lidar 方法
    new_get_lidar = '''    def get_lidar(self, idx):
        """
        加载点云数据，支持多种格式：.pcd, .bin, .npy
        """
        # 按优先级尝试不同格式
        point_cloud_formats = ['.pcd', '.bin', '.npy']

        for fmt in point_cloud_formats:
            lidar_file = self.root_path / 'points' / (f'{idx}{fmt}')
            if lidar_file.exists():
                if fmt == '.pcd':
                    # 读取 PCD 文件
                    point_features = self._load_pcd(lidar_file)
                elif fmt == '.bin':
                    # 读取 BIN 文件（KITTI 格式）
                    point_features = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
                elif fmt == '.npy':
                    # 读取 NPY 文件
                    point_features = np.load(lidar_file)
                return point_features

        raise FileNotFoundError(f"Point cloud file not found for {idx} in formats {point_cloud_formats}")

    def _load_pcd(self, pcd_file):
        """
        加载 PCD 文件（支持 ASCII 和 binary 格式）

        Args:
            pcd_file: PCD 文件路径

        Returns:
            points: (N, 4) numpy 数组 [x, y, z, intensity]
        """
        try:
            # 方法 1：使用 open3d（如果已安装）
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(pcd_file))
            points_xyz = np.asarray(pcd.points, dtype=np.float32)

            # 尝试获取强度信息
            if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
                colors = np.asarray(pcd.colors)
                intensity = colors.mean(axis=1, keepdims=True).astype(np.float32)
            else:
                intensity = np.ones((points_xyz.shape[0], 1), dtype=np.float32)

            points = np.concatenate([points_xyz, intensity], axis=1)

        except ImportError:
            # 方法 2：手动解析 ASCII PCD（如果没有 open3d）
            points = self._parse_pcd_ascii(pcd_file)

        return points

    def _parse_pcd_ascii(self, pcd_file):
        """
        手动解析 ASCII 格式的 PCD 文件

        Args:
            pcd_file: PCD 文件路径

        Returns:
            points: (N, 4) numpy 数组 [x, y, z, intensity]
        """
        with open(pcd_file, 'r') as f:
            lines = f.readlines()

        # 解析头部
        header = {}
        data_start = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('DATA'):
                data_start = i + 1
                break
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    header[parts[0]] = parts[1:]

        # 获取字段信息
        fields = header.get('FIELDS', ['x', 'y', 'z'])

        # 解析点云数据
        points_list = []
        for line in lines[data_start:]:
            line = line.strip()
            if line and not line.startswith('#'):
                values = list(map(float, line.split()))
                points_list.append(values)

        points = np.array(points_list, dtype=np.float32)

        # 提取或构造 [x, y, z, intensity]
        if points.shape[1] >= 4:
            # 假设前 4 列是 x, y, z, intensity
            points_xyzi = points[:, :4]
        else:
            # 如果只有 x, y, z，添加默认强度
            points_xyzi = np.concatenate([
                points[:, :3],
                np.ones((points.shape[0], 1), dtype=np.float32)
            ], axis=1)

        return points_xyzi
'''

    # 查找原来的 get_lidar 方法
    import re

    # 匹配原来的 get_lidar 方法（从 def 到下一个 def 或类结束）
    pattern = r'(    def get_lidar\(self, idx\):.*?(?=\n    def |\n\nclass |\Z))'

    match = re.search(pattern, content, re.DOTALL)

    if match:
        old_method = match.group(1)
        # 替换为新方法
        new_content = content.replace(old_method, new_get_lidar)

        # 写入文件
        with open(dataset_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✓ 成功修改 {dataset_file}")
        print(f"✓ 现在支持的格式: .pcd, .bin, .npy")
        print(f"\n提示:")
        print(f"  1. 如果使用 open3d 加载 PCD，请安装: pip install open3d")
        print(f"  2. 如果不安装 open3d，将使用内置的 ASCII PCD 解析器")
        print(f"  3. 备份文件位于: {dataset_file.with_suffix('.py.backup')}")
        return True
    else:
        print("✗ 未找到 get_lidar 方法，无法修改")
        return False


def main():
    print("=" * 60)
    print("修改 custom_dataset.py 以支持 PCD 格式")
    print("=" * 60)
    print()

    success = modify_custom_dataset()

    if success:
        print("\n" + "=" * 60)
        print("修改完成！")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 将您的 PCD 文件放入 data/custom/points/ 目录")
        print("  2. 文件命名格式: 000000.pcd, 000001.pcd, ...")
        print("  3. 继续按照 user_data_training.md 的步骤进行")
    else:
        print("\n修改失败，请检查错误信息")
        sys.exit(1)


if __name__ == '__main__':
    main()
