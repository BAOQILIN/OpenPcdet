# 使用 PCD 格式训练 PointPillar - 快速开始

## 核心要点

**您不需要转换 PCD 为 NPY！** 

就像 KITTI 数据集直接使用 `.bin` 文件一样，我们可以修改 `custom_dataset.py` 直接支持 `.pcd` 格式。

## 快速开始（3 步）

### 1. 修改 custom_dataset.py 支持 PCD

```bash
python tools/modify_custom_dataset_for_pcd.py
```

这会自动修改代码支持 `.pcd`, `.bin`, `.npy` 三种格式。

### 2. 准备 PCD 文件

```bash
# 批量重命名并复制 PCD 文件
python tools/rename_pcd_files.py /path/to/your/pcd/files data/custom/points
```

这会：
- 将 PCD 文件复制到 `data/custom/points/`
- 重命名为 `000000.pcd`, `000001.pcd`, ...
- 自动创建 `train.txt` 和 `val.txt`

### 3. 准备标注文件

将标注文件放入 `data/custom/labels/`，格式：

```
# 000000.txt
x y z dx dy dz heading_angle category_name
1.50 1.46 0.10 5.12 1.85 4.13 1.56 Vehicle
```

## 目录结构

```
data/custom/
├── ImageSets/
│   ├── train.txt          # 自动生成
│   └── val.txt            # 自动生成
├── points/
│   ├── 000000.pcd         # 您的 PCD 文件
│   ├── 000001.pcd
│   └── ...
└── labels/
    ├── 000000.txt         # 您的标注文件
    ├── 000001.txt
    └── ...
```

## 后续步骤

完成上述 3 步后，继续按照 `user_data_training.md` 的以下步骤：

1. **阶段 2**：配置文件调整
2. **阶段 3**：数据预处理（生成 info 文件）
3. **阶段 4**：模型训练
4. **阶段 5**：模型评估

## 常见问题

### Q: 需要安装 open3d 吗？

A: 不是必须的。代码支持两种方式：
- 如果安装了 open3d：`pip install open3d`（推荐，支持 binary PCD）
- 如果没有 open3d：使用内置的 ASCII PCD 解析器

### Q: PCD 文件的坐标系统需要转换吗？

A: 需要确保坐标系统为：**x-前，y-左，z-上**

如果您的 PCD 使用不同的坐标系统，需要在 `_load_pcd` 方法中添加转换代码。

### Q: 如何验证 PCD 文件加载正确？

A: 使用可视化脚本：

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载点云
import sys
sys.path.append('.')
from pcdet.datasets.custom.custom_dataset import CustomDataset
import yaml
from easydict import EasyDict

cfg = EasyDict(yaml.safe_load(open('tools/cfgs/dataset_configs/custom_dataset.yaml')))
dataset = CustomDataset(cfg, class_names=['Vehicle'], training=False, root_path='data/custom')

# 测试加载
points = dataset._load_pcd('data/custom/points/000000.pcd')

# 可视化（鸟瞰图）
plt.figure(figsize=(10, 10))
plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.1, cmap='viridis')
plt.colorbar(label='z (height)')
plt.xlabel('x (forward)')
plt.ylabel('y (left)')
plt.title('Point Cloud Bird Eye View')
plt.axis('equal')
plt.grid(True)
plt.show()

print(f"点云统计:")
print(f"  总点数: {points.shape[0]}")
print(f"  x 范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
print(f"  y 范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
print(f"  z 范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
```

## 与 KITTI 的对比

| 数据集 | 点云格式 | 加载方式 |
|--------|---------|---------|
| KITTI | `.bin` | `np.fromfile(...).reshape(-1, 4)` |
| Custom (原始) | `.npy` | `np.load(...)` |
| Custom (修改后) | `.pcd` / `.bin` / `.npy` | 自动检测格式 |

## 脚本说明

### modify_custom_dataset_for_pcd.py

- **功能**：修改 `custom_dataset.py` 支持多种点云格式
- **备份**：自动创建 `.backup` 备份文件
- **安全**：检查是否已修改，避免重复修改

### rename_pcd_files.py

- **功能**：批量重命名 PCD 文件
- **选项**：支持自定义起始索引
- **自动化**：可选自动创建 train.txt 和 val.txt

## 完整文档

详细步骤请参考：`user_data_training.md`
