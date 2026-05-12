# 使用自定义数据集训练 PointPillar 模型 - 完整指南

## 概述

本文档提供在 OpenPCDet 框架中使用自定义数据集训练 PointPillar 3D 目标检测模型的完整实施方案。PointPillar 是一种高效的基于柱状体素化的单阶段检测器，适合实时应用场景。

## 目标

- **输入**：原始点云数据（.npy 格式）及对应的 3D 边界框标注（.txt 格式）
- **输出**：训练好的模型检查点，可用于推理和评估
- **评估指标**：KITTI 风格的 AP（Average Precision）

---

## 阶段 1：数据准备

### 步骤 1：创建数据目录结构

在项目根目录下创建以下结构：

```bash
mkdir -p data/custom/ImageSets
mkdir -p data/custom/points
mkdir -p data/custom/labels
```

**最终目录结构**：
```
OpenPCDet/
├── data/
│   ├── custom/
│   │   ├── ImageSets/
│   │   │   ├── train.txt      # 训练集样本 ID 列表
│   │   │   └── val.txt        # 验证集样本 ID 列表
│   │   ├── points/
│   │   │   ├── 000000.npy     # 点云文件
│   │   │   ├── 000001.npy
│   │   │   └── ...
│   │   └── labels/
│   │       ├── 000000.txt     # 标注文件
│   │       ├── 000001.txt
│   │       └── ...
```

### 步骤 2：准备点云数据

**文件格式要求**：
- **格式**：NumPy 数组 (.npy)
- **数据形状**：(N, 4+) 其中 N 是点数
- **特征顺序**：[x, y, z, intensity, ...] 
- **坐标系统**：统一 LiDAR 坐标系（x-前，y-左，z-上）
- **命名规则**：6 位数字编号（000000.npy, 000001.npy, ...）

**转换示例代码**：
```python
import numpy as np

# 假设原始点云数据为 (N, 4) 数组
# 列顺序：x, y, z, intensity
points = load_your_point_cloud()  # 替换为实际加载函数

# 确保坐标系统正确（x-前，y-左，z-上）
# 如果需要坐标转换，在此处理

# 保存为 .npy 格式
np.save('data/custom/points/000000.npy', points.astype(np.float32))
```

### 步骤 3：准备标注数据

**文件格式要求**：
- **格式**：纯文本 (.txt)，每行一个目标
- **格式定义**：`x y z dx dy dz heading_angle category_name`
  - `x, y, z`：边界框中心坐标（LiDAR 坐标系）
  - `dx, dy, dz`：边界框尺寸（长、宽、高）
  - `heading_angle`：绕 z 轴的旋转角度（弧度）
  - `category_name`：类别名称（如 Vehicle, Pedestrian, Cyclist）
- **命名规则**：与对应点云文件同名（000000.txt, 000001.txt, ...）

**标注示例**：
```
# data/custom/labels/000000.txt
1.50 1.46 0.10 5.12 1.85 4.13 1.56 Vehicle
5.54 0.57 0.41 1.08 0.74 1.95 1.57 Pedestrian
-2.30 3.20 0.35 1.76 0.60 1.73 0.78 Cyclist
```

**转换示例代码**：
```python
def convert_box_to_openpcdet_format(boxes, labels):
    """
    将边界框转换为 OpenPCDet 格式
    
    Args:
        boxes: (N, 7+) 数组，格式可能因数据集而异
        labels: (N,) 类别标签数组
    
    Returns:
        格式化的标注字符串列表
    """
    lines = []
    for box, label in zip(boxes, labels):
        # 确保边界框格式为 [x, y, z, dx, dy, dz, heading]
        x, y, z, dx, dy, dz, heading = box[:7]
        
        # 类别名称映射
        category_map = {0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist'}
        category_name = category_map.get(label, 'Unknown')
        
        line = f"{x:.2f} {y:.2f} {z:.2f} {dx:.2f} {dy:.2f} {dz:.2f} {heading:.2f} {category_name}\n"
        lines.append(line)
    
    return lines

# 保存标注文件
with open('data/custom/labels/000000.txt', 'w') as f:
    f.writelines(lines)
```

### 步骤 4：创建数据集划分文件

**创建示例代码**：
```python
import os
import random
from pathlib import Path

# 获取所有样本 ID
points_dir = Path('data/custom/points')
sample_ids = sorted([f.stem for f in points_dir.glob('*.npy')])

# 随机划分（设置随机种子以保证可重复性）
random.seed(42)
random.shuffle(sample_ids)

# 80% 训练，20% 验证
split_idx = int(len(sample_ids) * 0.8)
train_ids = sample_ids[:split_idx]
val_ids = sample_ids[split_idx:]

# 保存划分文件
with open('data/custom/ImageSets/train.txt', 'w') as f:
    f.write('\n'.join(train_ids))

with open('data/custom/ImageSets/val.txt', 'w') as f:
    f.write('\n'.join(val_ids))

print(f"训练集样本数: {len(train_ids)}")
print(f"验证集样本数: {len(val_ids)}")
```

---

## 阶段 2：配置文件调整

### 步骤 5：调整数据集配置文件

**文件位置**：`tools/cfgs/dataset_configs/custom_dataset.yaml`

**需要调整的关键参数**：

#### a. 类别名称和映射
```yaml
# 根据实际数据集修改类别名称
CLASS_NAMES: ['Vehicle', 'Pedestrian', 'Cyclist']

# KITTI 评估映射（用于评估指标计算）
MAP_CLASS_TO_KITTI: {
    'Vehicle': 'Car',
    'Pedestrian': 'Pedestrian',
    'Cyclist': 'Cyclist',
}
```

#### b. 点云范围
```yaml
# 根据数据集的实际点云分布范围调整
# 格式: [x_min, y_min, z_min, x_max, y_max, z_max]
# 对于 PointPillar，需要满足：
# - (x_max - x_min) / voxel_size_x 是 16 的倍数
# - (y_max - y_min) / voxel_size_y 是 16 的倍数
POINT_CLOUD_RANGE: [-75.2, -75.2, -2, 75.2, 75.2, 4]

# 计算验证：
# x 方向: (75.2 - (-75.2)) / 0.16 = 940 (16 的倍数 ✓)
# y 方向: (75.2 - (-75.2)) / 0.16 = 940 (16 的倍数 ✓)
```

#### c. 点云特征
```yaml
POINT_FEATURE_ENCODING: {
    encoding_type: absolute_coordinates_encoding,
    # 使用的特征（用于模型输入）
    used_feature_list: ['x', 'y', 'z', 'intensity'],
    # 原始点云包含的特征
    src_feature_list: ['x', 'y', 'z', 'intensity'],
}
```

#### d. 数据增强配置
```yaml
DATA_AUGMENTOR:
    DISABLE_AUG_LIST: ['placeholder']
    AUG_CONFIG_LIST:
        - NAME: gt_sampling
          USE_ROAD_PLANE: False
          DB_INFO_PATH:
              - custom_dbinfos_train.pkl
          PREPARE: {
             # 根据实际类别调整最小点数过滤
             filter_by_min_points: ['Vehicle:5', 'Pedestrian:5', 'Cyclist:5'],
          }
          # 每个类别采样的目标数量
          SAMPLE_GROUPS: ['Vehicle:20', 'Pedestrian:15', 'Cyclist:15']
          NUM_POINT_FEATURES: 4
          DATABASE_WITH_FAKELIDAR: False
          REMOVE_EXTRA_WIDTH: [0.0, 0.0, 0.0]
          LIMIT_WHOLE_SCENE: True

        - NAME: random_world_flip
          ALONG_AXIS_LIST: ['x', 'y']

        - NAME: random_world_rotation
          WORLD_ROT_ANGLE: [-0.78539816, 0.78539816]  # ±45度

        - NAME: random_world_scaling
          WORLD_SCALE_RANGE: [0.95, 1.05]
```

#### e. 数据处理配置
```yaml
DATA_PROCESSOR:
    - NAME: mask_points_and_boxes_outside_range
      REMOVE_OUTSIDE_BOXES: True

    - NAME: shuffle_points
      SHUFFLE_ENABLED: {
        'train': True,
        'test': False
      }

    - NAME: transform_points_to_voxels
      # PointPillar 的柱状体素化参数
      VOXEL_SIZE: [0.16, 0.16, 4]  # [x, y, z] 体素尺寸
      MAX_POINTS_PER_VOXEL: 32     # 每个体素最多包含的点数
      MAX_NUMBER_OF_VOXELS: {
        'train': 16000,
        'test': 40000
      }
```

### 步骤 6：创建 PointPillar 模型配置文件

**文件位置**：`tools/cfgs/custom_models/pointpillar.yaml`

首先创建目录：
```bash
mkdir -p tools/cfgs/custom_models
```

**完整配置文件内容**：
```yaml
CLASS_NAMES: ['Vehicle', 'Pedestrian', 'Cyclist']

DATA_CONFIG: 
    _BASE_CONFIG_: cfgs/dataset_configs/custom_dataset.yaml
    POINT_CLOUD_RANGE: [-75.2, -75.2, -2, 75.2, 75.2, 4]
    
    DATA_PROCESSOR:
        - NAME: mask_points_and_boxes_outside_range
          REMOVE_OUTSIDE_BOXES: True

        - NAME: shuffle_points
          SHUFFLE_ENABLED: {
            'train': True,
            'test': False
          }

        - NAME: transform_points_to_voxels
          VOXEL_SIZE: [0.16, 0.16, 4]
          MAX_POINTS_PER_VOXEL: 32
          MAX_NUMBER_OF_VOXELS: {
            'train': 16000,
            'test': 40000
          }

MODEL:
    NAME: PointPillar

    VFE:
        NAME: PillarVFE
        WITH_DISTANCE: False
        USE_ABSLOTE_XYZ: True
        USE_NORM: True
        NUM_FILTERS: [64]

    MAP_TO_BEV:
        NAME: PointPillarScatter
        NUM_BEV_FEATURES: 64

    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [3, 5, 5]
        LAYER_STRIDES: [2, 2, 2]
        NUM_FILTERS: [64, 128, 256]
        UPSAMPLE_STRIDES: [1, 2, 4]
        NUM_UPSAMPLE_FILTERS: [128, 128, 128]

    DENSE_HEAD:
        NAME: AnchorHeadSingle
        CLASS_AGNOSTIC: False

        USE_DIRECTION_CLASSIFIER: True
        DIR_OFFSET: 0.78539
        DIR_LIMIT_OFFSET: 0.0
        NUM_DIR_BINS: 2

        ANCHOR_GENERATOR_CONFIG: [
            {
                'class_name': 'Vehicle',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Pedestrian',
                'anchor_sizes': [[0.8, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': 'Cyclist',
                'anchor_sizes': [[1.76, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }
        ]

        TARGET_ASSIGNER_CONFIG:
            NAME: AxisAlignedTargetAssigner
            POS_FRACTION: -1.0
            SAMPLE_SIZE: 512
            NORM_BY_NUM_EXAMPLES: False
            MATCH_HEIGHT: False
            BOX_CODER: ResidualCoder

        LOSS_CONFIG:
            LOSS_WEIGHTS: {
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'dir_weight': 0.2,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        SCORE_THRESH: 0.1
        OUTPUT_RAW_SCORE: False

        EVAL_METRIC: kitti

        NMS_CONFIG:
            MULTI_CLASSES_NMS: False
            NMS_TYPE: nms_gpu
            NMS_THRESH: 0.01
            NMS_PRE_MAXSIZE: 4096
            NMS_POST_MAXSIZE: 500

OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 4
    NUM_EPOCHS: 80

    OPTIMIZER: adam_onecycle
    LR: 0.003
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9

    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001

    LR_WARMUP: False
    WARMUP_EPOCH: 1

    GRAD_NORM_CLIP: 10
```

**锚框尺寸调整指南**：

使用以下脚本统计数据集中各类别目标的平均尺寸：
```python
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_box_sizes(labels_dir):
    """分析数据集中各类别目标的尺寸分布"""
    labels_dir = Path(labels_dir)
    class_boxes = defaultdict(list)
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                
                # 解析边界框
                x, y, z, dx, dy, dz, heading = map(float, parts[:7])
                class_name = parts[7]
                
                class_boxes[class_name].append([dx, dy, dz])
    
    # 计算每个类别的平均尺寸
    print("各类别目标尺寸统计：")
    print("-" * 60)
    for class_name, boxes in class_boxes.items():
        boxes = np.array(boxes)
        mean_size = boxes.mean(axis=0)
        std_size = boxes.std(axis=0)
        
        print(f"\n{class_name}:")
        print(f"  样本数量: {len(boxes)}")
        print(f"  平均尺寸 [长, 宽, 高]: [{mean_size[0]:.2f}, {mean_size[1]:.2f}, {mean_size[2]:.2f}]")
        print(f"  标准差: [{std_size[0]:.2f}, {std_size[1]:.2f}, {std_size[2]:.2f}]")
        print(f"  建议锚框尺寸: [[{mean_size[0]:.2f}, {mean_size[1]:.2f}, {mean_size[2]:.2f}]]")

# 运行分析
analyze_box_sizes('data/custom/labels')
```

---

## 阶段 3：数据预处理

### 步骤 7：修改数据集类中的类别名称

**文件位置**：`pcdet/datasets/custom/custom_dataset.py`（第 278-283 行）

```python
if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            # 修改此处的类别名称，与配置文件保持一致
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'custom',
            save_path=ROOT_DIR / 'data' / 'custom',
        )
```

**注意**：类别名称必须与配置文件中的 `CLASS_NAMES` 完全一致。

### 步骤 8：生成数据信息文件和 GT 数据库

**执行命令**：
```bash
cd /home/bql/ARS/OpenPcdetss0509/OpenPcdetss

python -m pcdet.datasets.custom.custom_dataset create_custom_infos \
    tools/cfgs/dataset_configs/custom_dataset.yaml
```

**预期输出**：
```
------------------------Start to generate data infos------------------------
train sample_idx: 000000
train sample_idx: 000001
...
Custom info train file is saved to /path/to/data/custom/custom_infos_train.pkl
val sample_idx: 000100
...
Custom info train file is saved to /path/to/data/custom/custom_infos_val.pkl
------------------------Start create groundtruth database for data augmentation------------------------
gt_database sample: 1/XXX
...
Database Vehicle: XXX
Database Pedestrian: XXX
Database Cyclist: XXX
------------------------Data preparation done------------------------
```

**生成的文件**：
- `data/custom/custom_infos_train.pkl` - 训练集信息文件
- `data/custom/custom_infos_val.pkl` - 验证集信息文件
- `data/custom/custom_dbinfos_train.pkl` - GT 数据库索引文件
- `data/custom/gt_database/` - GT 数据库目录

---

## 阶段 4：模型训练

### 步骤 9：单 GPU 训练（测试配置）

**执行命令**：
```bash
# 基础训练命令
python tools/train.py \
    --cfg_file tools/cfgs/custom_models/pointpillar.yaml \
    --batch_size 4 \
    --epochs 5 \
    --workers 4 \
    --extra_tag test_run

# 带有更多选项的命令
python tools/train.py \
    --cfg_file tools/cfgs/custom_models/pointpillar.yaml \
    --batch_size 4 \
    --epochs 5 \
    --workers 4 \
    --extra_tag test_run \
    --fix_random_seed \
    --use_tqdm_to_record
```

**预期输出**：
- 训练日志：`output/custom_models/pointpillar/test_run/log_train_*.txt`
- 检查点：`output/custom_models/pointpillar/test_run/ckpt/`
- TensorBoard：`output/custom_models/pointpillar/test_run/tensorboard/`

### 步骤 10：多 GPU 分布式训练（正式训练）

**执行命令**：
```bash
# 使用 4 个 GPU 训练
bash scripts/dist_train.sh 4 \
    --cfg_file tools/cfgs/custom_models/pointpillar.yaml \
    --batch_size 4 \
    --epochs 80 \
    --workers 8 \
    --extra_tag final_model \
    --sync_bn

# 从检查点恢复训练
bash scripts/dist_train.sh 4 \
    --cfg_file tools/cfgs/custom_models/pointpillar.yaml \
    --batch_size 4 \
    --epochs 80 \
    --workers 8 \
    --extra_tag final_model \
    --ckpt output/custom_models/pointpillar/final_model/ckpt/checkpoint_epoch_40.pth
```

**训练参数说明**：
- `--batch_size`：每个 GPU 的批次大小
- `--epochs`：训练轮数（建议 80）
- `--workers`：数据加载线程数
- `--extra_tag`：实验标签
- `--sync_bn`：同步批归一化（多 GPU 必需）
- `--fix_random_seed`：固定随机种子（可重复性）

### 步骤 11：训练监控

**使用 TensorBoard 监控**：
```bash
tensorboard --logdir=output/custom_models/pointpillar/final_model/tensorboard
```

在浏览器中打开 `http://localhost:6006` 查看：
- 训练损失曲线
- 验证指标
- 学习率变化

**查看训练日志**：
```bash
tail -f output/custom_models/pointpillar/final_model/log_train_*.txt
```

---

## 阶段 5：模型评估与测试

### 步骤 12：评估训练好的模型

**单个检查点评估**：
```bash
python tools/test.py \
    --cfg_file tools/cfgs/custom_models/pointpillar.yaml \
    --ckpt output/custom_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth
```

**评估所有检查点**：
```bash
python tools/test.py \
    --cfg_file tools/cfgs/custom_models/pointpillar.yaml \
    --eval_all \
    --ckpt_dir output/custom_models/pointpillar/final_model/ckpt
```

**多 GPU 评估**：
```bash
bash scripts/dist_test.sh 4 \
    --cfg_file tools/cfgs/custom_models/pointpillar.yaml \
    --ckpt output/custom_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth
```

**测量推理时间**：
```bash
python tools/test.py \
    --cfg_file tools/cfgs/custom_models/pointpillar.yaml \
    --ckpt output/custom_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth \
    --infer_time
```

### 步骤 13：可视化检测结果

**运行 Demo**：
```bash
python tools/demo.py \
    --cfg_file tools/cfgs/custom_models/pointpillar.yaml \
    --ckpt output/custom_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth \
    --data_path data/custom/points/000000.npy
```

**可视化要求**：
- 安装 Open3D：`pip install open3d`
- 或安装 Mayavi：`pip install mayavi`

---

## 常见问题排查

### 1. 数据加载错误

**问题**：`FileNotFoundError` 或 `KeyError`

**解决方案**：
- 检查数据目录结构是否正确
- 检查文件命名是否符合规范（6 位数字）
- 检查 `ImageSets/train.txt` 和 `val.txt` 中的样本 ID 是否存在

### 2. 坐标系统错误

**问题**：训练损失不收敛或检测结果异常

**解决方案**：
- 确认点云坐标系统为 x-前、y-左、z-上
- 检查边界框的 heading 角度是否正确（弧度制）
- 使用可视化工具检查标注是否正确对齐

### 3. 锚框匹配问题

**问题**：某些类别的 AP 很低

**解决方案**：
- 运行锚框尺寸分析脚本
- 根据统计结果调整 `anchor_sizes`
- 调整 `matched_threshold` 和 `unmatched_threshold`

### 4. 内存不足

**问题**：`CUDA out of memory`

**解决方案**：
- 减小 `--batch_size`
- 减小 `MAX_NUMBER_OF_VOXELS`
- 使用梯度累积（修改训练脚本）

### 5. 点云范围设置错误

**问题**：训练时报错 "feature map size must be divisible by 16"

**解决方案**：
- 确保 `(x_max - x_min) / voxel_size_x` 是 16 的倍数
- 确保 `(y_max - y_min) / voxel_size_y` 是 16 的倍数
- 调整 `POINT_CLOUD_RANGE` 或 `VOXEL_SIZE`

---

## 性能优化建议

### 1. 数据增强

- 根据数据集特点调整增强参数
- 如果数据量充足，可以减少 GT 采样数量
- 如果数据量不足，增加旋转和缩放范围

### 2. 训练策略

- 使用余弦退火学习率调度器
- 启用混合精度训练：`--use_amp`
- 使用更大的批次大小（如果 GPU 内存允许）

### 3. 模型调优

- 调整锚框尺寸以匹配数据集
- 调整损失权重以平衡不同类别
- 尝试不同的 NMS 阈值

---

## 总结

完成以上步骤后，您将获得：

1. ✅ 格式正确的自定义数据集
2. ✅ 配置好的 PointPillar 模型
3. ✅ 训练好的模型检查点
4. ✅ 评估结果和性能指标

**下一步**：
- 在测试集上评估模型性能
- 使用 Demo 脚本进行可视化验证
- 根据评估结果调整超参数
- 部署模型到实际应用场景

**参考资源**：
- OpenPCDet 官方文档：https://github.com/open-mmlab/OpenPCDet
- KITTI 数据集格式：http://www.cvlibs.net/datasets/kitti/
- PointPillar 论文：https://arxiv.org/abs/1812.05784
