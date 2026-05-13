# 使用自定义数据集训练 PointPillar 模型 - 完整指南

## 概述

本文档提供在 OpenPCDet 框架中使用自定义数据集训练 PointPillar 3D 目标检测模型的完整实施方案。PointPillar 是一种高效的基于柱状体素化的单阶段检测器，适合实时应用场景。

## 目标

- **输入**：原始点云数据（.pcd 格式,ascii）及对应的 3D 边界框标注（.json 格式）
- **输出**：训练好的模型检查点，可用于推理和评估
- **评估指标**：KITTI 风格的 AP（Average Precision）

---

## 说明
原始数据在/media/hirain/87e9cd68-1278-4bae-8d73-a0eb5d1ac165/home/jac/ars_train_data/origin_type目录下,内容如下：
├── 000000.json
├── 000000.pcd
├── 000005.json
├── 000005.pcd
├── 000010.json
├── 000010.pcd
├── 000015.json
├── 000015.pcd
├── 000020.json


**PCD 文件格式说明**：

典型的 ASCII PCD 文件格式：
```
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 1000
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 1000
DATA ascii
0.1 0.2 0.3 0.5
0.4 0.5 0.6 0.8
...
```
**坐标系统**：
   统一 LiDAR / 车辆坐标系（x-前，y-左，z-上）
   原始数据中的pcd已经转换到了车辆坐标系，所以只需要拷贝新的目录下就行

**原始数据中的标签json格式说明**
```
{
    "data_path": "",
    "file_name": "000000",
    "sync_time": 0.0,
    "object3D": {
        "lidar_main": {
            "annotation": [
                {
                    "acceleration": [
                        0,
                        0,
                        0
                    ],
                    "alpha": 0,
                    "cluster_id": 0,
                    "dimension": [
                        3.894,
                        2.118,
                        1.525
                    ],
                    "location_image": [
                        [
                            -1000,
                            -1000
                        ],
                        [
                            -1000,
                            -1000
                        ],
                        [
                            -1000,
                            -1000
                        ]
                    ],
                    "occluded": -1,
                    "position": [
                        44.8461,
                        13.6404,
                        -0.2534999
                    ],
                    "rotation": [
                        -1000,
                        -1000,
                        -0.0956855
                    ],
                    "score": 1.0,
                    "track_id": 1000,
                    "truncated": -1,
                    "type": 3,
                    "velocity": [
                        0,
                        0,
                        0
                    ]
                },
                {
                    "acceleration": [
                        0,
                        0,
                        0
                    ],
                    "alpha": 0,
                    "cluster_id": 0,
                    "dimension": [
                        4.17,
                        1.95,
                        1.36
                    ],
                    "location_image": [
                        [
                            -1000,
                            -1000
                        ],
                        [
                            -1000,
                            -1000
                        ],
                        [
                            -1000,
                            -1000
                        ]
                    ],
                    "occluded": -1,
                    "position": [
                        6.25,
                        9.34,
                        0.83
                    ],
                    "rotation": [
                        -1000,
                        -1000,
                        -0.0745256
                    ],
                    "score": 1.0,
                    "track_id": 1000,
                    "truncated": -1,
                    "type": 3,
                    "velocity": [
                        0,
                        0,
                        0
                    ]
                }
            ],
            "seq_id": 0,
            "time_stamp": 0.0
        }
    }
}
```
**转换之后的标签格式说明**
模型最终训练只是里面的一部分内容，annotation字段下的就是每一个目标的属性，但是训练的时候只需要三维框的尺寸（dimension）、中心点坐标（position）、类别（type）、和角度（rotation的第三个元素）。type有如下几类：'Obstacle', 'Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle'，分别对应0-5. 所以你在处理数据的时候需要从中提取这些数据，保存为.txt文本。
- **格式**：纯文本 (.txt)，每行一个目标
- **格式定义**：`type dimesion(x y z) position(x y z) heading`
  - `position(x, y, z)`：边界框中心坐标（LiDAR 坐标系）
  - `dimension(x, y, z)`：边界框尺寸（长、宽、高）
  - `heading`：绕 z 轴的旋转角度（弧度）
  - `type`：类别名称（如 Vehicle, Pedestrian, Cyclist）
- **命名规则**：与对应点云文件同名（000000.json, 000001.txt, ...）
txt中的排布示例：
Car 3.894 2.118 1.525 44.8461 13.6404 -0.2534999  -0.0956855
Car 4.17  1.95  1.36  6.25    9.34    0.83        -0.0745256

**最终目录结构**：
```
ars_train_data
├── data/
│   ├── ars/
│   │   ├── ImageSets/
│   │   │   ├── train.txt      # 训练集样本 ID 列表
│   │   │   └── val.txt        # 验证集样本 ID 列表
│   │   ├── points/
│   │   │   ├── 000000.pcd     # 点云文件
│   │   │   ├── 000001.pcd
│   │   │   └── ...
│   │   └── labels/
│   │       ├── 000000.txt     # 标注文件
│   │       ├── 000001.txt
│   │       └── ...
├── origin_type
    ├── 000000.json
    ├── 000000.pcd
    ├── 000005.json
    ├── 000005.pcd
    ├── 000010.json
    ├── 000010.pcd
    ├── 000015.json
    ├── 000015.pcd
    ├── 000020.json
```

## 具体步骤或要求
### 步骤 1：数据预处理数据脚本撰写

在这个工程的pcdet/datasets/ars下建立一个ars_dataset.py脚本，用于处理数据集，具体功能可以参考kitti的。我的数据中不区分训练集和验证集，验证和训练是同样的数据。处理后的数据集目录结构符合前面说的的`最终目录结构`要求，且为了快速加载，也可以生成一些.pkl文件。


**执行命令**：
```bash
cd /home/bql/ARS/OpenPcdetss0509/OpenPcdetss

python -m pcdet.datasets.ars.ars_dataset create_ars_infos \
    tools/cfgs/dataset_configs/ars_dataset.yaml
```

**预期输出**：
```
------------------------Start to generate data infos------------------------
train sample_idx: 000000
train sample_idx: 000001
...
Custom info train file is saved to /path/to/data/ars/ars_infos_train.pkl
val sample_idx: 000100
...
Custom info train file is saved to /path/to/data/ars/ars_infos_val.pkl
------------------------Start create groundtruth database for data augmentation------------------------
gt_database sample: 1/XXX
...
Database Vehicle: XXX
Database Pedestrian: XXX
Database Cyclist: XXX
------------------------Data preparation done------------------------
```

**生成的文件**：
- `data/ars/ars_infos_train.pkl` - 训练集信息文件
- `data/ars/ars_infos_val.pkl` - 验证集信息文件
- `data/ars/ars_dbinfos_train.pkl` - GT 数据库索引文件
- `data/ars/gt_database/` - GT 数据库目录

---

### 步骤 2：调整数据集配置文件

**文件位置**：`tools/cfgs/dataset_configs/ars_dataset.yaml` ，检查这个配置文件中的内容和我提供的数据集是否符合，尤其是一些类别、roi、网格尺寸等，如有不符合请指出并提出修改建议。

### 步骤 3： PointPillar 模型配置文件

**文件位置**：`tools/cfgs/ars_models/pointpillar.yaml`，检查这个配置文件中的内容和我提供的数据集是否符合，尤其是一些类别、roi、网格尺寸等如有不符合请指出并提出修改建议。


## 阶段 4：模型训练

### 单 GPU 训练（测试配置）

**执行命令**：
```bash
# 基础训练命令
python tools/train.py \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --batch_size 4 \
    --epochs 5 \
    --workers 4 \
    --extra_tag test_run

# 带有更多选项的命令
python tools/train.py \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --batch_size 4 \
    --epochs 5 \
    --workers 4 \
    --extra_tag test_run \
    --fix_random_seed \
    --use_tqdm_to_record
```

**预期输出**：
- 训练日志：`output/ars_models/pointpillar/test_run/log_train_*.txt`
- 检查点：`output/ars_models/pointpillar/test_run/ckpt/`
- TensorBoard：`output/ars_models/pointpillar/test_run/tensorboard/`

### 多 GPU 分布式训练（正式训练）

**执行命令**：
```bash
# 使用 4 个 GPU 训练
bash scripts/dist_train.sh 4 \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --batch_size 4 \
    --epochs 80 \
    --workers 8 \
    --extra_tag final_model \
    --sync_bn

# 从检查点恢复训练
bash scripts/dist_train.sh 4 \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --batch_size 4 \
    --epochs 80 \
    --workers 8 \
    --extra_tag final_model \
    --ckpt output/ars_models/pointpillar/final_model/ckpt/checkpoint_epoch_40.pth
```

**训练参数说明**：
- `--batch_size`：每个 GPU 的批次大小
- `--epochs`：训练轮数（建议 80）
- `--workers`：数据加载线程数
- `--extra_tag`：实验标签
- `--sync_bn`：同步批归一化（多 GPU 必需）
- `--fix_random_seed`：固定随机种子（可重复性）

### 训练监控

**使用 TensorBoard 监控**：
```bash
tensorboard --logdir=output/ars_models/pointpillar/final_model/tensorboard
```

在浏览器中打开 `http://localhost:6006` 查看：
- 训练损失曲线
- 验证指标
- 学习率变化

**查看训练日志**：
```bash
tail -f output/ars_models/pointpillar/final_model/log_train_*.txt
```

---

## 阶段 5：模型评估与测试

### 评估训练好的模型

**单个检查点评估**：
```bash
python tools/test.py \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --ckpt output/ars_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth
```

**评估所有检查点**：
```bash
python tools/test.py \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --eval_all \
    --ckpt_dir output/ars_models/pointpillar/final_model/ckpt
```

**多 GPU 评估**：
```bash
bash scripts/dist_test.sh 4 \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --ckpt output/ars_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth
```

**测量推理时间**：
```bash
python tools/test.py \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --ckpt output/ars_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth \
    --infer_time
```

### 步骤6 可视化检测结果

**运行 Demo**：
```bash
python tools/demo.py \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --ckpt output/ars_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth \
    --data_path data/ars/points/000000.npy
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
