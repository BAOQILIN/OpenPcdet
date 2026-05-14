# 使用自定义数据集训练 PointPillar 模型 - 完整指南

## 概述

本文档提供在 OpenPCDet 框架中使用自定义数据集训练 PointPillar 3D 目标检测模型的完整实施方案。PointPillar 是一种高效的基于柱状体素化的单阶段检测器，适合实时应用场景。

## 目标

- **输入**：原始点云数据（.pcd 格式,ascii）及对应的 3D 边界框标注（.json 格式）
- **输出**：训练好的模型检查点，可用于推理和评估
- **评估指标**：KITTI 风格的 AP（Average Precision）

---

## 说明
原始数据在/media/hirain/87e9cd68-1278-4bae-8d73-a0eb5d1ac165/home/jac/ars_hx_train_data/origin_type目录下,内容如下：
├── 1733898949738.json
├── 1733898949738.pcd
├── 1733898949938.json
├── 1733898949938.pcd
├── 1733898950139.json
├── 0000173389895013910.pcd



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
    "frameInfo": {
        "egoCar": {}, 
        "scene": {}, 
        "timestamp": 1733898949738.0
    }, 
    "movingObjects": [
        {
            "age": 10, 
            "annotationTool": {
                "bBox": {
                    "flag": 0, 
                    "value": []
                }, 
                "cuboid2D": {
                    "flag": 0, 
                    "value": []
                }, 
                "cuboid3D": {
                    "flag": 1, 
                    "value": {
                        "cuboidExtent": [
                            4.118916988372803, 
                            1.819737063185618, 
                            1.5159282684326172
                        ], 
                        "orientation": [
                            -1000.0, 
                            -1000.0, 
                            -2.930179832496942
                        ], 
                        "position": [
                            -15.222090992932346, 
                            30.12748853782225, 
                            -0.06523188736221153
                        ]
                    }
                }, 
                "ellipse2D": {
                    "flag": 0, 
                    "value": []
                }, 
                "point2D": {
                    "flag": 0, 
                    "value": []
                }, 
                "point3D": {
                    "flag": 0, 
                    "value": []
                }, 
                "pointCluster": {
                    "flag": 0, 
                    "value": []
                }, 
                "polygon2D": {
                    "flag": 0, 
                    "value": []
                }, 
                "polygon3D": {
                    "flag": 0, 
                    "value": []
                }, 
                "polyline2D": {
                    "flag": 0, 
                    "value": {
                        "interpolationMethod": "Spline", 
                        "vertices": []
                    }
                }, 
                "polyline3D": {
                    "flag": 0, 
                    "value": {
                        "interpolationMethod": "Cubic", 
                        "vertices": []
                    }
                }, 
                "rBBox": {
                    "flag": 0, 
                    "value": []
                }
            }, 
            "existenceProbability": 1.0, 
            "frameList": [], 
            "measurementStatus": "Measured", 
            "objectID": 190485, 
            "objectType": "Car", 
            "property": {
                "acceleration": {
                    "flag": 1, 
                    "value": [
                        0, 
                        0, 
                        0
                    ]
                }, 
                "groupID": {
                    "flag": 0, 
                    "value": 0
                }, 
                "laneAssociation": {
                    "flag": 0, 
                    "value": "Unknown"
                }, 
                "light": {
                    "flag": 0, 
                    "value": []
                }, 
                "personsPoses": {
                    "flag": 0, 
                    "value": []
                }, 
                "recognitionDifficulty": {
                    "flag": 1, 
                    "value": "Easy"
                }, 
                "ridingStatus": {
                    "flag": 0, 
                    "value": "Unknown"
                }, 
                "velocity": {
                    "flag": 1, 
                    "value": [
                        0.0, 
                        0.0, 
                        -1000.0
                    ]
                }
            }, 
            "typeConfidence": 1
        }
    ]
}
```
**转换之后的标签格式说明**
模型最终训练只是里面的一部分内容，annotation字段下的就是每一个目标的属性，但是训练的时候只需要三维框的尺寸（dimension）、中心点坐标（position）、类别（type）、和角度（rotation的第三个元素）。type有如下几类：, 'Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle'，分别对应0-4. 所以你在处理数据的时候需要从中提取这些数据，保存为.txt文本。
  ┌─────────────┬─────────────────────┬─────────────────────┬─────────────────┐
  │  模型输出 ID │  ObjectSubTypes 枚举 │      类别名称        │ 上级 ObjectType  │
  ├─────────────┼─────────────────────┼─────────────────────┼─────────────────┤
  │      0      │          1          │  行人 (Pedestrian)   │      行人       │
  ├─────────────┼─────────────────────┼─────────────────────┼─────────────────┤
  │      1      │          2          │ 摩托车 (Motorcycle)  │      摩托车      │
  ├─────────────┼─────────────────────┼─────────────────────┼─────────────────┤
  │      2      │          3          │     汽车 (Car)       │      车辆       │
  ├─────────────┼─────────────────────┼─────────────────────┼─────────────────┤
  │      3      │          4          │    公交车 (Bus)      │       车辆       │
  ├─────────────┼─────────────────────┼─────────────────────┼─────────────────┤
  │      4      │          5          │  三轮车 (Tricycle)   │      三轮车      │
  └─────────────┴─────────────────────┴─────────────────────┴─────────────────┘
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
│   │   ├── labels/
│   │   │     ├── 000000.txt     # 标注文件
│   │   │     ├── 000001.txt
│   │   │     └── ...
│   │   ├── ars_dbinfos_train.pkl
│   │   ├── ars_infos_train.pkl
│       └── ars_infos_val.pkl
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
cd tools
python test.py \
    --cfg_file cfgs/ars_models/pointpillar.yaml \
    --ckpt ../output/ars_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth
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
cd tools 
python test.py \
    --cfg_file cfgs/ars_models/pointpillar.yaml \
    --ckpt ../output/ars_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth \
    --infer_time
```

### 步骤6 可视化检测结果

**运行 Demo**：
```bash
cd tools 
python demo.py \
    --cfg_file cfgs/ars_models/pointpillar.yaml \
    --ckpt ../output/ars_models/pointpillar/final_model/ckpt/checkpoint_epoch_80.pth \
    --ext .npy \
    --data_path data/ars/points/000000.npy \
    --save_pred_dir ../output/demo_preds
```

**可视化要求**：
- 安装 Open3D：`pip install open3d`
- 或安装 Mayavi：`pip install mayavi`


### 步骤7 ONNX 导出

#### 7.1 导出方案：单 ONNX vs 多 ONNX

**结论：导出为一个 ONNX 文件。**

PointPillar 是单阶段检测器，4 个子模块紧密耦合，通过 `batch_dict` 传递中间结果：

```
voxels (M, 32, 4) + voxel_coords (M, 4) + voxel_num_points (M,)
    │
    ▼  PillarVFE
pillar_features (M, 64)
    │
    ▼  PointPillarScatter
spatial_features (1, 64, 320, 1280)
    │
    ▼  BaseBEVBackbone
spatial_features_2d (1, 384, 160, 640)
    │
    ▼  AnchorHeadSingle
cls_preds (1, 1024000, 6)   ← raw logits (未做 sigmoid)
box_preds (1, 1024000, 7)   ← decoded 3D boxes [x,y,z,dx,dy,dz,heading]
dir_cls_preds (1, 1024000, 2) ← direction classification
```

拆分成多个 ONNX 会在子模块边界产生额外的序列化/反序列化开销，且 scatter 操作（点云 → BEV 映射）必须存在于某一侧，拆分没有收益。

**NMS 和后处理留在 ONNX 外部**，因为 NMS 包含 while 循环和 CUDA kernel 调用，不适合 ONNX 导出。后处理流程：
1. `sigmoid(cls_preds)` → 得分
2. 得分阈值过滤（score > 0.3）
3. Top-K 选择（最多 4096 个）
4. 3D IoU NMS（阈值 0.2，类无关）
5. 限制输出数量（最多 500 个）

这些步骤应在部署框架（C++ TensorRT / Python ONNX Runtime）中实现。

#### 7.2 ONNX 输入输出规格

| 角色 | 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| 输入 | `voxels` | `(M, 32, 4)` | float32 | 体素化后的点云，[x, y, z, intensity] |
| 输入 | `voxel_coords` | `(M, 4)` | int32 | [batch_idx, z, y, x] 网格坐标 |
| 输入 | `voxel_num_points` | `(M,)` | int32 | 每个 pillar 的有效点数 |
| 输出 | `cls_preds` | `(1, 1024000, 6)` | float32 | 各类别 logits（未经 sigmoid） |
| 输出 | `box_preds` | `(1, 1024000, 7)` | float32 | 解码后的 3D 边界框 |
| 输出 | `dir_cls_preds` | `(1, 1024000, 2)` | float32 | 方向分类 logits |

其中 `M` 为动态维度（每帧 pillar 数量不同），ONNX 中标记为 `num_voxels`。

#### 7.3 关键修改点

**PointPillarScatter 的 ONNX 兼容适配**（`pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py`）：
- 原实现使用索引赋值 `spatial_feature[:, indices] = pillars`，ONNX 不支持
- 修改为在 ONNX 导出模式下使用 `scatter_` 操作替代
- 同时固定 `batch_size = 1`（单帧推理场景）

#### 7.4 导出命令

```bash
cd tools

# 导出 ONNX 模型
python export_onnx.py \
    --cfg_file cfgs/ars_models/pointpillar.yaml \
    --ckpt ../output/ars_models/pointpillar/default/ckpt/checkpoint_epoch_30.pth \
    --max_voxels 40000 \
    --opset_version 11

# 输出文件：
# output/ars_models/pointpillar/default/ckpt/checkpoint_epoch_30.onnx
```

**参数说明**：
- `--cfg_file`：模型配置文件
- `--ckpt`：PyTorch checkpoint 路径
- `--max_voxels`：dummy 输入的最大 pillar 数量（与训练配置的 `MAX_NUMBER_OF_VOXELS` 一致）
- `--opset_version`：ONNX opset 版本（默认 11，TensorRT 7.x 兼容）
- `--output`：可指定输出路径，默认与 checkpoint 同目录

#### 7.5 验证 ONNX 模型

```bash
cd tools

# 基本验证（模型结构 + ONNX Runtime 推理测试）
python verify_onnx.py \
    --onnx ../output/ars_models/pointpillar/default/ckpt/checkpoint_epoch_30.onnx \
    --num_voxels 1000
```

**验证内容**：
1. ONNX 模型结构合法性（`onnx.checker.check_model`）
2. 输入/输出名称和形状是否正确
3. ONNX Runtime 是否能成功加载并执行推理
4. 输出张量的形状和数值范围是否正常

**对比 PyTorch vs ONNX Runtime 输出**（精确验证）：
```bash
# 使用相同的随机输入，分别跑 PyTorch 和 ONNX Runtime，对比输出差异
python -c "
import torch
import numpy as np
import onnxruntime as ort
from pcdet.config import cfg, cfg_from_yaml_file

# 加载 PyTorch 模型和 ONNX 模型
# ... 对比推理结果，最大误差应 < 1e-4
"
```

#### 7.6 部署注意事项

1. **体素化预处理**：ONNX 模型只包含网络推理，不包含点云体素化（spconv VoxelGenerator）。部署时需在 ONNX 外部实现或使用单独的体素化模块。

2. **动态轴支持**：`voxels`、`voxel_coords`、`voxel_num_points` 的 dim 0 标记为动态（`num_voxels`），支持每帧不同数量的 pillar。如果使用 TensorRT 部署，构建引擎时需配置 `minShapes` / `optShapes` / `maxShapes`。

3. **后处理实现参考**：后处理逻辑参考 `pcdet/models/detectors/detector3d_template.py` 的 `post_processing` 方法（第 178 行起），核心步骤：sigmoid → score > 0.3 → TopK 4096 → 3D IoU NMS 0.2 → Top 500。

4. **TensorRT 加速建议**：
   - 使用 `trtexec` 或 ONNX-TensorRT 转换
   - 为 `num_voxels` 维度设置合理的 `minShapes`（如 100）和 `maxShapes`（如 40000）
   - NMS 可使用 TensorRT 的 `EfficientNMS` 插件或单独实现

5. **坐标系统**：输出 `box_preds` 中的 heading 已经过方向分类校正（`decode_torch` 逻辑），可直接使用。

#### 7.7 TensorRT 8.5.1.7 Engine 转换（方案A：3-Way 拆分 + Host Scatter）

**⚠ 单个 ONNX 无法直接转为 TRT engine。** 原因：

| 阻塞点 | 位置 | 说明 |
|--------|------|------|
| `ScatterND` (4个) | PillarScatter | TRT 8.5 以 plugin 代理实现，需自定义 PillarScatter 插件 |
| `NonZero` (2个) | map_to_bev 布尔掩码 | 输出形状数据依赖，TRT 无法静态推断 |
| `Where` (13个) | VFE 条件选择 | 下游继承 NonZero 的动态形状 |

**方案：3-Way 拆分，Scatter 放在 HOST 侧**

把 Pillar → BEV 的 scatter 操作从 ONNX 中移除，在 HOST 侧手动实现，三个子模型全部可转 TRT engine：

```
VFE (TRT engine)          HOST Scatter             Backbone2D (TRT engine)       RPN (TRT engine)
┌──────────────────┐    ┌──────────────┐    ┌──────────────────────────┐    ┌──────────────────┐
│ voxels (M,32,4)  │    │ pillar_feat  │    │ spatial_features         │    │ spatial_features │
│ voxel_coords     │    │ (M,64)       │    │ (1,64,320,1280) static   │    │ _2d (1,384,160,  │
│ voxel_num_points │    │   +          │    │       ↓                  │    │        640)      │
│       ↓          │    │ voxel_coords │    │ BaseBEVBackbone (CNN)    │    │       ↓          │
│ PillarVFE        │    │   ↓          │    │       ↓                  │    │ AnchorHeadSingle │
│       ↓          │    │ 零初始化grid  │     │ spatial_features_2d      │    │       ↓          │
│ pillar_features  │    │ 逐pillar索引  │    │ (1,384,160,640)          │    │ cls_preds        │
│ (M,64)           │    │ 赋值到BEV     │    │                          │    │ box_preds        │
│                  │    │   ↓          │    │                          │    │ dir_cls_preds    │
│ TRT ✓            │    │ spatial_feat │    │ TRT ✓                    │    │ TRT ✓            │
└──────────────────┘    └──────────────┘    └──────────────────────────┘    └──────────────────┘
```

**导出 3 个 ONNX**：
```bash
cd tools

python export_onnx_split.py \
    --cfg_file cfgs/ars_models/pointpillar.yaml \
    --ckpt ../output/ars_models/pointpillar/default/ckpt/checkpoint_epoch_30.pth

# 输出：
# checkpoint_epoch_30_vfe.onnx         (28 KB)   — PillarVFE only
# checkpoint_epoch_30_backbone2d.onnx  (19 MB)  — BaseBEVBackbone only
# checkpoint_epoch_30_rpn.onnx         (48 MB)  — AnchorHeadSingle only
```

**转 TensorRT Engine**（全部通过）：

```bash
export LD_LIBRARY_PATH=/home/hirain/ARS/ARS_Project/ARSThirdPartyLinux/TensorRT_8.5.1.7/lib:$LD_LIBRARY_PATH
TRT=/home/hirain/ARS/ARS_Project/ARSThirdPartyLinux/TensorRT_8.5.1.7/bin/trtexec

# VFE（动态形状，需配置 min/opt/max）
$TRT --onnx=..._vfe.onnx --saveEngine=..._vfe.engine --fp16 \
    --minShapes=voxels:1x32x4,voxel_coords:1x4,voxel_num_points:1 \
    --optShapes=voxels:12000x32x4,voxel_coords:12000x4,voxel_num_points:12000 \
    --maxShapes=voxels:40000x32x4,voxel_coords:40000x4,voxel_num_points:40000

# Backbone2D 和 RPN（静态形状，无需 shape 配置）
$TRT --onnx=..._backbone2d.onnx --saveEngine=..._backbone2d.engine --fp16
$TRT --onnx=..._rpn.onnx --saveEngine=..._rpn.engine --fp16
```

**实测结果 (RTX 3090, TensorRT 8.5.1.7)**：

| Engine | ONNX 大小 | Engine 大小 | GPU Compute | 状态 |
|--------|----------|------------|-------------|------|
| VFE | 28 KB | 1.1 MB | mean=0.56ms | ✅ PASSED |
| Backbone2D | 19 MB | 9.5 MB | — | ✅ PASSED |
| RPN | 48 MB | 21 MB | — | ✅ PASSED |

**HOST Scatter 实现参考**（Python 伪代码）：
```python
# VFE 输出: pillar_features (M, 64)
# 保存的 voxel_coords: (M, 4)  [batch_idx, z, y, x]

def host_scatter(pillar_features, voxel_coords, grid_size=(320, 1280), num_features=64):
    """在 GPU 上将稀疏 pillar 特征 scatter 到密集 BEV 网格。"""
    ny, nx = grid_size
    spatial_features = torch.zeros(1, num_features, ny, nx, 
                                    dtype=pillar_features.dtype, 
                                    device=pillar_features.device)
    # voxel_coords[:, 1] = z (always 0), [:, 2] = y_idx, [:, 3] = x_idx
    y_idx = voxel_coords[:, 2].long()
    x_idx = voxel_coords[:, 3].long()
    spatial_features[0, :, y_idx, x_idx] = pillar_features.t()
    return spatial_features
```

**关键修改点**：
- `pcdet/models/backbones_3d/vfe/pillar_vfe.py:121` — `squeeze()` 在 ONNX 导出时替换为 `view(-1, num_filters)`（TRT Squeeze 算子与动态形状不兼容）
- `pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py` — scatter 操作在 ONNX 导出时用 `scatter_` 替代索引赋值（保留给 2-Way 方案使用）

---

#### 7.8 方案对比：2-Way Split vs 3-Way Split（与另一个训练框架对照）

另一个训练框架导出了 3 个 ONNX（`/home/hirain/ARS/TEST_PROJECT/DongFeng/perception_model/pointpillar/pandarat128/`），全部成功编译为 TRT engine。以下对比两种方案的优劣。

**3-Way ONNX 功能**：

| ONNX | 大小 | 功能 | 输入 → 输出 |
|------|------|------|------------|
| `vfe.onnx` | 154KB | PillarVFE | `voxels(P,20,4)` → `pillar_features(1,P,64)` **稀疏** |
| `backbone2D.onnx` | 17MB | FPN Backbone | `spatial_features(1,64,512,512)` → `spatial_features_2d(1,384,128,128)` |
| `rpn.onnx` | 5MB | 检测头 | `(1,384,128,128)` → `cls_preds(1,163840,1)`, `box_deltas(1,163840,8)` |

**核心差异：Scatter 操作的位置**

```
3-Way Split（另一个框架）              2-Way Split（本方案）
                                        
VFE (TRT engine)                     ┌─ VFE
  ↓                                  │    ↓
pillar_features (1, P, 64) 稀疏      │  pillar_features
  ↓                                  │    ↓
══ HOST 侧 Scatter ══ ← 关键差异！   │  PointPillarScatter (ONNX内)
  手动零初始化+逐pillar索引           │    ↓
  ↓                                  └─ spatial_features (静态)
spatial_features (1,64,512,512)           ↓
  ↓                                  Backbone (ONNX内)
Backbone (TRT engine)                   ↓
  ↓                                  Head (ONNX内)
spatial_features_2d (1,384,128,128)     ↓
  ↓                                  cls_preds, box_preds, dir_cls_preds
RPN Head (TRT engine)                (decoded, 已含heading校正)
  ↓
box_deltas + cls_logits
(需外部解码+方向分类)
```

**3-Way 的核心设计理念**：将 scatter 操作从 ONNX/TRT 中**排除**，在 Host 侧手动实现，避免 ScatterND 进入 TRT，使三个子模型全部可转为 TRT engine。

**优劣对比**：

| 维度 | 3-Way Split | 2-Way Split（本方案） |
|------|-------------|----------------------|
| 子模型数 | 3 | 2 |
| TRT engine 数 | **3（全部 TRT）** | 1（仅 RPN 可 TRT） |
| Scatter 位置 | ❌ Host 侧（CPU/手动 GPU kernel） | ✅ ONNX 内（GPU scatter kernel）|
| Scatter 性能 | 慢：遍历 pillar 逐元素索引赋值 | 快：GPU scatter 一次性完成 |
| Host↔Device 传输 | 多：VFE输出→Host→回传 GPU | 少：voxel 入、prediction 出 |
| Engine 启动次数 | 3 次 | 1~2 次 |
| 部署复杂度 | 高：需实现 Host scatter + 管理中间张量 | 低：ORT + TRT 串联 |
| 调试 | 每个子模型可独立测试 | Encoder 内部耦合 |
| box 输出格式 | Delta（需外部解码+方向分类） | Decoded（可直接用 heading） |

**性能预估** (RTX 3090)：

| 方案 | VFE/Encoder | Scatter | Backbone+Head | 总计（估算） |
|------|-------------|---------|---------------|-------------|
| 3-Way | TRT ~1ms | Host ~3-5ms | TRT ~3ms | **~7-9ms** |
| 2-Way | ORT ~3ms (含scatter) | — | TRT ~3ms | **~6ms** |

3-Way 的 Host scatter 在高 pillar 数（>20000）时成为瓶颈。2-Way 的 ORT encoder 虽无 TRT 加速，但 scatter 在 GPU 上避免了 host round-trip。

**结论**：维持 2-Way 方案（encoder + rpn），理由：
1. 减少 Host↔Device 往返，整体延迟更低
2. 部署代码更简单
3. 未来可参考 [NVIDIA CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars) 为 encoder 编写自定义 PillarScatter 插件实现全 TRT
4. 本方案的 box 输出已解码（含 heading 校正），减少了外部后处理复杂度

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


---

## 步骤8：导出与 Legacy ONNX 结构兼容的模型

### 8.1 背景

已有 C++ 推理工程适配了另一个训练框架导出的 3 个 ONNX 模型，位于：

```
/home/hirain/ARS/TEST_PROJECT/DongFeng/perception_model/pointpillar/pandarat128/
├── vfe.onnx          (154KB, 72节点)
├── backbone2D.onnx   (17MB,  41节点)
└── rpn.onnx          (5MB,  130节点)
```

当前 `export_onnx_split.py` 导出的是 OpenPCDet 原生结构的 ONNX，需使其与 legacy 结构一致。Legacy 训练配置（`PointPillarConfig.yaml`）：

```yaml
SENSOR_TYPE: pandarat128
VOXEL_X_GRID: 0.2                  # voxel 0.2m
VOXEL_Y_GRID: 0.2
VOXEL_Z_GRID: 8.0
COORD_RANGE_X_MIN: 0               # x: [0, 102.4]
COORD_RANGE_X_MAX: 102.4
COORD_RANGE_Y_MIN: -51.2           # y: [-51.2, 51.2]
COORD_RANGE_Y_MAX: 51.2
COORD_RANGE_Z_MIN: -3.1            # z: [-3.1, 4.9]
COORD_RANGE_Z_MAX: 4.9
FEATURE_SIZE_X: 128                # 输出特征图 128×128
FEATURE_SIZE_Y: 128
CHANNEL_POINT_CLOUD: 4             # x, y, z, intensity
MAXINUM_POINT_PER_VOXEL: 20        # 每 pillar 最多 20 点
```

### 8.2 三种模型的详细对比

---

#### 8.2.1 VFE (`vfe.onnx`) 对比

| 维度 | Legacy | 当前 OpenPCDet 导出 |
| --- | --- | --- |
| 输入1 名称 | `voxel_features` | `voxels` |
| 输入1 形状 | `(P, 20, 4)` | `(M, 32, 4)` |
| 输入2 名称 | `point_num_per_voxel` | `voxel_num_points` |
| 输入3 名称 | `voxel_coords` | `voxel_coords` |
| 线性层类型 | Conv1d（权重 `[64, 10, 1]`） | MatMul / Linear（权重 `[10, 64]`） |
| BatchNorm | **无**（USE_NORM=False） | **有**（含 running_mean/var 共 4 个张量） |
| mask 实现 | 简单 `Greater → Cast → Mul` | 复杂 `Where + ScatterND + Expand` |
| 输出形状 | `(1, P, 64)` 3D 保留 batch 维 | `(M, 64)` 2D squeezed |
| 算子数 | **72** | **245** |

**Legacy VFE 内部流程**（10维 = 4 raw + 3 cluster + 3 center）：

```
voxel_features (P, 20, 4)
  → Slice xyz → ReduceSum → Div(point_num) → points_mean
  → Sub(xyz, points_mean) → f_cluster (3维)
  → Gather(coords) → Cast → Mul(voxel_size) → Add(offset) → f_center (3维)
  → Concat(raw, f_cluster, f_center) → (P, 20, 10)
  → Greater mask × Mul → 零化无效点
  → Transpose → Conv1d(10→64) → Transpose → Relu → ReduceMax
  → (1, P, 64)
```

**当前 PillarVFE 内部流程**（多了 BatchNorm + 复杂 mask）：

```
voxels (M, 32, 4)
  → 同上计算 f_cluster + f_center → Concat → (M, 32, 10)
  → get_paddings_indicator → mask (Where/ScatterND)
  → features × mask
  → Linear(10→64) → BatchNorm1d → Relu → ReduceMax
  → squeeze(-1) / view → (M, 64)
```

**关键差异**：

1. **Conv1d vs Linear**：两者数学等价但 ONNX 算子不同（Conv vs MatMul），TRT 优化路径不同
2. **BatchNorm**：Legacy 无 BN。当前有 BN，引入了 `running_mean`、`running_var`、`weight`、`bias` 4 个额外参数
3. **mask 实现**：Legacy 用简单 `Greater + Cast + Mul`，当前用复杂 `Where + ScatterND + Expand`

---

#### 8.2.2 Backbone2D (`backbone2D.onnx`) 对比

| 维度 | Legacy | 当前 OpenPCDet 导出 |
| --- | --- | --- |
| 输入形状 | `(1, 64, 512, 512)` | `(1, 64, 320, 1280)` |
| 输出形状 | `(1, 384, 128, 128)` **静态** | `(1, 384, 160, 640)` **动态** |
| 下采样结构 | Stage0(s=1, 4×Conv64) → 三分支 | Stage0(s=2)→Stage1(s=2)→Stage2(s=2) |
| 总下采样 | 4× (512→128) | 8× 然后上采样回 2× (320→160) |
| Pad 实现 | 无（Conv padding=1） | ZeroPad2d(1) + Conv(padding=0) |
| 上采样分支 | 直接从 Stage0 输出分三路 | 从各 Stage 输出独立 deblock |

**Legacy Backbone2D 拓扑**（三条分支从 Stage0 分叉，分别下采样）：

```
spatial_features (1, 64, 512, 512)
  │
  ↓ Stage 0: 4× Conv(64→64, 3×3, s=1)     (保持分辨率)
  │
  (1, 64, 512, 512)
  ├─────────────────────────────────────────────────────┐
  │                                                     │
  ↓ Conv(64→128, 2×2, s=4)                             │
  │  Branch 1: 直接一步下采样到 128×128                  │
  │                                                     │
  ↓──→ Concat_129 (1, 128, 128, 128)                   │
  │                                                     ↓ Conv(64→128, 3×3, s=4) → 5× Conv(128→128)
  │                                                       Stage1: 逐步下采样后 5 层 Conv
  │                                                       │
  │                                                       ↓ deblock2: ConvT(128→128, 1×1, s=1)
  │                                                          Branch 2: 不改变分辨率
  │                                                       │
  │                                                       ↓──→ Concat_150 (1, 128, 128, 128)
  │                                                       │
  │                                                       ↓ Conv(128→256, 3×3, s=2) → 5× Conv(256→256)
  │                                                         Stage2: 再 2× 下采样后 5 层 Conv
  │                                                         │
  │                                                         ↓ deblock3: ConvT(256→128, 2×2, s=2)
  │                                                            Branch 3: 2× 上采样回 128×128
  │                                                         │
  ↓                                                         ↓──→ Concat_171 (1, 128, 128, 128)
  │                                                         │
  └──────────────── Concat(dim=1) ──────────────────────────┘
                        ↓
               (1, 384, 128, 128)
```

**当前 Backbone 拓扑**（每个 Stage 独立下采样，deblock 分别上采样）：

```
spatial_features (1, 64, 320, 1280)
  │
  ↓ stage0: ZeroPad2d(1)+Conv(64→64,s=2)+2×Conv(64→64)
  (1, 64, 160, 640)  → deblock[0]: ConvT(64→128, s=1) → (1,128,160,640) ─┐
  │                                                                        │
  ↓ stage1: ZeroPad2d(1)+Conv(64→128,s=2)+4×Conv(128→128)                  │
  (1, 128, 80, 320) → deblock[1]: ConvT(128→128, s=2) → (1,128,160,640) ─┤
  │                                                                        │
  ↓ stage2: ZeroPad2d(1)+Conv(128→256,s=2)+4×Conv(256→256)                 │
  (1, 256, 40, 160) → deblock[2]: ConvT(256→128, s=4) → (1,128,160,640) ─┘
                                                                   ↓
                                                       Concat → (1, 384, 160, 640)
```

---

#### 8.2.3 RPN (`rpn.onnx`) 对比

| 维度 | Legacy | 当前 OpenPCDet 导出 |
| --- | --- | --- |
| 输入 | `(1, 384, 128, 128)` | `(1, 384, 160, 640)` |
| Head 结构 | **Multi-head**: 5 个类别专属 head | **Single head**: 共享权重 1×1 Conv |
| 特征预处理 | 2× Conv(384→64→64, 3×3) → 共享特征 | 无（直接 1×1 Conv 预测） |
| Box 预测 | **分解式**: reg(4) + height(2) + size(6) + angle(4) = 16ch/head | **统一式**: 7 params |
| Box 输出格式 | **raw deltas** `(1, 163840, 8)` | **decoded boxes** `(1, 1024000, 7)` |
| cls 输出 | 1/head（class-agnostic） `(1, 163840, 1)` | 6 类 `(1, 1024000, 6)` |
| dir_cls 输出 | **无**（direction 融入 angle 分支） | **有** `(1, 1024000, 2)` |
| 总 anchor 数 | 163,840 = 128×128×10 | 1,024,000 = 160×640×10 |
| BoxCoder | **ONNX 外**（Host C++ 侧解码） | **ONNX 内**（Exp/Floor/ArgMax） |

**Legacy RPN 内部结构**（5 个并行 head，共享 64 通道特征后分叉）：

```
spatial_features_2d (1, 384, 128, 128)
  │
  ↓ Conv(384→64, 3×3) → ReLU → Conv(64→64, 3×3) → ReLU
  │
  (1, 64, 128, 128)   ← 共享特征

  ├─ head1 (Car):
  │   conv_cls(64→2)
  │   conv_reg(64→4) ─┐
  │   conv_height(64→2)├→ Concat(16) → box_preds_1
  │   conv_size(64→6) ─┤
  │   conv_angle(64→4)─┘
  │
  ├─ head2 (Pedestrian): 同 head1
  ├─ head3 (Mbike):      同 head1
  ├─ head4 (Bus):        同 head1
  └─ head5 (Tricycle):   同 head1

  ↓ Reshape + Transpose → 按类别 Concat

batch_cls_preds  (1, 163840, 1)   ← 5× 128×128×2  → 逐 anchor 拼接
batch_box_preds  (1, 163840, 8)   ← 5× 128×128×16 → 逐 anchor 拼接
```

---

### 8.3 差异根因分析

差异分为两个层面：

#### 第一层：配置参数差异

| 参数 | Legacy | 当前 | 影响 |
| --- | --- | --- | --- |
| `VOXEL_SIZE` | `[0.2, 0.2, 8]` | `[0.16, 0.16, 8]` | grid: 512×512 vs 1280×320 |
| POINT_CLOUD_RANGE X | `[0, 102.4]` | `[-102.4, 102.4]` | BEV 坐标系原点不同 |
| POINT_CLOUD_RANGE Y | `[-51.2, 51.2]` | `[-25.6, 25.6]` | BEV 横向范围不同 |
| POINT_CLOUD_RANGE Z | `[-3.1, 4.9]` | `[-3, 5]` | 高度范围微小差异 |
| `MAX_POINTS_PER_VOXEL` | 20 | 32 | VFE 输入点维度不同 |
| VFE `USE_NORM` | False | True | BatchNorm 有无 |

#### 第二层：模型架构差异

| 组件 | Legacy | 当前 | 兼容性 |
| --- | --- | --- | --- |
| VFE 线性层 | Conv1d | Linear/MatMul | 需改 PFNLayer |
| Backbone 下采样 | Stage0(s=1)→分支(s=4,s=4,s=2) | Stage0(s=2)→Stage1(s=2)→Stage2(s=2) | **不兼容** |
| RPN Head | 5 个类别专属 Multi-head | 单一共享 head | **不兼容** |
| Box 输出 | 分解式 raw deltas | 统一式 decoded | **不兼容** |

**结论：仅修改 `export_onnx_split.py` 无法解决问题。** 导出脚本只能导出 checkpoint 中训练好的模型结构，不能改变架构。需要从配置 + 模型代码 + 训练全链路修改。

---

### 8.4 可行方案

#### 方案 A：完整对齐（推荐）

需要重新训练模型，但能彻底对齐。

**Step 1：创建 Legacy-compat 配置**

新建 `tools/cfgs/ars_models/pointpillar_legacy_compat.yaml`：

```yaml
CLASS_NAMES: ['Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle']

DATA_CONFIG:
    _BASE_CONFIG_: cfgs/dataset_configs/ars_dataset.yaml
    POINT_CLOUD_RANGE: [0, -51.2, -3.1, 102.4, 51.2, 4.9]
    DATA_PROCESSOR:
        - NAME: mask_points_and_boxes_outside_range
          REMOVE_OUTSIDE_BOXES: True
        - NAME: shuffle_points
          SHUFFLE_ENABLED: {'train': True, 'test': False}
        - NAME: transform_points_to_voxels
          VOXEL_SIZE: [0.2, 0.2, 8]
          MAX_POINTS_PER_VOXEL: 20
          MAX_NUMBER_OF_VOXELS: {'train': 16000, 'test': 40000}

MODEL:
    NAME: PointPillar
    VFE:
        NAME: PillarVFE
        WITH_DISTANCE: False
        USE_ABSLOTE_XYZ: True
        USE_NORM: False          # 关闭 BatchNorm
        NUM_FILTERS: [64]

    MAP_TO_BEV:
        NAME: PointPillarScatter
        NUM_BEV_FEATURES: 64

    BACKBONE_2D:
        NAME: BaseBEVBackboneLegacy  # 需新建，匹配 legacy 拓扑
        LAYER_NUMS: [3, 5, 5]       # Stage0: 3+1层, Stage1: 5+1层, Stage2: 5+1层
        LAYER_STRIDES: [1, 4, 2]    # 从 Stage0 保持分辨率 → 分支 s=4 + s=2
        NUM_FILTERS: [64, 128, 256]
        UPSAMPLE_STRIDES: [4, 1, 2] # Branch1=直接s=4, Branch2=deblock s=1, Branch3=deblock s=2
        NUM_UPSAMPLE_FILTERS: [128, 128, 128]

    DENSE_HEAD:
        NAME: MultiHeadAnchorHead  # 需新建
        NUM_HEADS: 5               # 每个类别一个 head
        SHARED_CONV_CHANNELS: 64
        BOX_BRANCHES:              # 分解式 box 预测
            reg: 4
            height: 2
            size: 6
            angle: 4
```

**Step 2：实现 Multi-head RPN Head**

新建 `pcdet/models/dense_heads/multihead_anchor_head.py`：

- 5 个类别专属子 head（对应 Car, Pedestrian, Mbike, Bus, Tricycle）
- 共享特征提取：`Conv(384→64, 3×3) → ReLU → Conv(64→64, 3×3) → ReLU`
- 每个 head 的 box 分支分解为 4 个子分支：
  - `conv_reg`: Conv(64→4, 3×3) — 中心点回归
  - `conv_height`: Conv(64→2, 3×3) — 高度
  - `conv_size`: Conv(64→6, 3×3) — 尺寸
  - `conv_angle`: Conv(64→4, 3×3) — 角度
- 每个 head 的 cls 分支：`Conv(64→2, 3×3)` — 每位置 2 anchors
- 导出时跳过 BoxCoder，输出 raw logits/deltas

**Step 3：实现 Legacy Backbone**

新建 `pcdet/models/backbones_2d/base_bev_backbone_legacy.py`：

- 不用 ZeroPad2d，改用 Conv padding=1
- Stage 0：stride=1 保持 512×512，3 层 Conv(64→64, 3×3)
- 从 Stage 0 输出分三路：
  - Branch 1: Conv(64→128, 2×2, stride=4) → 直接 128×128
  - Branch 2: Conv(64→128, 3×3, stride=4) → 5 层 Conv(128→128) → deblock(s=1)
  - Branch 3: Conv(128→256, 3×3, stride=2) → 5 层 Conv(256→256) → deblock(s=2)
- 三路 Concat(dim=1) → (B, 384, 128, 128)

**Step 4：修改 PFNLayer 使用 Conv1d 导出**

修改 `pcdet/models/backbones_3d/vfe/pillar_vfe.py` 中的 `PFNLayer`：

```python
def forward(self, inputs):
    if inputs.shape[0] > self.part:
        num_parts = inputs.shape[0] // self.part
        part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                           for num_part in range(num_parts+1)]
        x = torch.cat(part_linear_out, dim=0)
    else:
        # ONNX 导出时用 Conv1d 替代 Linear，使结构与 legacy 一致
        if torch.onnx.is_in_onnx_export():
            x = F.conv1d(inputs.transpose(1, 2),
                         self.linear.weight.unsqueeze(-1)).transpose(1, 2)
        else:
            x = self.linear(inputs)

    if self.use_norm:
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        torch.backends.cudnn.enabled = True
    x = F.relu(x)
    x_max = torch.max(x, dim=1, keepdim=True)[0]

    if self.last_vfe:
        return x_max
    else:
        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        x_concatenated = torch.cat([x, x_repeat], dim=2)
        return x_concatenated
```

**Step 5：简化 VFE 中的 mask 实现**

修改 `PillarVFE.forward()` 的 mask 部分，避免 Where/ScatterND 等复杂算子：

```python
# 替换前（当前）：
# mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
# mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
# features *= mask  # 实际已经这样做了，但 ONNX 导出时 trace 到的算子不同

# 确保 mask 应用方式简单直接：
mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
mask = mask.unsqueeze(-1).to(features.dtype)
features = features * mask  # 简单乘法，生成 Mul 算子而非 Where/ScatterND
```

**Step 6：修改导出脚本**

修改 `tools/export_onnx_split.py`：

- 新建 `LegacyVFEWrapper`：输出不 squeeze，保持 3D `(1, M, 64)`
- 新建 `LegacyRPNWrapper`：不调用 `generate_predicted_boxes()`，跳过 BoxCoder
- 输入名称对齐：`voxels` → `voxel_features`，`voxel_num_points` → `point_num_per_voxel`
- I/O 名称与 legacy 完全一致

**Step 7：重新训练并导出验证**

- 使用新配置 + 新模型代码训练
- 导出 ONNX 后与 legacy 对比结构
- 在 C++ 推理工程中加载测试

#### 方案 B：仅修改 export wrapper（不可行）

Backbone 的内部拓扑和 RPN 的输出语义完全不同，wrapper 无法通过 reshape/permute 转换。例如：
- 320×1280 的 BEV 不能 reshape 成 512×512
- Decoded 7D boxes 不能转换回 raw 8D deltas

#### 方案 C：C++ 推理侧增加适配层

在 C++ 推理代码中适配新 ONNX 格式。可实现但受限于：
- BEV 尺寸不同（512×512 vs 320×1280）意味着 anchor 数量和位置完全不同
- 后处理中的 NMS、anchor 解码等逻辑需适配

仅当 C++ 侧可接受不同 BEV 分辨率时才可行。

---

### 8.5 建议

**短期（快速验证）**：在 C++ 推理侧增加适配层（方案 C），接受 OpenPCDet 导出的 ONNX 格式，处理 I/O 名称和 shape 差异。

**长期（正式部署）**：执行方案 A，从配置 → 模型代码 → 训练 → 导出的完整对齐：

1. 创建 legacy-compat 模型配置（`pointpillar_legacy_compat.yaml`）
2. 实现 Multi-head RPN head（`multihead_anchor_head.py`）
3. 实现 legacy-compat backbone（`base_bev_backbone_legacy.py`）
4. 修改 PFNLayer 使用 Conv1d 导出 + 简化 mask 实现
5. 修改 `export_onnx_split.py` 匹配 legacy I/O 格式
6. 重新训练 → 导出 → 在 C++ 推理工程中验证

**保守替代方案**：保持两套导出独立——OpenPCDet 继续用于研究和开发，legacy 训练框架用于部署产线。
