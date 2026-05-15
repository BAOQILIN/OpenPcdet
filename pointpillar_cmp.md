# PointPillar 模型与 ONNX 导出对比分析

## 概述

本文档对比两个训练体系的 PointPillar 模型配置、架构实现、ONNX 导出方式：

| | ARS (OpenPcdetss) | Web_LidarDetector_test |
|---|---|---|
| 框架 | OpenPCDet (PyTorch) | 自定义框架 (PyTorch + Django) |
| 配置 | `tools/cfgs/ars_models/pointpillar.yaml` | `PointPillars/algo/algo_config.yaml` |
| 模型定义 | `pcdet/models/detectors/pointpillar.py` | `PointPillars/model/model/networks.py` |
| ONNX 导出 | `tools/export_onnx_split.py` | `PointPillars/algo/model_computers.py` |
| 已导出 ONNX | VFE/Backbone2D/RPN (epoch 1) | 运行中动态导出 (未找到静态文件) |

---

## 1. 输入数据与预处理

### 1.1 点云范围

| 参数 | ARS | Web_LidarDetector_test |
|------|-----|------------------------|
| X 范围 | **[0, 102.4]** (仅前向, 102.4m) | **[-102.4, 102.4]** (前后对称, 204.8m) |
| Y 范围 | [-51.2, 51.2] | [-51.2, 51.2] |
| Z 范围 | **[-3.1, 4.9]** | **[-3.1, 4.9]** |

**关键差异**: ARS 仅处理车辆前方点云 (x>0)，Web 处理前后对称范围 (x 范围是 ARS 的 2 倍)。

### 1.2 体素化参数

| 参数 | ARS | Web_LidarDetector_test |
|------|-----|------------------------|
| 体素尺寸 | [0.2, 0.2, 8.0] | [0.2, 0.2, 8.0] |
| 每体素最大点数 | 20 | 20 |
| 最大体素数 (训练) | **16000** | **60000** |
| 最大体素数 (测试) | **40000** | 60000 |
| **BEV 网格尺寸** | **[512, 512, 1]** | **[1024, 512, 1]** |
| BEV 特征图大小 | (1, 64, **512, 512**) | (1, 64, **1024, 512**) |

**关键差异**: Web 模型 X 范围是 ARS 的 2 倍 (204.8 vs 102.4)，因此在相同体素尺寸下，X 方向网格数是 ARS 的 2 倍 (1024 vs 512)。BEV 特征图尺寸完全不同。

### 1.3 点特征通道

两个系统的原始输入均为 4 通道 (x, y, z, intensity)。两者在 VFE 内部都会计算 `f_cluster` (相对体素簇心偏移, 3 通道) 和 `f_center` (相对体素网格中心偏移, 3 通道)，使得 PFN/Linear 层实际输入均为 **10 通道** (4 raw + 3 cluster + 3 center)。

```python
# ARS: pillar_vfe.py:104-114
points_mean = voxel_features[:, :, :3].sum(dim=1) / voxel_num_points
f_cluster = voxel_features[:, :, :3] - points_mean
f_center[:, 0] = x - (coords[:, 3] * voxel_x + x_offset)
f_center[:, 1] = y - (coords[:, 2] * voxel_y + y_offset)
f_center[:, 2] = z - (coords[:, 1] * voxel_z + z_offset)
# use_absolute_xyz=True → features = [raw_xyzi, f_cluster, f_center] → 10 通道
```

```python
# Web: PointPillars/model/layer/VFE.py (同逻辑, f_cluster + f_center)
# Concat([voxel_features, f_cluster, f_center]) → 10 通道
```

| 参数 | ARS | Web_LidarDetector_test |
|------|-----|------------------------|
| 原始输入 | x, y, z, intensity (4 通道) | x, y, z, intensity (4 通道) |
| 内部扩展 | f_cluster(3) + f_center(3) → **10 通道** | f_cluster(3) + f_center(3) → **10 通道** |
| 扩展方式 | 同逻辑, `use_absolute_xyz=True` 保留原始坐标 | 同逻辑, 硬编码保留原始坐标 |

---

## 2. 模型架构对比

### 2.1 VFE (体素特征编码器)

两个模型输入均为 (M, 20, 4)，都有相同的 f_cluster + f_center 计算逻辑，将特征扩展到 10 通道。关键差异在于 PFN 实现方式：

#### ARS: PillarVFE (`pcdet/models/backbones_3d/vfe/pillar_vfe.py`)

```
输入: (M, 20, 4) → f_cluster(3)+f_center(3) → Concat → (M, 20, 10)
  ├─ 掩码填充点(补零)
  └─ PFNLayer(10 → 64, use_norm=False, last_layer=True):
       Linear(10→64, bias=True) + ReLU + MaxPool(across points) → (M, 64)
       [ONNX 导出时 Linear → Conv1d(kernel=1) 替换，匹配 legacy 结构]
  
USE_NORM=False, WITH_DISTANCE=False, NUM_FILTERS: [64]
输出: pillar_features (M, 64) — 已做 max-pool 的体素级全局特征
```

#### Web_LidarDetector_test: Custom VFE + PFN (`PointPillars/model/layer/VFE.py` + `PFNLayers.py`)

```
输入: (M, 20, 4) → f_cluster(3)+f_center(3) → Concat → (M, 20, 10)
  ├─ 掩码填充点(补零)
  └─ PFN: Conv1d(10 → 64) + BatchNorm1d + ReLU → (M, 20, 64)

输出: (M, 20, 64) 逐点特征 — max-pool 推迟到 Scatter 层
```

**差异总结**:

| 差异点 | ARS | Web_LidarDetector_test |
|--------|-----|------------------------|
| 10 通道扩展 | ✓ (同逻辑) | ✓ (同逻辑) |
| PFN 操作 | **Linear(10→64)** | **Conv1d(10→64)** |
| BatchNorm | **无** (USE_NORM=False) | **有** (BatchNorm1d) |
| MaxPool 位置 | **VFE 内部** (PFNLayer last_layer) | **Scatter 层** |
| 输出形状 | **(M, 64)** | **(M, 20, 64)** |
| ONNX 导出适配 | Linear→Conv1d(kernel=1) | 无需转换 |

### 2.2 Scatter (特征散布到 BEV 伪图像)

#### ARS: PointPillarScatter

```
输入: pillar_features (M, 64), voxel_coords (M, 4)
操作: 直接按坐标索引 scatter 到 (1, 64, 512, 512)
输出: (1, 64, 512, 512)
```

#### Web_LidarDetector_test: Custom Scatter

```
输入: PFN 输出 (M, 20, 64), voxel_coords (M, 4)
操作: 先在 pillar 维度 max → (M, 64), 再 scatter 到 (1, 64, 1024, 512)
输出: (1, 64, 1024, 512)
```

### 2.3 Backbone (BEV 特征提取) — 核心差异

这是两个模型**最大的结构差异**。

#### ARS: BaseBEVBackboneLegacy (`pcdet/models/backbones_2d/base_bev_backbone_legacy.py`)

```
输入: spatial_features (1, 64, 512, 512)

Stage 0 (stride=1, 4 convs, 不下采样):
  Conv(64→64,3x3,s=1)×4 + BN + ReLU → (1, 64, 512, 512)
  ├── Branch1: Conv(64→128,2x2,s=4) + BN + ReLU → (1, 128, 128, 128)
  │                                   [直接 stride=4 跳连]
  ├── Stage 1 (stride=4):
  │     Conv(64→128,3x3,s=4) + BN + ReLU
  │     Conv(128→128,3x3,s=1)×5 + BN + ReLU → (1, 128, 128, 128)
  │     └── Branch2: ConvT(128→128,1x1,s=1) + BN + ReLU → (1, 128, 128, 128)
  │
  └── Stage 2 (stride=2):
        Conv(128→256,3x3,s=2) + BN + ReLU
        Conv(256→256,3x3,s=1)×5 + BN + ReLU → (1, 256, 64, 64)
        └── Branch3: ConvT(256→128,2x2,s=2) + BN + ReLU → (1, 128, 128, 128)

Concat(Branch1, Branch2, Branch3) → (1, 384, 128, 128)
```

**特点**: 自定义 backbone 专为匹配 pandarat128 legacy 拓扑设计。
LAYER_NUMS: [3, 5, 5], LAYER_STRIDES: [1, 4, 2], UPSAMPLE_STRIDES: [4, 1, 2]

#### Web_LidarDetector_test: Standard Blocks + Deblocks (`PointPillars/model/layer/Block.py` + `Deblock.py`)

```
输入: spatial_features (1, 64, 1024, 512)

Block1: Conv(64→64,s=2)×3 + BN + ReLU → (1, 64, 512, 256)
  └── Deblock1: Conv2d(64→128,s=2) [upsample_stride=0.5] → (1, 128, 256, 128)

Block2: Conv(64→128,s=2)×5 + BN + ReLU → (1, 128, 256, 128)
  └── Deblock2: ConvT(128→128,s=1) [upsample_stride=1.0] → (1, 128, 256, 128)

Block3: Conv(128→256,s=2)×5 + BN + ReLU → (1, 256, 128, 64)
  └── Deblock3: ConvT(256→128,s=2) [upsample_stride=2.0] → (1, 128, 256, 128)

Concat(Deblock1, Deblock2, Deblock3) → (1, 384, 256, 128)
```

LAYER_NUMS: [3, 5, 5], LAYER_STRIDES: [2, 2, 2], UPSAMPLE_STRIDES: [0.5, 1.0, 2.0]

#### Backbone 差异总结

| 差异点 | ARS | Web_LidarDetector_test |
|--------|-----|------------------------|
| BEV 输入尺寸 | **(1, 64, 512, 512)** | **(1, 64, 1024, 512)** |
| Stage 0 步长 | **1** (不下采样) | **2** (下采样) |
| Stage 下采样模式 | [1, 4, 2] | [2, 2, 2] |
| 上采样步长 | [4, 1, 2] | [0.5, 1.0, 2.0] |
| 上采样 stride<1 处理 | 无 | Conv2d (stride=2, 补偿降采样) |
| 初始 padding | Conv 自带 padding=1 | Conv 自带 padding=1 |
| **最终输出** | **(1, 384, 128, 128)** | **(1, 384, 256, 128)** |
| **位置数** | **16,384** (128×128) | **32,768** (256×128) |

**关键影响**: Web 模型的特征图分辨率是 ARS 的 2 倍 (256×128 vs 128×128)，意味着总 anchor 数也是 2 倍。

### 2.4 Detection Head

两者结构高度相似 (8D 分解式输出)，但输入尺寸和总 anchor 数不同。

#### ARS: MultiHeadAnchorHead (`pcdet/models/dense_heads/multihead_anchor_head.py`)

```
输入: (B, 384, 128, 128)
Shared Conv: Conv(384→64,3x3) + BN + ReLU + Conv(64→64,3x3) + BN + ReLU
5 个 class-agnostic heads (每类独立二分类):
  cls(64→1), reg(64→2), height(64→1), size(64→3), angle(64→2)
每类 2 朝向×1 anchor = 2 anchors/location
总 anchor: 128×128×10 = 163,840
输出: cls=(B, 163840, 1), box=(B, 163840, 8)
```

#### Web_LidarDetector_test: SharedConv + 5 Heads (`PointPillars/model/layer/SharedConv.py` + `Head.py`)

```
输入: (1, 384, 256, 128)
Shared Conv: Conv(384→64,3x3) + BN + ReLU
5 个类别 heads:
  Conv(64→64,3x3) + Conv(64→2,3x3) → cls (每位置 2 anchors)
  Conv(64→64,3x3) + Conv(64→16,3x3) → box (每位置 2 anchors × 8D)
  每 anchor 8D: reg(2) + height(1) + size(3) + angle(2)
每类 2 朝向×1 anchor = 2 anchors/location
总 anchor: 256×128×10 = 327,680
输出: cls=(B, 327680, 1), box=(B, 327680, 8)
```

| 差异点 | ARS | Web_LidarDetector_test |
|--------|-----|------------------------|
| 输入尺寸 | (384, 128, 128) | (384, 256, 128) |
| Shared Conv 层数 | 2 Conv | 1 Conv |
| Cls 输出通道/类 | 1 (class-agnostic) | 2 (per-anchor) |
| 总 anchor 数 | **163,840** | **327,680** |

---

## 3. Anchor 配置对比

| 类别 | 参数 | ARS | Web_LidarDetector_test |
|------|------|-----|------------------------|
| **Car** | 尺寸 [l,w,h] | **[3.9, 1.6, 1.56]** | **[4.63, 1.97, 1.74]** |
| | 底部高度 | **-1.78** | **-0.95** |
| | 匹配/未匹配阈值 | 0.6/0.45 | 0.6/0.45 |
| **Bus** | 尺寸 [l,w,h] | **[5.5, 2.0, 2.5]** | **[10.5, 2.94, 3.47]** |
| | 底部高度 | **-1.78** | **-0.085** |
| | 匹配/未匹配阈值 | 0.6/0.45 | **0.55/0.4** |
| **Tricycle** | 尺寸 [l,w,h] | **[1.76, 0.6, 1.73]** | **[2.30, 1.63, 1.83]** |
| | 底部高度 | **-0.6** | **-1.033** |
| | 匹配/未匹配阈值 | **0.6/0.45** | **0.55/0.4** |
| **Mbike** | 尺寸 [l,w,h] | **[1.76, 0.6, 1.73]** | **[1.80, 0.77, 1.47]** |
| | 底部高度 | **-0.6** | **-1.085** |
| | 匹配/未匹配阈值 | **0.5/0.35** | **0.5/0.3** |
| **Pedestrian** | 尺寸 [l,w,h] | **[0.8, 0.6, 1.73]** | **[0.73, 0.67, 1.67]** |
| | 底部高度 | **-0.6** | **-1.085** |
| | 匹配/未匹配阈值 | 0.5/0.35 | 0.5/0.3 |

**影响**: Anchor 尺寸和底部高度差异显著，尤其是 Bus (ARS 5.5m vs Web 10.5m) 和 Car (ARS 3.9m vs Web 4.63m)。

---

## 4. 训练超参数对比

| 参数 | ARS | Web_LidarDetector_test |
|------|-----|------------------------|
| Batch Size | **4** | **1** |
| Epochs | **80** | **5** |
| 学习率 | **0.003** | **0.001** |
| 优化器 | Adam + OneCycle | Adam + OneCycleLR |
| OneCycle MOMS | [0.95, 0.85] | (默认) |
| OneCycle pct_start | 0.4 | (默认 0.3) |
| 梯度裁剪 | 10 | 无 |
| 权重衰减 | 0.01 | (默认) |

### 损失函数

| 参数 | ARS | Web_LidarDetector_test |
|------|-----|------------------------|
| 分类损失 | 继承自 AnchorHeadTemplate | **SigmoidFocalLoss** (gamma=4.0, alpha=0.25) |
| 回归损失 | 继承自 AnchorHeadTemplate | **WeightedL1Loss** |
| cls/loc 权重 | cls=1.0, loc=**2.0** | cls=1.0, loc=**0.25** |
| 正/负样本权重 | (默认平衡) | pos=1.0, neg=**2.0** |
| head 权重 | 全 1.0 | **[0.3, 1.0, 1.0, 1.0, 1.0]** (行人 0.3) |

---

## 5. 数据增强对比

| 增强方式 | ARS | Web_LidarDetector_test |
|----------|-----|------------------------|
| GT Sampling | 启用 | 未提及 |
| 随机翻转 | **仅 x 轴** (ALONG_AXIS: ['x']) | **x 和 y 轴** |
| 随机旋转 | **±45°** (±0.785 rad) | **±45°** (推测) |
| 随机缩放 | [0.95, 1.05] | [0.95, 1.05] |

---

## 6. ONNX 导出对比

两者都采用 **3 路分离导出** (VFE + Backbone2D + RPN)，scatter 在 Host 侧完成。ops_version 均为 11。

### 6.1 ARS: `tools/export_onnx_split.py`

```
[VFE ONNX]  VFEWrapper:
  输入:  voxel_features (M, 20, 4), point_num_per_voxel (M), voxel_coords (M, 4)
  操作:  PillarVFE 内部 Linear→Conv1d(4→64, kernel=1), max_pool, 
         然后 broadcast 到 (M, 20, 64) ← legacy C++ 引擎兼容
  输出:  pillar_features (M, 20, 64)
  动态轴: num_voxels (dim 0)

[Backbone2D ONNX]  Backbone2DWrapper:
  输入:  spatial_features (1, 64, 512, 512)
  输出:  spatial_features_2d (1, 384, 128, 128)

[RPN ONNX]  RPNWrapper:
  输入:  spatial_features_2d (1, 384, 128, 128)
  输出:  batch_cls_preds (1, 163840, 1), batch_box_preds (1, 163840, 8) [raw deltas]

Host 侧: 体素化 → VFE ONNX → Scatter(spatial_features) → Backbone2D ONNX → RPN ONNX → 
         Anchor 生成 → BoxCoder 解码 → NMS
```

### 6.2 Web_LidarDetector_test: `PointPillars/algo/model_computers.py`

```python
[VFE ONNX]  OnnxVfePfnlayer:
  输入:  voxel (N, 20, 4), num_pts_voxel (N), voxel_coors (N, 4)
  操作:  VFE(f_cluster+f_center计算) → PFN(Conv1d(10→64)+Bn+ReLU)
  输出:  (N, 20, 64) 逐点特征
  动态轴: pillar_num (dim 0)
  样例输入: torch.ones(2, 20, 4)  # 仅 2 个体素做 dummy

[Backbone2D ONNX]  OnnxBackbone:
  输入:  spatial_features (1, 64, 1024, 512)
  操作:  Block1→Deblock1, Block2→Deblock2, Block3→Deblock3 → concat
  输出:  spatial_features_2d (1, 384, 256, 128)
  样例输入: torch.ones(1, 64, 1024, 512)

[RPN ONNX]  OnnxRPN:
  输入:  spatial_features_2d (1, 384, 256, 128)
  操作:  SharedConv → 5×Head → Merge → concat(cls_preds, box_preds)
  输出:  [batch_cls_preds, batch_box_preds]
  样例输入: torch.ones(1, 384, 256, 128)
```

### 6.3 ONNX 输入/输出尺寸差异汇总

| 子模型 | 维度 | ARS | Web_LidarDetector_test |
|--------|------|-----|------------------------|
| VFE | 输入 voxel | (M, 20, 4) | (N, 20, 4) ✓ 相同 |
| VFE | 输出 | (M, 20, 64) | (N, 20, 64) ✓ 相同 |
| Backbone2D | 输入 spatial | **(1, 64, 512, 512)** | **(1, 64, 1024, 512)** |
| Backbone2D | 输出 | **(1, 384, 128, 128)** | **(1, 384, 256, 128)** |
| RPN | 输入 | (1, 384, 128, 128) | (1, 384, 256, 128) |
| RPN | cls 输出 | (1, **163840**, 1) | (1, **327680**, 1) |
| RPN | box 输出 | (1, **163840**, 8) | (1, **327680**, 8) |

**所有 ONNX 尺寸均不兼容！** 两个模型不能互换使用。

---

## 7. 后处理对比

| 参数 | ARS | Web_LidarDetector_test |
|------|-----|------------------------|
| 分数阈值 | **0.3** | **0.2** |
| NMS 类型 | nms_gpu | nms_gpu |
| NMS 阈值 | 0.2 | 0.2 |
| NMS 前最大框数 | **4096** | **4000** |
| NMS 后最大框数 | **500** | **166** |
| 多类别 NMS | **False** | **True** |

---

## 8. 模型权重与已导出文件

### ARS

| 文件 | 说明 |
|------|------|
| `output/ars_models/pointpillar/default/ckpt/latest_model.pth` | 53.88 MB, epoch 1, 4,699,546 参数 |
| `latest_model_vfe.onnx` | 27 KB, 动态 M 维 |
| `latest_model_backbone2d.onnx` | 16.74 MB |
| `latest_model_rpn.onnx` | 1.20 MB |
| `latest_model_vfe.engine` | 791 KB (TensorRT) |
| `latest_model_backbone2d.engine` | 8.7 MB (TensorRT) |
| `latest_model_rpn.engine` | 861 KB (TensorRT) |

### Web_LidarDetector_test

| 文件 | 说明 |
|------|------|
| (无 .pth 文件) | 未找到已保存的 checkpoint |
| (无 .onnx 文件) | ONNX 运行时动态导出到 `model_onnx/` 目录 |

---

## 9. 兼容性结论

### 两模型 ONNX 不能互换，原因如下：

1. **BEV 网格尺寸不同**: ARS 512×512, Web 1024×512 → Backbone2D ONNX 输入形状不匹配
2. **Backbone 架构不同**: ARS 用 `BaseBEVBackboneLegacy` (stride [1,4,2])，Web 用标准 Block (stride [2,2,2]) → 内部计算图完全不同
3. **点云范围不同**: ARS 仅前方 [0, 102.4]，Web 前后对称 [-102.4, 102.4] → 相同物理目标的体素坐标不同
4. **VFE 实现不同**: ARS 用 Linear(4→64) 无 BN，Web 用 f_cluster+f_center + Conv1d(10→64)+BN → 权重形状不同
5. **特征图分辨率不同**: ARS 输出 (128×128)，Web 输出 (256×128) → RPN anchor 数 163,840 vs 327,680
6. **Anchor 尺寸不同**: 尤其是 Bus (5.5m vs 10.5m) 和 Car (3.9m vs 4.63m)

### 唯一的结构相似点：
- 都采用 3 路分离 ONNX 导出 (VFE/Backbone2D/RPN)，scatter 在 Host
- 都使用 8D 分解式 box 编码 (reg:2 + height:1 + size:3 + angle:2)
- opset_version 均为 11
- 5 类检测 (Car, Bus, Tricycle, Mbike, Pedestrian)

---

## 10. 验证方法

### ARS 模型验证

```bash
# 端到端 ONNX 推理测试
python tools/onnx_test.py \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --ckpt output/ars_models/pointpillar/default/ckpt/latest_model.pth \
    --data_path /path/to/test/data

# 导出新 ONNX
python tools/export_onnx_split.py \
    --cfg_file tools/cfgs/ars_models/pointpillar.yaml \
    --ckpt output/ars_models/pointpillar/default/ckpt/latest_model.pth
```

### Web 模型验证

```bash
# 训练并自动导出 ONNX
cd /home/hirain/ARS/Web_LidarDetector_test
python train_flow.py  # 每个 epoch 后自动导出 ONNX 到 model_onnx/
```
