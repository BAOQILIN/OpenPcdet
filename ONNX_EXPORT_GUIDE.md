# ONNX Export Guide

## 环境要求

**重要：** 本项目使用conda环境 `pcdet`，所有Python命令都需要在该环境中执行：

```bash
conda activate pcdet
# 或使用
conda run -n pcdet python <script>
```

## 导出模型到ONNX格式

### 基本用法

```bash
conda run -n pcdet python tools/export_onnx.py \
    --cfg_file tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml \
    --ckpt output/nuscenes_models/cbgs_pp_multihead/default/ckpt/checkpoint_epoch_20.pth
```

ONNX文件将自动保存到与checkpoint相同的目录，文件名为 `checkpoint_epoch_20.onnx`

### 高级选项

```bash
conda run -n pcdet python tools/export_onnx.py \
    --cfg_file tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml \
    --ckpt output/nuscenes_models/cbgs_pp_multihead/default/ckpt/checkpoint_epoch_20.pth \
    --output custom_path/model.onnx \
    --opset_version 11 \
    --max_points 30000
```

### 参数说明

- `--cfg_file`: 模型配置文件路径（必需）
- `--ckpt`: 训练好的checkpoint文件路径（必需）
- `--output`: 输出ONNX文件路径（可选，默认与checkpoint同目录）
- `--opset_version`: ONNX opset版本（默认11）
- `--max_points`: 最大点数（默认30000）

### 输入输出说明

**ONNX模型输入：**
- `voxels`: (num_voxels, 20, 5) [float32] - 体素化后的点云数据
- `voxel_coords`: (num_voxels, 4) [int32] - 体素坐标 [batch_idx, z, y, x]
- `voxel_num_points`: (num_voxels,) [int64] - 每个体素中的点数

**ONNX模型输出：**
- `cls_preds`: 类别预测
- `box_preds`: 3D边界框预测
- `dir_preds`: 方向预测
- 其他中间输出（用于多头检测）

### 验证导出的模型

使用提供的验证脚本：

```bash
conda run -n pcdet python tools/verify_onnx.py \
    --onnx output/nuscenes_models/cbgs_pp_multihead/default/ckpt/checkpoint_epoch_20.onnx
```

验证脚本会：
- 检查ONNX模型结构
- 显示输入输出信息
- 使用ONNX Runtime进行推理测试

### 使用ONNX Runtime进行推理

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])

# 准备输入数据（注意数据类型）
voxels = np.random.randn(1000, 20, 5).astype(np.float32)
voxel_coords = np.random.randint(0, 256, (1000, 4)).astype(np.int32)
voxel_num_points = np.random.randint(1, 21, (1000,)).astype(np.int64)

# 推理
outputs = session.run(
    None,
    {
        'voxels': voxels,
        'voxel_coords': voxel_coords,
        'voxel_num_points': voxel_num_points
    }
)

# outputs包含所有输出张量
cls_preds = outputs[0]
box_preds = outputs[1]
dir_preds = outputs[2]
```

### 注意事项

1. **环境要求**：必须在 `pcdet` conda环境中运行
2. **数据类型**：
   - voxels: float32
   - voxel_coords: int32
   - voxel_num_points: int64
3. **后处理**：ONNX导出不包含后处理步骤（NMS等），需要在推理时单独实现
4. **动态形状**：模型支持动态batch size和voxel数量
5. **GPU支持**：如需GPU推理，安装 `onnxruntime-gpu` 并确保CUDA版本匹配

### 故障排除

**问题1：CUDA错误**
- 确保voxel_coords在有效的grid范围内
- 检查grid_size配置是否正确

**问题2：数据类型不匹配**
- 严格按照上述数据类型要求准备输入
- voxel_coords必须是int32，voxel_num_points必须是int64

**问题3：权重加载失败**
- PyTorch 2.6+需要设置 `weights_only=False`
- 脚本已自动处理此问题
