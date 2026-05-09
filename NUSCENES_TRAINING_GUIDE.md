# NuScenes PointPillar 训练指南

## 问题总结

在尝试使用 NuScenes 数据训练 PointPillar 模型时，遇到了以下配置问题：

1. **配置文件路径问题**：原始配置使用相对路径 `../data/nuscenes`，需要改为 `data/nuscenes`
2. **数据文件路径问题**：INFO_PATH 和 DB_INFO_PATH 需要包含 `v1.0-trainval/` 前缀
3. **BACKUP_DB_INFO 配置缺失**：代码要求但原始配置中没有

## 解决方案

### 方案 1：使用修正后的配置文件（推荐）

已创建修正后的配置文件：
- `tools/cfgs/dataset_configs/nuscenes_dataset_simple.yaml` - 简化版（禁用 gt_sampling）
- `tools/cfgs/nuscenes_models/cbgs_pp_simple.yaml` - 对应的模型配置

### 方案 2：手动修改原始配置

修改 `tools/cfgs/dataset_configs/nuscenes_dataset.yaml`：

```yaml
# 1. 修改数据路径
DATA_PATH: 'data/nuscenes'  # 原来是 '../data/nuscenes'

# 2. 修改 INFO_PATH
INFO_PATH: {
    'train': [v1.0-trainval/nuscenes_infos_10sweeps_train.pkl],
    'test': [v1.0-trainval/nuscenes_infos_10sweeps_val.pkl],
}

# 3. 修改 DB_INFO_PATH（在 DATA_AUGMENTOR.AUG_CONFIG_LIST 中）
DB_INFO_PATH:
    - v1.0-trainval/nuscenes_dbinfos_10sweeps_withvelo.pkl

# 4. 添加 BACKUP_DB_INFO（在 gt_sampling 配置中）
BACKUP_DB_INFO:
    DB_INFO_PATH: v1.0-trainval/nuscenes_dbinfos_10sweeps_withvelo.pkl
    DB_DATA_PATH: v1.0-trainval/gt_database_10sweeps_withvelo
    NUM_POINT_FEATURES: 5
```

## 训练命令

### 激活环境
```bash
source /home/bql/miniconda3/bin/activate pcdet
cd /home/bql/OpenPCDet
```

### 单 GPU 训练（RTX 3080 10GB）
```bash
python tools/train.py \
    --cfg_file tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml \
    --batch_size 2 \
    --workers 4 \
    --epochs 20 \
    --extra_tag nuscenes_training
```

### 多 GPU 训练（如果有多个 GPU）
```bash
bash scripts/dist_train.sh 2 \
    --cfg_file tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml \
    --batch_size 2 \
    --extra_tag nuscenes_training
```

## 监控训练

### 查看训练日志
```bash
# 实时查看
tail -f output/nuscenes_models/cbgs_pp_multihead/nuscenes_training/log_train_*.txt

# 查看损失
grep "loss:" output/nuscenes_models/cbgs_pp_multihead/nuscenes_training/log_train_*.txt | tail -20
```

### 使用 TensorBoard
```bash
tensorboard --logdir output/nuscenes_models/cbgs_pp_multihead/nuscenes_training/tensorboard
# 浏览器打开 http://localhost:6006
```

### 监控 GPU
```bash
watch -n 1 nvidia-smi
```

## 预期训练时间

- 单 GPU (RTX 3080): ~80-100 小时（20 epochs）
- 建议先训练 5-10 epochs 验证配置正确

## 评估模型

训练完成后评估：
```bash
python tools/test.py \
    --cfg_file tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml \
    --ckpt output/nuscenes_models/cbgs_pp_multihead/nuscenes_training/ckpt/checkpoint_epoch_20.pth
```

## 预期性能

使用完整配置（包含 gt_sampling）训练 20 epochs：
- mAP: ~44-45%
- NDS: ~58-59%

## 故障排查

### 问题：FileNotFoundError
- 检查数据路径是否正确
- 确认 `data/nuscenes/v1.0-trainval/` 目录存在
- 确认 pkl 文件存在

### 问题：CUDA Out of Memory
- 减小 batch_size 到 1
- 减小 MAX_NUMBER_OF_VOXELS

### 问题：训练很慢
- 增加 --workers 参数（默认 4，可尝试 8）
- 检查数据是否在 SSD 上

## 下一步

1. 训练完成后，可以尝试其他模型：
   - CenterPoint: `cbgs_voxel01_res3d_centerpoint.yaml`
   - SECOND: `cbgs_second_multihead.yaml`

2. 调优超参数：
   - 学习率
   - batch size
   - 数据增强策略

3. 在测试集上评估并可视化结果
