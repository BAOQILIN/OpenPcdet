# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenPCDet is a PyTorch-based codebase for 3D object detection from LiDAR point clouds. It supports multiple state-of-the-art detection methods including PointRCNN, Part-A2-Net, PV-RCNN, Voxel R-CNN, PV-RCNN++, MPPNet, and others.

**Key Design Principles:**
- Data-Model separation with unified point cloud coordinate system
- Unified 3D box definition: (x, y, z, dx, dy, dz, heading)
- Flexible model structure supporting both one-stage and two-stage detection frameworks
- Multi-dataset support: KITTI, NuScenes, Waymo, Lyft, Pandaset, ONCE, Argo2, Custom

## Installation & Setup

### Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install spconv (version depends on PyTorch version)
# For PyTorch 1.1: spconv v1.0 (commit 8da6f96)
# For PyTorch 1.3+: spconv v1.2 or v2.x
# Latest: pip install spconv-cu118  # adjust cuda version

# Build and install pcdet with CUDA extensions
python setup.py develop
# OR
pip install -e . --config-settings editable_mode=compat --no-build-isolation
```

### CUDA Extensions
The project includes custom CUDA operations in `pcdet/ops/`:
- `iou3d_nms_cuda` - 3D IoU calculation and rotated NMS
- `roiaware_pool3d_cuda` - RoI-aware point cloud pooling
- `roipoint_pool3d_cuda` - RoI-grid point cloud pooling
- `pointnet2_stack_cuda` / `pointnet2_batch_cuda` - PointNet++ operations
- `bev_pool_ext` - BEV pooling for multi-modal fusion
- `ingroup_inds_cuda` - Group indexing operations

These are compiled during `setup.py develop` and require CUDA toolkit.

## Common Commands

### Training
```bash
# Single GPU training
python tools/train.py --cfg_file tools/cfgs/kitti_models/pointpillar.yaml

# Multi-GPU distributed training (recommended)
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml

# Training with specific options
python tools/train.py \
    --cfg_file tools/cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml \
    --batch_size 4 \
    --epochs 80 \
    --workers 8 \
    --extra_tag my_experiment \
    --ckpt path/to/checkpoint.pth \
    --sync_bn \
    --fix_random_seed

# Resume from checkpoint
python tools/train.py --cfg_file CONFIG --ckpt path/to/checkpoint.pth

# Use mixed precision training
python tools/train.py --cfg_file CONFIG --use_amp
```

### Testing/Evaluation
```bash
# Evaluate single checkpoint
python tools/test.py \
    --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt path/to/checkpoint.pth

# Evaluate all checkpoints in a directory
python tools/test.py \
    --cfg_file CONFIG \
    --eval_all \
    --ckpt_dir path/to/ckpt_dir

# Multi-GPU evaluation
bash scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file CONFIG \
    --ckpt path/to/checkpoint.pth

# Measure inference time
python tools/test.py --cfg_file CONFIG --ckpt CKPT --infer_time
```

### Demo/Visualization
```bash
# Run inference on custom data
python tools/demo.py \
    --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt path/to/checkpoint.pth \
    --data_path path/to/point_cloud_data

# Visualize with Open3D (preferred) or Mayavi
# Requires: pip install open3d
```

### Dataset Preparation
```bash
# KITTI dataset
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
    tools/cfgs/dataset_configs/kitti_dataset.yaml

# NuScenes dataset (lidar-only)
python -m pcdet.datasets.nuscenes.nuscenes_dataset \
    --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval

# NuScenes dataset (multi-modal with camera)
python -m pcdet.datasets.nuscenes.nuscenes_dataset \
    --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval \
    --with_cam

# Waymo dataset
python -m pcdet.datasets.waymo.waymo_dataset \
    --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

## Architecture Overview

### Core Components

**1. Dataset Layer (`pcdet/datasets/`)**
- `DatasetTemplate` - Base class for all datasets
- Dataset-specific implementations: `KittiDataset`, `NuScenesDataset`, `WaymoDataset`, etc.
- `augmentor/` - Data augmentation pipeline (GT sampling, random flip, rotation, scaling)
- `processor/` - Data preprocessing (point cloud range filtering, voxelization, etc.)
- Unified interface: `__getitem__` returns standardized `data_dict` with keys like `points`, `gt_boxes`, `gt_names`

**2. Model Layer (`pcdet/models/`)**

Models follow a modular pipeline architecture:

```
Input Point Cloud
    ↓
Backbone 3D (voxel/pillar/point-based feature extraction)
    ↓
Backbone 2D (optional, for BEV feature processing)
    ↓
Dense Head (anchor-based or anchor-free detection)
    ↓
RoI Head (optional, for two-stage refinement)
    ↓
Output: 3D Boxes + Scores
```

Key directories:
- `detectors/` - Complete model definitions (e.g., `PointPillar`, `PVRCNN`, `VoxelRCNN`)
  - `detector3d_template.py` - Base detector class with training/inference logic
- `backbones_3d/` - 3D feature extractors (VoxelNet, PointNet++, SECOND, etc.)
- `backbones_2d/` - 2D BEV feature extractors (BaseBEVBackbone, etc.)
- `backbones_image/` - Image backbones for multi-modal fusion (ResNet, VGG, etc.)
- `dense_heads/` - Detection heads (AnchorHead, CenterHead, TransFusionHead, etc.)
- `roi_heads/` - Two-stage refinement heads (PVRCNNHead, VoxelRCNNHead, etc.)
- `view_transforms/` - Multi-modal view transformation (LiftSplatShoot, BEVFusion, etc.)
- `model_utils/` - Shared utilities (target assignment, loss functions, etc.)

**3. Configuration System (`tools/cfgs/`)**
- `dataset_configs/` - Dataset-specific configs (data paths, augmentation, preprocessing)
- `{dataset}_models/` - Model configs for each dataset (architecture, training hyperparameters)
- YAML-based configuration with inheritance support
- Config structure: `MODEL`, `DATASET`, `OPTIMIZATION`, `DATA_AUGMENTOR`, `DATA_PROCESSOR`

**4. Training Infrastructure (`tools/train_utils/`)**
- `train_utils.py` - Main training loop with logging, checkpointing
- `optimization/` - Optimizers (Adam, AdamW) and learning rate schedulers (OneCycle, CosineAnnealing)
- Supports distributed training with PyTorch DDP
- TensorBoard logging for losses and metrics
- Automatic checkpoint saving with configurable intervals

**5. Evaluation (`tools/eval_utils/`)**
- Dataset-specific evaluation metrics (KITTI AP, NuScenes mAP/NDS, Waymo mAP/mAPH)
- `eval_utils.py` - Evaluation loop with result aggregation

### Model Flow

**Training:**
1. `build_dataloader()` creates dataset and dataloader from config
2. `build_network()` instantiates model from config
3. `build_optimizer()` and `build_scheduler()` set up optimization
4. `train_model()` runs training loop:
   - Forward pass: `model(batch_dict)` returns loss, tb_dict, disp_dict
   - Backward pass and optimizer step
   - Logging and checkpointing

**Inference:**
1. Load model and checkpoint
2. `model.eval()` mode
3. Forward pass: `model(batch_dict)` returns predictions in `batch_dict['final_boxes']`, `batch_dict['final_scores']`, `batch_dict['final_labels']`
4. Post-processing: NMS, score thresholding
5. Evaluation or visualization

### Key Abstractions

**batch_dict**: Dictionary passed through the entire pipeline containing:
- `points` - (N, 4+) point cloud [x, y, z, intensity, ...]
- `gt_boxes` - (B, M, 7+) ground truth boxes [x, y, z, dx, dy, dz, heading, ...]
- `gt_names` - (B, M) class names
- `voxels`, `voxel_coords`, `voxel_num_points` - voxelized representation
- `spatial_features`, `spatial_features_2d` - intermediate features
- `batch_cls_preds`, `batch_box_preds` - predictions
- `final_boxes`, `final_scores`, `final_labels` - post-processed outputs

**Model Config Structure:**
```yaml
MODEL:
  NAME: ModelName
  BACKBONE_3D: {...}
  BACKBONE_2D: {...}
  DENSE_HEAD: {...}
  ROI_HEAD: {...}  # optional for two-stage
  POST_PROCESSING: {...}

OPTIMIZATION:
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

## Development Guidelines

### Adding a New Model
1. Create detector class in `pcdet/models/detectors/` inheriting from `Detector3DTemplate`
2. Implement `forward()` method following the pipeline pattern
3. Add model config YAML in `tools/cfgs/{dataset}_models/`
4. Register model in `pcdet/models/detectors/__init__.py`

### Adding a New Dataset
1. Create dataset class in `pcdet/datasets/{dataset}/` inheriting from `DatasetTemplate`
2. Implement required methods: `__len__`, `__getitem__`, `generate_prediction_dicts`, `evaluation`
3. Add dataset config YAML in `tools/cfgs/dataset_configs/`
4. Register dataset in `pcdet/datasets/__init__.py`
5. See `docs/CUSTOM_DATASET_TUTORIAL.md` for detailed guide

### Modifying Training
- Training loop: `tools/train_utils/train_utils.py::train_model()`
- Loss computation: Model's `get_training_loss()` method
- Optimizer/scheduler: `tools/train_utils/optimization/`

### Debugging
- Set `cfg.LOCAL_RANK = 0` to disable distributed training
- Use `--fix_random_seed` for reproducibility
- Check `output/{dataset}_{model}/{tag}/log_train_{timestamp}.txt` for training logs
- Use `--use_tqdm_to_record` to see progress bar instead of file logging

## Important Notes

### Multi-GPU Training
- Always use `bash scripts/dist_train.sh` for multi-GPU training
- Single GPU training with `python tools/train.py` is slower and may have different batch norm behavior
- Set `--sync_bn` for synchronized batch normalization across GPUs

### Memory Management
- Adjust `--batch_size` based on GPU memory
- Use `DATA_CONFIG.SAMPLED_INTERVAL` in Waymo config to subsample data for limited GPU resources
- Enable `USE_SHARED_MEMORY` in dataset config to speed up data loading if I/O is bottleneck

### Coordinate Systems
- Unified coordinate: x-forward, y-left, z-up (LiDAR coordinate)
- Box format: (x, y, z, dx, dy, dz, heading) where heading is rotation around z-axis
- Different datasets may have different raw coordinates - conversion handled in dataset classes

### Version Compatibility
- PyTorch 1.1-1.10 supported
- spconv version must match PyTorch version (see INSTALL.md)
- CUDA 9.0+ required (CUDA 9.2+ for PyTorch 1.3+)

### Output Structure
```
output/
└── {dataset}_models/
    └── {model_name}/
        └── {extra_tag}/
            ├── ckpt/              # Saved checkpoints
            ├── log_train_*.txt    # Training logs
            ├── tensorboard/       # TensorBoard logs
            └── eval/              # Evaluation results
                └── epoch_{N}/
                    └── {split}/
                        └── {eval_tag}/
                            └── result.pkl
```

## Testing

The project uses pytest for testing. Run tests with:
```bash
pytest tests/
```

For specific test files:
```bash
pytest tests/test_dataset.py
pytest tests/test_model.py -v
```
