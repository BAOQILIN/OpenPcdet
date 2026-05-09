#!/usr/bin/env python3
"""Test multi-head predictions shape"""
import torch
import numpy as np
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network


class DummyDataset:
    """Dummy dataset for testing"""
    def __init__(self, cfg):
        self.class_names = cfg.CLASS_NAMES
        self.point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        voxel_size = cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE
        self.voxel_size = np.array(voxel_size)
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        ).astype(np.int64)
        self.depth_downsample_factor = None
        self.point_feature_encoder = type('obj', (object,), {
            'num_point_features': 5
        })()

    @property
    def mode(self):
        return 'TEST'


# Load config
cfg_file = 'output/cfgs/nuscenes_models/cbgs_pp_multihead/nuscenes_pointpillar_training/cbgs_pp_multihead.yaml'
ckpt_file = 'output/cfgs/nuscenes_models/cbgs_pp_multihead/nuscenes_pointpillar_training/ckpt/checkpoint_epoch_19.pth'

cfg_from_yaml_file(cfg_file, cfg)
dummy_dataset = DummyDataset(cfg)

# Build model
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dummy_dataset)
model.cuda()
model.eval()

# Load checkpoint
checkpoint = torch.load(ckpt_file, map_location='cuda', weights_only=False)
model.load_state_dict(checkpoint['model_state'], strict=True)
print(f'Loaded checkpoint')

# Create dummy inputs
max_voxels = 10000
max_points_per_voxel = 20
num_point_features = 5

dummy_voxels = torch.randn(max_voxels, max_points_per_voxel, num_point_features).cuda()
dummy_voxel_coords = torch.zeros(max_voxels, 4, dtype=torch.int32).cuda()
dummy_voxel_coords[:, 0] = 0
dummy_voxel_coords[:, 1] = torch.randint(0, 10, (max_voxels,))
dummy_voxel_coords[:, 2] = torch.randint(0, 496, (max_voxels,))
dummy_voxel_coords[:, 3] = torch.randint(0, 432, (max_voxels,))
dummy_voxel_num_points = torch.randint(1, max_points_per_voxel + 1, (max_voxels,)).cuda()

# Run forward
batch_dict = {
    'voxels': dummy_voxels,
    'voxel_coords': dummy_voxel_coords,
    'voxel_num_points': dummy_voxel_num_points,
    'batch_size': 1
}

with torch.no_grad():
    for cur_module in model.module_list:
        batch_dict = cur_module(batch_dict)

# Check predictions
batch_cls_preds = batch_dict['batch_cls_preds']
batch_box_preds = batch_dict['batch_box_preds']

print(f'\nPredictions type:')
print(f'  cls_preds: {type(batch_cls_preds)}')
print(f'  box_preds: {type(batch_box_preds)}')

if isinstance(batch_cls_preds, list):
    print(f'\nMulti-head predictions: {len(batch_cls_preds)} heads')
    for i, (cls_head, box_head) in enumerate(zip(batch_cls_preds, batch_box_preds)):
        print(f'  Head {i}:')
        print(f'    cls shape: {cls_head.shape}')
        print(f'    box shape: {box_head.shape}')
else:
    print(f'\nSingle-head predictions:')
    print(f'  cls shape: {batch_cls_preds.shape}')
    print(f'  box shape: {batch_box_preds.shape}')
