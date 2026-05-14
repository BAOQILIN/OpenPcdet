import argparse
import torch
import numpy as np
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network


class DummyDataset:
    """Dummy dataset for ONNX export."""

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
            'num_point_features': 4
        })()

    @property
    def mode(self):
        return 'TEST'


class VFEWrapper(torch.nn.Module):
    """
    Model A: PillarVFE (legacy-compatible).
    Input:  voxel_features (M, 20, 4), point_num_per_voxel (M,), voxel_coords (M, 4)
    Output: pillar_features (1, M, 64) — 3D sparse, batch dim preserved
    """

    def __init__(self, vfe):
        super().__init__()
        self.vfe = vfe

    def forward(self, voxel_features, voxel_coords, point_num_per_voxel):
        batch_dict = {
            'voxels': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': point_num_per_voxel,
            'batch_size': 1,
        }
        batch_dict = self.vfe(batch_dict)
        return batch_dict['pillar_features']


class Backbone2DWrapper(torch.nn.Module):
    """
    Model B: Backbone2D (legacy-compatible).
    Input:  spatial_features (1, 64, 512, 512)
    Output: spatial_features_2d (1, 384, 128, 128)
    """

    def __init__(self, backbone_2d):
        super().__init__()
        self.backbone_2d = backbone_2d

    def forward(self, spatial_features):
        batch_dict = {
            'spatial_features': spatial_features,
            'batch_size': 1,
        }
        batch_dict = self.backbone_2d(batch_dict)
        return batch_dict['spatial_features_2d']


class RPNWrapper(torch.nn.Module):
    """
    Model C: RPN head (legacy-compatible).
    Input:  spatial_features_2d (1, 384, 128, 128)
    Output: batch_cls_preds (1, 163840, 1), batch_box_preds (1, 163840, 8)
            - raw deltas, no BoxCoder inside ONNX
    """

    def __init__(self, dense_head):
        super().__init__()
        self.dense_head = dense_head

    def forward(self, spatial_features_2d):
        batch_dict = {
            'spatial_features_2d': spatial_features_2d,
            'batch_size': 1,
        }
        batch_dict = self.dense_head(batch_dict)
        return (
            batch_dict['batch_cls_preds'],
            batch_dict['batch_box_preds'],
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export PointPillar as 3 legacy-compatible ONNX models '
                    '(vfe + backbone2d + rpn) with scatter on HOST'
    )
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to model config yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same dir as checkpoint)')
    parser.add_argument('--max_voxels', type=int, default=40000,
                        help='Max number of voxels for dummy input')
    parser.add_argument('--opset_version', type=int, default=11,
                        help='ONNX opset version')
    args = parser.parse_args()
    return args


def export_split_onnx(args):
    cfg_from_yaml_file(args.cfg_file, cfg)
    dummy_dataset = DummyDataset(cfg)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dummy_dataset)
    model.cuda()
    model.eval()

    checkpoint = torch.load(args.ckpt, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    print(f'Loaded checkpoint from {args.ckpt} (epoch {checkpoint.get("epoch", "unknown")})')

    # module_list: [PillarVFE, PointPillarScatter, Backbone2D, DenseHead]
    vfe = model.module_list[0]
    scatter = model.module_list[1]
    backbone_2d = model.module_list[2]
    dense_head = model.module_list[3]

    # Output directory
    ckpt_path = Path(args.ckpt)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = ckpt_path.stem
    vfe_path = output_dir / f'{stem}_vfe.onnx'
    backbone_path = output_dir / f'{stem}_backbone2d.onnx'
    rpn_path = output_dir / f'{stem}_rpn.onnx'

    # Common dummy input (legacy-compatible)
    M = args.max_voxels
    P = 20  # legacy MAX_POINTS_PER_VOXEL = 20
    F = 4   # x, y, z, intensity
    grid_size = dummy_dataset.grid_size  # usually [512, 512, 1]

    dummy_voxels = torch.zeros(M, P, F, device='cuda')
    dummy_voxel_coords = torch.zeros(M, 4, dtype=torch.int32, device='cuda')
    dummy_voxel_coords[:, 2] = torch.randint(0, int(grid_size[1]), (M,), device='cuda')  # y_idx
    dummy_voxel_coords[:, 3] = torch.randint(0, int(grid_size[0]), (M,), device='cuda')  # x_idx
    dummy_voxel_num_points = torch.randint(1, P + 1, (M,), device='cuda', dtype=torch.int32)

    # ---- Export VFE (Model A) ----
    print(f'\n=== Exporting VFE (PillarVFE only) to {vfe_path} ===')
    vfe_wrapper = VFEWrapper(vfe)
    vfe_wrapper.cuda()
    vfe_wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            vfe_wrapper,
            (dummy_voxels, dummy_voxel_coords, dummy_voxel_num_points),
            str(vfe_path),
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=['voxel_features', 'voxel_coords', 'point_num_per_voxel'],
            output_names=['pillar_features'],
            dynamic_axes={
                'voxel_features': {0: 'num_voxels'},
                'voxel_coords': {0: 'num_voxels'},
                'point_num_per_voxel': {0: 'num_voxels'},
                'pillar_features': {1: 'num_voxels'},
            },
        )
    print(f'VFE exported: {vfe_path}')

    # Generate spatial_features via VFE + Scatter
    print('\n--- Generating spatial_features for Backbone2D input ---')
    with torch.no_grad():
        batch_dict = {
            'voxels': dummy_voxels,
            'voxel_coords': dummy_voxel_coords,
            'voxel_num_points': dummy_voxel_num_points,
            'batch_size': 1,
        }
        batch_dict = vfe(batch_dict)
        batch_dict = scatter(batch_dict)
        spatial_features = batch_dict['spatial_features']
    print(f'spatial_features shape: {spatial_features.shape}')  # expected: (1, 64, 512, 512)

    # ---- Export Backbone2D (Model B) ----
    print(f'\n=== Exporting Backbone2D to {backbone_path} ===')
    backbone_wrapper = Backbone2DWrapper(backbone_2d)
    backbone_wrapper.cuda()
    backbone_wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            backbone_wrapper,
            (spatial_features,),
            str(backbone_path),
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=['spatial_features'],
            output_names=['spatial_features_2d'],
        )
    print(f'Backbone2D exported: {backbone_path}')

    # Generate spatial_features_2d via Backbone
    print('\n--- Generating spatial_features_2d for RPN input ---')
    with torch.no_grad():
        batch_dict = {'spatial_features': spatial_features, 'batch_size': 1}
        batch_dict = backbone_2d(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
    print(f'spatial_features_2d shape: {spatial_features_2d.shape}')  # expected: (1, 384, 128, 128)

    # ---- Export RPN (Model C) ----
    print(f'\n=== Exporting RPN (MultiHeadAnchorHead) to {rpn_path} ===')
    rpn_wrapper = RPNWrapper(dense_head)
    rpn_wrapper.cuda()
    rpn_wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            rpn_wrapper,
            (spatial_features_2d,),
            str(rpn_path),
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=['spatial_features_2d'],
            output_names=['batch_cls_preds', 'batch_box_preds'],
        )
    print(f'RPN exported: {rpn_path}')

    # Validate
    try:
        import onnx
        for path in [vfe_path, backbone_path, rpn_path]:
            onnx_model = onnx.load(str(path))
            onnx.checker.check_model(onnx_model)
            print(f'ONNX check passed: {path.name}')
    except ImportError:
        print('Warning: onnx package not installed, skipping verification')
    except Exception as e:
        print(f'Warning: ONNX verification failed: {e}')

    # Print summary
    vfe_size = vfe_path.stat().st_size / 1024 / 1024
    backbone_size = backbone_path.stat().st_size / 1024 / 1024
    rpn_size = rpn_path.stat().st_size / 1024 / 1024

    print(f'\n{"="*70}')
    print(f'Export Summary — Legacy-Compatible 3-Way Split')
    print(f'{"="*70}')
    print(f'')
    print(f'[1] VFE ONNX:       {vfe_path} ({vfe_size:.1f} MB)')
    print(f'    Inputs:   voxel_features (M, 20, 4), voxel_coords (M, 4),')
    print(f'              point_num_per_voxel (M,)')
    print(f'    Outputs:  pillar_features (1, M, 64)')
    print(f'')
    print(f'    HOST Scatter: pillar_features + voxel_coords')
    print(f'              -> spatial_features (1, 64, 512, 512)')
    print(f'')
    print(f'[2] Backbone2D ONNX: {backbone_path} ({backbone_size:.1f} MB)')
    print(f'    Inputs:   spatial_features (1, 64, 512, 512)')
    print(f'    Outputs:  spatial_features_2d (1, 384, 128, 128)')
    print(f'')
    print(f'[3] RPN ONNX:        {rpn_path} ({rpn_size:.1f} MB)')
    print(f'    Inputs:   spatial_features_2d (1, 384, 128, 128)')
    print(f'    Outputs:  batch_cls_preds (1, 163840, 1),')
    print(f'              batch_box_preds (1, 163840, 8) [raw deltas]')
    print(f'')
    print(f'{"="*70}')


if __name__ == '__main__':
    args = parse_args()
    export_split_onnx(args)
