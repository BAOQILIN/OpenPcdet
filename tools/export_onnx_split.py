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
            'num_point_features': 4  # x, y, z, intensity
        })()

    @property
    def mode(self):
        return 'TEST'


class VFEWrapper(torch.nn.Module):
    """
    Model A: PillarVFE only (NO scatter inside — scatter on HOST)
    Input:  voxels (M, 32, 4), voxel_coords (M, 4), voxel_num_points (M,)
    Output: pillar_features (M, 64)  -- sparse, one feature vector per pillar

    Scatter (pillar_features + voxel_coords → spatial_features) is done
    on the HOST side, outside TensorRT/ONNX, to avoid ScatterND in TRT.
    """

    def __init__(self, vfe):
        super().__init__()
        self.vfe = vfe

    def forward(self, voxels, voxel_coords, voxel_num_points):
        batch_dict = {
            'voxels': voxels,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': 1,
        }
        batch_dict = self.vfe(batch_dict)
        return batch_dict['pillar_features']


class Backbone2DWrapper(torch.nn.Module):
    """
    Model B: BaseBEVBackbone only
    Input:  spatial_features (1, 64, 320, 1280)  -- static shape BEV pseudo-image
            (produced by HOST scatter from pillar_features + voxel_coords)
    Output: spatial_features_2d (1, 384, 160, 640)
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
    Model C: AnchorHeadSingle only
    Input:  spatial_features_2d (1, 384, 160, 640)
    Output: cls_preds (1, 1024000, 6), box_preds (1, 1024000, 7),
            dir_cls_preds (1, 1024000, 2)
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
            batch_dict['batch_dir_cls_preds'],
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export PointPillar as 3 ONNX models (vfe + backbone2d + rpn) '
                    'with scatter on HOST for full TRT acceleration'
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

    # module_list for PointPillar:
    #   [0] PillarVFE, [1] PointPillarScatter, [2] BaseBEVBackbone, [3] AnchorHeadSingle
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

    # Common dummy input
    M = args.max_voxels
    P = 32   # max points per pillar (matches config)
    F = 4    # x, y, z, intensity
    grid_size = dummy_dataset.grid_size  # [nx, ny, nz]

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
            input_names=['voxels', 'voxel_coords', 'voxel_num_points'],
            output_names=['pillar_features'],
            dynamic_axes={
                'voxels': {0: 'num_voxels'},
                'voxel_coords': {0: 'num_voxels'},
                'voxel_num_points': {0: 'num_voxels'},
                'pillar_features': {0: 'num_voxels'},
            },
        )
    print(f'VFE exported: {vfe_path}')

    # Generate spatial_features via VFE + Scatter (for Backbone2D dummy input)
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
    print(f'spatial_features shape: {spatial_features.shape}')

    # ---- Export Backbone2D (Model B) ----
    print(f'\n=== Exporting Backbone2D (BaseBEVBackbone only) to {backbone_path} ===')
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

    # Generate spatial_features_2d via Backbone (for RPN dummy input)
    print('\n--- Generating spatial_features_2d for RPN input ---')
    with torch.no_grad():
        batch_dict = {'spatial_features': spatial_features, 'batch_size': 1}
        batch_dict = backbone_2d(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
    print(f'spatial_features_2d shape: {spatial_features_2d.shape}')

    # ---- Export RPN (Model C) ----
    print(f'\n=== Exporting RPN (AnchorHeadSingle only) to {rpn_path} ===')
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
            output_names=['cls_preds', 'box_preds', 'dir_cls_preds'],
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
    print(f'Export Summary — 3-Way Split (方案A: Scatter on HOST)')
    print(f'{"="*70}')
    print(f'')
    print(f'[1] VFE ONNX:       {vfe_path} ({vfe_size:.1f} MB)')
    print(f'    Inputs:   voxels (M, 32, 4), voxel_coords (M, 4), voxel_num_points (M,)')
    print(f'    Outputs:  pillar_features (M, 64)')
    print(f'')
    print(f'    ╔════════════════════════════════════════╗')
    print(f'    ║  HOST Scatter (outside TRT/ONNX)      ║')
    print(f'    ║  pillar_features + voxel_coords       ║')
    print(f'    ║         ↓                             ║')
    print(f'    ║  spatial_features (1, 64, 320, 1280)  ║')
    print(f'    ╚════════════════════════════════════════╝')
    print(f'')
    print(f'[2] Backbone2D ONNX: {backbone_path} ({backbone_size:.1f} MB)')
    print(f'    Inputs:   spatial_features (1, 64, 320, 1280)')
    print(f'    Outputs:  spatial_features_2d (1, 384, 160, 640)')
    print(f'')
    print(f'[3] RPN ONNX:        {rpn_path} ({rpn_size:.1f} MB)')
    print(f'    Inputs:   spatial_features_2d (1, 384, 160, 640)')
    print(f'    Outputs:  cls_preds (1, 1024000, 6), box_preds (1, 1024000, 7),')
    print(f'              dir_cls_preds (1, 1024000, 2)')
    print(f'')
    print(f'{"="*70}')
    print(f'TensorRT conversion commands:')
    print(f'  trtexec --onnx={vfe_path}       --saveEngine={output_dir / f"{stem}_vfe.engine"}       --fp16 --minShapes=... --optShapes=... --maxShapes=...')
    print(f'  trtexec --onnx={backbone_path}  --saveEngine={output_dir / f"{stem}_backbone2d.engine"}  --fp16')
    print(f'  trtexec --onnx={rpn_path}       --saveEngine={output_dir / f"{stem}_rpn.engine"}       --fp16')
    print(f'{"="*70}')


if __name__ == '__main__':
    args = parse_args()
    export_split_onnx(args)
