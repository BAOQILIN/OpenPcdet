import argparse
import torch
import numpy as np
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils


class DummyDataset:
    """Dummy dataset for ONNX export, provides dataset-level config to model builder."""

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


class ONNXWrapper(torch.nn.Module):
    """
    Wraps the PointPillar model module_list for ONNX export.

    Inputs (after voxelization pre-processing):
        voxels:            (M, 32, 4)  - [x, y, z, intensity] per point in each pillar
        voxel_coords:      (M, 4)      - [batch_idx, z, y, x] grid coordinates
        voxel_num_points:  (M,)        - number of valid points per pillar

    Outputs (raw predictions, post-processing done externally):
        cls_preds:     (1, 1024000, 6)  - raw class logits (not sigmoid)
        box_preds:     (1, 1024000, 7)  - decoded 3D boxes [x, y, z, dx, dy, dz, heading]
        dir_cls_preds: (1, 1024000, 2)  - direction classification logits
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxels, voxel_coords, voxel_num_points):
        batch_dict = {
            'voxels': voxels,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': 1,
        }

        for cur_module in self.model.module_list:
            batch_dict = cur_module(batch_dict)

        return (
            batch_dict['batch_cls_preds'],
            batch_dict['batch_box_preds'],
            batch_dict['batch_dir_cls_preds'],
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Export PointPillar model to ONNX')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to model config yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--output', type=str, default=None, help='Output ONNX file path (default: next to ckpt)')
    parser.add_argument('--max_voxels', type=int, default=40000, help='Max number of voxels for dummy input')
    parser.add_argument('--opset_version', type=int, default=11, help='ONNX opset version')
    args = parser.parse_args()
    return args


def export_onnx(args):
    cfg_from_yaml_file(args.cfg_file, cfg)
    dummy_dataset = DummyDataset(cfg)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dummy_dataset)
    model.cuda()
    model.eval()

    checkpoint = torch.load(args.ckpt, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    print(f'Loaded checkpoint from {args.ckpt} (epoch {checkpoint.get("epoch", "unknown")})')

    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()

    # Dummy inputs
    M = args.max_voxels
    P = 32  # max points per pillar (matches config: MAX_POINTS_PER_VOXEL)
    F = 4   # num point features: x, y, z, intensity
    grid_size = dummy_dataset.grid_size  # [nx, ny, nz]

    dummy_voxels = torch.zeros(M, P, F, device='cuda')
    dummy_voxels[:, :, :4] = torch.randn(M, P, 4, device='cuda')

    # Coordinate ranges must be valid to avoid scatter out-of-bounds:
    #   z in [0, nz), y in [0, ny), x in [0, nx)
    # The scatter index = z * nx + y * nx + x. With nz=1 this is y * nx + x.
    dummy_voxel_coords = torch.zeros(M, 4, dtype=torch.int32, device='cuda')
    dummy_voxel_coords[:, 0] = 0  # batch_idx, always 0 for single batch
    dummy_voxel_coords[:, 1] = 0  # z_idx, always 0 since nz=1
    dummy_voxel_coords[:, 2] = torch.randint(0, int(grid_size[1]), (M,), device='cuda')  # y_idx
    dummy_voxel_coords[:, 3] = torch.randint(0, int(grid_size[0]), (M,), device='cuda')  # x_idx

    dummy_voxel_num_points = torch.randint(1, P + 1, (M,), device='cuda', dtype=torch.int32)

    # Output path
    if args.output is None:
        ckpt_path = Path(args.ckpt)
        output_path = ckpt_path.parent / f'{ckpt_path.stem}.onnx'
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Exporting ONNX to {output_path}...')
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (dummy_voxels, dummy_voxel_coords, dummy_voxel_num_points),
            str(output_path),
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=['voxels', 'voxel_coords', 'voxel_num_points'],
            output_names=['cls_preds', 'box_preds', 'dir_cls_preds'],
            dynamic_axes={
                'voxels': {0: 'num_voxels'},
                'voxel_coords': {0: 'num_voxels'},
                'voxel_num_points': {0: 'num_voxels'},
            },
        )

    print(f'Successfully exported ONNX model to {output_path}')

    # Validate
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print('ONNX model verification passed!')
    except ImportError:
        print('Warning: onnx package not installed, skipping verification')
    except Exception as e:
        print(f'Warning: ONNX model verification failed: {e}')

    return output_path


if __name__ == '__main__':
    args = parse_args()
    export_onnx(args)
