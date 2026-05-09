import argparse
import torch
import numpy as np
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils


class DummyDataset:
    """Dummy dataset for ONNX export"""
    def __init__(self, cfg):
        self.class_names = cfg.CLASS_NAMES
        self.point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)

        # Calculate grid size and voxel size from config
        voxel_size = cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE
        self.voxel_size = np.array(voxel_size)

        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        ).astype(np.int64)

        self.depth_downsample_factor = None

        # Point feature encoder
        self.point_feature_encoder = type('obj', (object,), {
            'num_point_features': 5  # x, y, z, intensity, timestamp
        })()

    @property
    def mode(self):
        return 'TEST'


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX format')
    parser.add_argument('--cfg_file', type=str, required=True, help='Config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file')
    parser.add_argument('--output', type=str, default=None, help='Output ONNX file path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for export')
    parser.add_argument('--max_points', type=int, default=30000, help='Maximum number of points')
    parser.add_argument('--opset_version', type=int, default=11, help='ONNX opset version')
    args = parser.parse_args()
    return args


class ONNXWrapper(torch.nn.Module):
    """Wrapper to handle batch_dict input/output for ONNX export"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxels, voxel_coords, voxel_num_points):
        batch_dict = {
            'voxels': voxels,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': 1  # Fixed batch size for export
        }

        for cur_module in self.model.module_list:
            batch_dict = cur_module(batch_dict)

        # Extract raw predictions
        batch_cls_preds = batch_dict['batch_cls_preds']
        batch_box_preds = batch_dict['batch_box_preds']

        # For CBGS: cls_preds is list of [1, N_i, C_i], box_preds is [N_total, 9]
        # We can't export list to ONNX, so just return box_preds
        # ARSBase will need to handle post-processing

        return batch_box_preds

        # NMS (using CPU-compatible operations for ONNX)
        if filtered_scores.shape[0] > 0:
            # Sort by score
            nms_pre_maxsize = self.post_process_cfg.NMS_CONFIG.NMS_PRE_MAXSIZE
            if filtered_scores.shape[0] > nms_pre_maxsize:
                _, indices = torch.topk(filtered_scores, k=nms_pre_maxsize)
                filtered_scores = filtered_scores[indices]
                filtered_labels = filtered_labels[indices]
                filtered_boxes = filtered_boxes[indices]

            # Simple NMS implementation (ONNX-compatible, CPU-only)
            keep_mask = self._nms_cpu(
                filtered_boxes[:, :7],  # [x, y, z, dx, dy, dz, heading]
                filtered_scores,
                self.post_process_cfg.NMS_CONFIG.NMS_THRESH
            )

            final_scores = filtered_scores[keep_mask]
            final_labels = filtered_labels[keep_mask]
            final_boxes = filtered_boxes[keep_mask]

            # Limit to post-NMS max size
            nms_post_maxsize = self.post_process_cfg.NMS_CONFIG.NMS_POST_MAXSIZE
            if final_scores.shape[0] > nms_post_maxsize:
                final_scores = final_scores[:nms_post_maxsize]
                final_labels = final_labels[:nms_post_maxsize]
                final_boxes = final_boxes[:nms_post_maxsize]
        else:
            # No detections
            final_scores = filtered_scores
            final_labels = filtered_labels
            final_boxes = filtered_boxes

        # Format outputs: labels [N], scores [N], boxes [N, 7]
        # Add batch dimension for compatibility: [1, N], [1, N], [1, N, 7]
        final_labels = final_labels.unsqueeze(0).float()  # [1, N]
        final_scores = final_scores.unsqueeze(0)  # [1, N]
        final_boxes = final_boxes.unsqueeze(0)  # [1, N, 7]

        return final_labels, final_scores, final_boxes

    def _nms_cpu(self, boxes, scores, iou_threshold):
        """
        CPU-based NMS implementation compatible with ONNX export.
        Args:
            boxes: [N, 7] (x, y, z, dx, dy, dz, heading)
            scores: [N]
            iou_threshold: float
        Returns:
            keep_mask: [N] boolean tensor
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=boxes.device)

        # Sort by score (descending)
        _, order = scores.sort(descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0].item()
            keep.append(i)

            # Compute IoU of the kept box with the rest
            ious = self._boxes_iou_bev(boxes[i:i+1], boxes[order[1:]])

            # Keep boxes with IoU less than threshold
            mask = ious[0] <= iou_threshold
            order = order[1:][mask]

        # Create boolean mask
        keep_mask = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)
        keep_mask[torch.tensor(keep, device=boxes.device)] = True

        return keep_mask

    def _boxes_iou_bev(self, boxes_a, boxes_b):
        """
        Compute BEV IoU between boxes (simplified, axis-aligned approximation).
        Args:
            boxes_a: [1, 7]
            boxes_b: [M, 7]
        Returns:
            ious: [1, M]
        """
        # Extract BEV coordinates (x, y, dx, dy)
        # Simplified: treat as axis-aligned boxes (ignore heading for ONNX compatibility)
        xa, ya, dxa, dya = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 3], boxes_a[:, 4]
        xb, yb, dxb, dyb = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 3], boxes_b[:, 4]

        # Compute box corners (axis-aligned)
        xa_min, xa_max = xa - dxa / 2, xa + dxa / 2
        ya_min, ya_max = ya - dya / 2, ya + dya / 2
        xb_min, xb_max = xb - dxb / 2, xb + dxb / 2
        yb_min, yb_max = yb - dyb / 2, yb + dyb / 2

        # Intersection
        inter_xmin = torch.max(xa_min, xb_min)
        inter_ymin = torch.max(ya_min, yb_min)
        inter_xmax = torch.min(xa_max, xb_max)
        inter_ymax = torch.min(ya_max, yb_max)

        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h

        # Union
        area_a = dxa * dya
        area_b = dxb * dyb
        union_area = area_a + area_b - inter_area

    # NMS (using CPU-compatible operations for ONNX)
        if filtered_scores.shape[0] > 0:
            # Sort by score
            nms_pre_maxsize = self.post_process_cfg.NMS_CONFIG.NMS_PRE_MAXSIZE
            if filtered_scores.shape[0] > nms_pre_maxsize:
                _, indices = torch.topk(filtered_scores, k=nms_pre_maxsize)
                filtered_scores = filtered_scores[indices]
                filtered_labels = filtered_labels[indices]
                filtered_boxes = filtered_boxes[indices]

            # Simple NMS implementation (ONNX-compatible, CPU-only)
            keep_mask = self._nms_cpu(
                filtered_boxes[:, :7],  # [x, y, z, dx, dy, dz, heading]
                filtered_scores,
                self.post_process_cfg.NMS_CONFIG.NMS_THRESH
            )

            final_scores = filtered_scores[keep_mask]
            final_labels = filtered_labels[keep_mask]
            final_boxes = filtered_boxes[keep_mask]

            # Limit to post-NMS max size
            nms_post_maxsize = self.post_process_cfg.NMS_CONFIG.NMS_POST_MAXSIZE
            if final_scores.shape[0] > nms_post_maxsize:
                final_scores = final_scores[:nms_post_maxsize]
                final_labels = final_labels[:nms_post_maxsize]
                final_boxes = final_boxes[:nms_post_maxsize]
        else:
            # No detections
            final_scores = filtered_scores
            final_labels = filtered_labels
            final_boxes = filtered_boxes

        # Format outputs: labels [N], scores [N], boxes [N, 7]
        # Add batch dimension for compatibility: [1, N], [1, N], [1, N, 7]
        final_labels = final_labels.unsqueeze(0).float()  # [1, N]
        final_scores = final_scores.unsqueeze(0)  # [1, N]
        final_boxes = final_boxes.unsqueeze(0)  # [1, N, 7]

        return final_labels, final_scores, final_boxes

    def _nms_cpu(self, boxes, scores, iou_threshold):
        """
        CPU-based NMS implementation compatible with ONNX export.
        Args:
            boxes: [N, 7] (x, y, z, dx, dy, dz, heading)
            scores: [N]
            iou_threshold: float
        Returns:
            keep_mask: [N] boolean tensor
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=boxes.device)

        # Sort by score (descending)
        _, order = scores.sort(descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0].item()
            keep.append(i)

            # Compute IoU of the kept box with the rest
            ious = self._boxes_iou_bev(boxes[i:i+1], boxes[order[1:]])

            # Keep boxes with IoU less than threshold
            mask = ious[0] <= iou_threshold
            order = order[1:][mask]

        # Create boolean mask
        keep_mask = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)
        keep_mask[torch.tensor(keep, device=boxes.device)] = True

        return keep_mask

    def _boxes_iou_bev(self, boxes_a, boxes_b):
        """
        Compute BEV IoU between boxes (simplified, axis-aligned approximation).
        Args:
            boxes_a: [1, 7]
            boxes_b: [M, 7]
        Returns:
            ious: [1, M]
        """
        # Extract BEV coordinates (x, y, dx, dy)
        # Simplified: treat as axis-aligned boxes (ignore heading for ONNX compatibility)
        xa, ya, dxa, dya = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 3], boxes_a[:, 4]
        xb, yb, dxb, dyb = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 3], boxes_b[:, 4]

        # Compute box corners (axis-aligned)
        xa_min, xa_max = xa - dxa / 2, xa + dxa / 2
        ya_min, ya_max = ya - dya / 2, ya + dya / 2
        xb_min, xb_max = xb - dxb / 2, xb + dxb / 2
        yb_min, yb_max = yb - dyb / 2, yb + dyb / 2

        # Intersection
        inter_xmin = torch.max(xa_min, xb_min)
        inter_ymin = torch.max(ya_min, yb_min)
        inter_xmax = torch.min(xa_max, xb_max)
        inter_ymax = torch.min(ya_max, yb_max)

        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h

        # Union
        area_a = dxa * dya
        area_b = dxb * dyb
        union_area = area_a + area_b - inter_area

        # IoU
        ious = inter_area / torch.clamp(union_area, min=1e-8)

        return ious.unsqueeze(0)  # [1, M]


def export_onnx(args):
    # Load config
    cfg_from_yaml_file(args.cfg_file, cfg)

    # Create dummy dataset
    dummy_dataset = DummyDataset(cfg)

    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dummy_dataset)
    model.cuda()
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    print(f'Loaded checkpoint from {args.ckpt}')
    print(f'Epoch: {checkpoint.get("epoch", "unknown")}')

    # Wrap model (no post-processing, just format conversion)
    wrapped_model = ONNXWrapper(model)

    # Create dummy inputs
    # These dimensions should match your actual data preprocessing
    max_voxels = 30000
    max_points_per_voxel = 20
    num_point_features = 5  # x, y, z, intensity, timestamp

    # Get grid size from dummy dataset
    grid_size = dummy_dataset.grid_size  # [x, y, z]

    dummy_voxels = torch.randn(max_voxels, max_points_per_voxel, num_point_features).cuda()
    # voxel_coords format: [batch_idx, z, y, x]
    # Make sure coordinates are within valid grid range
    dummy_voxel_coords = torch.zeros(max_voxels, 4, dtype=torch.int32).cuda()
    dummy_voxel_coords[:, 0] = 0  # batch_idx
    dummy_voxel_coords[:, 1] = torch.randint(0, int(grid_size[2]), (max_voxels,))  # z
    dummy_voxel_coords[:, 2] = torch.randint(0, int(grid_size[1]), (max_voxels,))  # y
    dummy_voxel_coords[:, 3] = torch.randint(0, int(grid_size[0]), (max_voxels,))  # x
    dummy_voxel_num_points = torch.randint(1, max_points_per_voxel + 1, (max_voxels,)).cuda()

    # Set output path
    if args.output is None:
        ckpt_path = Path(args.ckpt)
        output_path = ckpt_path.parent / f'{ckpt_path.stem}.onnx'
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    print(f'Exporting model to {output_path}...')

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (dummy_voxels, dummy_voxel_coords, dummy_voxel_num_points),
            str(output_path),
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=['voxels', 'voxel_coords', 'voxel_num_points'],
            output_names=['box_preds'],
            dynamic_axes={
                'voxels': {0: 'num_voxels'},
                'voxel_coords': {0: 'num_voxels'},
                'voxel_num_points': {0: 'num_voxels'}
            }
        )

    print(f'Successfully exported ONNX model to {output_path}')

    # Verify the exported model
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
