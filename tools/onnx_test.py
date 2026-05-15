#!/usr/bin/env python3
"""
ONNX Inference Test & Comparison Script for PointPillar (3-way split).

Runs end-to-end ONNX inference: voxelize → VFE → scatter → Backbone2D → RPN
then decodes boxes and applies NMS. Supports comparing user-exported ONNX
against reference ONNX and PyTorch model outputs.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils.box_coder_utils import ResidualCoder

# ---------------------------------------------------------------------------
# Section A: Data Loading & Voxelization
# ---------------------------------------------------------------------------

def load_point_cloud(file_path):
    """Load .npy point cloud, return (N, 4) float32 [x, y, z, intensity]."""
    pts = np.load(file_path)
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError(f'Expected (N, >=4) got {pts.shape}: {file_path}')
    pts = pts[:, :4].astype(np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


def voxelize_numpy(points, point_cloud_range, voxel_size,
                   max_voxels=40000, max_points_per_voxel=20):
    """
    Pure-numpy voxelization matching spconv VoxelGenerator behaviour.

    Args:
        points: (N, 4) float32 [x, y, z, intensity]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz]
        max_voxels: max number of non-empty voxels to keep
        max_points_per_voxel: pad/truncate to this many points per voxel

    Returns:
        voxels:       (M, max_points_per_voxel, 4) float32
        num_points:   (M,) int32
        voxel_coords: (M, 4) int32  [batch_idx=0, z_idx, y_idx, x_idx]
    """
    range_min = np.array(point_cloud_range[:3], dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)

    grid_size = np.round(
        (np.array(point_cloud_range[3:6]) - range_min) / voxel_size
    ).astype(np.int32)
    nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

    # Clip points to valid range
    for d in range(3):
        valid = (points[:, d] >= point_cloud_range[d]) & \
                (points[:, d] <= point_cloud_range[d + 3])
        points = points[valid]

    if len(points) == 0:
        return (np.zeros((0, max_points_per_voxel, 4), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros((0, 4), dtype=np.int32))

    # Grid indices
    grid_xyz = np.floor((points[:, :3] - range_min) / voxel_size).astype(np.int32)
    for d in range(3):
        grid_xyz[:, d] = np.clip(grid_xyz[:, d], 0, grid_size[d] - 1)

    # Unique voxel key: z * ny * nx + y * nx + x
    voxel_keys = grid_xyz[:, 2] * (ny * nx) + grid_xyz[:, 1] * nx + grid_xyz[:, 0]

    unique_keys, inverse_indices, counts = np.unique(
        voxel_keys, return_inverse=True, return_counts=True
    )

    # Sort voxels by point count (descending), keep top max_voxels
    sort_idx = np.argsort(-counts)
    if len(sort_idx) > max_voxels:
        sort_idx = sort_idx[:max_voxels]

    M = len(sort_idx)
    voxels = np.zeros((M, max_points_per_voxel, 4), dtype=np.float32)
    num_points = np.zeros(M, dtype=np.int32)
    voxel_coords = np.zeros((M, 4), dtype=np.int32)

    kept_keys = unique_keys[sort_idx]
    kept_counts = counts[sort_idx]

    # Build a mapping from old unique index -> new voxel index
    old_to_new = {old: new for new, old in enumerate(sort_idx)}

    # Assign points to voxels
    for new_idx, old_idx in enumerate(sort_idx):
        mask = inverse_indices == old_idx
        pts_in_voxel = points[mask]
        n = min(len(pts_in_voxel), max_points_per_voxel)
        if n < len(pts_in_voxel):
            # Random sample if too many
            choice = np.random.choice(len(pts_in_voxel), n, replace=False)
            pts_in_voxel = pts_in_voxel[choice]
        voxels[new_idx, :n, :] = pts_in_voxel
        num_points[new_idx] = n

        # Recover coordinates from the original key
        key = int(kept_keys[new_idx])
        z_idx = key // (ny * nx)
        remainder = key % (ny * nx)
        y_idx = remainder // nx
        x_idx = remainder % nx
        voxel_coords[new_idx] = [0, z_idx, y_idx, x_idx]

    return voxels, num_points, voxel_coords


# ---------------------------------------------------------------------------
# Section B: ONNX Inference Helpers
# ---------------------------------------------------------------------------

def load_onnx_session(model_path):
    """Load an ONNX model into an InferenceSession."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), opts,
                                   providers=['CUDAExecutionProvider',
                                              'CPUExecutionProvider'])
    return session


def _get_onnx_dtype_str(inp):
    """Get dtype string ('float32', 'int32', 'int64') from ONNX input info."""
    type_str = inp.type
    # onnxruntime >= 1.20 returns string like 'tensor(float)' or 'tensor(int32)'
    if isinstance(type_str, str):
        if 'float' in type_str:
            return 'float32'
        elif 'int64' in type_str:
            return 'int64'
        elif 'int32' in type_str or 'int' in type_str:
            return 'int32'
        return 'float32'
    # Older API: inp.type.tensor_type.elem_type (1=float32, 6=int32, 7=int64)
    elem = type_str.tensor_type.elem_type
    return {1: 'float32', 6: 'int32', 7: 'int64'}.get(elem, 'float32')


def _to_dtype_str(arr, dtype_str):
    """Convert numpy array to the requested dtype string."""
    return np.asarray(arr, dtype={'float32': np.float32, 'int32': np.int32,
                                   'int64': np.int64}[dtype_str])


def run_vfe(session, voxel_features, voxel_num_points, voxel_coords):
    """Run VFE ONNX. Returns pillar_features (1, M, 64)."""
    sess_inputs = session.get_inputs()
    feed = {}
    for inp in sess_inputs:
        name_lower = inp.name.lower()
        dt = _get_onnx_dtype_str(inp)
        if 'feature' in name_lower or ('voxel' in name_lower and
                'coord' not in name_lower and 'num' not in name_lower and 'point' not in name_lower):
            feed[inp.name] = _to_dtype_str(voxel_features, dt)
        elif 'num' in name_lower or 'point' in name_lower:
            feed[inp.name] = _to_dtype_str(voxel_num_points, dt)
        elif 'coord' in name_lower:
            feed[inp.name] = _to_dtype_str(voxel_coords, dt)
        else:
            feed[inp.name] = _to_dtype_str(voxel_features, dt)

    # Ensure all 3 expected inputs are populated (positional fallback)
    if len(feed) < 3:
        for i, inp in enumerate(sess_inputs):
            if inp.name not in feed:
                dt = _get_onnx_dtype_str(inp)
                if i == 0:
                    feed[inp.name] = _to_dtype_str(voxel_features, dt)
                elif i == 1:
                    feed[inp.name] = _to_dtype_str(voxel_num_points, dt)
                else:
                    feed[inp.name] = _to_dtype_str(voxel_coords, dt)

    outputs = session.run(None, feed)
    return outputs[0]


def scatter_host(pillar_features, voxel_coords, grid_size=(512, 512, 1),
                 num_bev_features=64):
    """
    Host-side scatter: pillar_features + voxel_coords → spatial_features.

    Handles two VFE output formats:
      - (1, M, C) → user ONNX (max-pooled inside VFE)
      - (M, 20, C) → reference ONNX (per-point, needs max-pool host-side)

    Args:
        pillar_features: (1, M, C), (M, C), or (M, 20, C)
        voxel_coords: (M, 4) [batch, z, y, x]
        grid_size: (nx, ny, nz)
        num_bev_features: C

    Returns:
        spatial_features: (1, C, H, W)  with H=ny, W=nx
    """
    nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

    pf = np.asarray(pillar_features)

    # Handle different VFE output formats
    if pf.ndim == 3:
        if pf.shape[0] == 1:
            # User format: (1, M, C) → squeeze batch
            pf = pf.squeeze(0)  # (M, C)
        elif pf.shape[1] == 20:
            # Reference format: (M, 20, C) → max-pool over point dim
            pf = pf.max(axis=1)  # (M, C)
        # else (M, C) → keep as-is

    M, C = pf.shape
    assert C == num_bev_features, f'Expected {num_bev_features} features, got {C}'

    spatial = np.zeros((C, nz * nx * ny), dtype=np.float32)

    # indices = z + y * nx + x
    coords = np.asarray(voxel_coords, dtype=np.int64)
    indices = coords[:, 1] * (ny * nx) + coords[:, 2] * nx + coords[:, 3]
    # With nz=1, this simplifies to y * nx + x; keep full formula for safety

    # Clamp to valid range
    indices = np.clip(indices, 0, nz * nx * ny - 1)

    # Scatter via advanced indexing (each grid cell has unique voxel)
    spatial[:, indices] = pf.T

    spatial = spatial.reshape((1, C * nz, ny, nx))
    return spatial


def run_backbone(session, spatial_features):
    """Run Backbone2D ONNX. Returns spatial_features_2d (1, 384, H', W')."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: spatial_features.astype(np.float32)})
    return outputs[0]


def run_rpn(session, spatial_features_2d):
    """Run RPN ONNX. Returns (cls_preds, box_preds)."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: spatial_features_2d.astype(np.float32)})
    # outputs: [batch_cls_preds, batch_box_preds]
    return outputs[0], outputs[1]


def run_onnx_pipeline(sessions, voxel_features, voxel_num_points, voxel_coords,
                      grid_size=(512, 512, 1), num_bev_features=64):
    """Full ONNX pipeline: VFE → scatter → Backbone2D → RPN."""
    # VFE
    pillar_features = run_vfe(
        sessions['vfe'], voxel_features, voxel_num_points, voxel_coords
    )
    # Scatter (host)
    spatial_features = scatter_host(
        pillar_features, voxel_coords, grid_size, num_bev_features
    )
    # Backbone2D
    spatial_features_2d = run_backbone(sessions['backbone'], spatial_features)
    # RPN
    cls_preds, box_preds = run_rpn(sessions['rpn'], spatial_features_2d)
    return {
        'pillar_features': pillar_features,
        'spatial_features': spatial_features,
        'spatial_features_2d': spatial_features_2d,
        'cls_preds': cls_preds,
        'box_preds': box_preds,
    }


# ---------------------------------------------------------------------------
# Section C: Anchor Generation & Box Decoding
# ---------------------------------------------------------------------------

def generate_anchors_from_config(anchor_config, point_cloud_range, grid_size,
                                  feature_map_stride=4):
    """
    Replicate AnchorGenerator.generate_anchors() + MultiHeadAnchorHead flatten.

    Returns:
        anchors_flat: (total_anchors, 8) — 8D (7 + encode_angle_by_sincos padding)
        num_anchors_per_location: list of ints
    """
    anchor_range = np.array(point_cloud_range, dtype=np.float32)
    feature_map_size = np.array(grid_size[:2]) // feature_map_stride

    all_anchors = []
    num_anchors_per_location = []

    for cfg in anchor_config:
        anchor_sizes = cfg['anchor_sizes']  # list of [l, w, h]
        anchor_rotations = cfg['anchor_rotations']  # list of radians
        anchor_heights = cfg['anchor_bottom_heights']  # list of z values
        align_center = cfg.get('align_center', False)
        class_stride = cfg.get('feature_map_stride', feature_map_stride)
        fm_size = np.array(grid_size[:2]) // class_stride

        n_sizes = len(anchor_sizes)
        n_rots = len(anchor_rotations)
        n_heights = len(anchor_heights)
        num_anchors_per_location.append(n_sizes * n_rots * n_heights)

        if align_center:
            x_stride = (anchor_range[3] - anchor_range[0]) / fm_size[0]
            y_stride = (anchor_range[4] - anchor_range[1]) / fm_size[1]
            x_offset, y_offset = x_stride / 2, y_stride / 2
        else:
            x_stride = (anchor_range[3] - anchor_range[0]) / (fm_size[0] - 1)
            y_stride = (anchor_range[4] - anchor_range[1]) / (fm_size[1] - 1)
            x_offset, y_offset = 0.0, 0.0

        x_shifts = np.arange(anchor_range[0] + x_offset,
                             anchor_range[3] + 1e-5, x_stride, dtype=np.float32)
        y_shifts = np.arange(anchor_range[1] + y_offset,
                             anchor_range[4] + 1e-5, y_stride, dtype=np.float32)

        anchors_per_class = []
        for h in anchor_heights:
            for sz in anchor_sizes:
                for rot in anchor_rotations:
                    # Meshgrid: shape (ny, nx, 7)
                    ny, nx = len(y_shifts), len(x_shifts)
                    grid = np.zeros((ny, nx, 7), dtype=np.float32)
                    yy, xx = np.meshgrid(y_shifts, x_shifts, indexing='ij')
                    grid[:, :, 0] = xx
                    grid[:, :, 1] = yy
                    grid[:, :, 2] = h
                    grid[:, :, 3] = sz[0]   # dx = length
                    grid[:, :, 4] = sz[1]   # dy = width
                    grid[:, :, 5] = sz[2]   # dz = height
                    grid[:, :, 6] = rot

                    # Shift z to box center: z += dz / 2
                    grid[:, :, 2] += grid[:, :, 5] / 2.0

                    anchors_per_class.append(grid)

        # Stack sizes and rotations: (n_heights, n_sizes, n_rots, ny, nx, 7)
        # Need to interleave anchors in the same order as PyTorch's permute logic:
        # permute(2, 1, 0, 3, 4, 5) → (z, y, x, sizes, rots, 7)
        # In numpy: result shape should be (ny, nx, n_heights*n_sizes*n_rots, 7)
        # But PyTorch does meshgrid with indexing='ij' (x, y, z) → shape (x_grid, y_grid, z_grid)
        # Then permute(2, 1, 0, ...) → (z_grid, y_grid, x_grid, ...) → z=1, y=ny, x=nx
        # Final anchors shape: (1, ny, nx, n_heights, n_sizes, n_rots, 7)
        ny_g, nx_g = len(y_shifts), len(x_shifts)
        anchors = np.zeros((1, ny_g, nx_g, n_heights, n_sizes, n_rots, 7),
                           dtype=np.float32)
        idx = 0
        for hi in range(n_heights):
            for si in range(n_sizes):
                for ri in range(n_rots):
                    anchors[0, :, :, hi, si, ri, :] = anchors_per_class[idx]
                    idx += 1

        # Flatten: permute(2,1,0,3,4,5) → (ny, nx, 1, n_heights, n_sizes, n_rots, 7)
        # Then view(-1, 7)
        anchors = anchors.transpose(2, 1, 0, 3, 4, 5, 6)  # (ny, nx, 1, nh, ns, nr, 7)
        anchors = anchors.reshape(-1, 7)

        all_anchors.append(anchors)

    # Concatenate per-class anchors
    all_anchors_flat = np.concatenate(all_anchors, axis=0)  # (total, 7)

    # Pad to 8D for encode_angle_by_sincos
    anchors_8d = np.pad(all_anchors_flat, ((0, 0), (0, 1)), mode='constant')  # (total, 8)

    return anchors_8d, num_anchors_per_location


def decode_boxes_numpy(box_deltas, anchors_8d):
    """
    Replicate ResidualCoder.decode_torch() with encode_angle_by_sincos=True.

    Args:
        box_deltas: (A, 8) [xt, yt, zt, dxt, dyt, dzt, cost, sint]
        anchors_8d: (A, 8) [xa, ya, za, dxa, dya, dza, ra, 0]

    Returns:
        decoded_boxes: (A, 7) [x, y, z, dx, dy, dz, heading]
    """
    xa = anchors_8d[:, 0:1]
    ya = anchors_8d[:, 1:2]
    za = anchors_8d[:, 2:3]
    dxa = np.maximum(anchors_8d[:, 3:4], 1e-5)
    dya = np.maximum(anchors_8d[:, 4:5], 1e-5)
    dza = np.maximum(anchors_8d[:, 5:6], 1e-5)
    ra = anchors_8d[:, 6:7]

    xt = box_deltas[:, 0:1]
    yt = box_deltas[:, 1:2]
    zt = box_deltas[:, 2:3]
    dxt = box_deltas[:, 3:4]
    dyt = box_deltas[:, 4:5]
    dzt = box_deltas[:, 5:6]
    cost = box_deltas[:, 6:7]
    sint = box_deltas[:, 7:8]

    diagonal = np.sqrt(dxa ** 2 + dya ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * dza + za
    dxg = np.exp(dxt) * dxa
    dyg = np.exp(dyt) * dya
    dzg = np.exp(dzt) * dza
    rg_cos = cost + np.cos(ra)
    rg_sin = sint + np.sin(ra)
    rg = np.arctan2(rg_sin, rg_cos)

    return np.concatenate([xg, yg, zg, dxg, dyg, dzg, rg], axis=1)


# ---------------------------------------------------------------------------
# Section D: Post-processing (NMS)
# ---------------------------------------------------------------------------

def post_process_nms(boxes, scores, score_thresh=0.3, nms_thresh=0.2,
                     pre_maxsize=4096, post_maxsize=500,
                     nms_type='class_agnostic'):
    """
    Apply sigmoid, score threshold, and NMS.

    Args:
        boxes: (A, 7) decoded boxes
        scores: (A, 1) or (A,) raw cls logits
        score_thresh: minimum sigmoid score
        nms_thresh: IoU threshold for NMS
        pre_maxsize: top-K before NMS
        post_maxsize: max boxes after NMS

    Returns:
        pred_boxes: (K, 7), pred_scores: (K,), pred_labels: (K,)
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    # Stable sigmoid
    scores_sigmoid = np.where(
        scores >= 0,
        1.0 / (1.0 + np.exp(-scores)),
        np.exp(scores) / (1.0 + np.exp(scores))
    ).astype(np.float32)

    # Score threshold
    mask = scores_sigmoid >= score_thresh
    if not mask.any():
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32))

    boxes = boxes[mask]
    scores_sigmoid = scores_sigmoid[mask]

    # Sort by score, keep top pre_maxsize
    order = np.argsort(-scores_sigmoid)
    if len(order) > pre_maxsize:
        order = order[:pre_maxsize]
    boxes = boxes[order]
    scores_sigmoid = scores_sigmoid[order]

    # Rotated NMS (try GPU first, fallback to CPU)
    try:
        from pcdet.ops.iou3d_nms import iou3d_nms_utils
        boxes_t = torch.from_numpy(boxes).float().cuda()
        scores_t = torch.from_numpy(scores_sigmoid).float().cuda()
        keep, _ = iou3d_nms_utils.nms_gpu(boxes_t, scores_t, nms_thresh,
                                          pre_maxsize=None)
        keep = keep.cpu().numpy()
    except Exception:
        # CPU fallback with simple IoU NMS
        keep = _nms_cpu_fallback(boxes, scores_sigmoid, nms_thresh)

    if len(keep) > post_maxsize:
        keep = keep[:post_maxsize]

    pred_boxes = boxes[keep]
    pred_scores = scores_sigmoid[keep]
    pred_labels = np.ones(len(keep), dtype=np.int32)  # class-agnostic → label=1

    return pred_boxes, pred_scores, pred_labels


def _nms_cpu_fallback(boxes, scores, thresh):
    """Simple CPU NMS using BEV IoU (axis-aligned approximation)."""
    from shapely.geometry import Polygon, box as sbox
    keep = []
    areas = boxes[:, 3] * boxes[:, 4]  # dx * dy
    for i in range(len(boxes)):
        suppressed = False
        for j in keep:
            # Axis-aligned BEV IoU
            i_x1 = boxes[i, 0] - boxes[i, 3] / 2.0
            i_y1 = boxes[i, 1] - boxes[i, 4] / 2.0
            i_x2 = boxes[i, 0] + boxes[i, 3] / 2.0
            i_y2 = boxes[i, 1] + boxes[i, 4] / 2.0
            j_x1 = boxes[j, 0] - boxes[j, 3] / 2.0
            j_y1 = boxes[j, 1] - boxes[j, 4] / 2.0
            j_x2 = boxes[j, 0] + boxes[j, 3] / 2.0
            j_y2 = boxes[j, 1] + boxes[j, 4] / 2.0
            inter_w = max(0, min(i_x2, j_x2) - max(i_x1, j_x1))
            inter_h = max(0, min(i_y2, j_y2) - max(i_y1, j_y1))
            inter = inter_w * inter_h
            iou = inter / (areas[i] + areas[j] - inter + 1e-8)
            if iou > thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Section E: PyTorch Reference
# ---------------------------------------------------------------------------

class DummyDataset:
    """Minimal dataset for model building."""

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


def build_pytorch_model(cfg_file, ckpt_path):
    """Build model from config, load weights, return in eval mode."""
    cfg_from_yaml_file(cfg_file, cfg)
    dummy = DummyDataset(cfg)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
                          dataset=dummy)
    model.cuda()
    model.eval()
    checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    for param in model.parameters():
        param.requires_grad = False
    return model


def run_pytorch_modules(model, voxel_features, voxel_num_points, voxel_coords):
    """
    Run PyTorch model's module_list with pre-voxelized input.

    Returns a dict with intermediate outputs and final detections.
    """
    vfe = model.module_list[0]
    scatter = model.module_list[1]
    backbone_2d = model.module_list[2]
    dense_head = model.module_list[3]

    voxels_t = torch.from_numpy(voxel_features).float().cuda()
    coords_t = torch.from_numpy(voxel_coords).int().cuda()
    num_pts_t = torch.from_numpy(voxel_num_points).int().cuda()

    batch_dict = {
        'voxels': voxels_t,
        'voxel_coords': coords_t,
        'voxel_num_points': num_pts_t,
        'batch_size': 1,
    }

    with torch.no_grad():
        batch_dict = vfe(batch_dict)
        pillar_features = batch_dict['pillar_features'].cpu().numpy()

        batch_dict = scatter(batch_dict)
        spatial_features = batch_dict['spatial_features'].cpu().numpy()

        batch_dict = backbone_2d(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d'].cpu().numpy()

        batch_dict = dense_head(batch_dict)
        # dense_head returns batch_cls_preds (raw logits in ONNX mode) and
        # batch_box_preds (decoded 7D in non-ONNX mode, raw 8D in ONNX mode).
        # Capture raw outputs from forward_ret_dict for comparison with ONNX.
        raw_cls_from_head = dense_head.forward_ret_dict.get('cls_preds')  # (B, C, H, W)
        raw_box_parts = dense_head.forward_ret_dict.get('box_parts')

        cls_preds = batch_dict['batch_cls_preds'].cpu().numpy()
        box_preds = batch_dict['batch_box_preds'].cpu().numpy()

        pred_dicts, _ = model.post_processing(batch_dict)

    # Build raw RPN outputs (matching ONNX format) from forward_ret_dict
    if raw_cls_from_head is not None and raw_box_parts is not None:
        B, C_cls, H, W = raw_cls_from_head.shape
        raw_cls = raw_cls_from_head.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
        box_chunks = []
        for reg_i, height_i, size_i, angle_i in raw_box_parts:
            box_chunks.append(torch.cat([reg_i, height_i, size_i, angle_i], dim=1))
        raw_box = torch.cat(box_chunks, dim=1)
        raw_box = raw_box.permute(0, 2, 3, 1).contiguous().view(B, -1, 8)
        cls_preds = raw_cls.cpu().numpy()
        box_preds = raw_box.cpu().numpy()

    preds = pred_dicts[0] if pred_dicts else {}
    return {
        'pillar_features': pillar_features,
        'spatial_features': spatial_features,
        'spatial_features_2d': spatial_features_2d,
        'cls_preds': cls_preds,
        'box_preds': box_preds,
        'pred_boxes': preds.get('pred_boxes', np.zeros((0, 7))).cpu().numpy()
            if isinstance(preds.get('pred_boxes'), torch.Tensor)
            else np.asarray(preds.get('pred_boxes', [])),
        'pred_scores': preds.get('pred_scores', np.zeros(0)).cpu().numpy()
            if isinstance(preds.get('pred_scores'), torch.Tensor)
            else np.asarray(preds.get('pred_scores', [])),
        'pred_labels': preds.get('pred_labels', np.zeros(0)).cpu().numpy()
            if isinstance(preds.get('pred_labels'), torch.Tensor)
            else np.asarray(preds.get('pred_labels', [])),
    }


# ---------------------------------------------------------------------------
# Section F: Comparison
# ---------------------------------------------------------------------------

def max_abs_diff(a, b):
    """Return (mean_abs_diff, max_abs_diff, shape_a, shape_b)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    shape_info = f'{list(a.shape)} vs {list(b.shape)}'
    if a.shape != b.shape:
        return float('nan'), float('nan'), shape_info
    diff = np.abs(a - b)
    return float(np.mean(diff)), float(np.max(diff)), shape_info


def compare_intermediate(results, label_a, label_b):
    """Compare intermediate feature maps side-by-side."""
    keys = ['pillar_features', 'spatial_features', 'spatial_features_2d',
            'cls_preds', 'box_preds']
    report = {}
    for k in keys:
        if k in results[label_a] and k in results[label_b]:
            a = results[label_a][k]
            b = results[label_b][k]
            mean_d, max_d, shape_info = max_abs_diff(a, b)
            report[k] = {'shape': shape_info,
                         'mean_abs_diff': mean_d, 'max_abs_diff': max_d}
    return report


def match_detections(boxes_a, boxes_b, iou_thresh=0.5):
    """Greedy match between two sets of 3D boxes using BEV IoU."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return [], [], []

    # Simple axis-aligned BEV IoU matrix
    iou_mat = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    for i in range(len(boxes_a)):
        for j in range(len(boxes_b)):
            i_x1 = boxes_a[i, 0] - boxes_a[i, 3] / 2
            i_y1 = boxes_a[i, 1] - boxes_a[i, 4] / 2
            i_x2 = boxes_a[i, 0] + boxes_a[i, 3] / 2
            i_y2 = boxes_a[i, 1] + boxes_a[i, 4] / 2
            j_x1 = boxes_b[j, 0] - boxes_b[j, 3] / 2
            j_y1 = boxes_b[j, 1] - boxes_b[j, 4] / 2
            j_x2 = boxes_b[j, 0] + boxes_b[j, 3] / 2
            j_y2 = boxes_b[j, 1] + boxes_b[j, 4] / 2
            iw = max(0, min(i_x2, j_x2) - max(i_x1, j_x1))
            ih = max(0, min(i_y2, j_y2) - max(i_y1, j_y1))
            inter = iw * ih
            area_a = boxes_a[i, 3] * boxes_a[i, 4]
            area_b = boxes_b[j, 3] * boxes_b[j, 4]
            iou_mat[i, j] = inter / (area_a + area_b - inter + 1e-8)

    matched_a, matched_b, ious = [], [], []
    remaining = list(range(len(boxes_b)))
    for i in range(len(boxes_a)):
        best_j = -1
        best_iou = iou_thresh
        for j in remaining:
            if iou_mat[i, j] > best_iou:
                best_iou = iou_mat[i, j]
                best_j = j
        if best_j >= 0:
            matched_a.append(i)
            matched_b.append(best_j)
            ious.append(best_iou)
            remaining.remove(best_j)

    return matched_a, matched_b, ious


# ---------------------------------------------------------------------------
# Section G: Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='ONNX Inference Test & Comparison for PointPillar'
    )
    p.add_argument('--cfg_file', type=str,
                   default='tools/cfgs/ars_models/pointpillar.yaml')
    p.add_argument('--ckpt', type=str,
                   default='output/ars_models/pointpillar/default/ckpt/latest_model.pth')
    p.add_argument('--data_dir', type=str, default='data/ars/points')
    p.add_argument('--user_onnx_dir', type=str,
                   default='output/ars_models/pointpillar/default/ckpt')
    p.add_argument('--user_vfe', type=str, default=None,
                   help='User VFE ONNX path (overrides --user_onnx_dir)')
    p.add_argument('--user_backbone', type=str, default=None)
    p.add_argument('--user_rpn', type=str, default=None)
    p.add_argument('--ref_onnx_dir', type=str,
                   default='/home/hirain/ARS/TEST_PROJECT/DongFeng/perception_model/pointpillar/pandarat128')
    p.add_argument('--num_files', type=int, default=5)
    p.add_argument('--max_voxels', type=int, default=40000)
    p.add_argument('--score_thresh', type=float, default=0.3)
    p.add_argument('--nms_thresh', type=float, default=0.2)
    p.add_argument('--save_dir', type=str,
                   default='output/onnx_test_results')
    p.add_argument('--skip_pt', action='store_true',
                   help='Skip PyTorch model comparison')
    p.add_argument('--skip_ref', action='store_true',
                   help='Skip reference ONNX comparison')
    p.add_argument('--first_only', action='store_true',
                   help='Process only the first point cloud file')
    return p.parse_args()


def main():
    args = parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger(__name__)

    # Resolve all paths to absolute early (before chdir)
    tools_dir = ROOT / 'tools'

    cfg_file_path = Path(args.cfg_file)
    if not cfg_file_path.is_absolute():
        if (ROOT / args.cfg_file).exists():
            cfg_file_path = ROOT / args.cfg_file
        elif (tools_dir / args.cfg_file).exists():
            cfg_file_path = tools_dir / args.cfg_file

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / args.ckpt

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / args.data_dir

    user_onnx_dir = Path(args.user_onnx_dir)
    if not user_onnx_dir.is_absolute():
        user_onnx_dir = ROOT / args.user_onnx_dir

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = ROOT / args.save_dir

    user_vfe_path = args.user_vfe or str(user_onnx_dir / 'latest_model_vfe.onnx')
    user_backbone_path = args.user_backbone or str(user_onnx_dir / 'latest_model_backbone2d.onnx')
    user_rpn_path = args.user_rpn or str(user_onnx_dir / 'latest_model_rpn.onnx')

    ref_dir = Path(args.ref_onnx_dir)
    ref_vfe_path = str(ref_dir / 'vfe.onnx')
    ref_backbone_path = str(ref_dir / 'backbone2D.onnx')
    ref_rpn_path = str(ref_dir / 'rpn.onnx')

    # Config loading must run from tools/ (due to relative _BASE_CONFIG_ paths)
    os.chdir(tools_dir)
    try:
        cfg_file_rel = str(cfg_file_path.relative_to(tools_dir))
    except ValueError:
        cfg_file_rel = str(cfg_file_path)

    # Load config for params
    cfg_from_yaml_file(cfg_file_rel, cfg)
    pcr = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    vs = cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE
    grid_size = np.round((pcr[3:6] - pcr[0:3]) / np.array(vs)).astype(np.int32)

    log.info(f'Config: point_cloud_range={pcr}, voxel_size={vs}, grid_size={grid_size}')
    log.info(f'Classes: {cfg.CLASS_NAMES}')

    # Locate point cloud files
    pc_files = sorted(data_dir.glob('*.npy'))
    if not pc_files:
        log.error(f'No .npy files found in {data_dir}')
        sys.exit(1)
    if args.first_only:
        pc_files = pc_files[:1]
    elif args.num_files > 0:
        pc_files = pc_files[:args.num_files]
    log.info(f'Processing {len(pc_files)} point cloud files')

    # Load ONNX sessions
    log.info('Loading ONNX sessions...')
    user_sessions = {
        'vfe': load_onnx_session(user_vfe_path),
        'backbone': load_onnx_session(user_backbone_path),
        'rpn': load_onnx_session(user_rpn_path),
    }
    log.info(f'  User VFE:    {user_vfe_path}')
    log.info(f'  User Backbone: {user_backbone_path}')
    log.info(f'  User RPN:    {user_rpn_path}')

    ref_sessions = None
    if not args.skip_ref and ref_vfe_path and Path(ref_vfe_path).exists():
        ref_sessions = {
            'vfe': load_onnx_session(ref_vfe_path),
            'backbone': load_onnx_session(ref_backbone_path),
            'rpn': load_onnx_session(ref_rpn_path),
        }
        log.info(f'  Ref VFE:     {ref_vfe_path}')
        log.info(f'  Ref Backbone: {ref_backbone_path}')
        log.info(f'  Ref RPN:     {ref_rpn_path}')
    else:
        log.info('  Reference ONNX: skipped')

    # Build PyTorch model (optional)
    pt_model = None
    if not args.skip_pt:
        log.info('Building PyTorch model...')
        pt_model = build_pytorch_model(cfg_file_rel, str(ckpt_path))
        log.info('  Model loaded successfully')

    # Generate anchors for user ONNX decoding
    anchor_config = cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG
    user_anchors_8d, num_per_loc = generate_anchors_from_config(
        anchor_config, pcr, grid_size
    )
    log.info(f'User anchors: {user_anchors_8d.shape[0]} total, '
             f'{num_per_loc} per location')

    # Generate reference anchors (approximate: 5 classes × 2 anchors = 10)
    # Use same parameters as user but with 5 classes (we don't have the exact ref config)
    # For comparison, the anchor details don't affect feature-level diffs
    ref_anchors_8d = None
    if ref_sessions is not None:
        # Infer anchor count from RPN output
        dummy_in = np.zeros((1, 384, 128, 128), dtype=np.float32)
        ref_cls_shape = ref_sessions['rpn'].run(None,
            {ref_sessions['rpn'].get_inputs()[0].name: dummy_in})[0].shape
        ref_A = ref_cls_shape[1]
        ref_anchors_per_loc = ref_A // (grid_size[0] // 4 * grid_size[1] // 4)
        log.info(f'Reference anchors: {ref_A} total, ~{ref_anchors_per_loc} per location')
        # Generate approximate ref anchors — uses same geometry but different class count
        # This is for rough comparison only; the exact anchor params differ
        ref_anchors_8d, _ = generate_anchors_from_config(
            anchor_config, pcr, grid_size
        )
        # Pad/truncate to match if needed (happens when class count differs)
        if ref_anchors_8d.shape[0] != ref_A:
            log.warning(f'Reference anchor count mismatch: generated '
                        f'{ref_anchors_8d.shape[0]} vs expected {ref_A}. '
                        f'Detection comparison may be inaccurate.')

    # Process each point cloud
    all_results = []
    for idx, pc_file in enumerate(pc_files):
        log.info(f'[{idx+1}/{len(pc_files)}] Processing {pc_file.name}')

        # Load & voxelize
        pts = load_point_cloud(str(pc_file))
        log.info(f'  Points: {pts.shape[0]}')

        voxels, num_pts, coords = voxelize_numpy(
            pts, pcr, vs, max_voxels=args.max_voxels, max_points_per_voxel=20
        )
        M = voxels.shape[0]
        log.info(f'  Voxels: {M}')

        if M == 0:
            log.warning('  No voxels generated, skipping')
            continue

        result = {'file': pc_file.name, 'num_points': len(pts), 'num_voxels': M}

        # ---- User ONNX ----
        t0 = time.time()
        user_out = run_onnx_pipeline(
            user_sessions, voxels, num_pts, coords,
            grid_size=tuple(grid_size), num_bev_features=64
        )
        user_time = time.time() - t0
        log.info(f'  User ONNX inference: {user_time*1000:.1f} ms')

        # Decode user ONNX
        user_cls = user_out['cls_preds'].squeeze(0)  # (A, 1)
        user_box = user_out['box_preds'].squeeze(0)  # (A, 8)
        user_decoded = decode_boxes_numpy(user_box, user_anchors_8d)
        user_boxes, user_scores, user_labels = post_process_nms(
            user_decoded, user_cls, score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh
        )
        log.info(f'  User detections: {len(user_boxes)} boxes')
        result['user_detections'] = len(user_boxes)

        # ---- Reference ONNX ----
        if ref_sessions is not None:
            t0 = time.time()
            ref_out = run_onnx_pipeline(
                ref_sessions, voxels, num_pts, coords,
                grid_size=tuple(grid_size), num_bev_features=64
            )
            ref_time = time.time() - t0
            log.info(f'  Ref ONNX inference: {ref_time*1000:.1f} ms')

            # Decode ref ONNX
            ref_cls = ref_out['cls_preds'].squeeze(0)
            ref_box = ref_out['box_preds'].squeeze(0)
            if ref_anchors_8d is not None and ref_anchors_8d.shape[0] == ref_cls.shape[0]:
                ref_decoded = decode_boxes_numpy(ref_box, ref_anchors_8d)
            else:
                ref_decoded = np.zeros((ref_cls.shape[0], 7), dtype=np.float32)
            ref_boxes, ref_scores, ref_labels = post_process_nms(
                ref_decoded, ref_cls, score_thresh=args.score_thresh,
                nms_thresh=args.nms_thresh
            )
            log.info(f'  Ref detections: {len(ref_boxes)} boxes')
            result['ref_detections'] = len(ref_boxes)

            # Feature-level comparison: user vs ref
            feat_diff = compare_intermediate(
                {'user': user_out, 'ref': ref_out}, 'user', 'ref'
            )
            for k, v in feat_diff.items():
                log.info(f'  Diff {k}: mean={v["mean_abs_diff"]:.6e} '
                         f'max={v["max_abs_diff"]:.6e}')
            result['feature_diffs_user_ref'] = feat_diff

        # ---- PyTorch Model ----
        if pt_model is not None:
            t0 = time.time()
            pt_out = run_pytorch_modules(pt_model, voxels, num_pts, coords)
            pt_time = time.time() - t0
            log.info(f'  PyTorch inference: {pt_time*1000:.1f} ms')
            log.info(f'  PyTorch detections: {len(pt_out["pred_boxes"])} boxes')
            result['pt_detections'] = len(pt_out['pred_boxes'])

            # Feature-level comparison: user ONNX vs PyTorch
            feat_diff = compare_intermediate(
                {'user': user_out, 'pt': pt_out}, 'user', 'pt'
            )
            for k, v in feat_diff.items():
                log.info(f'  Diff user_vs_pt {k}: mean={v["mean_abs_diff"]:.6e} '
                         f'max={v["max_abs_diff"]:.6e}')
            result['feature_diffs_user_pt'] = feat_diff

            # Detection comparison
            if len(user_boxes) > 0 and len(pt_out['pred_boxes']) > 0:
                ma, mb, ious = match_detections(user_boxes, pt_out['pred_boxes'])
                log.info(f'  Matched detections (user vs pt): {len(ma)} pairs, '
                         f'mean IoU={np.mean(ious):.4f}' if ious else '  No matches')
                result['matched_user_pt'] = {
                    'num_matches': len(ma),
                    'mean_iou': float(np.mean(ious)) if ious else 0.0,
                }

        all_results.append(result)

    # ---- Summary Report ----
    log.info(f'\n{"="*60}')
    log.info(f'Summary ({len(all_results)} files)')
    log.info(f'{"="*60}')

    for k in ['feature_diffs_user_ref', 'feature_diffs_user_pt']:
        # Aggregate across files
        stage_means = {}
        for r in all_results:
            if k in r:
                for stage, v in r[k].items():
                    if v['max_abs_diff'] is not None and not np.isnan(v['max_abs_diff']):
                        stage_means.setdefault(stage, []).append(v['max_abs_diff'])
        if stage_means:
            log.info(f'{k}:')
            for stage, vals in stage_means.items():
                log.info(f'  {stage}: avg_max_diff={np.mean(vals):.6e} '
                         f'(min={np.min(vals):.6e}, max={np.max(vals):.6e})')

    # Save detailed results (save_dir already resolved to absolute above)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / 'onnx_test_results.json'

    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    log.info(f'\nDetailed results saved to {out_path}')

    return all_results


if __name__ == '__main__':
    main()
