import argparse
import pickle
from pathlib import Path
import warnings
import numpy as np

# Suppress NumPy deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except ImportError:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

# Apply warning filters after numpy import
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Visualize KITTI evaluation results')
    parser.add_argument('--cfg_file', type=str, required=True, help='Config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file')
    parser.add_argument('--data_path', type=str, default='../data/kitti', help='KITTI data path')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Data split')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='Score threshold for predictions')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    logger.info('=' * 80)
    logger.info('KITTI Evaluation Visualization')
    logger.info('=' * 80)
    logger.info(f'Config: {args.cfg_file}')
    logger.info(f'Checkpoint: {args.ckpt}')
    logger.info(f'Data split: {args.split}')
    logger.info(f'Score threshold: {args.score_thresh}')
    logger.info('=' * 80)

    # Build dataset
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )

    logger.info(f'Total samples in {args.split} set: {len(test_set)}')

    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    num_to_process = min(args.num_samples, len(test_set) - args.start_idx)
    logger.info(f'Visualizing {num_to_process} samples starting from index {args.start_idx}')
    logger.info('=' * 80)

    with torch.no_grad():
        for idx in range(args.start_idx, args.start_idx + num_to_process):
            logger.info(f'\n{"=" * 60}')
            logger.info(f'Sample {idx} ({idx - args.start_idx + 1}/{num_to_process})')

            # Get data
            data_dict = test_set[idx]
            data_dict = test_set.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            # Get ground truth
            gt_boxes = data_dict['gt_boxes'][0].cpu().numpy()

            # Run inference
            pred_dicts, _ = model.forward(data_dict)

            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            # Filter by score threshold
            mask = pred_scores >= args.score_thresh
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]

            # Remove padding from gt_boxes (boxes with all zeros)
            valid_gt_mask = np.any(gt_boxes != 0, axis=1)
            gt_boxes = gt_boxes[valid_gt_mask]

            logger.info(f'Ground truth boxes: {len(gt_boxes)}')
            logger.info(f'Predicted boxes (score >= {args.score_thresh}): {len(pred_boxes)}')

            if len(pred_boxes) > 0:
                for class_idx, class_name in enumerate(cfg.CLASS_NAMES, start=1):
                    class_mask = pred_labels == class_idx
                    num_class = class_mask.sum()
                    if num_class > 0:
                        logger.info(f'  {class_name}: {num_class}')

            # Visualize
            # Blue boxes: ground truth
            # Green boxes: predictions
            logger.info('Visualizing: Blue = Ground Truth, Green = Predictions')
            V.draw_scenes(
                points=data_dict['points'][:, 1:].cpu().numpy(),
                gt_boxes=gt_boxes,  # Blue boxes
                ref_boxes=pred_boxes,  # Green boxes
                ref_scores=pred_scores,
                ref_labels=pred_labels
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('\n' + '=' * 80)
    logger.info('Visualization completed!')


if __name__ == '__main__':
    main()
