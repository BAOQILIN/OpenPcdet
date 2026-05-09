import argparse
import glob
from pathlib import Path
import os

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

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='NuScenes Model Evaluation Demo')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml',
                        help='Config file')
    parser.add_argument('--data_path', type=str, default='/home/bql/OpenPCDet/data/nuscenes/v1.0-trainval/samples/LIDAR_TOP',
                        help='Point cloud data directory')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file')
    parser.add_argument('--ext', type=str, default='.bin', help='Point cloud file extension')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--save_dir', type=str, default='demo_results', help='Directory to save results')
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    logger.info('=' * 80)
    logger.info('NuScenes Model Evaluation Demo')
    logger.info('=' * 80)
    logger.info(f'Config: {args.cfg_file}')
    logger.info(f'Checkpoint: {args.ckpt}')
    logger.info(f'Data path: {args.data_path}')
    logger.info(f'Score threshold: {args.score_thresh}')
    logger.info(f'Save directory: {args.save_dir}')

    os.makedirs(args.save_dir, exist_ok=True)

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    logger.info(f'Total samples: {len(demo_dataset)}')
    logger.info(f'Processing: {min(args.num_samples, len(demo_dataset))} samples')
    logger.info(f'Classes: {cfg.CLASS_NAMES}')
    logger.info('=' * 80)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    num_to_process = min(args.num_samples, len(demo_dataset))

    total_detections = {class_name: 0 for class_name in cfg.CLASS_NAMES}
    results = []

    with torch.no_grad():
        for idx in range(num_to_process):
            logger.info(f'\n{"=" * 60}')
            logger.info(f'Sample {idx + 1}/{num_to_process}')

            data_dict = demo_dataset[idx]
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_boxes = pred_dicts[0]['pred_boxes']
            pred_scores = pred_dicts[0]['pred_scores']
            pred_labels = pred_dicts[0]['pred_labels']

            mask = pred_scores >= args.score_thresh
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]

            logger.info(f'Detected: {len(pred_boxes)} objects')

            sample_result = {'sample_idx': idx, 'detections': {}}

            if len(pred_boxes) > 0:
                for class_idx, class_name in enumerate(cfg.CLASS_NAMES, start=1):
                    class_mask = pred_labels == class_idx
                    num_class = class_mask.sum().item()
                    if num_class > 0:
                        logger.info(f'  {class_name}: {num_class}')
                        total_detections[class_name] += num_class
                        sample_result['detections'][class_name] = num_class

                if not args.no_vis:
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:],
                        ref_boxes=pred_boxes,
                        ref_scores=pred_scores,
                        ref_labels=pred_labels
                    )
                    if not OPEN3D_FLAG:
                        mlab.show(stop=True)

            results.append(sample_result)

    logger.info('\n' + '=' * 80)
    logger.info('Summary Statistics')
    logger.info('=' * 80)
    logger.info(f'Total samples processed: {num_to_process}')
    logger.info(f'Total detections by class:')
    for class_name, count in total_detections.items():
        if count > 0:
            logger.info(f'  {class_name}: {count}')

    result_file = os.path.join(args.save_dir, 'detection_results.txt')
    with open(result_file, 'w') as f:
        f.write('Detection Results\n')
        f.write('=' * 80 + '\n')
        f.write(f'Checkpoint: {args.ckpt}\n')
        f.write(f'Score threshold: {args.score_thresh}\n')
        f.write(f'Samples processed: {num_to_process}\n\n')

        f.write('Total detections by class:\n')
        for class_name, count in total_detections.items():
            if count > 0:
                f.write(f'  {class_name}: {count}\n')

        f.write('\n' + '=' * 80 + '\n')
        f.write('Per-sample results:\n')
        for result in results:
            f.write(f"\nSample {result['sample_idx']}:\n")
            if result['detections']:
                for class_name, count in result['detections'].items():
                    f.write(f"  {class_name}: {count}\n")
            else:
                f.write("  No detections\n")

    logger.info(f'\nResults saved to: {result_file}')
    logger.info('=' * 80)
    logger.info('Demo completed!')


if __name__ == '__main__':
    main()
