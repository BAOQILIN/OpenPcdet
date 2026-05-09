import argparse
import glob
from pathlib import Path

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
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_pp_multihead.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/bql/OpenPCDet/data/nuscenes/v1.0-trainval/samples/LIDAR_TOP',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--num_samples', type=int, default=10, help='number of samples to visualize')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='score threshold for visualization')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------NuScenes Model Demo-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    logger.info(f'Will visualize: \t{min(args.num_samples, len(demo_dataset))} samples')
    logger.info(f'Class names: {cfg.CLASS_NAMES}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    num_to_process = min(args.num_samples, len(demo_dataset))

    with torch.no_grad():
        for idx in range(num_to_process):
            logger.info(f'Processing sample index: \t{idx + 1}/{num_to_process}')
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

            logger.info(f'Detected {len(pred_boxes)} objects (score >= {args.score_thresh})')

            if len(pred_boxes) > 0:
                for class_idx, class_name in enumerate(cfg.CLASS_NAMES, start=1):
                    class_mask = pred_labels == class_idx
                    num_class = class_mask.sum().item()
                    if num_class > 0:
                        logger.info(f'  {class_name}: {num_class}')

                V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    ref_boxes=pred_boxes,
                    ref_scores=pred_scores,
                    ref_labels=pred_labels
                )
                if not OPEN3D_FLAG:
                    mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
