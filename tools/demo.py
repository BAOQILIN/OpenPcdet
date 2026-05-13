import argparse
import glob
import json
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
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError(f'Invalid point shape: {points.shape}, file={self.sample_file_list[index]}')
        points = points[:, :4]
        points = points[np.isfinite(points).all(axis=1)]

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--save_pred_dir', type=str, default=None, help='directory to save prediction txt/json')

    # 新增参数：禁用可视化
    parser.add_argument('--no_vis', action='store_true', help='disable visualization to speed up inference')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def save_predictions(sample_path, pred_dict, class_names, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_stem = Path(sample_path).stem

    pred_boxes = pred_dict['pred_boxes'].detach().cpu().numpy()
    pred_scores = pred_dict['pred_scores'].detach().cpu().numpy()
    pred_labels = pred_dict['pred_labels'].detach().cpu().numpy()

    txt_path = save_dir / f'{sample_stem}.txt'
    with open(txt_path, 'w') as f:
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            name = class_names[int(label) - 1]
            f.write(
                f"{name} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f} {box[5]:.6f} {box[6]:.6f} {score:.6f}\n"
            )

    json_path = save_dir / f'{sample_stem}.json'
    json_data = []
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        json_data.append({
            'name': class_names[int(label) - 1],
            'score': float(score),
            'box_lidar': [float(x) for x in box.tolist()]
        })
    with open(json_path, 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Processing sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            if args.save_pred_dir is not None:
                save_predictions(
                    sample_path=demo_dataset.sample_file_list[idx],
                    pred_dict=pred_dicts[0],
                    class_names=cfg.CLASS_NAMES,
                    save_dir=args.save_pred_dir
                )

            # ------------------ Disable visualization if specified ------------------
            if not args.no_vis:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels']
                )
                if not OPEN3D_FLAG:
                    mlab.show(stop=True)
            # ------------------------------------------------------------------------

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
