import copy
import json
import pickle
from pathlib import Path

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate


class ArsDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        project_root = Path(__file__).resolve().parents[3]
        root_path = Path(root_path) if root_path is not None else Path(dataset_cfg.DATA_PATH)
        if not root_path.is_absolute():
            root_path = project_root / root_path
        if dataset_cfg.get('AUTO_PREPARE_FROM_RAW', False):
            self.prepare_from_raw_if_needed(dataset_cfg, root_path, class_names, logger)
        super().__init__(dataset_cfg, class_names, training=training, root_path=root_path, logger=logger)

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / f'{self.split}.txt'
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else []
        self.ars_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    @staticmethod
    def _read_ascii_pcd(pcd_path: Path) -> np.ndarray:
        with open(pcd_path, 'r') as f:
            lines = f.readlines()

        data_start = None
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('data '):
                if 'ascii' not in line.strip().lower():
                    raise ValueError(f'Only ascii pcd is supported: {pcd_path}')
                data_start = i + 1
                break
        if data_start is None:
            raise ValueError(f'Invalid pcd file without DATA header: {pcd_path}')

        points = []
        for line in lines[data_start:]:
            s = line.strip()
            if not s:
                continue
            vals = s.split()
            if len(vals) < 4:
                continue
            points.append([float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])])

        if len(points) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        arr = np.asarray(points, dtype=np.float32)
        finite_mask = np.isfinite(arr).all(axis=1)
        return arr[finite_mask]

    @classmethod
    def prepare_from_raw_if_needed(cls, dataset_cfg, root_path: Path, class_names, logger=None):
        points_dir = root_path / 'points'
        labels_dir = root_path / 'labels'
        image_sets = root_path / 'ImageSets'
        train_list = image_sets / 'train.txt'
        val_list = image_sets / 'val.txt'

        if points_dir.exists() and labels_dir.exists() and train_list.exists() and val_list.exists():
            return

        raw_path = Path(dataset_cfg.RAW_DATA_PATH)
        if not raw_path.exists():
            raise FileNotFoundError(f'RAW_DATA_PATH not found: {raw_path}')

        points_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_sets.mkdir(parents=True, exist_ok=True)

        type_map = {int(k): v for k, v in dict(dataset_cfg.TYPE_ID_TO_NAME).items()}
        pcd_files = sorted(raw_path.glob('*.pcd'))
        all_ids = []

        for pcd_file in pcd_files:
            sample_id = pcd_file.stem
            json_file = raw_path / f'{sample_id}.json'
            if not json_file.exists():
                continue

            pts = cls._read_ascii_pcd(pcd_file)
            np.save(points_dir / f'{sample_id}.npy', pts)

            with open(json_file, 'r') as f:
                raw = json.load(f)

            annos = raw.get('object3D', {}).get('lidar_main', {}).get('annotation', [])
            label_lines = []
            for a in annos:
                t = a.get('type', None)
                if t is None:
                    continue
                name = type_map.get(int(t), None)
                if name is None or name not in class_names:
                    continue
                dim = a.get('dimension', None)
                pos = a.get('position', None)
                rot = a.get('rotation', None)
                if dim is None or pos is None or rot is None or len(dim) < 3 or len(pos) < 3 or len(rot) < 3:
                    continue
                line = f"{pos[0]} {pos[1]} {pos[2]} {dim[0]} {dim[1]} {dim[2]} {rot[2]} {name}\n"
                label_lines.append(line)

            with open(labels_dir / f'{sample_id}.txt', 'w') as f:
                f.writelines(label_lines)

            all_ids.append(sample_id)

        all_ids = sorted(set(all_ids))
        with open(train_list, 'w') as f:
            f.write('\n'.join(all_ids) + ('\n' if len(all_ids) > 0 else ''))
        with open(val_list, 'w') as f:
            f.write('\n'.join(all_ids) + ('\n' if len(all_ids) > 0 else ''))

        if logger is not None:
            logger.info('Prepared ars dataset from raw path: %s, samples=%d', str(raw_path), len(all_ids))

    def include_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading ARS dataset.')
        ars_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                ars_infos.extend(infos)
        self.ars_infos.extend(ars_infos)
        if self.logger is not None:
            self.logger.info('Total samples for ARS dataset: %d' % len(ars_infos))

    def get_label(self, idx):
        label_file = self.root_path / 'labels' / ('%s.txt' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split()
            if len(line_list) < 8:
                continue
            gt_boxes.append(line_list[:7])
            gt_names.append(line_list[7])

        if len(gt_boxes) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=str)
        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
        lidar_file = self.root_path / 'points' / ('%s.npy' % idx)
        assert lidar_file.exists()
        point_features = np.load(lidar_file)
        if point_features.ndim != 2 or point_features.shape[1] < 4:
            raise ValueError(f'Invalid lidar shape: {point_features.shape}, file={lidar_file}')
        point_features = point_features[:, :4]
        point_features = point_features[np.isfinite(point_features).all(axis=1)]
        return point_features

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else []
        self.ars_infos = []
        self.include_data('train' if split == 'train' else 'test')

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs
        return len(self.ars_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.ars_infos)

        info = copy.deepcopy(self.ars_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'frame_id': sample_idx,
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('ars_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {name: [] for name in self.class_names}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))
                    db_info = {
                        'name': names[i], 'path': db_path, 'gt_idx': i,
                        'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                        'difficulty': -1
                    }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_ars_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = ArsDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('ars_infos_%s.pkl' % train_split)
    val_filename = save_path / ('ars_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    ars_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(ars_infos_train, f)
    print('Ars info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    ars_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(ars_infos_val, f)
    print('Ars info val file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys
    import yaml
    from easydict import EasyDict

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_ars_infos':
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        create_ars_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Obstacle', 'Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle'],
            data_path=ROOT_DIR / 'data' / 'ars',
            save_path=ROOT_DIR / 'data' / 'ars',
        )
