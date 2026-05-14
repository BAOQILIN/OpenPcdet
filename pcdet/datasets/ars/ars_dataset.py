import copy
import json
import pickle
from pathlib import Path

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate


def _read_ascii_pcd_static(pcd_path: Path) -> np.ndarray:
    """
    静态方法:读取Ascii格式的pcd点云文件(用于多进程)。
    Args:
        pcd_path: pcd文件路径
    Returns:
        np.ndarray: Nx4的点云数组 (x, y, z, intensity)
    """
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


def _process_sample_worker(args):
    """
    多进程worker函数:处理单个样本 pcd→npy, json→txt。
    Args:
        args: (pcd_file, raw_path, points_dir, labels_dir, use_new_format, type_map, class_names)
    Returns:
        (sample_id, success, skipped_invalid)
    """
    pcd_file, raw_path, points_dir, labels_dir, use_new_format, type_map, class_names = args

    sample_id = pcd_file.stem
    json_file = raw_path / f'{sample_id}.json'
    if not json_file.exists():
        return sample_id, False, 0

    # 读取并保存点云
    pts = _read_ascii_pcd_static(pcd_file)
    np.save(points_dir / f'{sample_id}.npy', pts)

    # 读取并解析标注
    with open(json_file, 'r') as f:
        raw = json.load(f)

    label_lines = []
    skipped = 0

    if use_new_format:
        moving_objects = raw.get('movingObjects', [])
        for obj in moving_objects:
            object_type = obj.get('objectType', None)
            if object_type is None:
                continue
            name = type_map.get(object_type, None)
            if name is None or name not in class_names:
                continue

            anno_tool = obj.get('annotationTool', {})
            cuboid = anno_tool.get('cuboid3D', {})
            if cuboid.get('flag', 0) != 1:
                skipped += 1
                continue

            value = cuboid.get('value', {})
            dim = value.get('cuboidExtent', None)
            pos = value.get('position', None)
            orient = value.get('orientation', None)

            if dim is None or pos is None or orient is None:
                continue
            if len(dim) < 3 or len(pos) < 3 or len(orient) < 3:
                continue

            heading = orient[2]
            line = f"{pos[0]} {pos[1]} {pos[2]} {dim[0]} {dim[1]} {dim[2]} {heading} {name}\n"
            label_lines.append(line)
    else:
        annos = raw.get('object3D', {}).get('lidar_main', {}).get('annotation', [])
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

    return sample_id, True, skipped


"""
ArsDataset: 针对ARS(自定义）数据集的数据处理类。
继承自 DatasetTemplate,负责数据集的解析、加载、划分以及数据预处理。
"""
class ArsDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        初始化ArsDataset。
        Args:
            dataset_cfg: 数据集配置对象。
            class_names: 需要检测的类别名称列表。
            training: 是否处于训练模式。
            root_path: 数据集根目录。
            logger: 日志记录器。
        """
        # 获取项目的绝对根路径
        project_root = Path(__file__).resolve().parents[3]
        # 解析真实根路径
        root_path = Path(root_path) if root_path is not None else Path(dataset_cfg.DATA_PATH)
        if not root_path.is_absolute():
            root_path = project_root / root_path
        
        # 自动从原始数据准备数据格式 (pcd和json -> npy和txt)
        if dataset_cfg.get('AUTO_PREPARE_FROM_RAW', False):
            self.prepare_from_raw_if_needed(dataset_cfg, root_path, class_names, logger)
        
        # 调用父类初始化生成数据增强和预处理组件
        super().__init__(dataset_cfg, class_names, training=training, root_path=root_path, logger=logger)

        # 获取划分方式 (train / val / test)
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # 对应划分文件的路径存放具体的样本名文件
        split_dir = self.root_path / 'ImageSets' / f'{self.split}.txt'
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else []
        
        self.ars_infos = []
        # 将相应模式(train或test)的info pkl文件载入到 self.ars_infos 中
        self.include_data(self.mode)
        # 是否映射到KITTI类别 (通常用于评估或转移时使用一致的名字)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    @staticmethod
    def _read_ascii_pcd(pcd_path: Path) -> np.ndarray:
        """
        读取Ascii格式的pcd点云文件。
        Args:
            pcd_path: pcd文件路径
        Returns:
            np.ndarray: Nx4的点云数组 (x, y, z, intensity)
        """
        with open(pcd_path, 'r') as f:
            lines = f.readlines()

        data_start = None
        # 查找数据开始行 (DATA ascii)
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('data '):
                if 'ascii' not in line.strip().lower():
                    raise ValueError(f'Only ascii pcd is supported: {pcd_path}')
                data_start = i + 1
                break
        if data_start is None:
            raise ValueError(f'Invalid pcd file without DATA header: {pcd_path}')

        points = []
        # 逐行解析点云坐标及强度
        for line in lines[data_start:]:
            s = line.strip()
            if not s:
                continue
            vals = s.split()
            if len(vals) < 4:
                continue
            points.append([float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])])

        # 返回Nx4格式的数据,并过滤非法坐标(如无穷大, NaN等)
        if len(points) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        arr = np.asarray(points, dtype=np.float32)
        finite_mask = np.isfinite(arr).all(axis=1)
        return arr[finite_mask]

    @classmethod
    def prepare_from_raw_if_needed(cls, dataset_cfg, root_path: Path, class_names, logger=None):
        """
        将原始JSON和PCD格式转换为本项目便于读取的TXT和NPY格式,同时生成划分的txt文件。
        在AUTO_PREPARE_FROM_RAW配置为True时触发。
        支持两种JSON格式：
          - 旧格式: object3D.lidar_main.annotation[] (type 为数字ID)
          - 新格式: movingObjects[].annotationTool.cuboid3D (objectType 为字符串)
        使用多进程并行处理以加速数据转换(绕过Python GIL限制)。
        """
        import concurrent.futures
        import multiprocessing

        points_dir = root_path / 'points'
        labels_dir = root_path / 'labels'
        image_sets = root_path / 'ImageSets'
        train_list = image_sets / 'train.txt'
        val_list = image_sets / 'val.txt'

        # 如果数据目录和划分文件都已存在,则认为已经准备就绪
        if points_dir.exists() and labels_dir.exists() and train_list.exists() and val_list.exists():
            return

        raw_path = Path(dataset_cfg.RAW_DATA_PATH)
        if not raw_path.exists():
            raise FileNotFoundError(f'RAW_DATA_PATH not found: {raw_path}')

        # 创建目标文件夹
        points_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_sets.mkdir(parents=True, exist_ok=True)

        # 检测 JSON 格式：RAW_OBJECT_TYPE_MAP 存在 → 新格式
        use_new_format = dataset_cfg.get('RAW_OBJECT_TYPE_MAP') is not None
        if use_new_format:
            type_map = dict(dataset_cfg.RAW_OBJECT_TYPE_MAP)
            if logger is not None:
                logger.info('Using new JSON format (movingObjects), type_map=%s', type_map)
        else:
            type_map = {int(k): v for k, v in dict(dataset_cfg.TYPE_ID_TO_NAME).items()}
            if logger is not None:
                logger.info('Using old JSON format (object3D.lidar_main.annotation), type_map=%s', type_map)

        pcd_files = sorted(raw_path.glob('*.pcd'))
        if logger is not None:
            logger.info('Found %d PCD files, starting multi-process conversion...', len(pcd_files))

        # 准备worker参数
        worker_args = [
            (pcd_file, raw_path, points_dir, labels_dir, use_new_format, type_map, class_names)
            for pcd_file in pcd_files
        ]

        all_ids = []
        total_skipped = 0
        # 使用进程池绕过GIL限制,实现真正的并行处理
        # CPU密集型任务使用CPU核心数
        num_workers = min(multiprocessing.cpu_count(), max(4, len(pcd_files)))

        if logger is not None:
            logger.info('Using %d worker processes for parallel processing', num_workers)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_sample_worker, args) for args in worker_args]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                sample_id, success, skipped = future.result()
                if success:
                    all_ids.append(sample_id)
                total_skipped += skipped
                completed += 1
                # 更频繁的进度反馈:每50个或每5%打印一次
                if logger is not None and (completed % 50 == 0 or completed % max(1, len(pcd_files) // 20) == 0):
                    logger.info('Processed %d / %d samples (%.1f%%), speed: ~%.1f samples/s',
                               completed, len(pcd_files), 100.0 * completed / len(pcd_files),
                               completed / max(1, completed // 10))

        all_ids = sorted(set(all_ids))
        # 因为没有单独对数据集做随机切分,此处简单的将所有样本写给train和val
        with open(train_list, 'w') as f:
            f.write('\n'.join(all_ids) + ('\n' if len(all_ids) > 0 else ''))
        with open(val_list, 'w') as f:
            f.write('\n'.join(all_ids) + ('\n' if len(all_ids) > 0 else ''))

        if logger is not None:
            logger.info('Prepared ars dataset from raw path: %s, samples=%d, skipped_invalid=%d',
                        str(raw_path), len(all_ids), total_skipped)

    def include_data(self, mode):
        """
        由 self.dataset_cfg.INFO_PATH 中配置的文件名加载 pkl 数据信息到 self.ars_infos 列表中。
        """
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
        """
        根据样本 idx 读取对应的物体标签 .txt 文件,返回解析后的 3D 边界框与类别名
        Returns:
            gt_boxes: N x 7 numpy数组 (x, y, z, dx, dy, dz, heading)
            gt_names: N 维 numpy数组 (包含每个框的类别名)
        """
        label_file = self.root_path / 'labels' / ('%s.txt' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split()
            # 需要满足长度>=8: 7个坐标信息+1个类别名
            if len(line_list) < 8:
                continue
            gt_boxes.append(line_list[:7])
            gt_names.append(line_list[7])

        if len(gt_boxes) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=str)
        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
        """
        根据样本 idx 读取对应的点云 .npy 文件,返回 Nx4 的点云信息 (x, y, z, intensity)
        """
        lidar_file = self.root_path / 'points' / ('%s.npy' % idx)
        assert lidar_file.exists()
        point_features = np.load(lidar_file)
        if point_features.ndim != 2 or point_features.shape[1] < 4:
            raise ValueError(f'Invalid lidar shape: {point_features.shape}, file={lidar_file}')
        
        # 仅取前4维 (x, y, z, intensity)
        point_features = point_features[:, :4]
        # 过滤掉非法坐标点
        point_features = point_features[np.isfinite(point_features).all(axis=1)]
        return point_features

    def set_split(self, split):
        """
        重置数据集对象使用的划分方式。相当于变更当前数据集为 training(train) 或者 validation(test)
        """
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else []
        self.ars_infos = []
        # 根据重置的 split 去装载对应的 info pkl 文件 (只有'train'走train分支,否则test分支)
        self.include_data('train' if split == 'train' else 'test')

    def __len__(self):
        """返回数据集的样本大小。是否合并所有的 iteration 到一个 epoch 中取决于配置设定。"""
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs
        return len(self.ars_infos)

    def __getitem__(self, index):
        """
        单次通过 index 读取样本的信息和数据字典。
        该数据字典后续将传给 prepare_data() 完成实际的数据增强和体素化处理。
        """
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.ars_infos)

        info = copy.deepcopy(self.ars_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        
        # 获得点云数据
        points = self.get_lidar(sample_idx)
        # 初始化基础的数据输入项
        input_dict = {
            'frame_id': sample_idx,
            'points': points
        }

        # 存在真值标签,则在字典中补全类别和框
        if 'annos' in info:
            annos = info['annos']
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        # 调用 DatasetTemplate 提供的 prepare_data 实现数据预处理与体素增强
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        """
        根据指定样本列表生成 info 列表对象(包含每帧的点云路径,真值等信息)。
        这在通过 `create_ars_infos` 操作时使用以固化数据索引及真值信息。
        """
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            # 设置基本 point cloud 属性信息
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # 解析标注加入 info 用于后续获取
            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        # 采用多线程方式并发解析获取
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        """
        提取包含目标的真值点云用来建立 GT database(用于 GT Sampling 数据增强）。
        会在数据根目录下生成 `gt_database` 文件夹以及关联的 `.pkl` dbinfo 描述文件。
        """
        import torch
        import concurrent.futures

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('ars_dbinfos_%s.pkl' % split)
        database_save_path.mkdir(parents=True, exist_ok=True)

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        def _process_sample(k):
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

            sample_db_infos = []
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
                    sample_db_infos.append(db_info)
            return sample_db_infos

        all_db_infos = {name: [] for name in self.class_names}
        # I/O密集型任务使用更多线程
        num_workers = min(64, max(4, len(infos)))
        print('Using %d worker threads for GT database creation' % num_workers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures_list = [executor.submit(_process_sample, k) for k in range(len(infos))]
            for i, future in enumerate(concurrent.futures.as_completed(futures_list)):
                sample_db_infos = future.result()
                # 更频繁的进度反馈:每100个或每10%打印一次
                if (i + 1) % 100 == 0 or (i + 1) % max(1, len(infos) // 10) == 0:
                    print('gt_database sample: %d/%d (%.1f%%)' % (i + 1, len(infos), 100.0 * (i + 1) / len(infos)))
                for db_info in sample_db_infos:
                    all_db_infos[db_info['name']].append(db_info)

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_ars_infos(dataset_cfg, class_names, data_path, save_path, workers=20):
    """
    一个完整的自动化入口流程,通过命令行调用。
    流程大致包括:
    1. 根据数据集根路径及划分信息生成划分好的 info pkl文件。
    2. 生成后续方便执行 GT Sampling 增强使用的 GT Database。
    """
    dataset = ArsDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('ars_infos_%s.pkl' % train_split)
    val_filename = save_path / ('ars_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    # 生成 Train_infos 的 Pkl
    dataset.set_split(train_split)
    ars_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(ars_infos_train, f)
    print('Ars info train file is saved to %s' % train_filename)

    # 生成 Val_infos 的 Pkl
    dataset.set_split(val_split)
    ars_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(ars_infos_val, f)
    print('Ars info val file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    # 结合生成的 Train_infos Pkl 生成包含相对坐标的GT objects以作 Data Augmentation
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    # 支持脚本执行,传参为: `create_ars_infos <对应的dataset_yaml配置文件路径>` 以开启该数据集的数据打标与抽取打包流程
    import sys
    import yaml
    from easydict import EasyDict

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_ars_infos':
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        data_path = Path(dataset_cfg.DATA_PATH)
        create_ars_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Pedestrian', 'Mbike', 'Car', 'Bus'],
            data_path=data_path,
            save_path=data_path,
        )
