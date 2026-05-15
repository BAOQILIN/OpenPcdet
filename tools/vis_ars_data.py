#!/usr/bin/env python3
"""可视化 data/ars 中的点云和 GT 框。"""

import argparse
import numpy as np
from pathlib import Path

# 每个类别的颜色
CLASS_COLORS = {
    'Car':        (0.0, 1.0, 0.0),   # 绿
    'Pedestrian': (1.0, 1.0, 0.0),   # 黄
    'Bus':        (0.3, 0.7, 1.0),   # 浅蓝
    'Mbike':      (1.0, 0.0, 0.0),   # 红
    'Tricycle':   (1.0, 0.0, 1.0),   # 品红
}


def load_label(label_path):
    """读取标签文件，返回 (N, 7) boxes 和 (N,) class_names。"""
    boxes, names = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            x, y, z, dx, dy, dz, heading, name = parts
            boxes.append([float(x), float(y), float(z), float(dx), float(dy),
                          float(dz), float(heading)])
            names.append(name)
    return np.array(boxes, dtype=np.float32), np.array(names)


def boxes_to_lines(boxes):
    """将 N×7 的 3D 框转为 N×8×3 的角点，再转为线段端点列表。"""
    if len(boxes) == 0:
        return np.zeros((0, 2, 3))

    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1],  [1, -1, 1],  [-1, -1, 1],  [-1, 1, 1],
    ]) / 2.0  # (8, 3)

    corners = boxes[:, None, 3:6] * template[None, :, :]  # (N, 8, 3)

    # 旋转 heading
    heading = boxes[:, 6]
    cos_a, sin_a = np.cos(heading), np.sin(heading)
    rot = np.zeros((len(boxes), 3, 3))
    rot[:, 0, 0] = cos_a
    rot[:, 0, 1] = sin_a
    rot[:, 1, 0] = -sin_a
    rot[:, 1, 1] = cos_a
    rot[:, 2, 2] = 1
    corners = np.matmul(corners, rot)  # (N, 8, 3)

    corners += boxes[:, None, 0:3]  # 平移到中心

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7),  # 竖线
    ]
    lines = []
    for c in corners:
        for i, j in edges:
            lines.append([c[i], c[j]])
    return np.array(lines)


def main():
    parser = argparse.ArgumentParser(description='可视化 ARS 数据集点云和 GT 框')
    parser.add_argument('--data_path', default=None,
                        help='数据集根目录（包含 points/ 和 labels/），默认 ../data/ars')
    parser.add_argument('--index', type=int, default=0,
                        help='显示第几个文件（按文件名排序）')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='直接指定 sample id（如 000000），覆盖 --index')
    parser.add_argument('--point_size', type=float, default=2.5,
                        help='点的大小')
    parser.add_argument('--no_box', action='store_true',
                        help='不显示 GT 框')
    parser.add_argument('--range', type=float, nargs=4, default=None,
                        help='裁剪范围 x_min y_min x_max y_max')
    args = parser.parse_args()

    data_dir = Path(args.data_path) if args.data_path else (Path(__file__).resolve().parent.parent / 'data' / 'ars')
    points_dir = data_dir / 'points'
    labels_dir = data_dir / 'labels'

    if not points_dir.exists():
        print(f'点云目录不存在: {points_dir}')
        return

    # 确定要显示的文件
    npy_files = sorted(points_dir.glob('*.npy'))
    if not npy_files:
        print(f'{points_dir} 中没有 .npy 文件')
        return

    if args.sample_id:
        pt_file = points_dir / f'{args.sample_id}.npy'
        lb_file = labels_dir / f'{args.sample_id}.txt'
    else:
        pt_file = npy_files[args.index % len(npy_files)]
        sample_id = pt_file.stem
        lb_file = labels_dir / f'{sample_id}.txt'

    if not pt_file.exists():
        print(f'点云文件不存在: {pt_file}')
        return

    print(f'加载: {pt_file.name}')
    pts = np.load(str(pt_file))[:, :4].astype(np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]

    # 裁剪
    if args.range:
        x_min, y_min, x_max, y_max = args.range
        mask = (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) & \
               (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
        pts = pts[mask]
        print(f'裁剪后点数: {len(pts)}')

    print(f'点数: {len(pts)}')

    # 加载标签
    gt_boxes, gt_names = None, None
    if lb_file.exists() and not args.no_box:
        gt_boxes, gt_names = load_label(lb_file)
        print(f'GT 框数: {len(gt_boxes)}')
        for name in np.unique(gt_names):
            print(f'  {name}: {np.sum(gt_names == name)}')

    # 构建文件列表
    file_list = []
    for nf in npy_files:
        sid = nf.stem
        lb = labels_dir / f'{sid}.txt'
        file_list.append((nf, lb))

    # 确定起始索引
    if args.sample_id:
        start_idx = next((i for i, (nf, _) in enumerate(file_list)
                          if nf.stem == args.sample_id), 0)
    else:
        start_idx = args.index % len(file_list)

    # 可视化（open3d 支持键盘翻页）
    _vis_all_open3d(file_list, start_idx, args.point_size,
                    args.range, args.no_box)


def _make_text_mesh(text, position, color, scale=0.08):
    """创建 3D 文字 mesh，放置在指定位置。"""
    import open3d as o3d
    mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=1).to_legacy()
    mesh.paint_uniform_color(color)
    # 默认文字非常大，缩放到合理尺寸
    bbox = mesh.get_axis_aligned_bounding_box()
    text_size = bbox.get_extent()  # [w, h, d]
    target_w = len(text) * scale * 1.2
    s = target_w / max(text_size[0], 1e-6)
    mesh.scale(s, center=mesh.get_center())
    # 移到目标位置（左上角对齐）
    mesh.translate(position)
    return mesh


def _build_scene(pts, gt_boxes, gt_names, point_size, x_range, show_details=False):
    """构建一帧的可视化几何体列表。"""
    import open3d as o3d

    if x_range:
        x_min, y_min, x_max, y_max = x_range
        mask = (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) & \
               (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
        pts = pts[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])

    colors = np.tile([0.0, 0.4, 1.0], (len(pts), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geoms = [pcd]

    if gt_boxes is not None and len(gt_boxes) > 0:
        for name, box in zip(gt_names, gt_boxes):
            color = CLASS_COLORS.get(name, (1.0, 1.0, 1.0))
            lines = boxes_to_lines(box[None, :])
            for line in lines:
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(line)
                ls.lines = o3d.utility.Vector2iVector([[0, 1]])
                ls.colors = o3d.utility.Vector3dVector([color])
                geoms.append(ls)

            # 3D 文字标签：框顶部上方
            top = box[:3].copy()
            top[2] += box[5] / 2.0 + 0.4  # 框顶部 + 偏移

            # 类别名
            geoms.append(_make_text_mesh(name, top + [0, 0, 0.3], color))

            if show_details:
                dims = f'{box[3]:.1f}x{box[4]:.1f}x{box[5]:.1f}'
                dist = np.linalg.norm(box[:2])
                heading = f'{np.degrees(box[6]) % 360:.0f}°'
                detail_color = (1.0, 1.0, 1.0)
                geoms.append(_make_text_mesh(dims, top + [0, 0, 0.0], detail_color, scale=0.06))
                geoms.append(_make_text_mesh(f'{dist:.1f}m', top + [0, 0, -0.3], detail_color, scale=0.06))
                geoms.append(_make_text_mesh(heading, top + [0, 0, -0.6], detail_color, scale=0.06))

    return geoms


def _print_frame_info(idx, total, sample_id, pts, gt_boxes, gt_names, show_details=False):
    info = f'[{idx+1}/{total}] {sample_id}  |  points: {len(pts)}'
    if gt_boxes is not None and len(gt_boxes) > 0:
        info += f'  |  boxes: {len(gt_boxes)}'
        class_counts = {n: int(np.sum(gt_names == n))
                        for n in np.unique(gt_names)}
        info += '  |  ' + '  '.join(f'{k}:{v}' for k, v in class_counts.items())
    else:
        info += '  |  no labels'
    print(info)

    if show_details and gt_boxes is not None and len(gt_boxes) > 0:
        print(f'  {"Class":12s} {"LxWxH":14s} {"Dist":>6s} {"Heading":>8s}  {"Center(x,y,z)"}')
        print(f'  {"-"*60}')
        for name, box in zip(gt_names, gt_boxes):
            dims = f'{box[3]:.1f}x{box[4]:.1f}x{box[5]:.1f}'
            dist = np.linalg.norm(box[:2])
            heading = f'{np.degrees(box[6]) % 360:.0f}°'
            center = f'({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f})'
            print(f'  {name:12s} {dims:14s} {dist:5.1f}m  {heading:>8s}  {center}')


def _vis_all_open3d(file_list, start_idx, point_size, x_range, no_box):
    """open3d 可视化，支持 ← → 键翻页，T 切换详细信息。"""
    import open3d as o3d

    state = {'idx': start_idx, 'need_update': True, 'show_details': False}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='ARS Data Viewer', width=1400, height=900)
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.line_width = 4.0
    render_opt.background_color = np.array([0.0, 0.0, 0.0])

    def load_frame(idx):
        nf, lb = file_list[idx]
        pts = np.load(str(nf))[:, :4].astype(np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)]
        gt_boxes, gt_names = None, None
        if lb.exists() and not no_box:
            gt_boxes, gt_names = load_label(lb)
        _print_frame_info(idx, len(file_list), nf.stem, pts, gt_boxes, gt_names,
                          show_details=state['show_details'])
        return _build_scene(pts, gt_boxes, gt_names, point_size, x_range,
                            show_details=state['show_details'])

    current_geoms = load_frame(state['idx'])
    for g in current_geoms:
        vis.add_geometry(g, reset_bounding_box=True)

    def switch_frame(delta):
        state['idx'] = (state['idx'] + delta) % len(file_list)
        state['need_update'] = True

    def toggle_details(_vis):
        state['show_details'] = not state['show_details']
        status = 'ON' if state['show_details'] else 'OFF'
        print(f'\n详细信息: {status}')
        state['need_update'] = True

    def on_prev(_vis):
        switch_frame(-1)

    def on_next(_vis):
        switch_frame(1)

    vis.register_key_callback(263, on_prev)  # ←
    vis.register_key_callback(262, on_next)  # →
    vis.register_key_callback(ord('A'), on_prev)
    vis.register_key_callback(ord('D'), on_next)
    vis.register_key_callback(ord('T'), toggle_details)
    vis.register_key_callback(ord('Q'), lambda v: v.close())

    label_info = 'T=详细信息' if not no_box else ''
    print(f'\n共 {len(file_list)} 帧，← → 或 A/D 翻页，{label_info} Q 退出\n')

    while vis.poll_events():
        if state['need_update']:
            vis.clear_geometries()
            current_geoms = load_frame(state['idx'])
            for g in current_geoms:
                vis.add_geometry(g, reset_bounding_box=False)
            state['need_update'] = False

        # Update status bar with hover info
        vis.update_renderer()

    vis.destroy_window()


if __name__ == '__main__':
    main()
