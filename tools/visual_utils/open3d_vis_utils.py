"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    add_measurement_grid(vis, points)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def add_measurement_grid(vis, points, major_step=10.0, minor_step=1.0):
    xyz = points[:, :3]
    x_min = np.floor(np.min(xyz[:, 0]) / major_step) * major_step
    x_max = np.ceil(np.max(xyz[:, 0]) / major_step) * major_step
    y_min = np.floor(np.min(xyz[:, 1]) / major_step) * major_step
    y_max = np.ceil(np.max(xyz[:, 1]) / major_step) * major_step

    z_plane = np.percentile(xyz[:, 2], 5)

    grid_points = []
    grid_lines = []
    grid_colors = []

    def is_major(v):
        return np.isclose((v / major_step) - np.round(v / major_step), 0.0)

    idx = 0
    x_values = np.arange(x_min, x_max + 1e-6, minor_step)
    for x in x_values:
        grid_points.append([x, y_min, z_plane])
        grid_points.append([x, y_max, z_plane])
        grid_lines.append([idx, idx + 1])
        color = [0.45, 0.45, 0.45] if is_major(x) else [0.2, 0.2, 0.2]
        grid_colors.append(color)
        idx += 2

    y_values = np.arange(y_min, y_max + 1e-6, minor_step)
    for y in y_values:
        grid_points.append([x_min, y, z_plane])
        grid_points.append([x_max, y, z_plane])
        grid_lines.append([idx, idx + 1])
        color = [0.45, 0.45, 0.45] if is_major(y) else [0.2, 0.2, 0.2]
        grid_colors.append(color)
        idx += 2

    grid = open3d.geometry.LineSet()
    grid.points = open3d.utility.Vector3dVector(np.asarray(grid_points, dtype=np.float64))
    grid.lines = open3d.utility.Vector2iVector(np.asarray(grid_lines, dtype=np.int32))
    grid.colors = open3d.utility.Vector3dVector(np.asarray(grid_colors, dtype=np.float64))

    vis.add_geometry(grid)


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    def to_open3d_color(c):
        c = np.asarray(c, dtype=np.float64)
        if np.max(c) > 1.0:
            c = c / 255.0
        return np.clip(c, 0.0, 1.0)

    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            draw_color = color
        else:
            label = int(ref_labels[i])
            if 0 <= label < len(box_colormap):
                draw_color = box_colormap[label]
            elif 1 <= label <= len(box_colormap):
                draw_color = box_colormap[label - 1]
            else:
                draw_color = color
        line_set.paint_uniform_color(to_open3d_color(draw_color))

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
