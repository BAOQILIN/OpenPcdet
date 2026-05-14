import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    """
    PointPillarScatter: 将 Pillar 特征散射(scatter)回 BEV (Bird's Eye View) 空间。

    在 PointPillar 中，点云被划分成若干个 pillar(柱体），
    每个 pillar 经过 PointNet 提取为固定长度的 pillar feature 向量。
    本模块负责将这些 pillar feature 根据其空间坐标 (x, y) 映射回
    2D BEV 伪图像格网中，形成类似图像的 2D 特征图供后续 2D 卷积使用。

    输入:
        pillar_features: (M, C)  —— M 个非空 pillar，每个有 C 维特征
        voxel_coords:    (M, 4)  —— [batch_idx, z_idx, y_idx, x_idx] 体素网格坐标

    输出:
        spatial_features: (B, C, H, W) —— BEV 特征图，C 为特征通道数，H/W 为格网尺寸
    """

    def __init__(self, model_cfg, grid_size, **kwargs):
        """
        Args:
            model_cfg: 模型配置对象，需包含 NUM_BEV_FEATURES
            grid_size: (nx, ny, nz) 体素网格各维度尺寸，其中 nz 必须为 1(2D BEV）
        """
        super().__init__()

        self.model_cfg = model_cfg
        # 每个 pillar 经散射后最终保留的特征通道数
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size  # nx: x方向网格数, ny: y方向网格数, nz: z方向网格数
        # PointPillar 是 2D BEV 方法，z 方向仅有一层，故断言 nz == 1
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
        将 pillar 特征散射回 BEV 空间，结果存入 batch_dict['spatial_features']
        """
        # 从 batch_dict 中取出 pillar 特征及其体素网格坐标
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []

        # ONNX 导出时 batch_size 固定为 1，否则从 coords 中推断实际 batch 大小
        if torch.onnx.is_in_onnx_export():
            batch_size = 1
        else:
            batch_size = coords[:, 0].max().int().item() + 1

        # 逐 batch 处理：为每个 batch 创建一个空的 BEV 特征图
        for batch_idx in range(batch_size):
            # 初始化全零特征图，形状: (C, nz * nx * ny) = (C, 1 * nx * ny)，即拉直成一维
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            # 筛选出属于当前 batch 的 pillar 坐标
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]

            # 计算每个 pillar 在 BEV 网格中的一维展平索引：
            # PointPillar 中 nz=1，所以索引公式为 z_idx + y_idx * nx + x_idx
            # 但由于 nz=1 且 z_idx 恒为 0，实际简化为 y_idx * nx + x_idx
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)

            # 取出对应的 pillar 特征并转置为 (C, M) 以匹配 spatial_feature 的 scatter 维度
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()  # (M, C) -> (C, M)

            # 将 pillar 特征散射(spatter)到 BEV 特征图的对应位置
            if torch.onnx.is_in_onnx_export():
                # ONNX 不支持直接的高级索引赋值 (spatial_feature[:, indices] = pillars)，
                # 因此使用 scatter_ 操作替代，需要将 indices 扩展为与特征通道数相同的维度
                indices_expanded = indices.unsqueeze(0).expand(self.num_bev_features, -1)
                spatial_feature.scatter_(1, indices_expanded, pillars)
            else:
                # PyTorch 模式下直接用高级索引赋值，效率更高
                spatial_feature[:, indices] = pillars

            batch_spatial_features.append(spatial_feature)

        # 将所有 batch 的结果堆叠起来 (B, C, nx*ny)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # 重塑为 2D 伪图像格式: (B, C * nz, ny, nx) -> (B, C, H, W)
        batch_spatial_features = batch_spatial_features.view(
            batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        # 存入 batch_dict 供后续 2D 骨干网络使用
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class PointPillarScatter3d(nn.Module):
    """
    PointPillarScatter3d: 将 Pillar 特征散射回 3D 体素空间(带 z 维度）。

    与 PointPillarScatter 不同，此类支持 nz > 1 的场景(如 VoxelNet、SECOND 等 3D 检测方法），
    将 pillar/voxel 特征按照三维网格坐标 (x, y, z) 散射回 3D 体素格网。

    输入:
        pillar_features: (M, C)  —— M 个非空 voxel，每个有 C 维特征
        voxel_coords:    (M, 4)  —— [batch_idx, z_idx, y_idx, x_idx]

    输出:
        spatial_features: (B, C', H, W) —— 压缩 z 维度后的 BEV 特征图
    """

    def __init__(self, model_cfg, grid_size, **kwargs):
        """
        Args:
            model_cfg: 模型配置，需包含 INPUT_SHAPE (nx, ny, nz) 和 NUM_BEV_FEATURES
            grid_size: 体素网格尺寸(此实现中从 model_cfg.INPUT_SHAPE 获取）
        """
        super().__init__()

        self.model_cfg = model_cfg
        # 从配置中直接读取 3D 体素网格的尺寸
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        # 最终输出的 BEV 特征通道数(含压缩后的所有 z 层特征）
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        # 压缩前的每层特征通道数(将总通道数均分到 nz 层）
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        """
        将 pillar/voxel 特征散射回 3D 体素空间
        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            # 创建全零 3D 体素特征图(拉直为一维）:
            # 形状: (C_per_layer, nz * nx * ny)，每个位置对应一个体素的 C_per_layer 维特征
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            # 筛选当前 batch 的坐标
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]

            # 3D 体素的一维展平索引公式：z_idx * ny * nx + y_idx * nx + x_idx
            # 与 PointPillarScatter 不同之处在于引入了 z_idx 项
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)

            # 取出特征并转置为 (C_per_layer, M)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()

            # 散射到 3D 体素特征图的对应位置
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        # 堆叠 batch: (B, C_per_layer, nz * nx * ny)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # 重塑为 (B, C_per_layer * nz, ny, nx)，将 z 维度压缩到特征通道中，
        # 输出 2D BEV 特征图供后续 Backbone 2D 处理
        batch_spatial_features = batch_spatial_features.view(
            batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict