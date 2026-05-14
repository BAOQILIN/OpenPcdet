import torch
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from ...utils import box_coder_utils


class MultiHeadAnchorHead(AnchorHeadTemplate):
    """
    Legacy-compatible multi-head anchor detection head.

    Matches the pandarat128 RPN ONNX structure:
      - Shared feature extractor: 2x Conv(384->64->64, 3x3) + BN + ReLU
      - 5 class-specific heads (one per class)
      - Each head: decomposed box prediction (reg + height + size + angle)
      - Class-agnostic cls per head, raw delta output (no BoxCoder in ONNX)

    Output format (legacy-compatible):
      batch_cls_preds: (B, H*W*sum_anchors, 1)   = (B, 163840, 1)
      batch_box_preds: (B, H*W*sum_anchors, 8)   = (B, 163840, 8)
    """

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True, **kwargs):
        # Must call Module.__init__ first to set up _modules
        nn.Module.__init__(self)
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = True  # signal to target assigner

        # Box coder with decomposed 8D format
        # code_size=7 + encode_angle_by_sincos=True = 8D:
        #   [xt, yt, zt, dxt, dyt, dzt, rt_cos, rt_sin]
        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            code_size=7, encode_angle_by_sincos=True,
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        # Anchor generation
        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = AnchorHeadTemplate.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

        self.num_heads = len(class_names)
        self.shared_conv_channels = model_cfg.get('SHARED_CONV_CHANNELS', 64)

        # ---- Shared feature extractor ----
        # Matches legacy: Conv(384->64, 3x3) + ReLU + Conv(64->64, 3x3) + ReLU
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, self.shared_conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.shared_conv_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(self.shared_conv_channels, self.shared_conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.shared_conv_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        # ---- Per-class heads ----
        self.heads = nn.ModuleList()
        for i in range(self.num_heads):
            num_a = self.num_anchors_per_location[i]  # anchors/location for this class
            head = nn.Module()
            head.conv_cls = nn.Conv2d(self.shared_conv_channels, num_a, kernel_size=3, padding=1)
            head.conv_reg = nn.Conv2d(self.shared_conv_channels, num_a * 2, kernel_size=3, padding=1)
            head.conv_height = nn.Conv2d(self.shared_conv_channels, num_a * 1, kernel_size=3, padding=1)
            head.conv_size = nn.Conv2d(self.shared_conv_channels, num_a * 3, kernel_size=3, padding=1)
            head.conv_angle = nn.Conv2d(self.shared_conv_channels, num_a * 2, kernel_size=3, padding=1)
            self.heads.append(head)

        self.total_anchors_per_location = sum(self.num_anchors_per_location)

    def _build_anchors_flat(self, batch_size):
        """Flatten per-class anchors to (1, total_anchors, 8)."""
        anchors_flat = []
        for anchor in self.anchors:
            # anchor: (z, y, x, 1, num_rot, 8) -> (z*y*x*num_rot, 8)
            a = anchor.permute(2, 1, 0, 3, 4, 5).contiguous()
            a = a.view(-1, 8)
            anchors_flat.append(a)
        anchors_flat = torch.cat(anchors_flat, dim=0)
        return anchors_flat.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']  # (B, 384, 128, 128)
        batch_size = spatial_features_2d.shape[0]

        x = self.shared_conv(spatial_features_2d)  # (B, 64, 128, 128)

        all_cls = []
        all_box_parts = []  # [(B, C_reg, H, W), (B, C_h, H, W), (B, C_sz, H, W), (B, C_ang, H, W)]

        for head in self.heads:
            cls_i = head.conv_cls(x)          # (B, num_a, 128, 128)
            reg_i = head.conv_reg(x)          # (B, num_a*2, 128, 128)
            height_i = head.conv_height(x)    # (B, num_a*1, 128, 128)
            size_i = head.conv_size(x)        # (B, num_a*3, 128, 128)
            angle_i = head.conv_angle(x)      # (B, num_a*2, 128, 128)

            all_cls.append(cls_i)
            all_box_parts.append((reg_i, height_i, size_i, angle_i))

        # Concatenate all heads' cls predictions: (B, total_cls_channels, H, W)
        cls_preds_all = torch.cat(all_cls, dim=1)

        # Store for loss computation
        self.forward_ret_dict['cls_preds'] = cls_preds_all
        self.forward_ret_dict['box_parts'] = all_box_parts

        # ---- Target assignment ----
        if self.training:
            targets_dict = self.target_assigner.assign_targets(
                self.anchors, data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        # ---- Box prediction ----
        if not self.training or self.predict_boxes_when_training:
            if torch.onnx.is_in_onnx_export():
                # ONNX export: raw conv results, no BoxCoder
                # cls: (B, 10, 128, 128) -> (B, 128, 128, 10) -> (B, -1, 1)
                B, _, H, W = cls_preds_all.shape
                cls_preds = cls_preds_all.permute(0, 2, 3, 1).contiguous()
                cls_preds = cls_preds.view(B, -1, 1)

                # box: concat all parts per head -> (B, total_box_ch, 128, 128)
                box_chunks = []
                for reg_i, height_i, size_i, angle_i in all_box_parts:
                    box_chunks.append(torch.cat([reg_i, height_i, size_i, angle_i], dim=1))
                box_preds = torch.cat(box_chunks, dim=1)  # (B, 80, 128, 128)
                box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
                box_preds = box_preds.view(B, -1, 8)

                data_dict['batch_cls_preds'] = cls_preds
                data_dict['batch_box_preds'] = box_preds
                data_dict['cls_preds_normalized'] = False
            else:
                batch_cls_preds, batch_box_preds = self._generate_predicted_boxes(
                    batch_size, cls_preds_all, all_box_parts
                )
                data_dict['batch_cls_preds'] = batch_cls_preds
                data_dict['batch_box_preds'] = batch_box_preds
                data_dict['cls_preds_normalized'] = False

        return data_dict

    def _generate_predicted_boxes(self, batch_size, cls_preds_all, all_box_parts):
        """Decode boxes for non-ONNX inference."""
        B, _, H, W = cls_preds_all.shape

        # Collect per-head box predictions and concatenate to 8D per anchor
        box_chunks = []
        cls_chunks = []
        anchor_offset = 0
        for i, (reg_i, height_i, size_i, angle_i) in enumerate(all_box_parts):
            num_a = self.num_anchors_per_location[i]

            # Permute to (B, H, W, C)
            reg = reg_i.permute(0, 2, 3, 1).contiguous()      # (B, H, W, num_a*2)
            height = height_i.permute(0, 2, 3, 1).contiguous() # (B, H, W, num_a*1)
            sz = size_i.permute(0, 2, 3, 1).contiguous()       # (B, H, W, num_a*3)
            ang = angle_i.permute(0, 2, 3, 1).contiguous()     # (B, H, W, num_a*2)

            # Reshape to (B, H*W*num_a, group_size)
            reg = reg.reshape(B, H * W * num_a, 2)
            height = height.reshape(B, H * W * num_a, 1)
            sz = sz.reshape(B, H * W * num_a, 3)
            ang = ang.reshape(B, H * W * num_a, 2)

            box_i = torch.cat([reg, height, sz, ang], dim=-1)  # (B, H*W*num_a, 8)
            box_chunks.append(box_i)

            # Cls: (B, H, W, num_a) -> (B, H*W*num_a, 1)
            cls_i = cls_preds_all[:, anchor_offset:anchor_offset+num_a, :, :]
            cls_i = cls_i.permute(0, 2, 3, 1).contiguous()
            cls_i = cls_i.reshape(B, H * W * num_a, 1)
            cls_chunks.append(cls_i)
            anchor_offset += num_a

        box_preds = torch.cat(box_chunks, dim=1)
        cls_preds = torch.cat(cls_chunks, dim=1)

        batch_anchors = self._build_anchors_flat(batch_size)
        batch_box_preds = self.box_coder.decode_torch(box_preds, batch_anchors)
        return cls_preds, batch_box_preds

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']  # (B, total_cls_c, H, W)
        box_cls_labels = self.forward_ret_dict['box_cls_labels']  # (B, total_anchors)
        batch_size = cls_preds.shape[0]

        # Permute to (B, H, W, total_cls_c) then view to (B, total_anchors, 1)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(batch_size, -1, 1)

        cared = box_cls_labels >= 0
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()

        # Class-agnostic per head: positive -> 1, negative -> 0
        box_cls_labels[positives] = 1
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights = positives.float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        # Binary classification per anchor
        cls_loss_src = self.cls_loss_func(cls_preds, cls_targets, weights=cls_weights)
        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

        # Store reg_weights for box loss
        self.forward_ret_dict['reg_weights'] = reg_weights
        self.forward_ret_dict['pos_normalizer'] = pos_normalizer

        return cls_loss, {'rpn_loss_cls': cls_loss.item()}

    def get_box_reg_layer_loss(self):
        all_box_parts = self.forward_ret_dict['box_parts']
        box_reg_targets = self.forward_ret_dict['box_reg_targets']  # (B, total_anchors, 8)
        batch_size = box_reg_targets.shape[0]
        reg_weights = self.forward_ret_dict['reg_weights']

        # Build box_preds in decomposed 8D format
        box_chunks = []
        offset = 0
        for i, (reg_i, height_i, size_i, angle_i) in enumerate(all_box_parts):
            B, _, H, W = reg_i.shape
            num_a = self.num_anchors_per_location[i]
            n = H * W * num_a

            reg = reg_i.permute(0, 2, 3, 1).contiguous().reshape(B, n, 2)
            height = height_i.permute(0, 2, 3, 1).contiguous().reshape(B, n, 1)
            sz = size_i.permute(0, 2, 3, 1).contiguous().reshape(B, n, 3)
            ang = angle_i.permute(0, 2, 3, 1).contiguous().reshape(B, n, 2)
            box_i = torch.cat([reg, height, sz, ang], dim=-1)
            box_chunks.append(box_i)

        box_preds = torch.cat(box_chunks, dim=1)  # (B, total_anchors, 8)

        loc_loss_src = self.reg_loss_func(box_preds, box_reg_targets, weights=reg_weights)
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

        return loc_loss, {'rpn_loss_loc': loc_loss.item()}
