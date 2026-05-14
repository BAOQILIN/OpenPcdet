import torch
import torch.nn as nn


class BaseBEVBackboneLegacy(nn.Module):
    """
    Legacy-compatible BEV backbone matching the pandarat128 training framework.

    Topology (matching legacy backbone2D.onnx):
        spatial_features (1, 64, 512, 512)
          -> Stage 0: 4x Conv(64->64, 3x3, s=1, p=1) + BN + ReLU
        (1, 64, 512, 512)
          +---> Branch 1: Conv(64->128, 2x2, s=4) + BN + ReLU -> (1,128,128,128)
          |
          +---> Stage 1: Conv(64->128, 3x3, s=4) + BN + ReLU
          |         -> 5x Conv(128->128, 3x3, s=1, p=1) + BN + ReLU
          |     (1, 128, 128, 128)
          |     +---> Branch 2: ConvT(128->128, 1x1, s=1) + BN + ReLU -> (1,128,128,128)
          |     |
          |     +---> Stage 2: Conv(128->256, 3x3, s=2) + BN + ReLU
          |               -> 5x Conv(256->256, 3x3, s=1, p=1) + BN + ReLU
          |           (1, 256, 64, 64)
          |               -> Branch 3: ConvT(256->128, 2x2, s=2) + BN + ReLU -> (1,128,128,128)
          |
          Concat(Branch1, Branch2, Branch3) -> (1, 384, 128, 128)
    """

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS          # [3, 5, 5]
        num_filters = self.model_cfg.NUM_FILTERS        # [64, 128, 256]
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES  # [4, 1, 2]
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS  # [128, 128, 128]

        assert len(layer_nums) == 3 and len(num_filters) == 3
        assert len(upsample_strides) == 3 and len(num_upsample_filters) == 3

        # ---- Stage 0: no downsampling, (layer_nums[0]+1) layers ----
        stage0 = []
        for k in range(layer_nums[0] + 1):
            stage0.extend([
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ])
        self.stage0 = nn.Sequential(*stage0)

        # ---- Branch 1: direct stride-4 Conv from stage 0 output ----
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, num_upsample_filters[0], kernel_size=2,
                      stride=upsample_strides[0], bias=False),
            nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        # ---- Stage 1: stride-4 Conv + (layer_nums[1]+1) Conv layers ----
        stage1 = [
            nn.Conv2d(64, num_filters[1], kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[1], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ]
        for k in range(layer_nums[1]):
            stage1.extend([
                nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters[1], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ])
        self.stage1 = nn.Sequential(*stage1)

        # ---- Branch 2: deblock from stage 1 (stride 1, no upsampling) ----
        self.deblock2 = nn.Sequential(
            nn.ConvTranspose2d(num_filters[1], num_upsample_filters[1],
                               kernel_size=upsample_strides[1],
                               stride=upsample_strides[1], bias=False),
            nn.BatchNorm2d(num_upsample_filters[1], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        # ---- Stage 2: stride-2 Conv + (layer_nums[2]+1) Conv layers ----
        stage2 = [
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ]
        for k in range(layer_nums[2]):
            stage2.extend([
                nn.Conv2d(num_filters[2], num_filters[2], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters[2], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ])
        self.stage2 = nn.Sequential(*stage2)

        # ---- Branch 3: deblock from stage 2 (stride 2 upsampling) ----
        self.deblock3 = nn.Sequential(
            nn.ConvTranspose2d(num_filters[2], num_upsample_filters[2],
                               kernel_size=upsample_strides[2],
                               stride=upsample_strides[2], bias=False),
            nn.BatchNorm2d(num_upsample_filters[2], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.num_bev_features = sum(num_upsample_filters)

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']  # (B, 64, 512, 512)

        x0 = self.stage0(spatial_features)                # (B, 64, 512, 512)

        b1 = self.branch1(x0)                             # (B, 128, 128, 128)
        x1 = self.stage1(x0)                              # (B, 128, 128, 128)
        b2 = self.deblock2(x1)                            # (B, 128, 128, 128)
        x2 = self.stage2(x1)                              # (B, 256, 64, 64)
        b3 = self.deblock3(x2)                            # (B, 128, 128, 128)

        x = torch.cat([b1, b2, b3], dim=1)                # (B, 384, 128, 128)

        data_dict['spatial_features_2d'] = x
        return data_dict
