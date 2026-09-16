"""
Multi-Class DynUNet Segmentation Model for PVS & Pathology.
Implements the 3-class segmentation head:
    0: Background / Normal Brain
    1: PVS (Perivascular Spaces)
    2: Pathology (WMH Lesions - Confounder suppression)
"""

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet
from monai.losses import DiceCELoss

class PVSSegmentationModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, spatial_dims=3):
        super().__init__()
        # Kernel and stride specifications for standard 128x128x128 patches
        kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides     = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        upsample_kernel_size = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        filters     = [32, 64, 128, 256, 320]

        self.net = DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            dropout=0.1,
            norm_name="instance",
            act_name="leakyrelu",
            deep_supervision=False,
            res_block=True
        )

    def forward(self, x):
        return self.net(x)

def get_loss_function():
    """
    Combined Soft Dice + Cross Entropy Loss.
    Handles severe foreground/background class imbalance.
    """
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,  # Exclude background from Dice calculation
        batch=True,
        lambda_dice=1.0,
        lambda_ce=1.0
    )
