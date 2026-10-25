import torch
import torch.nn as nn

from .net_utils import FeatureFusionModule


class ReliabilityGuidedFusionModule(FeatureFusionModule):
    """
    E3 prototype:
    Preserve CMX CrossPath and ChannelEmbed, but estimate spatially
    varying RGB/thermal fusion weights after cross-modal interaction.
    """

    def __init__(self, dim, reduction=1, num_heads=None,
                 norm_layer=nn.BatchNorm2d):
        super().__init__(
            dim=dim,
            reduction=reduction,
            num_heads=num_heads,
            norm_layer=norm_layer,
        )

        hidden = max(dim // 8, 8)
        self.reliability_head = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        # Original CMX cross-modal interaction.
        t1 = x1.flatten(2).transpose(1, 2)
        t2 = x2.flatten(2).transpose(1, 2)
        t1, t2 = self.cross(t1, t2)

        # Restore spatial maps for reliability estimation.
        f1 = t1.transpose(1, 2).reshape(B, C, H, W).contiguous()
        f2 = t2.transpose(1, 2).reshape(B, C, H, W).contiguous()

        # Two normalized weights per spatial location.
        logits = self.reliability_head(torch.cat([f1, f2], dim=1))
        weights = torch.softmax(logits, dim=1)

        w_rgb = weights[:, 0:1]
        w_thermal = weights[:, 1:2]

        t1 = (w_rgb * f1).flatten(2).transpose(1, 2)
        t2 = (w_thermal * f2).flatten(2).transpose(1, 2)

        merged = torch.cat([t1, t2], dim=-1)
        return self.channel_emb(merged, H, W)
