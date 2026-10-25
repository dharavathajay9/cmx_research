import torch
import torch.nn as nn

from .net_utils import FeatureFusionModule


class ReliabilityGatedFeatureRectifyModule(nn.Module):
    """
    E4 prototype:
    Estimate per-pixel RGB/thermal reliability and use it to
    gate cross-modal feature corrections.

    This module does not yet implement counterfactual supervision.
    """

    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()

        # Reuse the original CMX channel and spatial weighting mechanisms.
        from .net_utils import ChannelWeights, SpatialWeights

        self.lambda_c = lambda_c
        self.lambda_s = lambda_s

        self.channel_weights = ChannelWeights(
            dim=dim, reduction=reduction
        )
        self.spatial_weights = SpatialWeights(
            dim=dim, reduction=reduction
        )

        hidden = max(dim // 8, 8)
        self.reliability_head = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )

        # Start at equal modality reliability.
        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.zeros_(self.reliability_head[-1].bias)

    def forward(self, x_rgb, x_thermal):
        if x_rgb.shape != x_thermal.shape:
            raise ValueError(
                f"Feature shape mismatch: RGB={x_rgb.shape}, "
                f"thermal={x_thermal.shape}"
            )

        logits = self.reliability_head(
            torch.cat([x_rgb, x_thermal], dim=1)
        )
        reliability = torch.softmax(logits, dim=1)

        r_rgb = reliability[:, 0:1]
        r_thermal = reliability[:, 1:2]

        channel_weights = self.channel_weights(x_rgb, x_thermal)
        spatial_weights = self.spatial_weights(x_rgb, x_thermal)

        # RGB correction is gated by thermal reliability.
        rgb_correction = (
            self.lambda_c * channel_weights[1] * x_thermal
            + self.lambda_s * spatial_weights[1] * x_thermal
        )

        # Thermal correction is gated by RGB reliability.
        thermal_correction = (
            self.lambda_c * channel_weights[0] * x_rgb
            + self.lambda_s * spatial_weights[0] * x_rgb
        )

        out_rgb = x_rgb + r_thermal * rgb_correction
        out_thermal = x_thermal + r_rgb * thermal_correction

        return out_rgb, out_thermal
