import torch
import torch.nn as nn

from .net_utils import FeatureFusionModule


class SpatialReliabilityConditionedFusion(FeatureFusionModule):
    """
    SCR-CMX: Spatial Reliability-Conditioned Cross-Modal Interaction.

    Reuses the CMX CrossPath and ChannelEmbed modules.
    Reliability controls the cross-modal residual update rather than
    weighting the completed cross-modal features after interaction.
    """

    def __init__(
        self,
        dim,
        reduction=1,
        num_heads=None,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__(
            dim=dim,
            reduction=reduction,
            num_heads=num_heads,
            norm_layer=norm_layer,
        )

    def forward(self, x_rgb, x_thermal, reliability):
        if x_rgb.shape != x_thermal.shape:
            raise ValueError(
                f"RGB and thermal feature shapes differ: "
                f"{x_rgb.shape} vs {x_thermal.shape}"
            )

        if reliability.ndim != 4 or reliability.shape[1] != 2:
            raise ValueError(
                "Expected reliability shape (B, 2, H, W), "
                f"received {tuple(reliability.shape)}"
            )

        B, C, H, W = x_rgb.shape

        if reliability.shape[0] != B or reliability.shape[2:] != (H, W):
            raise ValueError(
                "Reliability batch/spatial dimensions must match features: "
                f"{tuple(reliability.shape)} vs {tuple(x_rgb.shape)}"
            )

        r_rgb = reliability[:, 0:1].clamp(0.0, 1.0)
        r_thermal = reliability[:, 1:2].clamp(0.0, 1.0)

        # Original features as token sequences.
        rgb_tokens = x_rgb.flatten(2).transpose(1, 2)
        thermal_tokens = x_thermal.flatten(2).transpose(1, 2)

        # Existing CMX cross-modal interaction.
        rgb_cross, thermal_cross = self.cross(rgb_tokens, thermal_tokens)

        # Return to spatial feature maps.
        rgb_cross = (
            rgb_cross.transpose(1, 2)
            .reshape(B, C, H, W)
            .contiguous()
        )
        thermal_cross = (
            thermal_cross.transpose(1, 2)
            .reshape(B, C, H, W)
            .contiguous()
        )

        # Interaction deltas: isolate the changes introduced by CrossPath.
        rgb_delta = rgb_cross - x_rgb
        thermal_delta = thermal_cross - x_thermal

        # Reliability-conditioned cross-modal updates.
        # RGB receives thermal information according to thermal reliability.
        # Thermal receives RGB information according to RGB reliability.
        rgb_out = x_rgb + r_thermal * rgb_delta
        thermal_out = x_thermal + r_rgb * thermal_delta

        # Keep the original CMX channel embedding interface.
        rgb_tokens = rgb_out.flatten(2).transpose(1, 2)
        thermal_tokens = thermal_out.flatten(2).transpose(1, 2)
        merged = torch.cat([rgb_tokens, thermal_tokens], dim=-1)

        return self.channel_emb(merged, H, W)
