import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

from engine.logger import get_logger

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        # import backbone and decoder
        if cfg.backbone == 'swin_s':
            logger.info('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'swin_b':
            logger.info('Using backbone: Swin-Transformer-Base')
            from .encoders.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b5':
            logger.info('Using backbone: Segformer-B5')
            from .encoders.dual_segformer import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2_e3':
            logger.info('Using E3: SegFormer-B2 with reliability-guided fusion')
            from .encoders.dual_segformer_e3 import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2_e4':
            logger.info('Using E4: SegFormer-B2 with reliability-gated rectification')
            from .encoders.dual_segformer_e4 import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2_e5':
            logger.info('Using E5: SegFormer-B2 with counterfactual reliability')
            from .encoders.dual_segformer_e5 import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2_e2':
            logger.info('Using E2: SegFormer-B2 with FRM and weighted cross-attention fusion')
            from .encoders.dual_segformer_e2 import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2_e1':
            logger.info('Using E1: SegFormer-B2 without FRM')
            from .encoders.dual_segformer_e1 import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        
        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        
        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        self.criterion = criterion

        # E5-only counterfactual reliability supervision.
        self.e5_counterfactual = (
            getattr(cfg, "backbone", "") == "mit_b2_e5"
        )
        self.e5_reliability_loss_weight = float(
            getattr(cfg, "e5_reliability_loss_weight", 0.1)
        )
        self.e5_corruption_fraction = float(
            getattr(cfg, "e5_corruption_fraction", 0.25)
        )
        self.e5_counterfactual_temperature = max(
            float(getattr(cfg, "e5_counterfactual_temperature", 1.0)),
            1e-6,
        )
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def _e5_corrupt_modality(self, image):
        """Mask one random rectangle per image after normalization."""
        import math

        batch, channels, height, width = image.shape
        fraction = min(max(self.e5_corruption_fraction, 0.01), 0.90)
        block_h = max(1, min(height, int(height * math.sqrt(fraction))))
        block_w = max(1, min(width, int(width * math.sqrt(fraction))))

        corrupted = image.clone()
        for b in range(batch):
            y0 = int(torch.randint(
                0, height - block_h + 1, (1,), device=image.device
            ).item())
            x0 = int(torch.randint(
                0, width - block_w + 1, (1,), device=image.device
            ).item())
            corrupted[b, :, y0:y0 + block_h, x0:x0 + block_w] = 0

        return corrupted

    def _e5_counterfactual_loss(self, rgb, modal_x, label, reliability_maps):
        """Robust, confidence-aware counterfactual reliability supervision."""
        if not reliability_maps:
            raise RuntimeError("E5 reliability maps are missing.")

        was_training = self.training

        try:
            self.eval()

            with torch.no_grad():
                clean_logits = self.encode_decode(rgb, modal_x)
                if isinstance(clean_logits, tuple):
                    clean_logits = clean_logits[0]

                rgb_logits = self.encode_decode(
                    self._e5_corrupt_modality(rgb), modal_x
                )
                if isinstance(rgb_logits, tuple):
                    rgb_logits = rgb_logits[0]

                thermal_logits = self.encode_decode(
                    rgb, self._e5_corrupt_modality(modal_x)
                )
                if isinstance(thermal_logits, tuple):
                    thermal_logits = thermal_logits[0]

                labels = label.long()
                valid = labels != 255

                clean_loss = F.cross_entropy(
                    clean_logits, labels,
                    ignore_index=255, reduction="none"
                )
                rgb_loss = F.cross_entropy(
                    rgb_logits, labels,
                    ignore_index=255, reduction="none"
                )
                thermal_loss = F.cross_entropy(
                    thermal_logits, labels,
                    ignore_index=255, reduction="none"
                )

                delta_rgb = (rgb_loss - clean_loss).clamp_min(0)
                delta_thermal = (thermal_loss - clean_loss).clamp_min(0)

                # Per-image robust median/IQR normalization.
                def robust_normalize(delta):
                    normalized = torch.zeros_like(delta)

                    for b in range(delta.shape[0]):
                        values = delta[b][valid[b]].float()
                        if values.numel() == 0:
                            continue

                        quantiles = torch.quantile(
                            values,
                            torch.tensor(
                                [0.25, 0.50, 0.75],
                                device=values.device,
                                dtype=values.dtype,
                            ),
                        )

                        q25, median, q75 = quantiles.unbind()
                        scale = q75 - q25

                        # Zero-heavy maps need a nonzero fallback scale.
                        if scale.item() < 1e-6:
                            positive = values[values > 1e-8]
                            if positive.numel():
                                scale = torch.quantile(
                                    positive, 0.90
                                ).clamp_min(1e-6)
                            else:
                                scale = values.new_tensor(1.0)

                        normalized[b] = (
                            (delta[b] - median) / scale
                        ).clamp(-3.0, 3.0)

                    return normalized

                norm_rgb = robust_normalize(delta_rgb)
                norm_thermal = robust_normalize(delta_thermal)

                # Confidence is based on raw counterfactual evidence.
                rho = 0.02
                eta = 0.01
                evidence_difference = (
                    delta_rgb - delta_thermal
                ).abs()

                confidence = torch.sigmoid(
                    (evidence_difference - rho) / eta
                )
                confidence = confidence * valid.float()

                temperature = max(
                    float(self.e5_counterfactual_temperature), 1e-6
                )

                normalized_deltas = torch.stack(
                    [norm_rgb, norm_thermal], dim=1
                )

                informative_target = torch.softmax(
                    normalized_deltas / temperature, dim=1
                )

                uniform_prior = torch.full_like(
                    informative_target, 0.5
                )

                targets = (
                    (1.0 - confidence.unsqueeze(1)) * uniform_prior
                    + confidence.unsqueeze(1) * informative_target
                ).detach()

        finally:
            self.train(was_training)

        losses = []

        for reliability in reliability_maps:
            target = F.interpolate(
                targets,
                size=reliability.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            target = target / target.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)

            confidence_stage = F.interpolate(
                confidence.unsqueeze(1),
                size=reliability.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            valid_stage = F.interpolate(
                valid.unsqueeze(1).float(),
                size=reliability.shape[-2:],
                mode="nearest",
            ).squeeze(1)

            pixel_loss = -(
                target * reliability.clamp_min(1e-6).log()
            ).sum(dim=1)

            weights = confidence_stage * valid_stage

            stage_loss = (
                (pixel_loss * weights).sum()
                / weights.sum().clamp_min(1e-6)
            )

            losses.append(stage_loss)

        return torch.stack(losses).mean()

    def forward(self, rgb, modal_x, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)

        if label is not None:
            loss = self.criterion(out, label.long())

            if self.aux_head:
                loss += self.aux_rate * self.criterion(
                    aux_fm, label.long()
                )

            if self.e5_counterfactual:
                # Preserve maps from the original gradient-enabled forward.
                reliability_maps = getattr(
                    self.backbone, "last_reliabilities", None
                )
                if reliability_maps is None or len(reliability_maps) != 4:
                    raise RuntimeError(
                        "Expected four reliability maps from the E5 backbone."
                    )

                reliability_loss = self._e5_counterfactual_loss(
                    rgb, modal_x, label, reliability_maps
                )
                loss = loss + (
                    self.e5_reliability_loss_weight * reliability_loss
                )

            return loss

        return out
