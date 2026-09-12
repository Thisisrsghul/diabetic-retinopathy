import torch.nn as nn
import timm


class EfficientNetB0FeatureExtractor(nn.Module):
    """timm EfficientNetB0 → 1280-D pooled features."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.out_dim = self.backbone.num_features  # 1280

    def forward(self, x):
        return self.backbone(x)