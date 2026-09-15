import torch
import torch.nn as nn
from config import Config
from efficientnet_features import EfficientNetB0FeatureExtractor
from attention_fusion import CrossFeatureAttention, GatedFusionAttention


class DRClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.cnn = EfficientNetB0FeatureExtractor(pretrained=True)
        cnn_dim = self.cnn.out_dim
        morph_dim = Config.MORPH_DIM
        fused_dim = cnn_dim + morph_dim

        self.morph_norm = nn.LayerNorm(morph_dim)
        self.attn = CrossFeatureAttention(dim=fused_dim,
                                          heads=Config.ATTN_HEADS,
                                          dropout=Config.DROPOUT)
        self.gate = GatedFusionAttention(dim=fused_dim,
                                         dropout=Config.DROPOUT)
        
        # Regression output: maps fused features to a single continuous grade
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512), 
            nn.BatchNorm1d(512), 
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256), 
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, 1),
        )

    def forward(self, image, morph_feats):
        cnn_feat = self.cnn(image)
        m = self.morph_norm(morph_feats)
        fused = torch.cat([cnn_feat, m], dim=1)
        fused = self.gate(fused)
        fused = self.attn(fused)
        return self.classifier(fused).squeeze(1)
