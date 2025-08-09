# models/swin.py
import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

class ViolenceModel(nn.Module):

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.embed_dim = self.backbone.head.in_features  
        self.backbone.head = nn.Identity()

        # Unfreeze last 
        for stage_idx in [0, 1, 2]:
            for p in self.backbone.features[stage_idx].parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.embed_dim, 1)

    def forward(self, x):  # x: [B, T, 3, 224, 224]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)                  # [B*T, D]
        feats = feats.view(B, T, self.embed_dim)  # [B, T, D]

        pooled = feats.mean(dim=1)                # [B, D]
        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(1)       # [B]
        return logits
