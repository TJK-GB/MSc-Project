# models/swin_gru.py
import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

class ViolenceModel(nn.Module):

    def __init__(self, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.embed_dim = self.backbone.head.in_features  # 768
        self.backbone.head = nn.Identity()

        # Unfreeze last
        for stage_idx in [0, 1, 2]:
            for p in self.backbone.features[stage_idx].parameters():
                p.requires_grad = False


        self.gru = nn.GRU(input_size=self.embed_dim,
                          hidden_size=hidden,
                          batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):  # x: [B, T, 3, 224, 224]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)                     # [B*T, D]
        feats = feats.view(B, T, -1)                 # [B, T, D]

        _, h = self.gru(feats)                       # h: [2, B, hidden]
        h_fwd = h[-2]
        h_bwd = h[-1]
        out = torch.cat([h_fwd, h_bwd], dim=1)       # [B, 2*hidden]

        out = self.dropout(out)
        logits = self.fc(out).squeeze(1)             # [B]
        return logits
