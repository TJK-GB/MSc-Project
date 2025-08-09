# models/resnet50_gru.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ViolenceModel(nn.Module):
    def __init__(self, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Unfreeze last CNN stage
        for p in base.parameters():
            p.requires_grad = False
        for p in base.layer4.parameters():
            p.requires_grad = True

        self.cnn = nn.Sequential(*list(base.children())[:-1])
        self.cnn_out_dim = 2048

        self.gru = nn.GRU(input_size=self.cnn_out_dim,
                          hidden_size=hidden,
                          batch_first=True,
                          bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):  # [B, T, 3, 224, 224]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)                    # Flatten batch & time
        feats = self.cnn(x)                           # [B*T, 2048, 1, 1]
        feats = feats.view(B, T, self.cnn_out_dim)    # [B, T, 2048]

        _, h = self.gru(feats)                        # h: [2, B, hidden]
        h_fwd = h[-2]
        h_bwd = h[-1]
        out = torch.cat([h_fwd, h_bwd], dim=1)        # [B, 2*hidden]

        out = self.dropout(out)
        logits = self.fc(out).squeeze(1)              # [B]
        return logits
