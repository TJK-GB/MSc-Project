import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch

class CueNet(nn.Module):
    def __init__(self):
        super().__init__()

        # === Local Branch: ResNet50 ===
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.local_cnn = nn.Sequential(*list(base.children())[:-1])
        self.local_out_dim = 2048

        # === Global Branch: GRU ===
        self.temporal_gru = nn.GRU(
            input_size=self.local_out_dim,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        self.global_out_dim = 512  # Bi-GRU output

        # === Project to common dimension (512)
        self.local_proj = nn.Linear(self.local_out_dim, 512)
        self.global_proj = nn.Identity()  # GRU already outputs 512

        # === Gating
        self.gate = nn.Sequential(
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )

        # === Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Local branch
        x = x.view(B * T, C, H, W)
        feats = self.local_cnn(x)                # [B*T, 768]
        feats = feats.view(B, T, -1)              # [B, T, 768]
        local_feat = feats.mean(dim=1)            # [B, 768]

        # Global branch
        global_seq, _ = self.temporal_gru(feats)  # [B, T, 512]
        global_feat = global_seq.mean(dim=1)      # [B, 512]

        # Project local to 512
        local_proj = self.local_proj(local_feat)  # [B, 512]
        global_proj = self.global_proj(global_feat)  # [B, 512]

        # Gate
        combined = torch.cat([local_proj, global_proj], dim=1)  # [B, 1024]
        gates = self.gate(combined)                             # [B, 2]
        g1, g2 = gates[:, 0].unsqueeze(1), gates[:, 1].unsqueeze(1)

        fused_feat = g1 * local_proj + g2 * global_proj  # [B, 512]

        # Classifier (concat with itself to match 1024 input)
        out = self.classifier(torch.cat([fused_feat, fused_feat], dim=1)).squeeze(1)
        return out
