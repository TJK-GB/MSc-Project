# To-Do
# Spatial Branch: ResNet18-based CNN
# Temporal Branch: LSTM (frame by frame motion)
# Fusion layer


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class DualBranchModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2):
        super(DualBranchModel, self).__init__()
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        """ Spatial Branch """
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)   # trained already on ImageNet(fast and good for feature extractor)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_output_dim = resnet.fc.in_features


        """ Temporal Branch(LSTM) """
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )

        """ Fusion + Classifier """
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + self.cnn_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.size()

        x = x.view(B * T, C, H, W) # Merge batch and time
        spatial_features = self.cnn_backbone(x) # [B*T, 512, 1, 1]
        spatial_features = spatial_features.view(B, T, -1) # [B, T, 512]


        """ LSTM: temporal branch """
        lstm_out, (h_n, c_n) = self.lstm(spatial_features)
        h_last = self.lstm_norm(h_n[-1])

        """ Mean pooling: spatial summary """
        spatial_mean = spatial_features.mean(dim=1)

        """ Fusion """
        fused = torch.cat((spatial_mean, h_last), dim=1)

        """ Classification """
        output = self.classifier(fused)
        return output
