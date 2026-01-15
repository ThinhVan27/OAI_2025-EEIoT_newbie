import torch
import torch.nn as nn
from torchvision import models

class efficientnetb6_model(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        self.efficientnetb6 = models.efficientnet_b6(
            weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.features = nn.Sequential(*list(self.efficientnetb6.features.children()))

        num_features = self.efficientnetb6.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        pooled = self.efficientnetb6.avgpool(features)
        x = torch.flatten(pooled, 1)
        x = self.classifier(x)
        return x
    
class efficientnetv2_m_model(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        self.efficientnetv2_m = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.features = nn.Sequential(*list(self.efficientnetv2_m.features.children()))

        num_features = self.efficientnetv2_m.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        pooled = self.efficientnetv2_m.avgpool(features)
        x = torch.flatten(pooled, 1)
        x = self.classifier(x)
        return x
    
class resnet101_model(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        self.resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)

        self.resnet101.fc = nn.Sequential(
            nn.Linear(self.resnet101.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet101(x)
        return x