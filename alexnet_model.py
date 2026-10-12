import torch
import torch.nn as nn
from torchvision import models


class ClassroomAlexNet(nn.Module):


    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_alexnet(num_classes, pretrained=False, freeze_features=False):
    model = ClassroomAlexNet(num_classes=1000 if pretrained else num_classes)

    if pretrained:
        try:
            weights = models.AlexNet_Weights.IMAGENET1K_V1
            official_model = models.alexnet(weights=weights)
        except AttributeError:
            official_model = models.alexnet(pretrained=True)
        model.load_state_dict(official_model.state_dict())
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    return model
