import torch

import torchvision.models as models
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        # Unfreeze only the final layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)