import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, pretrained):
        super(ResNet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)

    def forward(self, x):
        return self.resnet18(x)
