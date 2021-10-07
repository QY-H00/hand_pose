import torch.nn as nn


class RegressionDecoder(nn.Module):
    def __init__(self, features=1000, classes=21):
        super(RegressionDecoder, self).__init__()
        self.classes = classes
        self.linear = nn.Linear(features, classes * 2)

    def forward(self, x):
        out = self.linear(x)  # (batch_size, classes*2)
        out = out.reshape(out.shape[0], self.classes, 2)  # (batch_size, classes, 2)
        return out
