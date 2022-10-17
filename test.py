import torch
import torch.nn as nn


class fujiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Conv2d(3, 3, (3, 3))
    def forward(self, x):
        x = self.fc1(x)
        return x

test = torch.randn((32, 3, 240, 128))
model = fujiModel()
print(model(test).size())