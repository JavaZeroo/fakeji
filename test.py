import torch
import torch.nn as nn
from ssim import ssim

class fujiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4896, 4896)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(4896, 4896)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.mul(x)

test = torch.randn((4, 3, 4896, 4896))
model = fujiModel()
pred = model(test)
loss = 1 - ssim(test, pred)
print(loss)