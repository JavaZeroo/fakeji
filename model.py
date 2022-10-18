import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

from utils import *


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