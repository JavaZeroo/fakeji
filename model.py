from turtle import forward
import cv2
import numpy as np
import random
from utils import *
from pathlib import Path
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm.notebook import tqdm

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