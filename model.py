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
        self.fcr1 = nn.Linear(4896, 4896)
        self.fcg1 = nn.Linear(4896, 4896)
        self.fcb1 = nn.Linear(4896, 4896)
        self.fcr2 = nn.Linear(4896, 4896)
        self.fcg2 = nn.Linear(4896, 4896)
        self.fcb2 = nn.Linear(4896, 4896)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ret = torch.zeros_like(x)

        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]

        r_out = self.fcr1(r)
        g_out = self.fcg1(g)
        b_out = self.fcb1(b)

        r_out = self.sigmoid(r_out)
        g_out = self.sigmoid(g_out)
        b_out = self.sigmoid(b_out)

        r_out = self.fcr1(r)
        g_out = self.fcg1(g)
        b_out = self.fcb1(b)

        r_out = self.sigmoid(r_out)
        g_out = self.sigmoid(g_out)
        b_out = self.sigmoid(b_out)

        return r_out, g_out, b_out