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
        self.fcr = nn.Linear(4896, 4896)
        self.fcg = nn.Linear(4896, 4896)
        self.fcb = nn.Linear(4896, 4896)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shape = x.size()
        ret = torch.zeros_like(x)

        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]

        r_out = self.fcr(r)
        g_out = self.fcg(g)
        b_out = self.fcb(b)

        ret[:, 0, :, :] = r_out.mul(r)
        ret[:, 1, :, :] = g_out.mul(g)
        ret[:, 2, :, :] = b_out.mul(b)

        return self.sigmoid(ret)