import cv2
import numpy as np
import random
import torch
from utils import *
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path('data')
SOURCE_DIR = DATA_DIR / 'source'
TARGET_DIR = DATA_DIR / 'target'


def read_data(root_path):
    img_paths = root_path.glob('*.png')
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs




def main():
    set_seed()
    sources = read_data(SOURCE_DIR)
    targets = read_data(TARGET_DIR)
    print(sources[0].shape)


if __name__ == '__main__':
    main()