import gc
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from config import Config

config = Config()

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def save_model(name, model):
    model_path = config.MODEL_PATH / f'{name}.tph'
    torch.save(model.state_dict(), str(model_path))

def load_model(model, name, path='.'):
    data = torch.load(path / name, map_location=config.DEVICE)
    model.load_state_dict(data)
    return model

def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()


def read_data(root_path):
    img_paths = []
    for i in root_path.glob('*.png'):
        img_paths.append(i)
    return img_paths


def read_img(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.Tensor(np.transpose(img, (2, 0, 1)))


