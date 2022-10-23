import os
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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from dataset import fujiDataset
from tqdm import tqdm

import wandb
from config import Config
from model import fujiModel
from ssim import ssim
from utils import *

config = Config()
os.environ['WANDB_API_KEY'] = '67c99389e1ae37b747c40634c51802a4bf019d49'

def train_transform():
    transform_list = [
        transforms.Resize(size=(4896, 4896)),
    ]
    return transforms.Compose(transform_list)


def get_kfold_ds(fold, source_imgs, target_imgs):
    assert(len(source_imgs)==len(target_imgs))
    train_source_imgs = []
    valid_source_imgs = []
    train_target_imgs = []
    valid_target_imgs = []
    for i in range(config.N_FOLD):
        slc = slice(i, None, config.N_FOLD)
        if (i + 1) == fold:
            valid_source_imgs.extend(source_imgs[slc])
            valid_target_imgs.extend(target_imgs[slc])
        else:
            train_source_imgs.extend(source_imgs[slc])
            train_target_imgs.extend(target_imgs[slc])
    train_ds = fujiDataset(train_source_imgs, train_target_imgs, train_transform())
    valid_ds = fujiDataset(valid_source_imgs, valid_target_imgs, train_transform())
    return train_ds, valid_ds

def valid(model, valid_ds, shuffle=False):
    valid_dl = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, num_workers=int(os.cpu_count()/8), shuffle=shuffle)
    criterion = nn.MSELoss(reduction='mean')
    losses = []
    with torch.no_grad():
        model.eval()
        with tqdm(valid_dl, desc='Eval', miniters=10) as progress:
            for i, (source, target) in enumerate(progress):
                source = source.to(config.DEVICE)
                target = target.to(config.DEVICE)
                with autocast():
                    img_pred = model(source)
                    ssim_loss = criterion(img_pred, target)
                    # ssim_loss = 1 - ssim(img_pred, target)
                    # losses.append(ssim_loss)
                progress.set_description(f'Valid loss: {ssim_loss :.02f}')

        return np.mean(losses)





def train(train_ds, valid_ds, logger, name):
    print(len(train_ds))
    set_seed(11)
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, num_workers=int(os.cpu_count()/8), shuffle=True)
    model = fujiModel()
    model = model.to(config.DEVICE)
    optim = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config.ONE_CYCLE_MAX_LR, epochs=config.NUM_EPOCHS, steps_per_epoch=len(train_dl))
    scaler = GradScaler()
    
    criterion = nn.MSELoss(reduction='mean')

    for epoch in tqdm(range(config.NUM_EPOCHS)):
        with tqdm(train_dl, desc='Train', miniters=10) as progress:
            for batch_idx, (source, target) in enumerate(progress):
                optim.zero_grad()
                source = source.to(config.DEVICE)
                target = target.to(config.DEVICE)
                with autocast():
                    img_pred = model(source)
                    # ssim_loss = 1 - ssim(img_pred, target)
                    ssim_loss = criterion(img_pred, target)
                    if torch.isinf(ssim_loss).any() or torch.isnan(ssim_loss).any():
                        print(ssim_loss)
                        print(f'Bad loss: {ssim_loss}, skipping the batch {batch_idx}')
                        del ssim_loss, img_pred
                        gc_collect()
                        continue

                # scaler is needed to prevent "gradient underflow"
                scaler.scale(ssim_loss).backward()
                scaler.step(optim)
                scaler.update()
                # optim.steap()
                logger.log({'loss': (ssim_loss), 'lr': scheduler.get_last_lr()[0]})
                progress.set_description(f'Train loss: {ssim_loss :.02f}')

        scheduler.step()
        valid(model, valid_ds)
    save_model(name, model)
    return model
    


def main():
    sources = read_data(config.SOURCE_DIR)
    targets = read_data(config.TARGET_DIR)  
    train_ds, valid_ds = get_kfold_ds(1, sources, targets)

    # print(train_ds[10][0].shape)
    # ax = plt.subplot(1, 2, 1)
    # ax.imshow(train_ds[10][0])
    # ax.set_title('source')
    # ax.axis('off')
    # ax = plt.subplot(1, 2, 2)
    # ax.set_title('target')
    # ax.imshow(train_ds[10][1])
    # ax.axis('off')
    # plt.show()


    models = []
    for fold in range(config.N_FOLD):
        name = f'fakeji-fold{fold}'
        with wandb.init(project='fakeji', name=name, entity='jimmydut') as run:
            gc_collect()
            models.append(train(train_ds, valid_ds, run, name))


if __name__ == '__main__':
    main()