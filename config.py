from pathlib import Path

import torch


class Config:
    def __init__(self, ):
        self.DATA_DIR = Path('data_gen')
        self.SOURCE_DIR = self.DATA_DIR / 'source'
        self.TARGET_DIR = self.DATA_DIR / 'target'
        self.NUM_EPOCHS = 50
        self.N_FOLD = 5
        self.CROP_RATIO = 12
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.BATCH_SIZE = self.get_batch_size()
        # print(f'Batch Size: {self.BATCH_SIZE}')
        self.ONE_CYCLE_MAX_LR = 0.01
        self.MODEL_PATH = Path('model')
        self.check_path(self.MODEL_PATH)

    def get_batch_size(self, ):
        if self.DEVICE == 'cuda':
            self.device_name = torch.cuda.get_device_name(self.DEVICE)
            if self.device_name == 'NVIDIA GeForce RTX 3090':
                BATCH_SIZE = 2
            elif self.device_name == 'NVIDIA GeForce RTX 2060':
                BATCH_SIZE = 2
            else:
                BATCH_SIZE = 4
        else:
            BATCH_SIZE = 2
        return BATCH_SIZE

    def print_config(self):
        print(f'SOURCE_DIR: {self.SOURCE_DIR}')
        print(f'TARGET_DIR: {self.TARGET_DIR}')
        print(f'BATCH_SIZE: {self.BATCH_SIZE}')
        print(f'NUM_EPOCHS: {self.NUM_EPOCHS}')
        print(f'N_FOLD: {self.N_FOLD}')
        print(f'CROP_RATIO: {self.CROP_RATIO}')
        print(f'DEVICE: {self.DEVICE}')
        print(f'ONE_CYCLE_MAX_LR: {self.ONE_CYCLE_MAX_LR}')
    
    def check_path(self, path):
        path.mkdir(exist_ok=True)