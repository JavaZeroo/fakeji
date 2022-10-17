from pathlib import Path

import torch


class Config:
    def __init__(self, ):
        self.DATA_DIR = Path('data')
        self.SOURCE_DIR = self.DATA_DIR / 'source'
        self.TARGET_DIR = self.DATA_DIR / 'target'
        self.NUM_EPOCHS = 100
        self.N_FOLD = 5
        self.CROP_RATIO = 12
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.BATCH_SIZE = self.get_batch_size()
        print(f'Batch Size: {self.BATCH_SIZE}')
        self.ONE_CYCLE_MAX_LR = 0.0001
        self.MODEL_PATH = Path('model')

    def get_batch_size(self, ):
        if self.DEVICE == 'cuda':
            self.device_name = torch.cuda.get_device_name(self.DEVICE)
            if self.device_name == 'NVIDIA GeForce RTX 3090':
                BATCH_SIZE = 2
            elif self.device_name == 'NVIDIA GeForce RTX 2060':
                BATCH_SIZE = 8
            else:
                BATCH_SIZE = 4
        else:
            BATCH_SIZE = 2
        return BATCH_SIZE