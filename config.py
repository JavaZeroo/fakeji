from pathlib import Path
import torch
class Config:
    def __init__(self, ):
        self.DATA_DIR = Path('data_gen')
        self.SOURCE_DIR = self.DATA_DIR / 'source'
        self.TARGET_DIR = self.DATA_DIR / 'target'
        self.NUM_EPOCHS = 100
        self.N_FOLD = 5
        self.CROP_RATIO = 12
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.BATCH_SIZE = 64 if self.DEVICE == 'cuda' else 2
        self.ONE_CYCLE_MAX_LR = 0.0001
        self.MODEL_PATH = Path('model')