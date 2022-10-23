from torch.utils.data import DataLoader, Dataset
from utils import *


class fujiDataset(Dataset):
    def __init__(self, sources_imgs, targets_imgs, transforms):
        self.sources_imgs = sources_imgs
        self.targets_imgs = targets_imgs
        self.transforms = transforms

    def __len__(self):
        return len(self.sources_imgs)
    
    def __getitem__(self, idx):
        source = read_img(self.sources_imgs[idx])
        target = read_img(self.targets_imgs[idx])
        source = self.transforms(source)
        target = self.transforms(target)
        return source, target