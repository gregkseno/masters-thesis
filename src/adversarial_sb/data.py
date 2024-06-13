from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.distributions.distribution import Distribution

from sklearn.datasets import make_moons, make_circles


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class MoonsDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int):
        self.size = size
        self.moons = torch.tensor(make_moons(size, noise=0.05)[0], dtype=torch.float)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int):        
        return self.moons[idx]
    
class CirclesDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int):
        self.size = size
        self.circles = torch.tensor(make_circles(size, noise=0.03, factor=0.3)[0], dtype=torch.float)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int):        
        return self.circles[idx]
    


class OneVariateDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int, marg_x: Distribution, marg_y: Distribution):
        self.size = size
        self.x = marg_x.sample(torch.Size([size, 1]))
        self.y = marg_y.sample(torch.Size([size, 1]))

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]   
    

class LettersDataset(Dataset):
    def __init__(
            self, 
            base_path,
            size: tuple = (28, 28),
        ):
        self.size = size
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size),
                transforms.Normalize((127.5), (127.5)),
        ])
        self.letters = pd.read_csv(base_path, header=None)
        
    def  __getitem__(self, index):
        # letters = self.letters.iloc[index][1:].values.reshape(*self.size).astype(np.float32)
        letters = (self.letters.iloc[index][1:].values.astype(np.float32) - 127.5) / 127.5
        # letters = self.transform(letters)
        return letters # .transpose(2, 1)
    
    def __len__(self):
        return len(self.letters)
    
class DigitsDataset(Dataset):
    def __init__(
            self, 
            base_path,
            size: tuple = (28, 28),
        ):
        self.size = size
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size),
                transforms.Normalize((127.5), (127.5)),
        ])
        self.digits = pd.read_csv(base_path, header=None)
        
    def  __getitem__(self, index):
        # digits = self.digits.iloc[index][1:].values.reshape(*self.size).astype(np.float32)
        digits = (self.digits.iloc[index][1:].values.astype(np.float32) - 127.5) / 127.5
        # digits = self.transform(digits)
        return digits # .transpose(2, 1)
    
    def __len__(self):
        return len(self.digits)
