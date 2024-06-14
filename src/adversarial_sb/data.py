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
            base_path: str
        ):
        self.letters = pd.read_csv(base_path, header=None)
        self.letters.drop(columns=self.letters.columns[0], axis=1, inplace=True)
        self.letters = ((self.letters - 127.5) / 127.5).astype(np.float32)
        self.letters = torch.tensor(self.letters.values).reshape(-1, 1, 28, 28)
        
    def  __getitem__(self, index):
        return self.letters[index] # .transpose(2, 1)
    
    def __len__(self):
        return len(self.letters)
    
class DigitsDataset(Dataset):
    def __init__(
            self, 
            base_path: str
        ):
        self.digits = pd.read_csv(base_path, header=None)
        self.digits.drop(columns=self.digits.columns[0], axis=1, inplace=True)
        self.digits = ((self.digits - 127.5) / 127.5).astype(np.float32)
        self.digits = torch.tensor(self.digits.values).reshape(-1, 1, 28, 28)
        
    def  __getitem__(self, index):
        return self.digits[index] # .transpose(2, 1)
    
    def __len__(self):
        return len(self.digits)
