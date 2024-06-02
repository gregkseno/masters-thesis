import os
import glob

from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import EMNIST

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.distributions.distribution import Distribution

from sklearn.datasets import make_moons, make_circles


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class MoonCircleDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int):
        self.size = size
        self.circles = torch.tensor(make_circles(size, noise=0.05, factor=0.5)[0], dtype=torch.float)
        self.moons = torch.tensor(make_moons(size, noise=0.05)[0], dtype=torch.float)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int):        
        return self.circles[idx], self.moons[idx]
    


class OneVariateDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int, marg_x: Distribution, marg_y: Distribution):
        self.size = size
        self.x = marg_x.sample(torch.Size([size, 1]))
        self.y = marg_y.sample(torch.Size([size, 1]))

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]   
    

class ImageDataset(Dataset):
    def __init__(
            self, 
            base_path,
            size: tuple = (28, 28),
        ):
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size),
        ])
        self.digits = EMNIST(base_path , split='mnist', download=True, transform=transform)
        self.letters = EMNIST(base_path, split='letters', download=True, transform=transform)
        
    def  __getitem__(self, index):
        x, _ = self.letters[index]
        y, _ = self.digits[index]
        return x, y
    
    def __len__(self):
        return len(self.digits)