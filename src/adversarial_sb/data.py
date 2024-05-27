import os
import glob

from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

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
            size: tuple = (256, 256),
            unaligned=False,
            mode='train'
        ):
        self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                
        ])

        self.unaligned = unaligned
        self.mode = mode
        if self.mode == 'train':
            self.files_a = sorted(glob.glob(os.path.join(base_path+'/monet_jpg')+'/*.*')[:250])
            self.files_b = sorted(glob.glob(os.path.join(base_path+'/photo_jpg')+'/*.*')[:250])
        elif self.mode == 'test':
            self.files_a = sorted(glob.glob(os.path.join(base_path+'/monet_jpg')+'/*.*')[250:])
            self.files_b = sorted(glob.glob(os.path.join(base_path+'/photo_jpg')+'/*.*')[250:301])

    def  __getitem__(self, index):
        image_a = Image.open(self.files_a[index % len(self.files_a)])
        
        if self.unaligned:
            image_b = Image.open(self.files_b[np.random.randint(0, len(self.files_b)-1)])
        else:
            image_b = Image.open(self.files_b[index % len(self.files_b)])

        if image_a.mode != 'RGB':
            image_a = to_rgb(image_a)
        if image_b.mode != 'RGB':
            image_b = to_rgb(image_b)
            
        x = self.transform(image_a)
        y = self.transform(image_b)
        return x, y
    
    def __len__(self):
        return max(len(self.files_a), len(self.files_b))