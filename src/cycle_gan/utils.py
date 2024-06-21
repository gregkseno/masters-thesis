import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset



def visualize_cycle_gan(gan_x: nn.Module, gan_y: nn.Module, x: Dataset, y: Dataset):
    with torch.no_grad():
        fake_x = pd.DataFrame(gan_x.gen(y[:]).numpy(), columns=['x', 'y'])
        fake_y = pd.DataFrame(gan_y.gen(x[:]).numpy(), columns=['x', 'y'])
    x = pd.DataFrame(x[:], columns=['x', 'y']) # type: ignore
    y = pd.DataFrame(y[:], columns=['x', 'y']) # type: ignore
    
    _, axs = plt.subplots(2, 2, figsize=(12, 8))
    sns.kdeplot(x, x='x', y='y', fill=True, ax=axs[0][0]) # type: ignore
    sns.kdeplot(fake_y, x='x', y='y', fill=True, color='r', ax=axs[0][1])
    sns.kdeplot(y, x='x', y='y', fill=True, ax=axs[1][0]) # type: ignore
    sns.kdeplot(fake_x, x='x', y='y', fill=True, color='r', ax=axs[1][1])
    plt.show()