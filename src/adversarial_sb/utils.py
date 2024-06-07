from typing import Any

import pandas as pd
import numpy as np
import random

import seaborn as sns
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.utils import make_grid


def visualize_sb(
    cond_p: nn.Module, 
    cond_q: nn.Module, 
    x: Dataset,
    y: Dataset,
    num_samples: int = 1_000,
    x_title: str = 'X',
    y_title: str = 'Y',
):
    num_samples -= 1
    assert num_samples < len(x), 'Number of samples is larger than dataset length' # type: ignore
    assert num_samples < len(y), 'Number of samples is larger than dataset length' # type: ignore
    
    x, y = x[:num_samples], y[:num_samples]
    columns: dict[str, Any] = {'x': 'x', 'y': 'y'} if x.shape[1] == 2 else {'x': 'x'} # type: ignore

    fake_x = pd.DataFrame(cond_p(y).detach().numpy(), columns=list(columns.values()))
    fake_y = pd.DataFrame(cond_q(x).detach().numpy(), columns=list(columns.values()))
    x = pd.DataFrame(x.numpy(), columns=list(columns.values())) # type: ignore
    y = pd.DataFrame(y.numpy(), columns=list(columns.values())) # type: ignore
    
    _, axs = plt.subplots(2, 2, figsize=(12, 8))
    sns.kdeplot(x, fill=True, ax=axs[0][0], **columns) # type: ignore
    axs[0][0].set_title(f'{x_title}')
    sns.kdeplot(fake_y, fill=True, color='r', ax=axs[0][1], **columns)
    axs[0][1].set_title(f'Fake {y_title}')

    sns.kdeplot(y, fill=True, ax=axs[1][0], **columns) # type: ignore
    axs[1][0].set_title(f'{y_title}')
    sns.kdeplot(fake_x, fill=True, color='r', ax=axs[1][1],  **columns)
    axs[1][1].set_title(f'Fake {x_title}')


def visualize_gan(
    gan: nn.Module, 
    x: Dataset,
    y: Dataset,
    num_samples: int = 1_000,
):
    num_samples -= 1
    assert num_samples < len(x), 'Number of samples is larger than dataset length' # type: ignore
    assert num_samples < len(y), 'Number of samples is larger than dataset length' # type: ignore
    
    x = x[:num_samples]
    y = y[:num_samples]
    columns: dict[str, Any] = {'x': 'x', 'y': 'y'} if x.shape[1] == 2 else {'x': 'x'} # type: ignore

    generated = pd.DataFrame(gan(x).detach().numpy(), columns=list(columns.values())).assign(Type='Generated')
    y = pd.DataFrame(y.numpy(), columns=list(columns.values())).assign(Type='Real') # type: ignore

    both = pd.concat([generated, y], axis=0) # type: ignore
    
    sns.kdeplot(both, hue='Type', fill=True, **columns)


def visualize_losses(
    losses: dict[str, list[float]],
    loss_titles: dict[str, str],
    inner_steps: int = 1
):
    for loss_key, loss, in losses.items():
        plt.plot(np.arange(0, len(loss) // inner_steps, step=1/inner_steps), loss, label=loss_titles[loss_key])
    plt.grid()
    plt.legend()
    plt.show()

def visualize_sb_images(
    cond_p: nn.Module, 
    cond_q: nn.Module, 
    x: Dataset,
    y: Dataset,
    num_samples: int = 1,
    title: str = 'Samples',
    x_title: str = 'X',
    y_title: str = 'Y',
    figsize: tuple[int, int] | None = None,
):
    idx = random.choice(range(len(x))) # type: ignore
    x, y = x[idx], y[idx]
    with torch.no_grad():
        x_fake = cond_p(y.unsqueeze(0)).detach()
        y_fake = cond_q(x.unsqueeze(0)).detach()

    x = (make_grid(x, nrow=num_samples).permute(1, 2, 0) + 1) / 2
    y = (make_grid(y, nrow=num_samples).permute(1, 2, 0) + 1) / 2

    x_fake = (make_grid(x_fake, nrow=num_samples).permute(1, 2, 0) + 1) / 2
    y_fake = (make_grid(y_fake, nrow=num_samples).permute(1, 2, 0) + 1) / 2

    if figsize is None:
        figsize = (6, 6)

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title)

    axs[0][0].set_title(f'{x_title}')
    axs[0][0].imshow(x)
    axs[0][0].axis('off')

    axs[1][0].set_title(f'{y_title}')
    axs[1][0].imshow(y)
    axs[1][0].axis('off')

    axs[0][1].set_title(f'Generated {y_title}')
    axs[0][1].imshow(x_fake)
    axs[0][1].axis('off')

    axs[1][1].set_title(f'Generated {x_title}')
    axs[1][1].imshow(y_fake)
    axs[1][1].axis('off')
    
    plt.show()


def visualize_gan_images(
    gan: nn.Module, 
    x: Dataset,
    num_samples: int = 1,
    title: str = 'Y',
    figsize: tuple[int, int] | None = None,
):
    idx = random.choice(range(len(x))) # type: ignore
    x = x[idx]
    with torch.no_grad():
        y_fake = gan(x.unsqueeze(0)).detach()

    x = (make_grid(x, nrow=num_samples).permute(1, 2, 0) + 1) / 2
    y_fake = (make_grid(y_fake, nrow=num_samples).permute(1, 2, 0) + 1) / 2

    if figsize is None:
        figsize = (6, 6)

    _, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].set_title(f'{title}')
    axs[0].imshow(x)
    axs[0].axis('off')

    axs[1].set_title(f'Generated {title}')
    axs[1].imshow(y_fake)
    axs[1].axis('off')
    
    plt.show()