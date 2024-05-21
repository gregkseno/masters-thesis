from typing import Any

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from torch import nn
from torch.utils.data import Dataset


def visualize_sb(
    cond_p: nn.Module, 
    cond_q: nn.Module, 
    dataset: Dataset,
    num_samples: int = 1_000,
    x_title: str = 'X',
    y_title: str = 'Y',
):
    num_samples -= 1
    assert num_samples < len(dataset), 'Number of samples is larger than dataset length' # type: ignore
    
    x, y = dataset[:num_samples, :num_samples]
    columns: dict[str, Any] = {'x': 'x', 'y': 'y'} if x.shape[1] == 2 else {'x': 'x'}

    fake_x = pd.DataFrame(cond_p(y).detach().numpy(), columns=list(columns.values()))
    fake_y = pd.DataFrame(cond_q(x).detach().numpy(), columns=list(columns.values()))
    x = pd.DataFrame(x.numpy(), columns=list(columns.values()))
    y = pd.DataFrame(y.numpy(), columns=list(columns.values()))
    
    _, axs = plt.subplots(2, 2, figsize=(12, 8))
    sns.kdeplot(x, fill=True, ax=axs[0][0], **columns)
    axs[0][0].set_title(f'{x_title}')
    sns.kdeplot(fake_y, fill=True, color='r', ax=axs[0][1], **columns)
    axs[0][1].set_title(f'Fake {y_title}')

    sns.kdeplot(y, fill=True, ax=axs[1][0], **columns)
    axs[1][0].set_title(f'{y_title}')
    sns.kdeplot(fake_x, fill=True, color='r', ax=axs[1][1],  **columns)
    axs[1][1].set_title(f'Fake {x_title}')

def visualize_gan(
    gan: nn.Module, 
    dataset: Dataset,
    num_samples: int = 1_000,
):
    num_samples -= 1
    assert num_samples < len(dataset), 'Number of samples is larger than dataset length' # type: ignore
    
    x, y = dataset[:num_samples, :num_samples]
    columns: dict[str, Any] = {'x': 'x', 'y': 'y'} if x.shape[1] == 2 else {'x': 'x'}

    generated = pd.DataFrame(gan(x).detach().numpy(), columns=list(columns.values())).assign(Type='Generated')
    y = pd.DataFrame(y.numpy(), columns=list(columns.values())).assign(Type='Real')

    both = pd.concat([generated, y], axis=0)
    
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