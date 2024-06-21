from typing import Any

import pandas as pd
import numpy as np
import random

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.collections import LineCollection

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
    idx = random.choice(range(len(y))) # type: ignore
    x_ = []
    y_ = []
    for i in range(idx, idx + num_samples):
        x_.append(torch.tensor(x[i]))
        y_.append(torch.tensor(y[i]))
    x = torch.stack(x_, dim=0) # type: ignore
    y = torch.stack(y_, dim=0) # type: ignore

    with torch.no_grad():
        x_fake = cond_p(y).detach().reshape(num_samples, 1, 28, 28).transpose(3, 2)
        y_fake = cond_q(x).detach().reshape(num_samples, 1, 28, 28).transpose(3, 2)

    x = (make_grid(x.reshape(num_samples, 1, 28, 28).transpose(3, 2), nrow=int((num_samples)**0.5)).permute(1, 2, 0) + 1) / 2 # type: ignore
    y = (make_grid(y.reshape(num_samples, 1, 28, 28).transpose(3, 2), nrow=int((num_samples)**0.5)).permute(1, 2, 0) + 1) / 2 # type: ignore

    x_fake = (make_grid(x_fake, nrow=int((num_samples)**0.5)).permute(1, 2, 0) + 1) / 2
    y_fake = (make_grid(y_fake, nrow=int((num_samples)**0.5)).permute(1, 2, 0) + 1) / 2

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

    axs[1][1].set_title(f'Generated {x_title}')
    axs[1][1].imshow(x_fake)
    axs[1][1].axis('off')

    axs[0][1].set_title(f'Generated {y_title}')
    axs[0][1].imshow(y_fake)
    axs[0][1].axis('off')
    
    plt.show()


def visualize_gamma(
    conditionals: list[nn.Module], 
    x: Dataset, 
    y: Dataset,
    titles: list[str],
    num_traslations: int = 10,
    figsize: tuple[int, int] | None = None
):
    
    def alpha_color(color_name: str, alpha=0.2):
        color_rgb = np.asarray(mpl_colors.hex2color(mpl_colors.cnames[color_name]))
        alpha_color_rgb = 1. - (1. - color_rgb) * alpha
        return alpha_color_rgb.tolist()
    
    if figsize is None:
        figsize = (5 * len(conditionals), 5)
    _, axs = plt.subplots(1, len(conditionals), figsize=figsize)
    
    idexes = [28, 30, 1337]
    points_to_transfer = torch.stack([x[index] for index in idexes], dim=0).numpy()

    for i, (cond, title) in enumerate(zip(conditionals, titles)):
        axs[i].set_title(title)
        axs[i].scatter(
            y[:, 0], y[:, 1],
            color=alpha_color('red'), s=48, edgecolors=alpha_color('black'), zorder=0, label=r'Реальные сэмплы $y \sim \pi_T(y)$'
        )
        axs[i].scatter(
            points_to_transfer[:, 0], points_to_transfer[:, 1],
            c="blue", s=48, edgecolors="black", zorder=2, label=r'Исходыне элементы $x \sim \mathcal{N}(0, \mathbb{I}_2)$'
        )

        for _i, index in enumerate(idexes):
            with torch.no_grad():
                y_fake = cond(x[index].repeat(num_traslations, 1)).numpy()
            
            lines_energy = np.concatenate([x[index].repeat(num_traslations, 1).numpy(), y_fake], axis=-1).reshape(-1, 2, 2)
            lc_energy = LineCollection(
                lines_energy, color='black', linewidths=1., alpha=0.4, zorder=1) # type: ignore
            axs[i].add_collection(lc_energy)

            axs[i].scatter(
                y_fake[:, 0], y_fake[:, 1],
                c="green", s=48, edgecolors="black", zorder=2, label=r'Сгенерированные элементы $y \sim q(y | x)$' if _i == 0 else None
            )
            axs[i].legend(fontsize=8)
    