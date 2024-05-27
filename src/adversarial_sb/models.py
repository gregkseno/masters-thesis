import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import wandb
from tqdm.auto import tqdm

from adversarial_sb.style_gan.models import Generator, Discriminator


def get_simple_model(
    model_dims: list[int], 
    activation: nn.Module = nn.ReLU()
):
    assert len(model_dims) > 1
    modules = []
    for in_, out_ in zip(model_dims[:-2], model_dims[1:-1]):
        modules.extend([nn.Linear(in_, out_), nn.LayerNorm(out_), activation])
    modules.append(nn.Linear(model_dims[-2], model_dims[-1]))
    return nn.Sequential(*modules)


class Conjugate(nn.Module):
    conjugates = {
        'forward_kl': lambda x: torch.exp(x - 1),
        'reverse_kl': lambda x: -1 - torch.log(-x),
        'js': lambda x: -torch.log(1 - torch.exp(x))
    }
    
    def __init__(self, divergence: str = 'forward_kl'):
        super().__init__()
        assert divergence in self.conjugates.keys() 
        self.conjugate = self.conjugates[divergence]

    def forward(self, x):
        return self.conjugate(x)


class Activation(nn.Module):
    activations = {
        'forward_kl': lambda x: x,
        'reverse_kl': lambda x: -torch.exp(-x),
        'js': lambda x: -torch.log(1 + torch.exp(-x))
    }
    
    def __init__(self, divergence: str = 'forward_kl'):
        super().__init__()
        assert divergence in self.activations.keys() 
        self.activation = self.activations[divergence]

    def forward(self, x):
        return self.activation(x)
    

class SimpleConditional(nn.Module):
    loss_titles = {
        'loss_gen': 'Generator loss', 
        'loss_disc_real': 'Critic real loss', 
        'loss_disc_fake': 'Critic fake loss'
    }

    def __init__(
        self, 
        in_dim: int,
        hidden_dims: list[int],
        latent_dim: int = 64
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims

        self.embed = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU())
        self.gen = get_simple_model([2 * latent_dim] + hidden_dims + [in_dim])
        self.latent_dist = lambda num: 2 * torch.rand(size=(num, latent_dim)) - 1

    def forward(self, x: torch.Tensor):
        bs = x.shape[0]
        z_x = torch.cat([self.latent_dist(bs).to(x.device), self.embed(x)], dim=1)
        return self.gen(z_x)
    
    def reset_params(self):
        @torch.no_grad()
        def weight_reset(module: nn.Module):
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                module.reset_parameters()
        self.apply(fn=weight_reset)        

    def init_conditional(
        self, 
        gamma: float,
        epochs: int, 
        loader: DataLoader,
        lr_disc: float,
        lr_gen: float
    ):
        self._device = next(self.gen.parameters()).device
        self._disc = get_simple_model([2 * self.in_dim] + self.hidden_dims + [1], nn.LeakyReLU(0.2)).to(self._device)
        self._real_dist = lambda mean: torch.normal(mean=mean, std=gamma)
        self._criterion = nn.BCELoss()
        self._optim_disc = torch.optim.AdamW(self._disc.parameters(), lr=lr_disc)
        self._optim_gen = torch.optim.AdamW(self.gen.parameters(), lr=lr_gen)
        
        losses = {'loss_gen': [], 'loss_disc_real': [], 'loss_disc_fake': []}

        for epoch in tqdm(range(epochs)):
            total_disc_real, total_disc_fake = 0, 0
            total_gen = 0
            for x, _ in loader:
                x = x.to(self._device)
                loss_disc_real, loss_disc_fake = self._train_step_disc(x)
                loss_gen = self._train_step_gen(x)
                
                total_disc_real += loss_disc_real
                total_disc_fake += loss_disc_fake
                total_gen += loss_gen
            losses['loss_gen'].append(total_gen / len(loader))
            losses['loss_disc_real'].append(total_disc_real / len(loader))
            losses['loss_disc_fake'].append(total_disc_fake / len(loader))
            if epoch % 20 == 0:
                print(f"gen Loss: {losses['loss_gen'][-1]}, disc Real Loss: {losses['loss_disc_real'][-1]}, disc Fake Loss: {losses['loss_disc_fake'][-1]}")
        return losses

    def _train_step_gen(self, x: torch.Tensor):
        bs = x.shape[0]
        self._optim_gen.zero_grad()
        
        y_x = torch.cat([self(x), x], dim=1)
        loss = self._criterion(
            F.sigmoid(self._disc(y_x)), 
            torch.ones((bs, 1), device=self._device)
        )
        loss.backward()
        self._optim_gen.step()
        return loss.item()

    def _train_step_disc(self, x: torch.Tensor):
        bs = x.shape[0]
        self._optim_disc.zero_grad()
        
        y_x = torch.cat([self._real_dist(x), x], dim=1)
        loss_real = self._criterion(
            F.sigmoid(self._disc(y_x)), 
            torch.ones((bs, 1), device=self._device)
        )

        with torch.no_grad():
            y_x = torch.cat([self(x), x], dim=1)
        
        loss_fake = self._criterion(
            F.sigmoid(self._disc(y_x)), 
            torch.zeros((bs, 1), device=self._device)
        )

        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self._optim_disc.step()
        
        return loss_real.item(), loss_fake.item()
    

class SimpleCritic(nn.Module):
    def __init__(
        self, 
        in_dim: int,
        hidden_dims: list[int],
        divergence: str = 'forward_kl'
    ):
        super().__init__()
        self.in_dim = in_dim

        self.net = get_simple_model([2 * in_dim] + hidden_dims + [1],  nn.LeakyReLU(0.2))
        self.activation = Activation(divergence)
        self.conjugate = Conjugate(divergence)

    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        conjugate: bool = False
    ):
        x_y = torch.cat([x, y], dim=1)
        x_y = self.activation(self.net(x_y))
        return x_y if not conjugate else self.conjugate(x_y)
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    

class Critic(nn.Module):
    def __init__(self, divergence: str = 'forward_kl'):
        super().__init__()

        self.net = Discriminator(in_channels=6)
        self.activation = Activation(divergence)
        self.conjugate = Conjugate(divergence)

    def forward(self, x, y, conjugate=False):
        x_y = torch.cat([x, y], dim=1)
        x_y = self.activation(self.net(x_y))
        return x_y if not conjugate else self.conjugate(x_y)    


class Conditional(nn.Module):
    loss_titles = {
        'loss_gen': 'Generator loss', 
        'loss_disc_real': 'Critic real loss', 
        'loss_disc_fake': 'Critic fake loss'
    }

    def __init__(self):
        super().__init__()
        self.gen = Generator(in_channels=6, out_channels=3)
        self.latent_dist = lambda shape: torch.normal(mean=0., std=1., size=shape)

    def forward(self, x):
        z_x = torch.cat([self.latent_dist(x.shape).to(x.device), x], dim=1)
        return self.gen(z_x)
    
    @torch.no_grad
    def _log(
        self,
        losses: dict[str, list[float]],
        dataset: Dataset
    ):
        self.gen.eval()
        x, y = dataset[42]
        x = x.to(self._device).unsqueeze(0)

        y_fake = wandb.Image(self(x).cpu().squeeze().permute(1, 2, 0).detach().numpy(), caption="Fake Photo")
        y = wandb.Image(y.squeeze().permute(1, 2, 0).numpy(), caption="Photo")
        wandb.log({'Photo': y, 'Fake Photo': y_fake})
        wandb.log({key: loss[-1] for key, loss in losses.items()})

        self.gen.train()
    
    def init_conditional(
        self, 
        gamma: float,
        epochs: int, 
        loader: DataLoader,
        lr_disc: float,
        lr_gen: float
    ):
        self._device = next(self.gen.parameters()).device
        self._disc = Critic().to(self._device)
        self._real_dist = lambda mean: torch.normal(mean=mean, std=gamma)
        self._criterion = nn.BCELoss()
        self._optim_disc = torch.optim.Adam(self._disc.parameters(), lr=lr_disc)
        self._optim_gen = torch.optim.Adam(self.gen.parameters(), lr=lr_gen)
        
        losses = {'loss_gen': [], 'loss_disc_real': [], 'loss_disc_fake': []}

        for epoch in tqdm(range(epochs)):
            total_disc_real, total_disc_fake = 0, 0
            total_gen = 0
            for x, _ in loader:
                x = x.to(self._device)
                loss_disc_real, loss_disc_fake = self._train_step_disc(x)
                loss_gen = self._train_step_gen(x)
                
                total_disc_real += loss_disc_real
                total_disc_fake += loss_disc_fake
                total_gen += loss_gen
            losses['loss_gen'].append(total_gen / len(loader))
            losses['loss_disc_real'].append(total_disc_real / len(loader))
            losses['loss_disc_fake'].append(total_disc_fake / len(loader))

            self._log(losses, loader.dataset)
            if epoch % 20 == 0:
                print(f"gen Loss: {losses['loss_gen'][-1]}, disc Real Loss: {losses['loss_disc_real'][-1]}, disc Fake Loss: {losses['loss_disc_fake'][-1]}")
        return losses

    def _train_step_gen(self, x: torch.Tensor):
        bs = x.shape[0]
        self._optim_gen.zero_grad()
        
        loss = self._criterion(
            F.sigmoid(self._disc(self(x), x)), 
            torch.ones((bs, 1), device=self._device)
        )
        loss.backward()
        self._optim_gen.step()
        return loss.item()

    def _train_step_disc(self, x: torch.Tensor):
        bs = x.shape[0]
        self._optim_disc.zero_grad()
        
        loss_real = self._criterion(
            F.sigmoid(self._disc(self._real_dist(x), x)), 
            torch.ones((bs, 1), device=self._device)
        )

        with torch.no_grad():
            y = self(x)
        
        loss_fake = self._criterion(
            F.sigmoid(self._disc(y, x)), 
            torch.zeros((bs, 1), device=self._device)
        )

        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self._optim_disc.step()
        
        return loss_real.item(), loss_fake.item()
        