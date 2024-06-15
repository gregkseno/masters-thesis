import torch
from torch import nn

from adversarial_sb.style_gan.models import Generator, Discriminator


def get_simple_model(
    model_dims: list[int], 
    activation: nn.Module = nn.ReLU()
):
    assert len(model_dims) > 1
    modules = []
    for in_, out_ in zip(model_dims[:-2], model_dims[1:-1]):
        modules.extend([nn.Linear(in_, out_), activation])
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
        latent_dim: int = 64,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims

        self.embed = nn.Linear(latent_dim, in_dim)
        self.gen = get_simple_model([2 * in_dim] + hidden_dims + [in_dim])
        self.latent_dist = lambda num: 2 * torch.randn(size=(num, latent_dim))

    def forward(self, x: torch.Tensor):
        bs = x.shape[0]
        z_x = torch.cat([self.embed(self.latent_dist(bs).to(x.device)), x], dim=1)
        return self.gen(z_x)       
    

class SimpleCritic(nn.Module):
    def __init__(
        self, 
        in_dim: int,
        hidden_dims: list[int],
        divergence: str = 'forward_kl'
    ):
        super().__init__()
        self.in_dim = in_dim

        self.net = get_simple_model([2 * in_dim] + hidden_dims + [1], nn.LeakyReLU(0.2))
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
    

class Critic(nn.Module):
    def __init__(
        self, 
        hidden_dims: list[int],
        in_dim: int = 784,
        divergence: str = 'forward_kl'
    ):
        super().__init__()

        self.net = Discriminator(in_channels=2)
        # self.net = get_simple_model([2 * in_dim] + hidden_dims + [1], nn.ELU())
        self.activation = Activation(divergence)
        self.conjugate = Conjugate(divergence)

    def forward(self, x: torch.Tensor, y: torch.Tensor, conjugate=False):
        x_y = torch.cat([x, y], dim=1)
        x_y = self.activation(self.net(x_y))
        return x_y if not conjugate else self.conjugate(x_y)    


class Conditional(nn.Module):
    loss_titles = {
        'loss_gen': 'Generator loss', 
        'loss_disc_real': 'Critic real loss', 
        'loss_disc_fake': 'Critic fake loss'
    }

    def __init__(
            self, 
            hidden_dims: list[int], 
            in_dim: int = 784, 
            latent_dim: int = 100, 
        ):
        super().__init__()

        self.embed = nn.Linear(latent_dim, in_dim)
        # self.gen = nn.Sequential(get_simple_model([in_dim] + hidden_dims + [in_dim]), nn.Tanh())
        self.gen = Generator(in_channels=2, out_channels=1)
        self.latent_dist = lambda num: 2 * torch.rand(size=(num, latent_dim)) - 1


    def forward(self, x: torch.Tensor):
        bs = x.shape[0]
        z = self.embed(self.latent_dist(bs).to(x.device)).reshape(x.shape)
        z_x = torch.cat([z, x], dim=1)
        return self.gen(z_x)     
    
        