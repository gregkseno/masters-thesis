import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm


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
        'loss_disc_real': 'Discriminator real loss', 
        'loss_disc_fake': 'Discriminator fake loss'
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
    

class SimpleDiscriminator(nn.Module):
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
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.identity_map = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs):
        x = inputs.clone().detach()
        out = self.layer(x)
        residual  = self.identity_map(inputs)
        skip = out + residual
        return self.relu(skip)
    

class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels)
        )

    def forward(self, inputs):
        return self.layer(inputs)
    

class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block = ResBlock(in_channels + out_channels, out_channels)
        
    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, divergence: str = 'forward_kl'):
        super().__init__()

        def critic_block(in_filters, out_filters, normalization=True):
            """Returns layers of each critic block"""
            layers: list[nn.Module] = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *critic_block(in_channels, 64, normalization=False),
            *critic_block(64, 128),
            *critic_block(128, 256),
            *critic_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
        self.activation = Activation(divergence)
        self.conjugate = Conjugate(divergence)

    def forward(self, x, y, conjugate=False):
        x_y = torch.cat([x, y], dim=1)
        x_y = self.activation(self.net(x_y))
        return x_y if not conjugate else self.conjugate(x_y)    

class Generator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.encoding_layer1_ = ResBlock(in_channels, 64)
        self.encoding_layer2_ = DownSampleConv(64, 128)
        self.encoding_layer3_ = DownSampleConv(128, 256)
        self.bridge = DownSampleConv(256, 512)
        self.decoding_layer3_ = UpSampleConv(512, 256)
        self.decoding_layer2_ = UpSampleConv(256, 128)
        self.decoding_layer1_ = UpSampleConv(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout2d(0.2)
        
    def forward(self, x):
        ###################### Enocoder #########################
        e1 = self.encoding_layer1_(x)
        e1 = self.dropout(e1)
        e2 = self.encoding_layer2_(e1)
        e2 = self.dropout(e2)
        e3 = self.encoding_layer3_(e2)
        e3 = self.dropout(e3)
        
        ###################### Bridge #########################
        bridge = self.bridge(e3)
        bridge = self.dropout(bridge)
        
        ###################### Decoder #########################
        d3 = self.decoding_layer3_(bridge, e3)
        d2 = self.decoding_layer2_(d3, e2)
        d1 = self.decoding_layer1_(d2, e1)
        
        ###################### Output #########################
        output = self.tanh(self.output(d1))
        return output
    

class Conditional(nn.Module):
    loss_titles = {
        'loss_gen': 'Generator loss', 
        'loss_disc_real': 'Discriminator real loss', 
        'loss_disc_fake': 'Discriminator fake loss'
    }

    def __init__(self):
        super().__init__()
        self.gen = Generator(6, 3)
        self.latent_dist = lambda shape: torch.normal(mean=0., std=1., size=shape)

    def forward(self, x):
        z_x = torch.cat([self.latent_dist(x.shape).to(x.device), x], dim=1)
        return self.gen(z_x)
    
    def init_conditional(
        self, 
        gamma: float,
        epochs: int, 
        loader: DataLoader,
        lr_disc: float,
        lr_gen: float
    ):
        self._device = next(self.gen.parameters()).device
        self._disc = Discriminator(6).to(self._device)
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