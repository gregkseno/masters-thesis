import torch
from torch import nn

class GAN(nn.Module):
    def __init__(self, n_latent: int = 2):
        super().__init__()
        
        self.gen = self._get_simple_model([n_latent, 32, 32, 32, 2])
        self.disc = nn.Sequential(*self._get_simple_model([2, 32, 32, 32, 1], True), nn.Sigmoid())
        self.noise_fn = lambda num: torch.normal(0, 1, size=(num, self.n_latent))

        self.n_latent = n_latent

    def forward(self, x: torch.Tensor):
        latent_vec = self.noise_fn(x.shape[0]).to(x.device)
        return self.gen(latent_vec)
        
    def sample(self, num: int = 1000, device: str = 'cpu'):
        latent_vec = self.noise_fn(num).to(device)
        self.gen = self.gen.to(device)
        with torch.no_grad():
            samples = self.gen(latent_vec)
        self.gen = self.gen.cpu()
        return samples.cpu().numpy()

    def _get_simple_model(self, hiddens: list[int], disc: bool = False):
        assert len(hiddens) > 1
        activation = nn.LeakyReLU() if disc else nn.ReLU()
        modules = []
        for in_, out_ in zip(hiddens[:-2], hiddens[1:-1]):
            modules.extend([nn.Linear(in_, out_), activation])
        modules.append(nn.Linear(hiddens[-2], hiddens[-1]))
        return nn.Sequential(*modules)