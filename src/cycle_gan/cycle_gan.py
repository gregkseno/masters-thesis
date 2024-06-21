import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

class CycleGANTrainer():
    loss_titles = {
        'loss_gen': 'Generator loss', 
        'loss_disc_real': 'Real discriminator loss', 
        'loss_disc_fake': 'Fake discriminator loss'
    }

    def __init__(
        self, 
        gan_x: nn.Module, 
        gan_y: nn.Module, 
        lr_disc: float = 1e-3, 
        lr_gen: float = 2e-4,
        device: str = 'cpu'
    ):
        self.gan_x = gan_x
        self.gan_y = gan_y
        self.criterion = nn.BCELoss()
        self.cycle_criterion = nn.MSELoss()
        self.optim_disc = torch.optim.Adam([{'params': gan_x.disc.parameters()},
                                            {'params': gan_y.disc.parameters()}], lr=lr_disc)
        self.optim_gen = torch.optim.Adam([{'params': gan_x.gen.parameters()},
                                           {'params': gan_y.gen.parameters()}], lr=lr_gen)
        
        self.device = device

    def _train_step_gen(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Train the generator one step and return the loss."""
        self.gan_x.gen.zero_grad()
        self.gan_y.gen.zero_grad()
        
        fake_x = self.gan_x.gen(y)
        fake_y = self.gan_y.gen(x)

        loss_identity_x = self.cycle_criterion(fake_y, x)
        loss_identity_y = self.cycle_criterion(fake_x, y)
        loss_identity = (loss_identity_x + loss_identity_y) / 2                        
        
        cls_x = self.gan_x.disc(fake_x)
        cls_y = self.gan_y.disc(fake_y)
        loss_gan_x = self.criterion(cls_x, torch.ones((x.shape[0], 1)).to(self.device))
        loss_gan_y = self.criterion(cls_y, torch.ones((x.shape[0], 1)).to(self.device))
        loss_gan = (loss_gan_x + loss_gan_y) / 2

        recov_x = self.gan_x.gen(fake_y)
        recov_y = self.gan_y.gen(fake_x)
        loss_cycle_x = self.cycle_criterion(recov_x, x)
        loss_cycle_y = self.cycle_criterion(recov_y, y)
        loss_cycle = (loss_cycle_x + loss_cycle_y) / 2

        loss = loss_gan + 10*loss_cycle + 5*loss_identity
        loss.backward()
        self.optim_gen.step()
        return loss.item()

    def _train_step_disc(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        """Train the discriminator one step and return the losses."""
        self.gan_x.disc.zero_grad()
        self.gan_y.disc.zero_grad()
        
        cls_x = self.gan_x.disc(x)
        cls_y = self.gan_y.disc(y)
        loss_real_x = self.criterion(cls_x, torch.ones((x.shape[0], 1)).to(self.device))
        loss_real_y = self.criterion(cls_y, torch.ones((x.shape[0], 1)).to(self.device))
        loss_real = (loss_real_x + loss_real_y) / 2

        with torch.no_grad():
            fake_x = self.gan_x.gen(y)
            fake_y = self.gan_y.gen(x)
        cls_fake_x = self.gan_x.disc(fake_x)
        cls_fake_y = self.gan_y.disc(fake_y)
        loss_fake_x = self.criterion(cls_fake_x, torch.zeros((x.shape[0], 1)).to(self.device))
        loss_fake_y = self.criterion(cls_fake_y, torch.zeros((x.shape[0], 1)).to(self.device))
        loss_fake = (loss_fake_x + loss_fake_y) / 2
        
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_disc.step()
        return loss_real.item(), loss_fake.item()

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, tuple[float, float]]:
        """Train both networks and return the losses."""
        loss_disc = self._train_step_disc(x, y)
        loss_gen = self._train_step_gen(x, y)
        return loss_gen, loss_disc

    def train(self, epochs: int, x_loader: DataLoader, y_loader: DataLoader) -> dict[str, list[float]]:
        losses = {'loss_gen': [], 'loss_disc_real': [], 'loss_disc_fake': []}
        for epoch in tqdm(range(epochs)):
            avg_loss_gen = 0
            avg_loss_disc_real = 0
            avg_loss_disc_fake = 0
            y_iter = iter(y_loader)
            for x in x_loader:
                try:
                    y = next(y_iter)
                except StopIteration:
                    y_iter = iter(y_loader)
                    y = next(y_iter)
                    
                x = x.to(self.device)
                y = y.to(self.device)

                loss_gen, (loss_disc_real, loss_disc_fake) = self._train_step(x, y)
                avg_loss_gen += loss_gen
                avg_loss_disc_real += loss_disc_real
                avg_loss_disc_fake += loss_disc_fake

            losses['loss_gen'].append(avg_loss_gen / len(x_loader))
            losses['loss_disc_real'].append(avg_loss_disc_real / len(x_loader))
            losses['loss_disc_fake'].append(avg_loss_disc_fake / len(x_loader))
            print(f'gen Loss: {losses["loss_gen"][-1]}, disc Real Loss: {losses["loss_disc_real"][-1]}, disc Fake Loss: {losses["loss_disc_fake"][-1]}')
        return losses