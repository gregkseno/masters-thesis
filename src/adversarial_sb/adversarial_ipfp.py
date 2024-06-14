from typing import Callable

import torch
from torch import nn
import torch.monitor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

import wandb
from tqdm.auto import tqdm


class AdversarialIPFPTrainer:

    loss_titles = {
        'cond_p': 'p(x|y) loss',
        'cond_q': 'q(y|x) loss',
        'disc_b_fixed': 'Backward fixed critic loss',
        'disc_f_fixed': 'Forward fixed critic loss',
        'disc_b_training': 'Backward training critic loss',
        'disc_f_training': 'Forward training critic loss'
    }

    def __init__(
        self,
        cond_p: nn.Module,
        cond_q: nn.Module,
        disc_b: nn.Module,
        disc_f: nn.Module,
        gamma: float,
        lr_gen: dict[str, float],
        lr_disc: dict[str, float],
        clip: float = 0.1,
        device: str = 'cpu',
        log_path: str = './'
    ):
        self.cond_p = cond_p  # x|y
        self.cond_q = cond_q  # y|x
        self.disc_b = disc_b
        self.disc_f = disc_f
        self.clip = clip
        self.gamma = gamma

        self.optim_gen_f = Adam(cond_q.parameters(), lr=lr_gen['forward'])
        self.optim_gen_b = Adam(cond_p.parameters(), lr=lr_gen['backward'])
        self.optim_disc_f = Adam(disc_f.parameters(), lr=lr_disc['forward']) # , weight_decay=0.5)
        self.optim_disc_b = Adam(disc_b.parameters(), lr=lr_disc['backward']) # , weight_decay=0.5)
        
        self.device = device
        self.log_path = log_path

    def _backward_step(self, x: torch.Tensor, y:  torch.Tensor, init: bool):
        cond_q = self.cond_q
        if init:
            cond_q = lambda mean: torch.normal(mean, self.gamma)
        loss_disc_fixed, loss_disc_training = self._train_step_disc(y, x, self.cond_p, cond_q, self.disc_b, self.optim_disc_b)
        loss_cond = self._train_step_gen(y, self.cond_p, self.disc_b, self.optim_gen_b)
        return loss_cond, loss_disc_fixed, loss_disc_training

    def _forward_step(self, x: torch.Tensor, y:  torch.Tensor):
        loss_disc_fixed, loss_disc_training = self._train_step_disc(x, y, self.cond_q, self.cond_p, self.disc_f, self.optim_disc_f)
        loss_cond = self._train_step_gen(x, self.cond_q, self.disc_f, self.optim_gen_f)
        return loss_cond, loss_disc_fixed, loss_disc_training

    def _train_step_gen(
        self,
        latent: torch.Tensor,
        cond: nn.Module,
        disc: nn.Module,
        optim: Optimizer,
    ):
        optim.zero_grad()

        # Generate fake samples
        generated = cond(latent)

        loss = disc(latent, generated).mean()

        loss.backward()
        optim.step()
        return loss.detach().cpu().item()

    def _train_step_disc(
        self,
        latent: torch.Tensor,
        latent_fixed: torch.Tensor,
        cond: nn.Module,
        cond_fixed: nn.Module | Callable,
        disc: nn.Module,
        optim: Optimizer,
    ):
        optim.zero_grad()

        # calc training cond loss
        with torch.no_grad():
            generated = cond(latent)

        loss_training = disc(latent, generated).mean()

        # calc fixed cond loss
        with torch.no_grad():
            generated = cond_fixed(latent_fixed)

        loss_fixed = disc(generated, latent_fixed, conjugate=True).mean()

        loss = loss_fixed - loss_training
        loss.backward()
        clip_grad_norm_(disc.parameters(), self.clip)
        optim.step()

        return loss_fixed.detach().cpu().item(), -loss_training.detach().cpu().item()

    def _train_backward(
        self,
        x_loader: DataLoader,
        y_loader: DataLoader,
        losses: dict[str, list[float]],
        inner_steps: int = 10,
        init: bool = False
    ):
        for step in range(inner_steps):
            total_loss_cond_p = 0
            total_loss_disc_b_fixed = 0
            total_loss_disc_b_training = 0

            y_iter = iter(y_loader)
            for x in x_loader:
                try:
                    y = next(y_iter)
                except StopIteration:
                    y_iter = iter(y_loader)
                    y = next(y_iter)
                    
                x = x.to(self.device)
                y = y.to(self.device)

                loss_cond_p, loss_disc_b_fixed, loss_disc_b_training = self._backward_step(x, y, init)
                total_loss_cond_p += loss_cond_p
                total_loss_disc_b_fixed += loss_disc_b_fixed
                total_loss_disc_b_training += loss_disc_b_training

            losses['cond_p'].append(total_loss_cond_p / len(x_loader))
            losses['disc_b_fixed'].append(total_loss_disc_b_fixed / len(x_loader))
            losses['disc_b_training'].append(total_loss_disc_b_training / len(x_loader))

            if step % 2 == 0:
                print(f'Backward cond_p: {losses["cond_p"][-1]:.5f}, disc_b_fixed: {losses["disc_b_fixed"][-1]:.5f}, disc_b_training: {losses["disc_b_training"][-1]:.5f}')

        return losses
    

    def _train_forward(
        self,
        x_loader: DataLoader,
        y_loader: DataLoader,
        losses: dict[str, list[float]],
        inner_steps: int = 10,
    ):
        for step in range(inner_steps):
            total_loss_cond_q = 0
            total_loss_disc_f_fixed = 0
            total_loss_disc_f_training = 0
            
            y_iter = iter(y_loader)
            for x in x_loader:
                try:
                    y = next(y_iter)
                except StopIteration:
                    y_iter = iter(y_loader)
                    y = next(y_iter)
                x = x.to(self.device)
                y = y.to(self.device)

                loss_cond_q, loss_disc_f_fixed, loss_disc_f_training = self._forward_step(x, y)
                total_loss_cond_q += loss_cond_q
                total_loss_disc_f_fixed += loss_disc_f_fixed
                total_loss_disc_f_training += loss_disc_f_training

            losses['cond_q'].append(total_loss_cond_q / len(x_loader))
            losses['disc_f_fixed'].append(total_loss_disc_f_fixed / len(x_loader))
            losses['disc_f_training'].append(total_loss_disc_f_training / len(x_loader))

            if step % 2 == 0:
                print(f'Forward cond_q: {losses["cond_q"][-1]:.5f}, disc_f_fixed: {losses["disc_f_fixed"][-1]:.5f}, disc_f_training: {losses["disc_f_training"][-1]:.5f}')
        return losses
    
    @torch.no_grad
    def _log(
        self,
        losses: dict[str, list[float]],
        x: Dataset,
        y: Dataset,
        step: int
    ):
        self.cond_p.eval()
        self.cond_q.eval()
        x, y = x[42], y[42]
        x, y = x.to(self.device).unsqueeze(0), y.to(self.device).unsqueeze(0) # type: ignore

        y_fake = wandb.Image(self.cond_q(x).squeeze(0).cpu().permute(1, 2, 0).detach().numpy(), caption="Fake Digit")
        x = wandb.Image(x.cpu().squeeze(0).permute(1, 2, 0).detach().numpy(), caption="Letter") # type: ignore
        x_fake = wandb.Image(self.cond_p(y).cpu().squeeze(0).permute(1, 2, 0).detach().numpy(), caption="Fake Letter")
        y = wandb.Image(y.cpu().squeeze(0).permute(1, 2, 0).detach().numpy(), caption="Digit") # type: ignore
        
        wandb.log({'Letter': x, 'Fake Digit': y_fake, 'Digit': y, 'Fake Letter': x_fake}, step=step)
        wandb.log({key: loss[-1] for key, loss in losses.items() if len(loss) != 0}, step=step)

        torch.save(self.cond_p.state_dict(), self.log_path + 'conditional_p.pt')
        torch.save(self.cond_q.state_dict(), self.log_path + 'conditional_q.pt')
        # wandb.save('../models/conditional_p.pt')
        # wandb.save('../models/conditional_q.pt')

        self.cond_p.train()
        self.cond_q.train()
    

    def train(
        self,
        epochs: int,
        x_loader: DataLoader,
        y_loader: DataLoader,
        inner_steps: int = 10
    ) -> dict[str, list[float]]:
        losses = {
            'cond_p': [],
            'cond_q': [],
            'disc_b_fixed': [],
            'disc_f_fixed': [],
            'disc_b_training': [],
            'disc_f_training': []
        }
        init = True

        for epoch in tqdm(range(epochs), desc='Epochs'):
            print(f'======= Epoch {epoch} =======')
            losses = self._train_backward(x_loader, y_loader, losses, inner_steps, init=init)
            if wandb.run is not None:
                self._log(losses, x_loader.dataset, y_loader.dataset, epoch)
            
            losses = self._train_forward(x_loader, y_loader, losses, inner_steps)
            if wandb.run is not None:
                self._log(losses, x_loader.dataset, y_loader.dataset, epoch)

            init = False
        
        return losses