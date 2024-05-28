from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
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
        lr_gen: dict[str, float],
        lr_disc: dict[str, float],
        clip: float = 0.1,
        device: str = 'cpu'
    ):
        self.cond_p = cond_p  # x|y
        self.cond_q = cond_q  # y|x
        self.disc_b = disc_b
        self.disc_f = disc_f
        self.clip = clip

        self.optim_gen: dict[str, Optimizer] = {
            'forward': AdamW(cond_q.parameters(), lr=lr_gen['forward']),
            'backward': AdamW(cond_p.parameters(), lr=lr_gen['backward'])
        }
        self.optim_disc: dict[str, Optimizer] = {
            'forward': AdamW(disc_f.parameters(), lr=lr_disc['forward'], weight_decay=0.5),
            'backward': AdamW(disc_b.parameters(), lr=lr_disc['backward'], weight_decay=0.5),
        }
        
        self.device = device

    def _backward_step(self, x: torch.Tensor, y:  torch.Tensor):
        loss_cond = self._train_step_gen(y, self.cond_p, self.disc_b, self.optim_gen, step='backward')
        loss_disc_fixed, loss_disc_training = self._train_step_disc(y, x, self.cond_p, self.cond_q, self.disc_b, self.optim_disc, step='backward')
        return loss_cond, loss_disc_fixed, loss_disc_training

    def _forward_step(self, x: torch.Tensor, y:  torch.Tensor):
        loss_cond = self._train_step_gen(x, self.cond_q, self.disc_f, self.optim_gen, step='forward')
        loss_disc_fixed, loss_disc_training = self._train_step_disc(x, y, self.cond_q, self.cond_p, self.disc_f, self.optim_disc, step='forward')
        return loss_cond, loss_disc_fixed, loss_disc_training

    def _train_step_gen(
        self,
        latent: torch.Tensor,
        cond: nn.Module,
        disc: nn.Module,
        optim: dict[str, Optimizer],
        step: str
    ):
        optim[step].zero_grad()

        # Generate fake samples
        generated = cond(latent)

        # if step == 'backward':
        #     loss = disc(generated, latent).mean()
        # else:
        loss = disc(latent, generated).mean()

        loss.backward()
        clip_grad_norm_(cond.parameters(), self.clip)
        optim[step].step()
        return loss.detach().cpu().item()

    def _train_step_disc(
        self,
        latent: torch.Tensor,
        latent_fixed: torch.Tensor,
        cond: nn.Module,
        cond_fixed: nn.Module,
        disc: nn.Module,
        optim: dict[str, Optimizer],
        step: str
    ):
        optim[step].zero_grad()

        # calc training cond loss
        with torch.no_grad():
            generated = cond(latent)

        # if step == 'backward':
        #     loss_training = disc(generated, latent).mean()
        # else:
        loss_training = disc(latent, generated).mean()

        # calc fixed cond loss
        with torch.no_grad():
            generated = cond_fixed(latent_fixed)

        # if step == 'backward':
        #     loss_fixed = disc(latent_fixed, generated, conjugate=True).mean()
        # else:
        loss_fixed = disc(generated, latent_fixed, conjugate=True).mean()

        loss = loss_fixed - loss_training
        loss.backward()
        clip_grad_norm_(disc.parameters(), self.clip)
        optim[step].step()

        return -loss_fixed.detach().cpu().item(), loss_training.detach().cpu().item()

    def _train_backward(
        self,
        dataloader: DataLoader,
        losses: dict[str, list[float]],
        inner_steps: int = 10
    ):
        # self.cond_p.reset_params()
        # self.disc_b.reset_params()
        for step in range(inner_steps):
            total_loss_cond_p = 0
            total_loss_disc_b_fixed = 0
            total_loss_disc_b_training = 0

            for batch in dataloader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                loss_cond_p, loss_disc_b_fixed, loss_disc_b_training = self._backward_step(x, y)
                total_loss_cond_p += loss_cond_p
                total_loss_disc_b_fixed += loss_disc_b_fixed
                total_loss_disc_b_training += loss_disc_b_training

            losses['cond_p'].append(total_loss_cond_p / len(dataloader))
            losses['disc_b_fixed'].append(total_loss_disc_b_fixed / len(dataloader))
            losses['disc_b_training'].append(total_loss_disc_b_training / len(dataloader))

            if step % 2 == 0:
                print(f'Backward cond_p: {losses["cond_p"][-1]:.5f}, disc_b_fixed: {losses["disc_b_fixed"][-1]:.5f}, disc_b_training: {losses["disc_b_training"][-1]:.5f}')

        return losses
    

    def _train_forward(
        self,
        dataloader: DataLoader,
        losses: dict[str, list[float]],
        inner_steps: int = 10,
    ):
        # self.cond_q.reset_params()
        # self.disc_f.reset_params()
        for step in range(inner_steps):
            total_loss_cond_q = 0
            total_loss_disc_f_fixed = 0
            total_loss_disc_f_training = 0
            
            for batch in dataloader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                loss_cond_q, loss_disc_f_fixed, loss_disc_f_training = self._forward_step(x, y)
                total_loss_cond_q += loss_cond_q
                total_loss_disc_f_fixed += loss_disc_f_fixed
                total_loss_disc_f_training += loss_disc_f_training

            losses['cond_q'].append(total_loss_cond_q / len(dataloader))
            losses['disc_f_fixed'].append(total_loss_disc_f_fixed / len(dataloader))
            losses['disc_f_training'].append(total_loss_disc_f_training / len(dataloader))

            if step % 2 == 0:
                print(f'Forward cond_q: {losses["cond_q"][-1]:.5f}, disc_f_fixed: {losses["disc_f_fixed"][-1]:.5f}, disc_f_training: {losses["disc_f_training"][-1]:.5f}')
        return losses
    
    @torch.no_grad
    def _log(
        self,
        losses: dict[str, list[float]],
        dataset: Dataset
    ):
        self.cond_p.eval()
        self.cond_q.eval()
        x, y = dataset[42]
        x, y = x.to(self.device).unsqueeze(0), y.to(self.device).unsqueeze(0)

        y_fake = wandb.Image(self.cond_q(x).squeeze().cpu().permute(1, 2, 0).detach().numpy(), caption="Fake Photo")
        x = wandb.Image(x.cpu().squeeze().permute(1, 2, 0).detach().numpy(), caption="Monet")
        x_fake = wandb.Image(self.cond_p(y).cpu().squeeze().permute(1, 2, 0).detach().numpy(), caption="Fake Monet")
        y = wandb.Image(y.cpu().squeeze().permute(1, 2, 0).detach().numpy(), caption="Photo")
        
        wandb.log({'Monet': x, 'Fake Photo': y_fake, 'Photo': y, 'Fake Monet': x_fake})
        wandb.log({key: loss[-1] for key, loss in losses.items() if len(loss) != 0})

        torch.save(self.cond_p.state_dict(), '../models/conditional_p.pt')
        torch.save(self.cond_q.state_dict(), '../models/conditional_q.pt')

        self.cond_p.train()
        self.cond_q.train()
    

    def train(
        self,
        epochs: int,
        dataloader: DataLoader,
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

        for epoch in tqdm(range(epochs), desc='Epochs'):
            print(f'======= Epoch {epoch} =======')
            losses = self._train_backward(dataloader, losses, inner_steps)
            if epoch % 2 == 0:
                self._log(losses, dataloader.dataset)
            
            losses = self._train_forward(dataloader, losses, inner_steps)
            if epoch % 2 == 0:
                self._log(losses, dataloader.dataset)
        
        return losses
