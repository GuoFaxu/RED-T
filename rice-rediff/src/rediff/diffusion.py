from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ema import EMA
from .utils import extract


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        img_size,
        img_channels: int,
        num_classes: int,
        betas,
        loss_type: str = "l2",
        ema_decay: float = 0.9999,
        ema_start: int = 0,
        ema_update_rate: int = 1,
        pred_type: str = "v",
        use_ema: bool = True,
        loss_weight_type: str = "none",
        min_snr_gamma: float = 5.0,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.ema_model = deepcopy(model)
        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        if pred_type not in ("eps", "v"):
            raise ValueError("pred_type must be 'eps' or 'v'")
        self.pred_type = pred_type

        if loss_type not in ("l1", "l2"):
            raise ValueError("loss_type must be 'l1' or 'l2'")
        self.loss_type = loss_type

        self.use_ema = bool(use_ema)
        self.loss_weight_type = str(loss_weight_type or "none").lower()
        self.min_snr_gamma = float(min_snr_gamma)

        betas = np.array(betas, dtype=np.float32)
        self.num_timesteps = len(betas)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_var = np.clip(posterior_var, 1e-20, None)
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1.0 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))
        self.register_buffer("posterior_sigma", to_torch(np.sqrt(posterior_var)))

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    def ema_to_model(self):
        self.model.load_state_dict(self.ema_model.state_dict(), strict=False)

    def _pred_eps_and_x0(self, x_t, t, out):
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1mab = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        if self.pred_type == "eps":
            eps_pred = out
            x0_pred = (x_t - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)
        else:
            eps_pred = sqrt_1mab * x_t + sqrt_ab * out
            x0_pred = sqrt_ab * x_t - sqrt_1mab * out
        return eps_pred, x0_pred

    def _loss_weight_for_t(self, t, batch_size):
        if self.loss_weight_type != "min_snr":
            return torch.ones(batch_size, device=self.alphas_cumprod.device, dtype=torch.float32)

        alpha_bar = extract(self.alphas_cumprod, t, (batch_size, 1, 1, 1))
        snr = alpha_bar / torch.clamp(1.0 - alpha_bar, min=1e-20)
        snr = snr.view(batch_size)
        gamma = torch.full_like(snr, self.min_snr_gamma)
        return (torch.minimum(snr, gamma) / (snr + 1.0)).to(dtype=torch.float32)

    @torch.no_grad()
    def remove_noise(self, x: torch.Tensor, t: torch.Tensor, use_ema: bool = None) -> torch.Tensor:
        if t.dtype != torch.long:
            t = t.long()
        if use_ema is None:
            use_ema = self.use_ema
        net = self.ema_model if use_ema else self.model
        out = net(x, t)
        eps_pred, _ = self._pred_eps_and_x0(x, t, out)
        return (
            x - extract(self.remove_noise_coeff, t, x.shape) * eps_pred
        ) * extract(self.reciprocal_sqrt_alphas, t, x.shape)

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, use_ema: bool = None) -> torch.Tensor:
        if use_ema is None:
            use_ema = self.use_ema
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.remove_noise(x, t_batch, use_ema=use_ema)
            if t > 0:
                noise = torch.randn_like(x)
                x = x + extract(self.posterior_sigma, t_batch, x.shape) * noise
        return x.cpu().detach()

    def perturb_x(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.long:
            t = t.long()
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def _main_loss_reduce(self, out, target, reduction="mean"):
        if self.loss_type == "l1":
            loss_map = F.l1_loss(out, target, reduction="none")
        else:
            loss_map = F.mse_loss(out, target, reduction="none")
        if reduction == "mean":
            return loss_map.mean()
        if reduction == "batch":
            return loss_map.view(loss_map.shape[0], -1).mean(dim=1)
        return loss_map

    def get_losses(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.long:
            t = t.long()
        noise = torch.randn_like(x)
        x_t = self.perturb_x(x, t, noise)
        out = self.model(x_t, t)
        if self.pred_type == "eps":
            return self._main_loss_reduce(out, noise, reduction="mean")
        v_true = (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
        return self._main_loss_reduce(out, v_true, reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        if (height, width) != tuple(self.img_size):
            raise ValueError("input spatial size does not match diffusion parameters")
        if channels != self.img_channels:
            raise ValueError("input channels does not match diffusion parameters")
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device, dtype=torch.long)
        return self.get_losses(x, t)

    def training_step(self, x: torch.Tensor, aux_weights=None, aux_temp: float = 1.0):
        batch_size, _, height, _ = x.shape
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_t = self.perturb_x(x, t, noise)
        out = self.model(x_t, t)

        if self.pred_type == "eps":
            target = noise
        else:
            target = (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * noise
                - extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
            )

        raw_loss = self._main_loss_reduce(out, target, reduction="batch")
        weight = self._loss_weight_for_t(t, batch_size)
        main_loss = (weight * raw_loss).mean()
        aux_total = torch.zeros((), device=x.device, dtype=main_loss.dtype)
        aux_dict = {"main": main_loss.detach().item()}

        if not aux_weights or sum(float(v) for v in aux_weights.values()) <= 0:
            aux_dict["total"] = main_loss.detach().item()
            return main_loss, aux_dict

        _, x0_pred = self._pred_eps_and_x0(x_t, t, out)
        logits = x0_pred.squeeze(-1)
        p_pred = F.softmax(logits / max(1e-6, aux_temp), dim=1).transpose(1, 2)

        x01 = (x + 1.0) / 2.0
        x01 = torch.clamp(x01, 0.0, 1.0)[:, :, :, 0].transpose(1, 2)
        p_true = torch.zeros_like(x01)
        p_true.scatter_(2, x01.argmax(dim=2, keepdim=True), 1.0)

        if aux_weights.get("mono", 0.0) > 0:
            pred_mono = p_pred.mean(dim=(0, 1))
            true_mono = p_true.mean(dim=(0, 1))
            loss_mono = F.mse_loss(pred_mono, true_mono)
            aux_total = aux_total + aux_weights["mono"] * loss_mono
            aux_dict["mono"] = loss_mono.detach().item()

        if aux_weights.get("mono_global", 0.0) > 0 and hasattr(self, "mono_target"):
            pred_mono = p_pred.mean(dim=(0, 1))
            loss_mono_global = F.mse_loss(pred_mono, self.mono_target)
            aux_total = aux_total + aux_weights["mono_global"] * loss_mono_global
            aux_dict["mono_global"] = loss_mono_global.detach().item()

        if aux_weights.get("markov1", 0.0) > 0 and height >= 2:
            pred_1, pred_2 = p_pred[:, :-1, :], p_pred[:, 1:, :]
            pred_t = torch.einsum("bhi,bhj->ij", pred_1, pred_2)
            pred_t = pred_t / (pred_1.shape[0] * pred_1.shape[1] + 1e-6)
            pred_t = torch.clamp(pred_t, 1e-7, 1.0)
            pred_t = pred_t / pred_t.sum()

            true_1, true_2 = p_true[:, :-1, :], p_true[:, 1:, :]
            true_t = torch.einsum("bhi,bhj->ij", true_1, true_2)
            true_t = true_t / (true_1.shape[0] * true_1.shape[1] + 1e-6)
            true_t = torch.clamp(true_t, 1e-7, 1.0)
            true_t = true_t / true_t.sum()

            loss_markov = F.kl_div(pred_t.log(), true_t, reduction="batchmean")
            loss_markov = loss_markov + F.kl_div(true_t.log(), pred_t, reduction="batchmean")
            loss_markov = 0.5 * loss_markov
            aux_total = aux_total + aux_weights["markov1"] * loss_markov
            aux_dict["markov1"] = loss_markov.detach().item()

        if aux_weights.get("tri", 0.0) > 0 and height >= 3:
            pred_0, pred_1, pred_2 = p_pred[:, :-2, :], p_pred[:, 1:-1, :], p_pred[:, 2:, :]
            pred_tri = torch.einsum("bhi,bhj,bhk->ikj", pred_0, pred_1, pred_2)
            pred_tri = pred_tri / (pred_0.shape[0] * pred_0.shape[1] + 1e-6)
            pred_tri = torch.clamp(pred_tri.reshape(-1), 1e-7, 1.0)
            pred_tri = pred_tri / pred_tri.sum()

            true_0, true_1, true_2 = p_true[:, :-2, :], p_true[:, 1:-1, :], p_true[:, 2:, :]
            true_tri = torch.einsum("bhi,bhj,bhk->ikj", true_0, true_1, true_2)
            true_tri = true_tri / (true_0.shape[0] * true_0.shape[1] + 1e-6)
            true_tri = torch.clamp(true_tri.reshape(-1), 1e-7, 1.0)
            true_tri = true_tri / true_tri.sum()

            loss_tri = F.mse_loss(pred_tri, true_tri)
            aux_total = aux_total + aux_weights["tri"] * loss_tri
            aux_dict["tri"] = loss_tri.detach().item()

        if aux_weights.get("rc", 0.0) > 0:
            rc_idx = torch.tensor([3, 2, 1, 0], device=x.device, dtype=torch.long)
            rc = x0_pred.index_select(1, rc_idx)
            rc = torch.flip(rc, dims=(2,))
            loss_rc = F.mse_loss(x0_pred, rc)
            aux_total = aux_total + aux_weights["rc"] * loss_rc
            aux_dict["rc"] = loss_rc.detach().item()

        total = main_loss + aux_total
        aux_dict["total"] = total.detach().item()
        return total, aux_dict


def generate_cosine_schedule(num_timesteps: int, s: float = 0.008):
    def f(t, total):
        return (np.cos((t / total + s) / (1.0 + s) * np.pi / 2.0)) ** 2

    alphas = []
    f0 = f(0, num_timesteps)
    for t in range(num_timesteps + 1):
        alphas.append(f(t, num_timesteps) / f0)

    betas = []
    for t in range(1, num_timesteps + 1):
        betas.append(min(1.0 - alphas[t] / alphas[t - 1], 0.999))
    return np.array(betas, dtype=np.float32)


def generate_linear_schedule(num_timesteps: int, low: float, high: float):
    return np.linspace(low, high, num_timesteps, dtype=np.float32)
