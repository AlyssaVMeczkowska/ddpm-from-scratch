"""
DDPM Noise Scheduler
Implements the forward and reverse diffusion processes from
"Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
with cosine schedule from "Improved DDPMs" (Nichol & Dhariwal, 2021)
"""

import torch
import torch.nn.functional as F
import numpy as np


class DDPMScheduler:
    def __init__(self, num_timesteps: int = 1000, beta_schedule: str = "cosine"):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule

        # Compute betas
        if beta_schedule == "linear":
            self.betas = self._linear_schedule()
        elif beta_schedule == "cosine":
            self.betas = self._cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        # Pre-compute diffusion quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Forward process: q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Reverse process: q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _linear_schedule(self):
        """Original DDPM linear schedule."""
        beta_start, beta_end = 1e-4, 0.02
        return torch.linspace(beta_start, beta_end, self.num_timesteps)

    def _cosine_schedule(self):
        """
        Cosine schedule from Nichol & Dhariwal (2021).
        Produces less aggressive noise early in the process.
        """
        s = 0.008
        steps = self.num_timesteps + 1
        t = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((t / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract values from a 1D tensor at timesteps t, reshape to broadcast."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Forward diffusion: sample x_t given x_0 and t.
        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Compute posterior q(x_{t-1} | x_t, x_0) mean and variance."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """Reconstruct x_0 from x_t and predicted noise."""
        sqrt_recip_alphas_cumprod = self._extract(
            1.0 / self.sqrt_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod = self._extract(
            torch.sqrt(1.0 / self.alphas_cumprod - 1), t, x_t.shape
        )
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: int):
        """
        Reverse diffusion step: sample x_{t-1} given x_t.
        p_theta(x_{t-1} | x_t)
        """
        device = x_t.device
        t_tensor = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x_t, t_tensor)

        # Reconstruct x_0
        x_start = self.predict_start_from_noise(x_t, t_tensor, predicted_noise)
        x_start = torch.clamp(x_start, -1.0, 1.0)

        # Get posterior distribution parameters
        model_mean, _, model_log_variance = self.q_posterior_mean_variance(x_start, x_t, t_tensor)

        # No noise at t=0
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        return model_mean + torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape: tuple, device: torch.device, return_intermediates: bool = False):
        """Full reverse diffusion: generate samples from pure noise."""
        img = torch.randn(shape, device=device)
        intermediates = [img.clone()] if return_intermediates else None

        for t in reversed(range(self.num_timesteps)):
            img = self.p_sample(model, img, t)
            if return_intermediates and t % (self.num_timesteps // 10) == 0:
                intermediates.append(img.clone())

        if return_intermediates:
            return img, intermediates
        return img

    def p_losses(self, model, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Training loss: predict noise added at timestep t.
        L_simple = E[||eps - eps_theta(x_t, t)||^2]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)
