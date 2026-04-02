"""
DDPM Visualization
Generates publication-quality figures showing:
1. Forward diffusion process (noising trajectory)
2. Reverse diffusion process (denoising trajectory)
3. Generated samples grid
4. Training loss curve
5. Noise schedule comparison (linear vs cosine)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms

from ddpm.noise_scheduler import DDPMScheduler
from ddpm.unet import UNet

# Dark theme for all plots
plt.style.use("dark_background")
ACCENT = "#a78bfa"   # purple
ACCENT2 = "#34d399"  # green
FIG_BG = "#0f0f11"
PANEL_BG = "#1a1a1f"


def fig_forward_process(scheduler: DDPMScheduler, save_path: str = "outputs/forward_process.png"):
    """Visualize q(x_t | x_0) — how images become noise."""
    dataset = datasets.MNIST(
        root="./data", train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x * 2) - 1)])
    )
    x0, _ = dataset[0]
    x0 = x0.unsqueeze(0)

    timesteps = [0, 100, 250, 500, 750, 999]
    fig, axes = plt.subplots(1, len(timesteps), figsize=(14, 3))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Forward Diffusion Process  q(x_t | x₀)", color="white", fontsize=13, y=1.02)

    for ax, t in zip(axes, timesteps):
        t_tensor = torch.tensor([t])
        x_t = scheduler.q_sample(x0, t_tensor)
        img = (x_t.squeeze().numpy() + 1) / 2
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t = {t}", color=ACCENT, fontsize=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(PANEL_BG)
        ax.set_facecolor(PANEL_BG)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"Saved: {save_path}")


def fig_noise_schedules(save_path: str = "outputs/noise_schedules.png"):
    """Compare linear vs cosine noise schedules."""
    linear = DDPMScheduler(num_timesteps=1000, beta_schedule="linear")
    cosine = DDPMScheduler(num_timesteps=1000, beta_schedule="cosine")
    T = 1000
    t = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Noise Schedule Comparison", color="white", fontsize=13)

    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Alpha bar (signal retention)
    axes[0].plot(t, linear.alphas_cumprod.numpy(), color=ACCENT, lw=2, label="Linear")
    axes[0].plot(t, cosine.alphas_cumprod.numpy(), color=ACCENT2, lw=2, label="Cosine")
    axes[0].set_title("ᾱ_t  (signal retained)", color="white")
    axes[0].set_xlabel("timestep t", color="gray")
    axes[0].legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor="white")
    axes[0].grid(alpha=0.1)

    # Beta (noise added per step)
    axes[1].plot(t, linear.betas.numpy(), color=ACCENT, lw=2, label="Linear")
    axes[1].plot(t, cosine.betas.numpy(), color=ACCENT2, lw=2, label="Cosine")
    axes[1].set_title("β_t  (noise per step)", color="white")
    axes[1].set_xlabel("timestep t", color="gray")
    axes[1].legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor="white")
    axes[1].grid(alpha=0.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"Saved: {save_path}")


def fig_training_loss(losses: list, save_path: str = "outputs/training_loss.png"):
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(PANEL_BG)

    epochs = np.arange(1, len(losses) + 1)
    ax.plot(epochs, losses, color=ACCENT, lw=2, marker="o", markersize=5)
    ax.fill_between(epochs, losses, alpha=0.15, color=ACCENT)
    ax.set_title("Training Loss (Simple MSE)", color="white", fontsize=13)
    ax.set_xlabel("Epoch", color="gray")
    ax.set_ylabel("Loss", color="gray")
    ax.tick_params(colors="gray")
    ax.grid(alpha=0.1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"Saved: {save_path}")


def fig_generated_samples(
    model: UNet,
    scheduler: DDPMScheduler,
    device: torch.device,
    n: int = 64,
    save_path: str = "outputs/generated_samples.png",
):
    """Generate a grid of samples via reverse diffusion."""
    model.eval()
    samples = scheduler.p_sample_loop(model, shape=(n, 1, 28, 28), device=device)
    samples = (samples.clamp(-1, 1) + 1) / 2  # back to [0,1]
    samples = samples.cpu().numpy()

    cols = 8
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Generated Samples  (DDPM from scratch)", color="white", fontsize=13, y=1.01)

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"Saved: {save_path}")


def fig_denoising_trajectory(
    model: UNet,
    scheduler: DDPMScheduler,
    device: torch.device,
    save_path: str = "outputs/denoising_trajectory.png",
):
    """Visualize the reverse diffusion trajectory for one sample."""
    model.eval()
    _, intermediates = scheduler.p_sample_loop(
        model, shape=(1, 1, 28, 28), device=device, return_intermediates=True
    )

    fig, axes = plt.subplots(1, len(intermediates), figsize=(len(intermediates) * 1.8, 2.5))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Reverse Diffusion: Noise → Image", color="white", fontsize=13, y=1.02)

    labels = ["Pure noise"] + [f"Step {i}" for i in range(1, len(intermediates) - 1)] + ["Final"]
    for ax, img_tensor, label in zip(axes, intermediates, labels):
        img = (img_tensor[0, 0].cpu().numpy() + 1) / 2
        img = np.clip(img, 0, 1)
        ax.imshow(img, cmap="gray")
        ax.set_title(label, color=ACCENT, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_static_figures():
    """Generate figures that don't require a trained model."""
    os.makedirs("outputs", exist_ok=True)
    scheduler = DDPMScheduler(num_timesteps=1000, beta_schedule="cosine")
    fig_forward_process(scheduler)
    fig_noise_schedules()
    print("Static figures done.")


def generate_model_figures(checkpoint_path: str = "outputs/checkpoint.pt"):
    """Generate figures that require a trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, model_channels=64, channel_mults=(1, 2, 4)).to(device)
    scheduler = DDPMScheduler(num_timesteps=1000, beta_schedule="cosine")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    losses = ckpt.get("losses", [])

    if losses:
        fig_training_loss(losses)
    fig_generated_samples(model, scheduler, device)
    fig_denoising_trajectory(model, scheduler, device)
    print("Model figures done.")


if __name__ == "__main__":
    generate_all_static_figures()
