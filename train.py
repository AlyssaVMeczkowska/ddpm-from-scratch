"""
DDPM Training Loop
Trains the U-Net denoising network on MNIST.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ddpm.unet import UNet
from ddpm.noise_scheduler import DDPMScheduler


def train(
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 2e-4,
    num_timesteps: int = 1000,
    image_size: int = 28,
    device: str = None,
    save_dir: str = "outputs",
):
    os.makedirs(save_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2) - 1),  # scale to [-1, 1]
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Model + scheduler + optimizer
    model = UNet(in_channels=1, model_channels=64, channel_mults=(1, 2, 4)).to(device)
    scheduler = DDPMScheduler(num_timesteps=num_timesteps, beta_schedule="cosine")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(dataloader)
    )

    print(f"Model parameters: {model.count_parameters():,}")

    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for x, _ in pbar:
            x = x.to(device)
            t = torch.randint(0, num_timesteps, (x.shape[0],), device=device).long()

            loss = scheduler.p_losses(model, x, t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "losses": losses,
        }, os.path.join(save_dir, "checkpoint.pt"))

    return model, scheduler, losses


if __name__ == "__main__":
    model, scheduler, losses = train()
