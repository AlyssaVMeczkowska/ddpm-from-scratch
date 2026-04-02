# DDPM from Scratch

A clean PyTorch implementation of Denoising Diffusion Probabilistic Models built entirely from first principles. No diffusers library used.

<img width="2085" height="450" alt="forward_process" src="https://github.com/user-attachments/assets/6dbfc0b4-f4e9-4ff3-9546-14154bdf2c6d" />
<img width="1785" height="593" alt="noise_schedules" src="https://github.com/user-attachments/assets/1c402337-07bf-45e3-b715-a5b81b0b44ff" />

## Implementation

| Component | File | Description |
|---|---|---|
| Noise Scheduler | `ddpm/noise_scheduler.py` | Forward/reverse diffusion math, linear + cosine schedules |
| U-Net | `ddpm/unet.py` | Full U-Net with sinusoidal embeddings, residual blocks, self-attention |
| Training | `ddpm/train.py` | Training loop with OneCycleLR, gradient clipping |
| Visualization | `ddpm/visualize.py` | Forward process, denoising trajectory, generated samples |

## Architecture

The denoising network is a U-Net conditioned on timestep $t$ via sinusoidal embeddings:

```
Input (x_t, t)
    │
    ├── Sinusoidal Embedding → MLP → time_emb
    │
    ├── Encoder: ResBlock → ResBlock → Downsample (×3 levels)
    │            [scale-shift conditioning from time_emb]
    │
    ├── Bottleneck: ResBlock → Self-Attention → ResBlock
    │
    └── Decoder: ResBlock (+ skip) → ResBlock → Upsample (×3 levels)
                 → Conv1x1 → predicted noise ε
```

## Math

Forward process adds Gaussian noise over $T$ steps:

$$q(x_t | x_0) = \mathcal{N}\left(\sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t)\mathbf{I}\right)$$

Cosine schedule (Nichol & Dhariwal, 2021):

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

Training objective — predict the noise $\epsilon$:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

Sampling iterates the reverse step $T$ times:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$

## Quick start

```bash
pip install -r requirements.txt

# train on MNIST
python main.py --mode train --epochs 30

# generate visualizations
python main.py --mode visualize

# both
python main.py --mode all
```

## Results

After 30 epochs on MNIST:
- Training loss converges to ~0.02
- Generated digits are sharp and varied
- Full reverse diffusion trajectory visible in `outputs/denoising_trajectory.png`

## References

- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021) — [Improved DDPMs](https://arxiv.org/abs/2102.09672)
- Song et al. (2020) — [Score-Based Generative Modeling](https://arxiv.org/abs/2011.13456)
