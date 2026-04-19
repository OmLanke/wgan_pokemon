# Pokemon WGAN-GP

Generates new Pokemon sprites using a Wasserstein GAN with Gradient Penalty (WGAN-GP), implemented in PyTorch. Trained on ~820 official Pokemon sprites at 64×64 resolution.

## Architecture

### Why WGAN-GP over DCGAN

The original project used DCGAN (TensorFlow 1.x). This rewrite upgrades to WGAN-GP for several practical reasons:

| | DCGAN | WGAN-GP |
|---|---|---|
| Loss stability | Prone to mode collapse | Stable by design |
| Training signal | Binary cross-entropy | Wasserstein distance |
| Gradients | Vanish when D is too good | Lipschitz constraint via gradient penalty |
| Output quality | OK | Noticeably sharper |

### Generator

Maps a latent vector **z** ∈ ℝ¹²⁸ to a 3×64×64 RGB image in [-1, 1].

| Layer | Output shape | Notes |
|---|---|---|
| Linear | 512 × 4 × 4 | Projection + reshape |
| ConvTranspose2d | 256 × 8 × 8 | BatchNorm + ReLU |
| ConvTranspose2d | 128 × 16 × 16 | BatchNorm + ReLU |
| ConvTranspose2d | 64 × 32 × 32 | BatchNorm + ReLU |
| ConvTranspose2d | 3 × 64 × 64 | Tanh |

**3.81M parameters**

### Critic

Maps a 3×64×64 image to a scalar Wasserstein score. No BatchNorm (required for WGAN-GP). SpectralNorm on all layers.

| Layer | Output shape | Notes |
|---|---|---|
| Conv2d | 64 × 32 × 32 | SpectralNorm + LeakyReLU(0.2) |
| Conv2d | 128 × 16 × 16 | SpectralNorm + LeakyReLU(0.2) |
| Conv2d | 256 × 8 × 8 | SpectralNorm + LeakyReLU(0.2) |
| Conv2d | 512 × 4 × 4 | SpectralNorm + LeakyReLU(0.2) |
| Linear | 1 (scalar) | SpectralNorm, no sigmoid |

**2.76M parameters**

### Hyperparameters

| Parameter | Value |
|---|---|
| Batch size | 128 |
| Learning rate | 1e-4 (Adam, β₁=0, β₂=0.9) |
| Noise dim (z) | 128 |
| Critic steps per G step | 5 |
| Gradient penalty λ | 10 |
| Weight init | N(0, 0.02) |

---

## Dataset

~820 official Pokemon sprites (64×64 JPG). PNG sprites with transparency are composited onto a white background at load time — no preprocessing step needed.

Sources:
- https://www.kaggle.com/dollarakshay/pokemon-images
- https://veekun.com/dex/downloads

---

## Project structure

```
├── model_wgan.py      # Generator, Critic, gradient_penalty, weights_init
├── train_wgan.py      # Training script
├── generate_wgan.py   # Inference script
├── data/
│   └── pokemon/       # Training images (PNG or JPG)
└── pyproject.toml
```

---

## Setup

### Local (uv)

```bash
# Install uv if you don't have it
curl -Lsf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# For GPU support, install the CUDA-enabled PyTorch build instead:
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Google Colab

PyTorch and torchvision are pre-installed on Colab. Just upload or clone the repo files and mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then upload your Pokemon images to `MyDrive/pokemon_dataset/` and create an empty `MyDrive/pokemon_checkpoints/` folder.

---

## Training

### Local

```bash
uv run python train_wgan.py \
    --data_dir      data/pokemon \
    --checkpoint_dir ./checkpoints \
    --epochs 500
```

### Google Colab (T4 GPU — recommended)

```python
!python train_wgan.py \
    --data_dir       /content/drive/MyDrive/pokemon_dataset \
    --checkpoint_dir /content/drive/MyDrive/pokemon_checkpoints \
    --epochs 500 \
    --compile
```

`--compile` enables `torch.compile` for an extra ~15% speed on PyTorch 2.0+.

**Expected training time on T4:** ~3–4 minutes for 500 epochs with AMP + compile.
Recognisable Pokemon shapes typically appear around epoch 150–200.

### Resuming after a Colab disconnect

Checkpoints are saved to Drive every 50 epochs. Resume from the latest one:

```python
!python train_wgan.py \
    --data_dir       /content/drive/MyDrive/pokemon_dataset \
    --checkpoint_dir /content/drive/MyDrive/pokemon_checkpoints \
    --resume /content/drive/MyDrive/pokemon_checkpoints/checkpoint_epoch0300.pt \
    --epochs 500
```

### Key training flags

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 500 | Total training epochs |
| `--batch_size` | 128 | 128 fits T4 comfortably with AMP |
| `--n_critic` | 5 | Critic updates per generator update |
| `--sample_interval` | 25 | Save a sample grid every N epochs |
| `--save_interval` | 50 | Save a checkpoint every N epochs |
| `--lr` | 1e-4 | Adam learning rate |
| `--compile` | off | Enable torch.compile (~15% speedup) |

Training output looks like:
```
Epoch [  1/500]  C: -12.3451  G:  -3.2109  | 0.4s/epoch  elapsed 0.0m  ETA 3.3m
Epoch [ 25/500]  C:  -8.1234  G:  -5.4321  | 0.4s/epoch  elapsed 0.2m  ETA 3.1m
  → sample  /content/drive/MyDrive/pokemon_checkpoints/samples/sample_epoch_0025.png
```

The Critic loss (Wasserstein distance estimate) should trend toward 0 as training stabilises. The Generator loss should decrease steadily.

---

## Generating images

```bash
# Basic — 64 images, 4× upscale, truncation 0.8
python generate_wgan.py \
    --checkpoint /content/drive/MyDrive/pokemon_checkpoints/checkpoint_final.pt

# Custom
python generate_wgan.py \
    --checkpoint /content/drive/MyDrive/pokemon_checkpoints/checkpoint_final.pt \
    --num_images 100 \
    --truncation 0.7 \
    --upscale 6 \
    --output ./my_pokemon.png
```

### Key generation flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.pt` checkpoint |
| `--num_images` | 64 | Images to generate (rounded to perfect square) |
| `--truncation` | 0.8 | z clipping threshold (see below) |
| `--upscale` | 4 | LANCZOS upscale multiplier on the output grid |
| `--seed` | None | Fix seed for reproducible outputs |
| `--output` | auto | Output PNG path |

### Truncation guide

Truncation clips each z dimension to `[-t, t]`, biasing samples toward the centre of the latent space where the generator is most confident:

| Value | Effect |
|---|---|
| 0.5 | Very clean, low diversity |
| 0.7 | Good balance — recommended starting point |
| 0.8 | More diversity, slight quality drop |
| 1.0 | Full random normal, maximum diversity |

---

## References

- Arjovsky et al. — [Wasserstein GAN](https://arxiv.org/abs/1701.07875) (2017)
- Gulrajani et al. — [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) (2017)
- Radford et al. — [DCGAN](https://arxiv.org/abs/1511.06434) (2015)
- Original DCGAN TensorFlow implementation: [@carpedm20](https://github.com/carpedm20/DCGAN-tensorflow)
