# AI Agent Guidelines for Pokemon WGAN

This project implements an SN-GAN (hinge loss + spectral norm) to generate Pokemon sprites at 64×64. It was rewritten from a TensorFlow 1.x DCGAN to PyTorch.

## Project Architecture

### Core Components

| File                | Purpose                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| **train_wgan.py**   | Training loop — hinge loss, AMP, torch.compile, resume, sample saving  |
| **model_wgan.py**   | Generator + Critic architectures, weights_init, gradient_penalty (unused) |
| **generate_wgan.py**| Inference script — truncation, LANCZOS upscale, batch generation        |
| **pyproject.toml**  | uv deps: torch, torchvision, Pillow, numpy (Python 3.12)                |

### Directory Structure (Colab)

- `/content/drive/MyDrive/pokemon_dataset/` — 819 PNG/JPG training images
- `/content/drive/MyDrive/pokemon_checkpoints/` — `.pt` checkpoints + `samples/` grids

## Quick Start (Colab)

### Training from scratch

```bash
uv run train_wgan.py \
    --data_dir /content/drive/MyDrive/pokemon_dataset \
    --checkpoint_dir /content/drive/MyDrive/pokemon_checkpoints \
    --epochs 500
```

### Resume

```bash
uv run train_wgan.py \
    --data_dir /content/drive/MyDrive/pokemon_dataset \
    --checkpoint_dir /content/drive/MyDrive/pokemon_checkpoints \
    --resume /content/drive/MyDrive/pokemon_checkpoints/checkpoint_epoch0150.pt \
    --epochs 500
```

### Generate images

```bash
uv run generate_wgan.py \
    --checkpoint /content/drive/MyDrive/pokemon_checkpoints/checkpoint_final.pt \
    --n 100
```

## Architecture

**Generator** (3.81M params):
- Input: z_dim=128 noise vector
- Linear → reshape 4×4×512 → 4× ConvTranspose2d blocks → 3×64×64 (Tanh)
- BatchNorm2d after each block, ReLU activations (`inplace=False` — required for torch.compile)

**Critic** (2.76M params):
- Input: 3×64×64 image
- 4× Conv2d blocks → Flatten → Linear → scalar (no sigmoid)
- All layers wrapped with `spectral_norm` — enforces Lipschitz constraint
- LeakyReLU(0.2, `inplace=False`) — required for torch.compile

## Training Details

- **Loss**: Hinge loss (not Wasserstein) — no gradient penalty needed
  - Critic: `relu(1 - C(real)).mean() + relu(1 + C(fake)).mean()`
  - Generator: `-C(G(z)).mean()`
- **Optimiser**: Adam, betas=(0.0, 0.9), lr=1e-4
- **AMP**: float16 on both G and C steps
- **n_critic**: 2 (hinge is stable with fewer critic steps)
- **Augmentation**: RandomHorizontalFlip, RandomAffine(±10°, translate 5%), ColorJitter

## Known torch.compile Constraints — READ THIS

These were discovered through painful trial and error. Do not revert them.

1. **Only compile the Generator.** The Critic uses `spectral_norm`, which has weight hooks with inplace ops that break torch.compile's AOT Autograd backend. Compiling the Critic always causes:
   ```
   RuntimeError: one of the variables needed for gradient computation has been
   modified by an inplace operation: [torch.cuda.FloatTensor [1]] is at version
   3; expected version 2
   ```

2. **Never use `inplace=True`** on any activation (ReLU, LeakyReLU) in either model. Inplace ops conflict with torch.compile's graph tracing.

3. **Never use gradient penalty (`create_graph=True`) with torch.compile.** Even with an "uncompiled reference" trick, torch.compile wraps the module in-place so both references point to the compiled graph. GP + compile = always crashes.

4. **Never wrap `torch.autocast` around the critic step when using GP.** Mixing AMP with `create_graph=True` causes the same inplace error in eager mode too.

## Why Hinge Loss (not WGAN-GP)

WGAN-GP requires `create_graph=True` in the gradient penalty, which:
- Cannot be used with `torch.compile` (crashes)
- Is very slow even in eager mode (~20s/epoch on T4 vs ~3s with hinge)
- Is redundant — the Critic already has `spectral_norm` on every layer which enforces the Lipschitz constraint on its own

Hinge loss + spectral norm = SN-GAN, which is stable and fast.

## Checkpoint Format

Checkpoints are saved as:
```python
{
    "epoch": int,
    "generator": G.state_dict(),   # may have "_orig_mod." prefix if saved from compiled model
    "critic": C.state_dict(),
    "opt_G": opt_G.state_dict(),
    "opt_C": opt_C.state_dict(),
    "args": vars(args),
}
```

When loading, always strip `_orig_mod.` prefix from state dict keys:
```python
state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["generator"].items()}
G.load_state_dict(state_dict)
```

This is already handled in both `train_wgan.py` (resume) and `generate_wgan.py`.

## uv / Python Version

- Python 3.12 — required because torchvision CUDA wheels (cu121) have no cp313 builds
- Install CUDA torch on Colab: `uv add torch torchvision --default-index https://download.pytorch.org/whl/cu121`
- DataLoader `num_workers=2` — Colab warns if higher
